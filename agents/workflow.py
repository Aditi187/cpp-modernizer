"""
LangGraph Workflow for Code Modernization Engine  # This top-level string briefly explains that this file defines the multi-step modernization workflow.
Implements a 3-node workflow: Analyzer -> Modernizer -> Verifier (with feedback loop)  # This line describes the three main nodes and the feedback loop structure.
"""  # This closing triple quote ends the module-level documentation string.

import sys
import os as os_module
# Force UTF-8 output on Windows to prevent UnicodeEncodeError with emoji characters.
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional
import difflib
import hashlib
import importlib
import json
import os
import re
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os_module.path.join(_PROJECT_ROOT, ".env"),
    override=False,
)  # Load .env defaults but keep explicit runtime env var overrides.

from core.parser import (
    CppParser,
    detect_legacy_patterns,
)
from core.graph import (
    DependencyGraph,
    build_analysis_report,
)
from core.differential_tester import (
    run_differential_test,
    compile_cpp_source,
)
from core.openai_bridge import CPP_MODERNIZATION_SYSTEM_PROMPT, OpenAIBridge
from core.inspect_parser import score_cpp23_compliance
from core.similarity import code_similarity_ratio
from core.logger import get_logger

logger = get_logger(__name__)


def _has_provider_api_key() -> bool:
    return bool(os.environ.get("API_KEY", "").strip())


def _build_provider_bridge(provider: str) -> tuple[Any, str]:
    if provider == "openai":
        original_provider = os.environ.get("WORKFLOW_MODEL_PROVIDER")
        os.environ["WORKFLOW_MODEL_PROVIDER"] = provider
        bridge = OpenAIBridge.from_env(log_fn=logger.info)
        if original_provider is None:
            os.environ.pop("WORKFLOW_MODEL_PROVIDER", None)
        else:
            os.environ["WORKFLOW_MODEL_PROVIDER"] = original_provider
        provider_name = str(getattr(getattr(bridge, "config", None), "provider", provider) or provider)
        if provider_name != provider:
            raise ValueError(f"Requested provider '{provider}' resolved to '{provider_name}'")
        return bridge, provider_name
    raise ValueError(f"Unsupported provider '{provider}'")


def _provider_candidates() -> list[str]:
    preferred_provider = os.environ.get("WORKFLOW_MODEL_PROVIDER", "").strip().lower()
    has_api_key = _has_provider_api_key()
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(provider: str) -> None:
        if provider not in seen:
            seen.add(provider)
            candidates.append(provider)

    if preferred_provider == "openai":
        _add(preferred_provider)

    if has_api_key:
        _add("openai")

    if has_api_key and not candidates:
        _add("openai")

    if not candidates:
        _add("openai")

    return candidates


def _select_model_bridge() -> tuple[Any, str]:
    last_error: Exception | None = None
    for provider in _provider_candidates():
        try:
            bridge, provider_name = _build_provider_bridge(provider)
            logger.info("Selected provider '%s' for workflow", provider_name)
            return bridge, provider_name
        except Exception as exc:
            last_error = exc
            logger.warning("Provider '%s' initialization failed: %r", provider, exc)

    if last_error is not None:
        raise RuntimeError(f"No model provider could be initialized. Last error: {last_error!r}") from last_error
    raise RuntimeError("No model provider could be initialized. Set API_KEY.")


_FALLBACK_BRIDGES: dict[str, Any] = {}


def _get_or_create_bridge(provider: str) -> Any:
    if provider not in _FALLBACK_BRIDGES:
        bridge, _provider_name = _build_provider_bridge(provider)
        _FALLBACK_BRIDGES[provider] = bridge
    return _FALLBACK_BRIDGES[provider]


_MODEL_BRIDGE, _MODEL_PROVIDER = _select_model_bridge()
_MODEL_SYSTEM_PROMPT = "\n\n".join(
    [
        CPP_MODERNIZATION_SYSTEM_PROMPT,
        "MUST FOLLOW STRICTLY:",
        "0) Target C++17 only. Absolutely no C++20/C++23 features.",
        "0.1) Forbidden: concepts, std::ranges, coroutines, std::format, std::span, modules.",
        "1) You MAY refactor function signatures and local data structures to safer modern C++17 constructs when behavior is preserved.",
        "1.1) If a signature changes, update all in-file call sites consistently.",
        "2) Preserve observable behavior.",
        "3) Rewrite only the requested function(s); do not edit unrelated code.",
        "4) Perform deep modernization inside target functions: RAII, std::unique_ptr, std::string, modern loops, nullptr, and safer ownership.",
        "4.1) Replace C-style ownership patterns when safe: malloc/free/new/delete -> smart pointers, std::string, std::vector, std::array, or stack objects.",
        "4.2) Do not keep manual heap management unless required by external ABI or binary interface constraints visible in this file.",
        "4.3) Prefer expressive modern signatures (const refs, string_view where appropriate, return-by-value with move semantics) when behavior remains equivalent.",
        "5) When compiler errors are provided, fix them and retain deep-modernized style.",
        "6) Return ONLY code (no prose, no markdown fences, no lists, no commentary).",
        "Output must compile as valid C++17 in the existing translation unit context.",
        "Few-shot examples:",
        "Example 1\nBefore:\nfor (int i = 0; i < v.size(); ++i) { sum += v[i]; }\nAfter:\nfor (const auto& item : v) { sum += item; }",
        "Example 2\nBefore:\nif (ptr == NULL) return -1;\nAfter:\nif (ptr == nullptr) return -1;",
        "Example 3\nBefore:\nint* p = new int(1);\nuse(*p);\ndelete p;\nAfter:\nauto p = std::make_unique<int>(1);\nuse(*p);",
    ]
)
_MODERNIZER_NODE_SIMILARITY_GUARD = 0.90
_MAX_MODERNIZER_FUNCTION_CHARS = 3000
_MODEL_OUTPUT_ATTEMPTS = 3
_WORKFLOW_ALLOW_SIGNATURE_REFACTOR = os.environ.get("WORKFLOW_ALLOW_SIGNATURE_REFACTOR", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
try:
    _MODEL_MIN_SCORE_DELTA = max(0, min(100, int(os.environ.get("WORKFLOW_MIN_SCORE_DELTA", "3").strip() or "3")))
except ValueError:
    _MODEL_MIN_SCORE_DELTA = 3


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _read_bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _read_bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


_WORKFLOW_BATCH_SIZE = _read_bounded_int_env("WORKFLOW_BATCH_SIZE", 1, 1, 5)
_WORKFLOW_MAX_ATTEMPTS_PER_FUNCTION = _read_bounded_int_env("MAX_ATTEMPTS_PER_FUNCTION", 10, 1, 100)
_WORKFLOW_STAGNANT_SCORE_LIMIT = _read_bounded_int_env("WORKFLOW_STAGNANT_SCORE_LIMIT", 10, 1, 100)
_WORKFLOW_MIN_FINAL_SCORE = _read_bounded_int_env("WORKFLOW_MIN_FINAL_SCORE", 30, 0, 100)


def _workflow_use_llm() -> bool:
    explicit = os.environ.get("WORKFLOW_USE_LLM", "").strip()
    if explicit:
        return _read_bool_env("WORKFLOW_USE_LLM", True)
    return not _read_bool_env("WORKFLOW_DISABLE_LLM", False)


def _workflow_model_temperature() -> float:
    return _read_bounded_float_env("LLM_TEMPERATURE", 0.1, 0.0, 1.0)


def _safe_score_percent(payload: dict[str, Any]) -> int:
    raw = payload.get("percent", 0)
    try:
        return int(float(str(raw)))
    except (TypeError, ValueError):
        return 0


def _is_provider_failure_message(message: str) -> bool:
    upper_message = str(message or "").upper()
    return any(
        token in upper_message
        for token in (
            "PROVIDER_QUOTA_EXHAUSTED",
            "MODEL_UNAVAILABLE",
            "ALL_MODEL_PROVIDERS_FAILED",
            "OLLAMA_REQUEST_FAILED",
            "OLLAMA_CHAT_FAILED",
            "429",
            "402",
            "TIMEOUT",
        )
    )


_STRICT_CPP23_MODE = (
    _read_bool_env("WORKFLOW_STRICT_MODE", False)
    or _read_bool_env("CPP23_STRICT_MODE", False)
)
_STRICT_CPP23_TARGET_PERCENT = _read_bounded_int_env(
    "CPP23_STRICT_TARGET_PERCENT",
    70,
    0,
    100,
)


def call_model(system_prompt: str, user_prompt: str) -> str:
    """Call the configured model bridge and let the bridge own generation tracing."""
    try:
        return _MODEL_BRIDGE.chat_completion(
            system_prompt,
            user_prompt,
            temperature=_workflow_model_temperature(),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as primary_exc:
        primary_message = str(primary_exc)
        should_fallback = _is_provider_failure_message(primary_message)
        if not should_fallback:
            raise

        last_exc: Exception | None = None
        for provider in _provider_candidates():
            if provider == _MODEL_PROVIDER:
                continue
            try:
                fallback_bridge = _get_or_create_bridge(provider)
                logger.warning(
                    "Primary provider '%s' failed (%s); trying fallback provider '%s'.",
                    _MODEL_PROVIDER,
                    primary_message,
                    provider,
                )
                return fallback_bridge.chat_completion(
                    system_prompt,
                    user_prompt,
                    temperature=_workflow_model_temperature(),
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as fallback_exc:
                last_exc = fallback_exc
                logger.warning("Fallback provider '%s' failed: %r", provider, fallback_exc)
                continue
        raise RuntimeError(
            "ALL_MODEL_PROVIDERS_FAILED "
            f"primary={primary_message}"
        ) from (last_exc or primary_exc)


def check_model_health() -> bool:
    """Verify the configured model provider is reachable and ready."""
    is_healthy, message = _MODEL_BRIDGE.check_health()
    status = "✅ " if is_healthy else "❌ "
    log_fn = logger.info if is_healthy else logger.error
    log_fn("[%s] %s%s", _MODEL_PROVIDER, status, message)
    return is_healthy


def _health_failure_guidance() -> str:
    """Return actionable guidance for common provider startup failures."""
    _is_healthy, message = _MODEL_BRIDGE.check_health()
    normalized = str(message or "")
    if "429" in normalized:
        return (
            "Provider rate limit detected (429). Wait exactly 60 seconds and rerun "
            "`python agents/workflow.py`. Avoid repeated retries during that cooldown window."
        )
    if "402" in normalized:
        return (
            "Provider billing/quota issue detected (402). The configured provider rejected the request. "
            "Verify billing/quota for your OpenAI key, then rerun "
            "`python agents/workflow.py`."
        )
    if "UNREACHABLE" in normalized.upper() or "CONNECTION REFUSED" in normalized.upper():
        return (
            "OpenAI API endpoint is not reachable. Check network access and OPENAI_ENDPOINT_BASE, then rerun the workflow."
        )
    return f"Provider startup failed: {normalized}"


def _parse_functions_from_source(source_code: str) -> list[dict[str, Any]]:
    """Parse C++ source from text and return function metadata."""
    try:
        parser = CppParser()
        project_map = parser.parse_string(source_code)
        functions = project_map.get("functions") or {}
        if isinstance(functions, dict):
            return list(functions.values())
        if isinstance(functions, list):
            return [item for item in functions if isinstance(item, dict)]
        return []
    except Exception:
        return []


def _parse_project_map_from_source(source_code: str) -> dict[str, Any]:
    try:
        parser = CppParser()
        project_map = parser.parse_string(source_code)
        if isinstance(project_map, dict):
            return project_map
        return {}
    except Exception:
        return {}


def _index_functions_by_identifier(functions_info: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for fn in functions_info:
        identifier = _function_identifier(fn)
        if identifier and identifier not in index:
            index[identifier] = fn
        stable_key = _function_stable_key(fn)
        if stable_key and stable_key not in index:
            index[stable_key] = fn
    return index


def _function_identifier(function_info: dict[str, Any]) -> str:
    return str(function_info.get("unique_fqn") or function_info.get("fqn") or function_info.get("name") or "")


def _function_stable_key(function_info: dict[str, Any]) -> str:
    """Build a stable function key from declarative identity.

    Format: <fqn-or-name-without-trailing-hash>#<signature_hash>
    """
    base_name = str(function_info.get("fqn") or function_info.get("name") or "")
    if "#" in base_name:
        base_name = base_name.split("#", 1)[0]

    signature_hash = str(function_info.get("signature_hash") or "")
    if not signature_hash:
        signature_text = str(function_info.get("signature") or "")
        normalized_signature = re.sub(r"/\*.*?\*/", " ", signature_text, flags=re.DOTALL)
        normalized_signature = re.sub(r"//[^\n]*", " ", normalized_signature)
        normalized_signature = re.sub(r"\s+", " ", normalized_signature).strip()
        if normalized_signature:
            signature_hash = hashlib.md5(normalized_signature.encode("utf-8")).hexdigest()[:8]
        else:
            parameters = function_info.get("parameters") or []
            type_text = ",".join(str(p.get("type") or "") for p in parameters if isinstance(p, dict))
            type_text = re.sub(r"\s+", " ", type_text).strip()
            signature_hash = hashlib.md5(type_text.encode("utf-8")).hexdigest()[:8] if type_text else ""

    if base_name and signature_hash:
        return f"{base_name}#{signature_hash}"
    if signature_hash:
        return signature_hash
    return base_name


def _function_matches_identifier(function_info: dict[str, Any], identifier: str) -> bool:
    ident = str(identifier or "")
    if not ident:
        return False
    candidates = {
        str(function_info.get("unique_fqn") or ""),
        str(function_info.get("fqn") or ""),
        str(function_info.get("name") or ""),
        _function_stable_key(function_info),
    }
    return ident in candidates


def _find_function_by_identifier(functions_info: list[dict[str, Any]], identifier: str) -> dict[str, Any]:
    ident = str(identifier or "")
    if not ident:
        return {}
    for fn in functions_info:
        if _function_matches_identifier(fn, ident):
            return fn
    return {}


def _build_callers_map(dependency_map: dict[str, list[str]]) -> dict[str, list[str]]:
    callers_map: dict[str, list[str]] = {name: [] for name in dependency_map.keys()}
    for caller_name, callees in dependency_map.items():
        for callee_name in callees:
            if callee_name in callers_map and caller_name not in callers_map[callee_name]:
                callers_map[callee_name].append(caller_name)
    return callers_map


# Regex to match model special tokens like <｜begin▁of▁sentence｜> that can leak
# into generated code. Handles both fullwidth (U+FF5C ｜) and ASCII pipe (|).
_MODEL_SPECIAL_TOKEN_RE = re.compile(r"<[｜|][^>]{1,40}[｜|]>")


def _clean_model_code_block(text: str) -> str:
    """Extract code from the model's response, stripping markdown fences and prose.

    If the response contains a fenced code block (```), only the content inside
    the *first* such block is returned.  This prevents explanatory text that the
    model sometimes emits before or after the code from polluting the source.
    """
    # Try to extract the first fenced code block.
    fence_match = re.search(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Fallback: strip any stray triple-backtick lines.
        cleaned = re.sub(r"```(?:[^\n]*)\n?", "", text)
        cleaned = cleaned.strip()
    # Strip model special tokens (e.g. <｜begin▁of▁sentence｜>) that the
    # model sometimes injects into its output.
    cleaned = _MODEL_SPECIAL_TOKEN_RE.sub("", cleaned)
    return cleaned


def _looks_like_prose_response(raw_text: str, cleaned_text: str) -> bool:
    raw_without_comments = re.sub(r"/\*.*?\*/", "", raw_text, flags=re.DOTALL)
    raw_without_comments = re.sub(r"//[^\n]*", "", raw_without_comments)
    cleaned_without_comments = re.sub(r"/\*.*?\*/", "", cleaned_text, flags=re.DOTALL)
    cleaned_without_comments = re.sub(r"//[^\n]*", "", cleaned_without_comments)
    lower_raw = raw_without_comments.lower()
    lower_cleaned = cleaned_without_comments.lower()
    prose_markers = (
        "to modernize this function",
        "we can follow these steps",
        "here is",
        "here's",
        "below is",
        "replace the line",
        "output only valid c++ code",
        "example transformation",
    )
    if any(marker in lower_raw for marker in prose_markers):
        return True
    if re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", cleaned_without_comments):
        return True
    if re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", raw_text) and "```" not in raw_text:
        return True
    if "{" not in cleaned_without_comments or "}" not in cleaned_without_comments:
        return True
    if re.search(r"\b(?:steps?|explanation|summary|replace the line)\b", lower_cleaned):
        return True
    return False


def _build_retry_prompt(base_prompt: str, rejection_reason: str, rejected_response: str) -> str:
    response_preview = rejected_response.strip()[:500]
    return (
        base_prompt
        + "\n\nRETRY MODE (STRICT): The previous response was rejected."
        + f"\nReason: {rejection_reason}."
        + "\nReturn ONLY one full function definition for the same target function."
        + (
            "\nSignature refactors are allowed when they improve safety and modern C++17 usage; if changed, keep call-site compatibility within this file."
            if _WORKFLOW_ALLOW_SIGNATURE_REFACTOR
            else "\nKeep the exact function signature unchanged."
        )
        + "\nIf compiler errors are present in the prompt, enter FIX-ONLY mode: resolve only those errors with the smallest possible edits."
        + "\nDo not introduce unrelated refactors, renames, reordering, or structural changes."
        + "\nDo not include explanations, lists, markdown fences, comments describing steps, or placeholders."
        + "\nStart with the function signature and end with its matching closing brace."
        + f"\nRejected response preview:\n{response_preview}"
    )


def _extract_json_object_text(raw_text: str) -> str:
    cleaned = _clean_model_code_block(raw_text).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return match.group(0)
    return ""


def _batch_prompt_for_functions(
    batch_entries: list[dict[str, Any]],
    dependency_map: dict[str, list[str]],
    previous_error_log: str,
) -> str:
    sections: list[str] = [
        "Modernize the following C++ functions while preserving behavior and exact signatures.",
        "Return STRICT JSON only in this exact shape:",
        "{\"functions\":[{\"id\":\"<identifier>\",\"code\":\"<full_function_code>\"}]}",
        "Rules: one full function definition per id, no markdown fences, no prose.",
        "Rules: minimal/surgical diffs only, no unrelated changes outside each target function.",
        "Rules: when compiler feedback is present, fix only those errors with minimal edits.",
    ]

    if previous_error_log:
        sections.append("Recent compiler feedback:\n" + previous_error_log)

    for entry in batch_entries:
        function_name = str(entry.get("name") or "")
        function_id = str(entry.get("identifier") or "")
        called_by = ", ".join(sorted(dependency_map.get(function_name, []))) or "none"
        function_code = str(entry.get("source") or "")
        sections.append(
            "\n".join(
                [
                    f"ID: {function_id}",
                    f"Function: {function_name}",
                    f"Direct callers: {called_by}",
                    "Source:",
                    "```cpp",
                    function_code,
                    "```",
                ]
            )
        )
    return "\n\n".join(sections)


def _collect_batch_entries(
    functions_info: list[dict[str, Any]],
    functions_index: dict[str, dict[str, Any]],
    modernization_order: list[str],
    current_index: int,
    source_to_improve: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 1:
        return []

    source_bytes = source_to_improve.encode("utf-8")
    collected: list[dict[str, Any]] = []
    scan_index = current_index
    while scan_index < len(modernization_order) and len(collected) < batch_size:
        identifier = str(modernization_order[scan_index] or "")
        if not identifier:
            scan_index += 1
            continue

        fn_info = functions_index.get(identifier, {})
        if not fn_info:
            for fn in functions_info:
                if _function_matches_identifier(fn, identifier):
                    fn_info = fn
                    break
        if not fn_info:
            scan_index += 1
            continue

        start_byte = fn_info.get("start_byte")
        end_byte = fn_info.get("end_byte")
        if not (
            isinstance(start_byte, int)
            and isinstance(end_byte, int)
            and 0 <= start_byte <= end_byte <= len(source_bytes)
        ):
            scan_index += 1
            continue

        function_source = source_bytes[start_byte:end_byte].decode("utf-8", errors="strict")
        if len(function_source) > _MAX_MODERNIZER_FUNCTION_CHARS:
            scan_index += 1
            continue

        parameters = fn_info.get("parameters") or []
        param_count = len(parameters) if isinstance(parameters, list) else -1
        collected.append(
            {
                "identifier": identifier,
                "name": str(fn_info.get("name") or ""),
                "start_byte": start_byte,
                "end_byte": end_byte,
                "param_count": param_count,
                "source": function_source,
            }
        )
        scan_index += 1
    return collected


def _apply_batched_replacements(
    source_to_improve: str,
    batch_entries: list[dict[str, Any]],
    candidate_by_id: dict[str, str],
) -> tuple[str, list[str]]:
    updated_source = source_to_improve
    applied_ids: list[str] = []
    for entry in sorted(batch_entries, key=lambda item: int(item["start_byte"]), reverse=True):
        identifier = str(entry.get("identifier") or "")
        function_name = str(entry.get("name") or "")
        original_param_count = int(entry.get("param_count", -1))
        candidate = str(candidate_by_id.get(identifier, "") or "").strip()
        if not candidate:
            continue

        validation_error = _validate_single_function_candidate(candidate, function_name, original_param_count)
        if validation_error:
            continue

        updated_source = _replace_function_by_span(
            updated_source,
            int(entry["start_byte"]),
            int(entry["end_byte"]),
            candidate,
        )
        updated_source = IncludeManager().update_file_includes(updated_source, candidate)
        applied_ids.append(identifier)
    return updated_source, applied_ids


def _split_function_signature_and_body(function_source: str) -> tuple[str, str]:
    """Split a full function definition into signature prefix and body text."""
    first_brace = function_source.find("{")
    last_brace = function_source.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return function_source, ""
    signature = function_source[: first_brace + 1]
    body = function_source[first_brace + 1:last_brace]
    return signature, body


def _transform_char_ptr_literal_declarations(body: str) -> tuple[str, int]:
    """Convert local `char* name = \"...\";` into `std::string name = \"...\";`."""
    pattern = re.compile(
        r"(^|\n)([ \t]*)char\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"[^\"\\]*(?:\\.[^\"\\]*)*\")\s*;"
    )

    replacements = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        prefix, indent, name, literal = match.groups()
        return f"{prefix}{indent}std::string {name} = {literal};"

    return pattern.sub(_repl, body), replacements


def _transform_null_macro_to_nullptr(body: str) -> tuple[str, int]:
    """Convert legacy NULL macro tokens to nullptr in function bodies."""
    pattern = re.compile(r"\bNULL\b")
    replacements = len(pattern.findall(body))
    return pattern.sub("nullptr", body), replacements


def _transform_malloc_to_unique_ptr(body: str) -> tuple[str, int]:
    """Convert local malloc + free pairs to std::make_unique for simple local buffers."""
    pattern = re.compile(
        r"(^|\n)([ \t]*)([A-Za-z_][A-Za-z0-9_:<>]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\([^)]+\)\s*)?malloc\(([^\)]*)\)\s*;"
    )

    replacements = 0
    freed_vars: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        prefix, indent, base_type, var_name, malloc_args = match.groups()
        freed_vars.append(var_name)

        size_pattern = re.compile(
            r"^\s*sizeof\s*\(\s*" + re.escape(base_type) + r"\s*\)\s*\*\s*(.+)$"
        )
        size_match = size_pattern.match(malloc_args.strip())
        if size_match:
            count_expr = size_match.group(1).strip()
            replacement = f"{indent}auto {var_name} = std::make_unique<{base_type}[]>({count_expr});"
        else:
            replacement = f"{indent}auto {var_name} = std::make_unique<{base_type}>();"
        return f"{prefix}{replacement}"

    updated = pattern.sub(_repl, body)
    for var_name in freed_vars:
        updated = re.sub(
            r"(^|\n)[ \t]*free\s*\(\s*" + re.escape(var_name) + r"\s*\)\s*;",
            r"\1",
            updated,
        )
    return updated, replacements


def _transform_index_for_to_range_loop(body: str) -> tuple[str, int]:
    """Convert simple index-based loops over one container into range-based loops."""
    loop_pattern = re.compile(
        r"for\s*\(\s*int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\s*;\s*\1\s*<\s*([^;\)]+)\s*;\s*(?:\+\+\1|\1\+\+)\s*\)\s*\{([\s\S]*?)\}",
        re.MULTILINE,
    )

    replacements = 0

    def _loop_repl(match: re.Match[str]) -> str:
        nonlocal replacements
        index_var = match.group(1)
        limit_expr = match.group(2).strip()
        loop_body = match.group(3)

        # Support common patterns: i < count or i < container.size().
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", limit_expr):
            candidates = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*" + re.escape(index_var) + r"\s*\]", loop_body)
            if len(set(candidates)) != 1:
                return match.group(0)
            container_name = candidates[0]
        else:
            size_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*size\s*\(\s*\)", limit_expr)
            if not size_match:
                return match.group(0)
            container_name = size_match.group(1)

        loop_body_rewritten = re.sub(
            r"\b" + re.escape(container_name) + r"\s*\[\s*" + re.escape(index_var) + r"\s*\]",
            "item",
            loop_body,
        )
        replacements += 1
        return f"for (auto& item : {container_name}) {{{loop_body_rewritten}}}"

    return loop_pattern.sub(_loop_repl, body), replacements


def _apply_rule_based_function_transforms(function_source: str) -> tuple[str, list[str]]:
    """Apply deterministic C++ modernization rewrites before calling the LLM."""
    signature, body = _split_function_signature_and_body(function_source)
    if not body:
        return function_source, []

    notes: list[str] = []

    body, malloc_count = _transform_malloc_to_unique_ptr(body)
    if malloc_count:
        notes.append(f"malloc->unique_ptr: {malloc_count}")

    body, loop_count = _transform_index_for_to_range_loop(body)
    if loop_count:
        notes.append(f"index-loop->range-loop: {loop_count}")

    body, char_ptr_count = _transform_char_ptr_literal_declarations(body)
    if char_ptr_count:
        notes.append(f"char*-literal->std::string: {char_ptr_count}")

    body, null_count = _transform_null_macro_to_nullptr(body)
    if null_count:
        notes.append(f"NULL->nullptr: {null_count}")

    first_brace = signature
    transformed = first_brace + body + "}"
    return transformed, notes


def _extract_error_line_numbers(compiler_text: str) -> list[int]:
    """Extract likely source line numbers from compiler stderr/stdout text."""

    line_numbers: set[int] = set()
    for match in re.finditer(r":(\d+):(\d+)?:", compiler_text):
        try:
            line_numbers.add(int(match.group(1)))
        except ValueError:
            continue
    return sorted(line_numbers)


def _get_code_snippet_by_line(code_text: str, line_number: int, radius: int = 2) -> str:
    lines = code_text.splitlines()
    if line_number <= 0 or line_number > len(lines):
        return ""

    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    snippet_lines: list[str] = []
    for idx in range(start, end + 1):
        snippet_lines.append(f"{idx:4d}: {lines[idx - 1]}")
    return "\n".join(snippet_lines)


def _replace_function_by_span(
    source_code: str,
    start_byte: int,
    end_byte: int,
    replacement: str,
) -> str:
    # Guard against corrupting the file when the model returns whole-file output.
    if re.search(r"(?m)^\s*#\s*include\b", replacement):
        raise ValueError("Replacement contains '#include' directives; expected a single function body.")
    parsed_replacement_functions = _parse_functions_from_source(replacement)
    if len(parsed_replacement_functions) != 1:
        raise ValueError(
            f"Replacement must contain exactly one function definition, got {len(parsed_replacement_functions)}."
        )

    source_bytes = source_code.encode("utf-8")
    prefix = source_bytes[:start_byte]
    suffix = source_bytes[end_byte:]
    replacement_bytes = replacement.encode("utf-8")
    updated_bytes = prefix + replacement_bytes + suffix
    try:
        return updated_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Replacement produced non-UTF-8 output. Ensure the input file is valid UTF-8 before modernization."
        ) from exc


def _validate_single_function_candidate(
    candidate_code: str,
    expected_function_name: str,
    original_param_count: int,
) -> str | None:
    """Return None when candidate is valid, else return rejection reason."""
    if re.search(r"(?m)^\s*#\s*include\b", candidate_code):
        return "model returned whole-file output with include directives"

    parsed_map = _parse_project_map_from_source(candidate_code)
    parsed_functions_obj = parsed_map.get("functions") or {}
    if isinstance(parsed_functions_obj, dict):
        parsed_functions = list(parsed_functions_obj.values())
    else:
        parsed_functions = [item for item in parsed_functions_obj if isinstance(item, dict)] if isinstance(parsed_functions_obj, list) else []
    if len(parsed_functions) != 1:
        return f"model output must contain exactly one function definition (got {len(parsed_functions)})"

    candidate_name = str(parsed_functions[0].get("name") or "")
    if candidate_name != expected_function_name:
        return (
            "model output function name mismatch "
            f"(expected '{expected_function_name}', got '{candidate_name or 'unknown'}')"
        )

    if not _WORKFLOW_ALLOW_SIGNATURE_REFACTOR:
        candidate_params = parsed_functions[0].get("parameters") or []
        candidate_param_count = len(candidate_params) if isinstance(candidate_params, list) else -1
        if (
            original_param_count >= 0
            and candidate_param_count >= 0
            and original_param_count != candidate_param_count
        ):
            return (
                "model changed parameter compatibility "
                f"(expected {original_param_count} params, got {candidate_param_count})"
            )

    extra_types = parsed_map.get("types") or []
    extra_globals = parsed_map.get("global_variables") or []
    if extra_types or extra_globals:
        return "model output contains extra top-level declarations"

    # Fast heuristic checks for common LLM output mistakes before compile stage.
    body_text = str(parsed_functions[0].get("body") or "")
    if not body_text.strip():
        return "model output has empty function body"

    stripped_lines = [line.strip() for line in candidate_code.splitlines() if line.strip()]
    for line in stripped_lines:
        if line.startswith(("//", "/*", "*", "#", "return ")):
            continue
        if line.endswith(("{", "}", ";", ":")):
            continue
        if line.startswith(("if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "else", "do")):
            continue
        if re.search(r"\bstd::(cout|cerr|clog)\b", line) and "<<" in line and not line.endswith(";"):
            return "likely missing semicolon after stream output statement"

    has_iostream_include = bool(re.search(r"(?m)^\s*#\s*include\s*<iostream>\s*$", candidate_code))
    uses_iostream_streams = bool(re.search(r"\bstd::(cout|cerr|clog)\b", candidate_code))
    if uses_iostream_streams and "#include" in candidate_code and not has_iostream_include:
        return "uses std::cout/cerr/clog but missing #include <iostream>"

    return None


def _extract_function_text_from_code(source_code: str, function_identifier: str) -> str:
    functions_info = _parse_functions_from_source(source_code)
    for function_info in functions_info:
        if not _function_matches_identifier(function_info, function_identifier):
            continue
        start_byte = function_info.get("start_byte")
        end_byte = function_info.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int):
            code_bytes = source_code.encode("utf-8")
            return code_bytes[start_byte:end_byte].decode("utf-8", errors="strict")
    return ""


def _remove_functions_by_name(source_code: str, function_names: set[str]) -> str:
    """Remove full function definitions by name using parser byte spans."""
    if not function_names:
        return source_code

    source_bytes = source_code.encode("utf-8")
    functions_info = _parse_functions_from_source(source_code)
    spans_to_remove: list[tuple[int, int]] = []

    for fn in functions_info:
        fn_name = str(fn.get("name") or "")
        if fn_name not in function_names:
            continue
        start_byte = fn.get("start_byte")
        end_byte = fn.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int) and 0 <= start_byte <= end_byte <= len(source_bytes):
            spans_to_remove.append((start_byte, end_byte))

    if not spans_to_remove:
        return source_code

    spans_to_remove.sort(key=lambda pair: pair[0])
    new_chunks: list[bytes] = []
    cursor = 0
    for start_byte, end_byte in spans_to_remove:
        if start_byte < cursor:
            continue
        if start_byte > cursor:
            new_chunks.append(source_bytes[cursor:start_byte])
        cursor = end_byte

    if cursor < len(source_bytes):
        new_chunks.append(source_bytes[cursor:])

    return b"".join(new_chunks).decode("utf-8", errors="strict")


class IncludeManager:
    """Maps modern C++ feature patterns to the headers they require and injects
    missing ``#include`` directives into source files.

    Usage::

        manager = IncludeManager()
        updated_file = manager.update_file_includes(full_file_content, new_function_code)
    """

    # (regex_pattern, header) — each pattern is tested independently so one
    # header can be triggered by multiple patterns (e.g. both std::variant and
    # std::visit map to <variant>).
    _FEATURE_HEADERS: list[tuple[str, str]] = [
        # C++23
        (r"\bstd::print\b",                "<print>"),
        (r"\bstd::println\b",              "<print>"),
        # C++20 — formatting
        (r"\bstd::format\b",               "<format>"),
        (r"\bstd::vformat\b",              "<format>"),
        # C++20 — ranges / views
        (r"\bstd::ranges::",               "<ranges>"),
        (r"\bstd::views::",                "<ranges>"),
        # C++20 — span
        (r"\bstd::span\b",                 "<span>"),
        # C++20 — bit operations
        (r"\bstd::popcount\b",             "<bit>"),
        (r"\bstd::countl_zero\b",          "<bit>"),
        (r"\bstd::bit_cast\b",             "<bit>"),
        # C++20 — math constants
        (r"\bstd::numbers::",              "<numbers>"),
        # C++20 — concepts
        (r"\bstd::integral\b",             "<concepts>"),
        (r"\bstd::floating_point\b",       "<concepts>"),
        (r"\bstd::same_as\b",              "<concepts>"),
        (r"\bstd::convertible_to\b",       "<concepts>"),
        (r"\bstd::invocable\b",            "<concepts>"),
        # C++17 — string_view, optional, variant, any, filesystem, execution
        (r"\bstd::string_view\b",          "<string_view>"),
        (r"\bstd::optional\b",             "<optional>"),
        (r"\bstd::variant\b",              "<variant>"),
        (r"\bstd::visit\b",                "<variant>"),
        (r"\bstd::any\b",                  "<any>"),
        (r"\bstd::filesystem::",           "<filesystem>"),
        (r"\bstd::execution::",            "<execution>"),
        # Smart pointers / memory management
        (r"\bstd::unique_ptr\b",           "<memory>"),
        (r"\bstd::shared_ptr\b",           "<memory>"),
        (r"\bstd::weak_ptr\b",             "<memory>"),
        (r"\bstd::make_unique\b",          "<memory>"),
        (r"\bstd::make_shared\b",          "<memory>"),
        # Containers
        (r"\bstd::vector\b",               "<vector>"),
        (r"\bstd::array\b",                "<array>"),
        (r"\bstd::map\b",                  "<map>"),
        (r"\bstd::unordered_map\b",        "<unordered_map>"),
        (r"\bstd::set\b",                  "<set>"),
        (r"\bstd::unordered_set\b",        "<unordered_set>"),
        (r"\bstd::deque\b",                "<deque>"),
        (r"\bstd::list\b",                 "<list>"),
        (r"\bstd::forward_list\b",         "<forward_list>"),
        (r"\bstd::stack\b",                "<stack>"),
        (r"\bstd::queue\b",                "<queue>"),
        (r"\bstd::priority_queue\b",       "<queue>"),
        # Strings / streams
        (r"\bstd::string\b",               "<string>"),
        (r"\bstd::stringstream\b",         "<sstream>"),
        (r"\bstd::ostringstream\b",        "<sstream>"),
        (r"\bstd::istringstream\b",        "<sstream>"),
        # I/O
        (r"\bstd::cout\b",                 "<iostream>"),
        (r"\bstd::cin\b",                  "<iostream>"),
        (r"\bstd::cerr\b",                 "<iostream>"),
        # Algorithms
        (r"\bstd::sort\b",                 "<algorithm>"),
        (r"\bstd::find\b",                 "<algorithm>"),
        (r"\bstd::transform\b",            "<algorithm>"),
        (r"\bstd::for_each\b",             "<algorithm>"),
        (r"\bstd::copy\b",                 "<algorithm>"),
        (r"\bstd::fill\b",                 "<algorithm>"),
        (r"\bstd::count_if\b",             "<algorithm>"),
        # Numeric
        (r"\bstd::accumulate\b",           "<numeric>"),
        (r"\bstd::iota\b",                 "<numeric>"),
        (r"\bstd::reduce\b",               "<numeric>"),
        # Utility
        (r"\bstd::move\b",                 "<utility>"),
        (r"\bstd::forward\b",              "<utility>"),
        (r"\bstd::pair\b",                 "<utility>"),
        (r"\bstd::swap\b",                 "<utility>"),
        (r"\bstd::exchange\b",             "<utility>"),
        # Tuple
        (r"\bstd::tuple\b",                "<tuple>"),
        (r"\bstd::tie\b",                  "<tuple>"),
        # Concurrency
        (r"\bstd::thread\b",               "<thread>"),
        (r"\bstd::mutex\b",                "<mutex>"),
        (r"\bstd::lock_guard\b",           "<mutex>"),
        (r"\bstd::unique_lock\b",          "<mutex>"),
        (r"\bstd::condition_variable\b",   "<condition_variable>"),
        (r"\bstd::atomic\b",               "<atomic>"),
        (r"\bstd::chrono::",               "<chrono>"),
        # Functional
        (r"\bstd::function\b",             "<functional>"),
        (r"\bstd::bind\b",                 "<functional>"),
        # Exceptions
        (r"\bstd::exception\b",            "<exception>"),
        (r"\bstd::runtime_error\b",        "<stdexcept>"),
        (r"\bstd::logic_error\b",          "<stdexcept>"),
        (r"\bstd::out_of_range\b",         "<stdexcept>"),
        (r"\bstd::invalid_argument\b",     "<stdexcept>"),
        # Type traits
        (r"\bstd::is_same\b",              "<type_traits>"),
        (r"\bstd::is_same_v\b",            "<type_traits>"),
        (r"\bstd::enable_if\b",            "<type_traits>"),
        (r"\bstd::decay\b",                "<type_traits>"),
        (r"\bstd::decay_t\b",              "<type_traits>"),
        (r"\bstd::remove_reference\b",     "<type_traits>"),
        # Numeric limits
        (r"\bstd::numeric_limits\b",       "<limits>"),
    ]

    def _get_existing_includes(self, file_content: str) -> set[str]:
        """Return the set of header strings already present in file_content."""
        return {
            m.group(1).strip()
            for m in re.finditer(
                r'^\s*#\s*include\s*([<"][^>"]+[>"])',
                file_content,
                re.MULTILINE,
            )
        }

    def required_headers(self, code: str) -> set[str]:
        """Return the set of headers required by feature patterns found in code."""
        needed: set[str] = set()
        for pattern, header in self._FEATURE_HEADERS:
            if re.search(pattern, code):
                needed.add(header)
        return needed

    def update_file_includes(self, file_content: str, modernized_code: str) -> str:
        """Return file_content with missing headers injected after the last #include.

        Scans *modernized_code* for known C++ feature patterns to determine which
        headers are needed.  Any header not already present in *file_content* is
        injected right after the last existing ``#include`` line in alphabetical
        order.  If no ``#include`` exists, the directives are prepended at the top.
        """
        existing = self._get_existing_includes(file_content)
        needed = self.required_headers(modernized_code)
        missing = needed - existing
        if not missing:
            return file_content

        new_directives = sorted(f"#include {h}" for h in missing)
        lines = file_content.splitlines()

        # Locate the last #include within the first 100 lines (safe header-block limit).
        last_include_idx = -1
        scan_limit = min(len(lines), 100)
        for i in range(scan_limit):
            if lines[i].strip().startswith("#include"):
                last_include_idx = i

        if last_include_idx >= 0:
            insert_at = last_include_idx + 1
            result_lines = lines[:insert_at] + new_directives + lines[insert_at:]
        else:
            result_lines = new_directives + lines

        # Preserve trailing newline if the original had one.
        joined = "\n".join(result_lines)
        if file_content.endswith("\n") and not joined.endswith("\n"):
            joined += "\n"
        return joined


class ModernizationState(TypedDict):
    """State object passed through the workflow"""
    code: str  # This field holds either the raw source code text or a path to a source file, depending on how the workflow is invoked.
    language: str  # This field records the language name (for example "cpp"), so nodes can adjust behavior based on the language.
    analysis: str  # This field stores a JSON-formatted string describing analysis results such as discovered functions and graph metrics.
    dependency_map: dict[str, list[str]]  # This field stores a call-graph style dependency map: function name -> list of functions it calls.
    call_graph_data: dict[str, Any]  # This field stores a JSON-serializable view of the call graph (nodes and edges) for visualization and analysis.
    impact_map: dict[str, list[str]]  # This field stores which functions call which (alias of dependency_map for clarity in prompts).
    orphans: list[str]  # This field stores a list of functions that have no callers in the local call graph.
    analysis_report: str  # This field stores a concise human-readable summary of the codebase / translation unit structure.
    modernized_code: str  # This field holds the most recent modernized version of the code produced by the modernizer node.
    verification_result: dict  # This field records the outcome of compiling the modernized code, including success flag and any errors.
    error_log: str  # This field accumulates human-readable error messages that should be fed back into the model for retries.
    attempt_count: int  # This field tracks how many times the modernizer node has been run, which controls the retry loop.
    is_parity_passed: bool  # This field records whether the most recent differential test confirmed functional parity.
    is_functionally_equivalent: bool  # This field records whether the latest tester run declared the modernized code functionally equivalent.
    diff_output: str  # This field stores the unified diff from the tester when outputs differ.
    feedback_loop_count: int  # This field counts how many times we have routed back into the modernizer for self-healing.
    modernization_order: list[str]  # This field records the order in which functions should be modernized.
    modernized_functions: dict[str, str]  # This field tracks which functions have already been modernized.
    current_function_index: int  # This field tracks which function in the modernization_order is currently being modernized.
    partial_success: bool  # This field records whether the workflow finished with only a subset of functions modernized.
    last_working_code: str  # This field stores the last code snapshot that successfully compiled.
    current_target_function: str  # This field stores the function being surgically modernized in the current iteration.
    source_file: str  # Path to the original source file; used to derive the output file path.
    output_file_path: str  # Explicit output path; when non-empty it overrides the auto-derived path.
    legacy_findings: list[dict[str, Any]]  # Static-analysis tags for legacy C/C++ regions needing C++23 overhaul.
    compliance_report: dict[str, Any]  # C++23 compliance score details for the current modernized candidate.
    functions_info: list[dict[str, Any]]  # Cached parser output for the current working code to avoid repeated full re-parsing.
    functions_index: dict[str, dict[str, Any]]  # Index of function metadata keyed by unique identifier (unique_fqn preferred).
    current_function_name: str  # Name of the function currently being modernized.
    current_function_span: tuple[int, int]  # Byte span (start, end) of the current target function in the current working code.
    project_map: dict[str, Any]  # Cached Tree-sitter semantic map for the current working source.
    use_llm: bool  # Whether LLM-based modernization is enabled for this run.
    llm_available: bool  # Whether at least one model provider is currently available.
    llm_fallback_reason: str  # Reason text when the workflow switches to deterministic no-LLM fallback mode.
    batched_target_functions: list[str]  # Function identifiers modernized in the current batched request.
    current_target_stable_key: str  # Stable signature-based key for the current target function.
    last_function_scores: dict[str, int]  # Last modernization score observed for each target function.
    function_stagnation_counts: dict[str, int]  # Consecutive attempts with unchanged score per function.


def _update_function_stagnation(state: ModernizationState, code_snapshot: str) -> None:
    target_stable_key = str(state.get("current_target_stable_key") or "")
    target_identifier = str(state.get("current_target_function") or "")
    if not target_stable_key and not target_identifier:
        return

    function_text = ""
    if target_identifier:
        function_text = _extract_function_text_from_code(code_snapshot, target_identifier)
    if not function_text and target_stable_key:
        parsed_functions = _parse_functions_from_source(code_snapshot)
        matched = _find_function_by_identifier(parsed_functions, target_stable_key)
        if matched:
            start_byte = matched.get("start_byte")
            end_byte = matched.get("end_byte")
            if isinstance(start_byte, int) and isinstance(end_byte, int):
                code_bytes = code_snapshot.encode("utf-8")
                function_text = code_bytes[start_byte:end_byte].decode("utf-8", errors="strict")
    if not function_text:
        return

    function_score = _safe_score_percent(score_cpp23_compliance(function_text))
    last_scores = dict(state.get("last_function_scores") or {})
    stagnation_counts = dict(state.get("function_stagnation_counts") or {})

    score_key = target_stable_key or target_identifier
    previous_score = last_scores.get(score_key)
    if previous_score is not None and previous_score == function_score:
        stagnation_counts[score_key] = int(stagnation_counts.get(score_key, 0)) + 1
    else:
        stagnation_counts[score_key] = 0

    last_scores[score_key] = function_score
    state["last_function_scores"] = last_scores
    state["function_stagnation_counts"] = stagnation_counts

    logger.info(
        "Function progress: %s score=%d stagnation=%d/%d body=%s",
        score_key,
        function_score,
        int(stagnation_counts.get(score_key, 0)),
        _WORKFLOW_STAGNANT_SCORE_LIMIT,
        hashlib.sha256(function_text.encode("utf-8")).hexdigest()[:10],
    )


def analyzer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 1: Analyzer - Graph-aware analysis of legacy code.

    This node:
      - Uses CppParser to extract function definitions and their calls.
      - Builds a DependencyGraph of functions using networkx.
      - Identifies orphan functions (no callers) and circular recursions (cycles).
      - Produces both a machine-friendly JSON analysis and a human-readable analysis_report.
    """
    logger.info("\n🔍 ANALYZER NODE")
    logger.info("Language: %s", state['language'])
    logger.debug("Input Code (first 200 chars):\n%s...", state['code'][:200])

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++17", "c++20", "c++23"}  # This line checks whether the language label indicates C++, which is when we want to use the Tree-sitter parser.

    functions_info = []  # type: ignore[var-annotated]  # This variable will hold the list of functions discovered by the parser, if any.
    dependency_map: dict[str, list[str]] = {}  # This dictionary will map each function name to the list of functions it calls.
    parser_error: str = ""  # This string will store any error message encountered while trying to parse the code.
    orphans: list[str] = []  # This list will store functions with no callers.
    cycles: list[list[str]] = []  # This list will store any detected circular recursion cycles.
    analysis_report: str = ""  # This string will hold a concise human-readable summary of the code structure.
    call_graph_data: dict[str, Any] = {}  # This dictionary will store a JSON-serializable view of the call graph.
    legacy_findings: list[dict[str, Any]] = []
    project_map: dict[str, Any] = {}
    dep_graph: DependencyGraph | None = None

    if is_cpp:  # This condition ensures we only run the C++ parser when we are actually working with C++ code.
        try:  # This try block attempts to run the parser on the current C++ source text and build a dependency graph.
            parser = CppParser()
            project_map = parser.parse_string(
                state["code"],
                source_file=state.get("source_file") or "",
            )  # Parse once and reuse this semantic map across nodes.
            parsed_functions = project_map.get("functions", {})
            if isinstance(parsed_functions, dict):
                functions_info = list(parsed_functions.values())
            else:
                functions_info = []
            legacy_findings = detect_legacy_patterns(state["code"])

            dep_graph = DependencyGraph(functions_info)  # This line constructs a DependencyGraph object from the function metadata.
            dependency_map = dep_graph.dependency_map  # This line reads the caller -> callees adjacency list.

            graph_metrics = dep_graph.analyze()  # This line analyzes orphans and cycles in the call graph.
            orphans = list(graph_metrics.get("orphans", []))  # This line records the set of orphan functions.
            cycles = list(graph_metrics.get("cycles", []))  # This line records the list of cycles.

            analysis_report = build_analysis_report(functions_info, dependency_map, orphans, cycles)  # This line builds a concise human-readable report.
            call_graph_data = dep_graph.to_dict()  # This line exports the call graph as simple nodes/edges data for visualization and prompts.
        except Exception as exc:  # This except block catches any parsing- or graph-related errors so they do not crash the workflow.
            parser_error = f"Analyzer failed to parse or analyze C++ file: {exc!r}"  # This line records a human-readable message explaining why analysis failed.

    analysis: dict[str, Any] = {  # This dictionary will hold the structured analysis information that we will turn into JSON.
        "language": state["language"],  # This entry records the language label that was used for this analysis.
        "functions": functions_info,  # This entry stores detailed per-function metadata including names, calls, and byte spans.
        "function_summary": {  # This nested dictionary describes the functions we discovered (if any).
            "count": len(functions_info),  # This entry records how many functions the parser found in the C++ file.
            "names": [fn.get("name", "") for fn in functions_info],  # This list comprehension collects just the function names from the parser output.
        },  # This closing brace ends the function_summary dictionary.
        "dependency_map": dependency_map,  # This entry captures the call-graph style function dependency map for the current translation unit.
        "call_graph_data": call_graph_data,  # This entry stores a simple nodes/edges representation of the call graph.
        "orphans": orphans,  # This entry records functions that have no callers.
        "cycles": cycles,  # This entry records any circular recursion cycles in the call graph.
        "analysis_report": analysis_report,  # This entry stores a concise human-readable description of the code structure.
        "parser_error": parser_error,  # This entry stores any error message from the parser, or an empty string if everything went fine.
        "legacy_findings": legacy_findings,
    }  # This closing brace ends the analysis dictionary.

    state["analysis"] = json.dumps(analysis, indent=2, sort_keys=True)  # This line converts the analysis dictionary into a stable, human-readable JSON string and stores it in the state.
    state["dependency_map"] = dependency_map  # This line stores the raw dependency map separately so downstream nodes can consume it without re-parsing JSON.
    state["call_graph_data"] = call_graph_data  # This line stores the simple call graph data for downstream nodes and visualization.
    state["impact_map"] = dependency_map  # This line aliases the dependency_map as impact_map to emphasize its use in prompts.
    state["orphans"] = orphans  # This line stores the list of orphan functions so downstream nodes can choose to prune them.
    state["analysis_report"] = analysis_report  # This line stores the human-readable report so it can be included in prompts if desired.
    state["legacy_findings"] = legacy_findings
    state["functions_info"] = list(functions_info)
    state["functions_index"] = _index_functions_by_identifier(functions_info)
    state["project_map"] = dict(project_map)

    if is_cpp and functions_info:
        if dep_graph is not None:
            state["modernization_order"] = dep_graph.get_modernization_order()
        else:
            state["modernization_order"] = [
                _function_identifier(fn)
                for fn in functions_info
                if _function_identifier(fn)
            ]
    else:
        state["modernization_order"] = []
    state["modernized_functions"] = {}
    state["current_function_index"] = 0
    state["current_target_stable_key"] = ""
    state["current_function_name"] = ""
    state["current_function_span"] = (0, 0)
    state["partial_success"] = False
    logger.debug("Analysis (JSON):\n%s", state['analysis'])

    return state  # This return passes the updated state object along to the next node in the workflow.


def pruner_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Pruner - Removes orphan code (except main) before modernization.

    This node:
      - Reads the function metadata and orphan list from the deep analyzer.
      - Uses AST byte ranges to surgically remove orphan function definitions from the
        source text, except for the entry point 'main'.
      - Updates state["code"] with a pruned version to reduce the LLM's workload.
    """
    logger.info("\n✂️  PRUNER NODE")

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20", "c++23"}  # This line checks whether we are working with C++ code.
    if not is_cpp:  # This condition skips pruning for non-C++ languages.
        logger.info("Pruner: skipping (non-C++ language).")
        return state  # This return leaves the state unchanged.

    if not state.get("analysis"):
        try:
            pre_functions = _parse_functions_from_source(state["code"])
            pre_graph = DependencyGraph(pre_functions)
            pre_orphans = list(pre_graph.analyze().get("orphans", []))
            pre_analysis = {
                "functions": pre_functions,
                "orphans": pre_orphans,
            }
            state["analysis"] = json.dumps(pre_analysis)
            state["orphans"] = pre_orphans
        except Exception:
            logger.warning("Pruner: no analysis found and pre-analysis failed, skipping pruning.")
            return state

    try:
        analysis_obj = json.loads(state["analysis"])  # This line parses the JSON analysis string into a Python dictionary.
    except json.JSONDecodeError:
        logger.warning("Pruner: failed to parse analysis JSON, skipping pruning.")
        return state  # This return leaves the state unchanged.

    functions_info = state.get("functions_info") or analysis_obj.get("functions") or []  # This line reads cached function metadata first.
    orphans = state.get("orphans") or analysis_obj.get("orphans") or []  # This line prefers the orphans recorded on the state but falls back to the analysis JSON.
    if not functions_info or not orphans:  # This condition checks whether there is anything to prune.
        logger.info("Pruner: no functions or no orphans detected, skipping pruning.")
        return state  # This return leaves the state unchanged.

    # We never prune the program entry point 'main', even if it has no callers.
    orphans_to_prune = {str(name) for name in orphans if str(name) != "main"}  # This set records orphan function names that are eligible for pruning.
    if not orphans_to_prune:
        logger.info("Pruner: orphan list only contains 'main' or is empty, nothing to prune.")
        return state  # This return leaves the state unchanged.

    original_code = state["code"]  # This line reads the original source code text from the state.
    original_bytes = original_code.encode("utf-8")  # This line encodes the code as UTF-8 bytes so we can use Tree-sitter byte offsets.

    # Collect byte-span ranges for every function definition that should be removed.
    spans_to_remove: list[tuple[int, int]] = []  # This list will hold (start_byte, end_byte) pairs for orphan functions.
    for fn in functions_info:  # This loop inspects each function metadata entry.
        name = str(fn.get("name") or "")  # This line normalizes the function name.
        if name not in orphans_to_prune:  # This condition ensures we only consider orphan functions for pruning.
            continue
        start_byte = fn.get("start_byte")
        end_byte = fn.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int) and 0 <= start_byte <= end_byte <= len(original_bytes):  # This condition validates the byte span.
            spans_to_remove.append((start_byte, end_byte))  # This line records the valid span.

    if not spans_to_remove:  # This condition checks whether we have any valid spans to delete.
        logger.warning("Pruner: no valid byte spans found for orphan functions, skipping pruning.")
        return state  # This return leaves the state unchanged.

    spans_to_remove.sort(key=lambda pair: pair[0])  # This line sorts the spans by starting offset so we can rebuild the code in order.

    # Rebuild the code bytes, skipping over the regions marked for removal.
    new_chunks: list[bytes] = []  # This list will hold the surviving segments of the original code.
    cursor = 0  # This integer tracks our current position in the original byte array.
    for start_byte, end_byte in spans_to_remove:  # This loop processes each span to remove.
        if start_byte < cursor:  # This condition skips spans that overlap with or are fully contained in a previously removed region.
            continue
        if start_byte > cursor:  # This condition preserves the code between the previous cursor and the start of this span.
            new_chunks.append(original_bytes[cursor:start_byte])
        cursor = end_byte  # This line moves the cursor past the removed span.

    if cursor < len(original_bytes):  # This condition ensures we keep any trailing code after the last removed span.
        new_chunks.append(original_bytes[cursor:])

    pruned_bytes = b"".join(new_chunks)  # This line concatenates all preserved segments into a single byte string.
    pruned_code = pruned_bytes.decode("utf-8", errors="strict")  # This line decodes the pruned bytes back into text.

    def _prune_dead_includes(code_text: str, removed_functions: list[str]) -> str:
        lines = code_text.splitlines()
        kept_lines: list[str] = []
        removed_lower = [name.lower() for name in removed_functions]
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#include"):
                lowered = stripped.lower()
                if any(name in lowered for name in removed_lower):
                    continue
            kept_lines.append(line)
        return "\n".join(kept_lines)

    pruned_code = _prune_dead_includes(pruned_code, sorted(orphans_to_prune))

    state["code"] = pruned_code  # This line updates the state with the pruned source code so the modernizer works on a smaller, focused input.
    state["last_working_code"] = pruned_code  # Keep subsequent passes anchored to the pruned baseline so removed orphans do not come back.
    pruned_project_map = _parse_project_map_from_source(pruned_code)
    pruned_functions_obj = pruned_project_map.get("functions") or {}
    if isinstance(pruned_functions_obj, dict):
        pruned_functions = list(pruned_functions_obj.values())
    else:
        pruned_functions = [item for item in pruned_functions_obj if isinstance(item, dict)] if isinstance(pruned_functions_obj, list) else []
    state["project_map"] = dict(pruned_project_map)
    state["functions_info"] = pruned_functions
    state["functions_index"] = _index_functions_by_identifier(pruned_functions)

    # Remove pruned functions from the modernization order so the modernizer
    # does not attempt to re-add them.
    existing_order = state.get("modernization_order") or []
    state["modernization_order"] = [
        name for name in existing_order if name not in orphans_to_prune
    ]

    logger.info("✂️  Pruning %d orphan function(s): %s", len(spans_to_remove), ', '.join(sorted(orphans_to_prune)))

    return state  # This return passes the updated state object along to the next node in the workflow.


def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Modernizer - Rewrites code to modern standards
    Receives error_log if coming from verifier feedback
    """
    logger.info("\n✏️  MODERNIZER NODE")
    logger.info("Attempt: %d (per-current-function)", state['attempt_count'])

    if state["error_log"]:  # This condition checks whether the verifier or tester reported any previous issues.
        logger.info("Previous Feedback:\n%s", state['error_log'])

    # On retry after verifier feedback, keep the last failed candidate so the model can
    # perform a minimal compile-fix patch instead of restarting from an older snapshot.
    if state.get("error_log") and state.get("modernized_code"):
        source_to_improve = state["modernized_code"]
    else:
        source_to_improve = (
            state.get("last_working_code")
            or state["modernized_code"]
            or state["code"]
        )

    logger.debug("Source Snapshot (preview):\n%s...", source_to_improve[:200])

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++17", "c++20", "c++23"}  # This line checks whether we are working with C++ so we know whether to ask for C++ modernization.

    modernized = source_to_improve  # This line initializes the modernized code to the source as a safe default in case the model call fails.
    state["current_target_function"] = ""
    state["current_target_stable_key"] = ""
    state["current_function_name"] = ""
    state["current_function_span"] = (0, 0)

    if is_cpp:
        functions_info = state.get("functions_info") or []
        if not functions_info:
            functions_info = _parse_functions_from_source(source_to_improve)
            state["functions_info"] = list(functions_info)

        modernization_order = state.get("modernization_order") or [
            str(fn.get("name") or "")
            for fn in functions_info
            if fn.get("name")
        ]
        current_index = int(state.get("current_function_index", 0))

        while current_index < len(modernization_order):
            candidate_identifier = str(modernization_order[current_index])
            has_candidate = any(_function_matches_identifier(fn, candidate_identifier) for fn in functions_info)
            if has_candidate:
                break
            current_index += 1

        if current_index >= len(modernization_order):
            logger.info("No remaining functions in modernization_order; leaving code unchanged.")
            state["modernized_code"] = source_to_improve
            state["attempt_count"] += 1
            return state

        state["current_function_index"] = current_index
        current_function_identifier = str(modernization_order[current_index]) if modernization_order else ""
        state["current_target_function"] = current_function_identifier

        functions_index = state.get("functions_index") or _index_functions_by_identifier(functions_info)
        current_function_info: dict[str, Any] = functions_index.get(current_function_identifier, {})
        if not current_function_info:
            for fn in functions_info:
                if _function_matches_identifier(fn, current_function_identifier):
                    current_function_info = fn
                    break

        current_function_name = str(current_function_info.get("name") or "")
        state["current_target_stable_key"] = _function_stable_key(current_function_info)
        state["current_function_name"] = current_function_name
        original_parameters = current_function_info.get("parameters") or []
        original_param_count = len(original_parameters) if isinstance(original_parameters, list) else -1

        start_byte = current_function_info.get("start_byte")
        end_byte = current_function_info.get("end_byte")
        source_bytes = source_to_improve.encode("utf-8")
        if not (
            isinstance(start_byte, int)
            and isinstance(end_byte, int)
            and 0 <= start_byte <= end_byte <= len(source_bytes)
        ):
            warning = f"Missing valid byte span for function '{current_function_identifier}', skipping."
            logger.warning("%s", warning)
            state["error_log"] = warning
            state["attempt_count"] += 1
            state["modernized_code"] = source_to_improve
            return state

        state["current_function_span"] = (start_byte, end_byte)
        function_source = source_bytes[start_byte:end_byte].decode("utf-8", errors="strict")
        original_quality_score = _safe_score_percent(score_cpp23_compliance(function_source))
        if len(function_source) > _MAX_MODERNIZER_FUNCTION_CHARS:
            warning = (
                f"Skipping large function '{current_function_identifier}' "
                f"({len(function_source)} chars > {_MAX_MODERNIZER_FUNCTION_CHARS})."
            )
            logger.warning("%s", warning)
            state["error_log"] = warning
            state["modernized_code"] = source_to_improve
            state["attempt_count"] += 1
            return state

        legacy_pattern_findings = detect_legacy_patterns(function_source)
        legacy_patterns = [
            str(item.get("pattern") or "")
            for item in legacy_pattern_findings
            if item.get("pattern")
        ]
        legacy_pattern_summary = ", ".join(sorted(set(legacy_patterns))) if legacy_patterns else "none"

        rule_based_candidate, transform_notes = _apply_rule_based_function_transforms(function_source)

        if transform_notes:
            logger.info("Rule-based transforms applied: %s", ", ".join(transform_notes))
        else:
            logger.debug("Rule-based transforms applied: none")

        callers_map = _build_callers_map(state.get("dependency_map", {}))
        callers = sorted(callers_map.get(current_function_name, []))
        callers_display = ", ".join(callers) if callers else "none"
        prompt_parts: list[str] = []
        prompt_parts.append(
            "Rewrite ONLY this function to deeply modern C++17 while preserving behavior. "
            "You MUST apply meaningful modernization and avoid legacy ownership/manual memory patterns. Returning identical code is NOT allowed. "
            "Prefer RAII, std::unique_ptr, std::string, nullptr, and safer interfaces. "
            "Signature refactors are allowed when they improve safety; update in-file call sites as needed. "
            "Return ONLY the updated function."
        )
        prompt_parts.append(f"Function name: {current_function_name}")
        prompt_parts.append(f"Direct callers: {callers_display}")
        prompt_parts.append(f"Detected legacy patterns: {legacy_pattern_summary}")
        prompt_parts.append("Target function:\n```cpp\n" + function_source + "\n```")

        if transform_notes:
            prompt_parts.append(
                "Deterministic rule-based rewrites that can be applied when safe: "
                + ", ".join(transform_notes)
                + ". Preserve behavior and correctness."
            )

        if state["error_log"]:
            prompt_parts.append(
                "Compiler/parity feedback from the previous attempt for this same function:\n"
                f"{state['error_log']}\n\n"
                "FIX-FOCUSED MODE: resolve reported issues while preserving deep-modernized style. "
                "Avoid unrelated edits, but do not regress to legacy ownership patterns."
            )

        prompt_parts.append(
            "Rules: no markdown fences, do not output the whole file, preserve surrounding structure, keep C++17 compatibility only, and prioritize deep modernization over minimal diff."
        )

        full_prompt = "\n\n".join(prompt_parts)  # This line joins all prompt sections together with blank lines so the text is readable and well structured.
        state["batched_target_functions"] = []

        llm_enabled = bool(state.get("use_llm", True))
        llm_available = bool(state.get("llm_available", True))
        if not llm_enabled or not llm_available:
            fallback_reason = state.get("llm_fallback_reason") or (
                "LLM mode disabled by WORKFLOW_USE_LLM/WORKFLOW_DISABLE_LLM"
                if not llm_enabled
                else "Model providers unavailable; using deterministic rule-based fallback"
            )
            logger.warning("Modernizer running in no-LLM fallback mode: %s", fallback_reason)

            if transform_notes and rule_based_candidate != function_source:
                try:
                    modernized = _replace_function_by_span(
                        source_to_improve,
                        start_byte,
                        end_byte,
                        rule_based_candidate,
                    )
                    modernized = IncludeManager().update_file_includes(modernized, rule_based_candidate)
                    refreshed_project_map = _parse_project_map_from_source(modernized)
                    refreshed_functions_obj = refreshed_project_map.get("functions") or {}
                    if isinstance(refreshed_functions_obj, dict):
                        refreshed_functions = list(refreshed_functions_obj.values())
                    else:
                        refreshed_functions = [item for item in refreshed_functions_obj if isinstance(item, dict)] if isinstance(refreshed_functions_obj, list) else []
                    state["project_map"] = dict(refreshed_project_map)
                    state["functions_info"] = refreshed_functions
                    state["functions_index"] = _index_functions_by_identifier(refreshed_functions)
                    state["error_log"] = "no-LLM fallback applied: " + ", ".join(transform_notes)
                except ValueError as replacement_exc:
                    warning = f"No-LLM fallback replacement failed for '{current_function_identifier}': {replacement_exc}"
                    logger.warning("%s", warning)
                    state["error_log"] = warning
                    modernized = source_to_improve
            else:
                warning = (
                    f"No-LLM fallback found no deterministic rewrite for '{current_function_identifier}'."
                )
                logger.warning("%s", warning)
                state["error_log"] = warning
                modernized = source_to_improve

            state["modernized_code"] = modernized
            state["attempt_count"] += 1
            return state

        if _WORKFLOW_BATCH_SIZE > 1:
            batch_entries = _collect_batch_entries(
                functions_info=functions_info,
                functions_index=functions_index,
                modernization_order=modernization_order,
                current_index=current_index,
                source_to_improve=source_to_improve,
                batch_size=_WORKFLOW_BATCH_SIZE,
            )
            if len(batch_entries) > 1:
                batch_prompt = _batch_prompt_for_functions(
                    batch_entries=batch_entries,
                    dependency_map=callers_map,
                    previous_error_log=state.get("error_log", ""),
                )
                try:
                    raw_batch_text = call_model(_MODEL_SYSTEM_PROMPT, batch_prompt)
                    batch_json_text = _extract_json_object_text(raw_batch_text)
                    parsed_batch = json.loads(batch_json_text) if batch_json_text else {}
                    batch_items = parsed_batch.get("functions") if isinstance(parsed_batch, dict) else []
                    candidate_by_id: dict[str, str] = {}
                    for item in batch_items if isinstance(batch_items, list) else []:
                        if not isinstance(item, dict):
                            continue
                        identifier = str(item.get("id") or "").strip()
                        code = str(item.get("code") or "").strip()
                        if identifier and code:
                            candidate_by_id[identifier] = code

                    batched_source, applied_ids = _apply_batched_replacements(
                        source_to_improve=source_to_improve,
                        batch_entries=batch_entries,
                        candidate_by_id=candidate_by_id,
                    )
                    if len(applied_ids) > 1 and batched_source != source_to_improve:
                        logger.info("Batch modernization applied for %d functions in one API call.", len(applied_ids))
                        refreshed_project_map = _parse_project_map_from_source(batched_source)
                        refreshed_functions_obj = refreshed_project_map.get("functions") or {}
                        if isinstance(refreshed_functions_obj, dict):
                            refreshed_functions = list(refreshed_functions_obj.values())
                        else:
                            refreshed_functions = [item for item in refreshed_functions_obj if isinstance(item, dict)] if isinstance(refreshed_functions_obj, list) else []
                        state["project_map"] = dict(refreshed_project_map)
                        state["functions_info"] = refreshed_functions
                        state["functions_index"] = _index_functions_by_identifier(refreshed_functions)
                        state["modernized_code"] = batched_source
                        state["batched_target_functions"] = applied_ids
                        state["attempt_count"] += 1
                        return state
                except Exception as batch_exc:
                    logger.warning("Batch modernization attempt failed; falling back to single-function mode: %r", batch_exc)

        try:
            cleaned_candidate = ""
            raw_text = ""
            prompt_for_attempt = full_prompt
            rejection_reason = ""
            for model_attempt in range(1, _MODEL_OUTPUT_ATTEMPTS + 1):
                raw_text = call_model(_MODEL_SYSTEM_PROMPT, prompt_for_attempt)
                logger.debug("Model response attempt %d (first 200 chars): %s...", model_attempt, raw_text[:200])

                cleaned_candidate = _clean_model_code_block(raw_text).strip()
                if not cleaned_candidate:
                    rejection_reason = "model returned empty function output"
                elif _looks_like_prose_response(raw_text, cleaned_candidate):
                    rejection_reason = "model returned prose or non-code instructions"
                elif current_function_name and current_function_name not in cleaned_candidate:
                    rejection_reason = f"model output does not contain function name '{current_function_name}'"
                else:
                    validation_error = _validate_single_function_candidate(
                        cleaned_candidate,
                        current_function_name,
                        original_param_count,
                    )
                    if validation_error:
                        rejection_reason = validation_error
                    elif code_similarity_ratio(function_source, cleaned_candidate) > _MODERNIZER_NODE_SIMILARITY_GUARD:
                        rejection_reason = "model returned nearly identical code"
                    else:
                        rejection_reason = ""
                        break

                logger.warning("Rejecting model output attempt %d: %s.", model_attempt, rejection_reason)
                if model_attempt < _MODEL_OUTPUT_ATTEMPTS:
                    prompt_for_attempt = _build_retry_prompt(full_prompt, rejection_reason, raw_text)

            if rejection_reason:
                warning = f"Model returned unusable output for '{current_function_identifier}': {rejection_reason}."
                logger.warning("%s", warning)
                state["error_log"] = warning
                modernized = source_to_improve
            else:
                diff_lines = list(
                    difflib.unified_diff(
                        function_source.splitlines(),
                        cleaned_candidate.splitlines(),
                        fromfile=f"{current_function_name}_before",
                        tofile=f"{current_function_name}_after",
                        lineterm="",
                    )
                )
                if diff_lines:
                    logger.debug("\n".join(diff_lines))
                try:
                    modernized = _replace_function_by_span(
                        source_to_improve,
                        start_byte,
                        end_byte,
                        cleaned_candidate,
                    )
                except ValueError as replacement_exc:
                    replacement_rejection = (
                        f"Model output rejected during replacement validation: {replacement_exc}"
                    )
                    logger.warning("%s", replacement_rejection)
                    state["error_log"] = replacement_rejection
                    modernized = source_to_improve
                    state["attempt_count"] += 1
                    state["modernized_code"] = modernized
                    return state

                # Keep header updates in place, but only derive required includes from
                # the rewritten function content.
                modernized = IncludeManager().update_file_includes(modernized, cleaned_candidate)

                # Refresh cached metadata immediately after a successful replacement
                # so subsequent function lookups always use current byte spans.
                refreshed_project_map = _parse_project_map_from_source(modernized)
                refreshed_functions_obj = refreshed_project_map.get("functions") or {}
                if isinstance(refreshed_functions_obj, dict):
                    refreshed_functions = list(refreshed_functions_obj.values())
                else:
                    refreshed_functions = [item for item in refreshed_functions_obj if isinstance(item, dict)] if isinstance(refreshed_functions_obj, list) else []
                state["project_map"] = dict(refreshed_project_map)
                state["functions_info"] = refreshed_functions
                state["functions_index"] = _index_functions_by_identifier(refreshed_functions)
        except Exception as exc:
            error_message = f"Model call failed in modernizer: {exc!r}"
            logger.error("%s", error_message)
            if _is_provider_failure_message(error_message):
                state["llm_available"] = False
                state["llm_fallback_reason"] = str(exc)
                if transform_notes and rule_based_candidate != function_source:
                    try:
                        modernized = _replace_function_by_span(
                            source_to_improve,
                            start_byte,
                            end_byte,
                            rule_based_candidate,
                        )
                        modernized = IncludeManager().update_file_includes(modernized, rule_based_candidate)
                        refreshed_project_map = _parse_project_map_from_source(modernized)
                        refreshed_functions_obj = refreshed_project_map.get("functions") or {}
                        if isinstance(refreshed_functions_obj, dict):
                            refreshed_functions = list(refreshed_functions_obj.values())
                        else:
                            refreshed_functions = [item for item in refreshed_functions_obj if isinstance(item, dict)] if isinstance(refreshed_functions_obj, list) else []
                        state["project_map"] = dict(refreshed_project_map)
                        state["functions_info"] = refreshed_functions
                        state["functions_index"] = _index_functions_by_identifier(refreshed_functions)
                    except ValueError:
                        modernized = source_to_improve
            if state["error_log"]:
                state["error_log"] += f"\n{error_message}"
            else:
                state["error_log"] = error_message
    else:
        # For non-C++ languages, apply a minimal placeholder transformation.
        modernized = source_to_improve.replace("var ", "auto ")

    state["modernized_code"] = modernized  # This line stores the modernized (or minimally transformed) code back into the shared workflow state.
    state["attempt_count"] += 1  # This line increments the attempt counter so the router can track how many modernization passes we have performed.

    logger.debug("Modernized Code (first 200 chars):\n%s...", modernized[:200])

    return state  # This return passes the updated state on to the verifier node.


def verifier_node(state: ModernizationState) -> ModernizationState:
    """
    Node 3: Verifier - Validates modernized code via MCP compiler tool
    """
    logger.info("\n✅ VERIFIER NODE")
    logger.info("Validating modernized code with real g++ compilation...")

    code_to_verify = state["modernized_code"]  # This line uses only the modernized_code field, because we want to compile exactly what the modernizer produced.

    if not code_to_verify.strip():  # This condition checks if the AI returned no code (empty or whitespace-only); if so, we stop the workflow entirely.
        message = "CRITICAL: Model returned no code. Check your model selection, API key, and max token settings."  # This line sets a precise failure message for empty AI output.
        logger.error("%s", message)
        verification_result = {  # This dictionary records a failure result so the workflow can finish without calling the compiler.
            "success": False,  # This field marks the verification as a failure because there was nothing to compile.
            "errors": [message],  # This list stores the same message for the verification result.
            "warnings": [],  # This empty list indicates that there were no compiler warnings.
            "compilation_time_ms": 0,  # This field records zero time because we never invoked the compiler.
            "raw_stdout": "",  # This field is empty because the compiler never ran.
            "raw_stderr": "",  # This field is empty for the same reason.
        }  # This closing brace ends the verification_result dictionary for the empty-code case.
        state["verification_result"] = verification_result  # This line stores the failure result in the shared state.
        state["error_log"] = message  # This line copies the exact error message into the error log for visibility in later inspection.
        state["attempt_count"] = _WORKFLOW_MAX_ATTEMPTS_PER_FUNCTION  # This line sets attempts to max retry budget so the router safely terminates.
        logger.error("❌ Verification FAILED: %s", message)
        return state  # This return exits early so we do not try to call g++ with empty input.

    verification_result = compile_cpp_source(code_to_verify)
    state["verification_result"] = verification_result
    _update_function_stagnation(state, code_to_verify)

    if verification_result["success"]:  # This condition checks whether compilation finished successfully without errors.
        logger.info("✅ Verification PASSED")
        state["error_log"] = ""  # This line clears the error log because there are no compiler errors to feed back into the model.
        state["last_working_code"] = state["modernized_code"]
        state["compliance_report"] = score_cpp23_compliance(state["modernized_code"])
        compliance_raw = state["compliance_report"].get("percent", 0)
        try:
            compliance_percent = int(float(str(compliance_raw)))
        except (TypeError, ValueError):
            compliance_percent = 0
        logger.info("Modernization score: %d%%", compliance_percent)

        # Refresh parser/cache + dependency graph/order after each successful function replacement.
        previous_functions_info = list(state.get("functions_info") or [])
        previous_target_identifier = str(state.get("current_target_function") or "")
        previous_target_stable_key = str(state.get("current_target_stable_key") or "")
        previous_target_info = _find_function_by_identifier(previous_functions_info, previous_target_identifier)
        if not previous_target_stable_key and previous_target_info:
            previous_target_stable_key = _function_stable_key(previous_target_info)

        parser = CppParser()
        refreshed_project_map = parser.parse_string(
            state["modernized_code"],
            source_file=state.get("source_file") or "",
        )
        parsed_functions = refreshed_project_map.get("functions", {})
        if isinstance(parsed_functions, dict):
            refreshed_functions = list(parsed_functions.values())
        else:
            refreshed_functions = []
        state["project_map"] = dict(refreshed_project_map)
        state["functions_info"] = refreshed_functions
        state["functions_index"] = _index_functions_by_identifier(refreshed_functions)

        refreshed_dep_graph = DependencyGraph(refreshed_functions)
        state["dependency_map"] = refreshed_dep_graph.dependency_map
        state["impact_map"] = refreshed_dep_graph.dependency_map
        state["call_graph_data"] = refreshed_dep_graph.to_dict()
        if refreshed_functions:
            state["modernization_order"] = refreshed_dep_graph.get_modernization_order()

        target_function = state.get("current_target_function") or ""
        target_stable_key = previous_target_stable_key
        if target_function:
            logger.debug(
                "Completion mapping before refresh: target_identifier=%s target_stable_key=%s",
                target_function,
                target_stable_key,
            )

        updated_function_code = ""
        updated_key = ""
        if target_stable_key:
            refreshed_target = _find_function_by_identifier(refreshed_functions, target_stable_key)
            if refreshed_target:
                start_byte = refreshed_target.get("start_byte")
                end_byte = refreshed_target.get("end_byte")
                if isinstance(start_byte, int) and isinstance(end_byte, int):
                    code_bytes = state["modernized_code"].encode("utf-8")
                    updated_function_code = code_bytes[start_byte:end_byte].decode("utf-8", errors="strict")
                    updated_key = _function_stable_key(refreshed_target)
        if not updated_function_code and target_function:
            updated_function_code = _extract_function_text_from_code(
                state["modernized_code"],
                target_function,
            )
            if updated_function_code:
                refreshed_target = _find_function_by_identifier(refreshed_functions, target_function)
                updated_key = _function_stable_key(refreshed_target) if refreshed_target else (target_stable_key or target_function)

        if updated_function_code:
            modernized_functions = state.get("modernized_functions") or {}
            modernized_functions[updated_key or target_stable_key or target_function] = updated_function_code
            state["modernized_functions"] = modernized_functions
            logger.debug(
                "Completion mapping after refresh: recorded_key=%s completed_count=%d",
                updated_key or target_stable_key or target_function,
                len(modernized_functions),
            )

        batched_targets = [str(item) for item in (state.get("batched_target_functions") or []) if str(item)]
        if batched_targets:
            modernized_functions = state.get("modernized_functions") or {}
            for identifier in batched_targets:
                batch_target = _find_function_by_identifier(refreshed_functions, identifier)
                updated_function_code = ""
                stable_batch_key = ""
                if batch_target:
                    start_byte = batch_target.get("start_byte")
                    end_byte = batch_target.get("end_byte")
                    if isinstance(start_byte, int) and isinstance(end_byte, int):
                        code_bytes = state["modernized_code"].encode("utf-8")
                        updated_function_code = code_bytes[start_byte:end_byte].decode("utf-8", errors="strict")
                    stable_batch_key = _function_stable_key(batch_target)
                if not updated_function_code:
                    updated_function_code = _extract_function_text_from_code(
                        state["modernized_code"],
                        identifier,
                    )
                if updated_function_code:
                    modernized_functions[stable_batch_key or identifier] = updated_function_code
            state["modernized_functions"] = modernized_functions
        state["batched_target_functions"] = []

        completed_functions = set((state.get("modernized_functions") or {}).keys())
        next_index = 0
        updated_order = state.get("modernization_order") or []
        while next_index < len(updated_order):
            order_identifier = str(updated_order[next_index])
            order_target = _find_function_by_identifier(refreshed_functions, order_identifier)
            order_stable_key = _function_stable_key(order_target) if order_target else order_identifier
            if order_stable_key not in completed_functions:
                break
            next_index += 1
        state["current_function_index"] = next_index
    else:  # This branch handles the case where g++ returned at least one error.
        logger.error("❌ Verification FAILED")
        state["compliance_report"] = {}
        raw_stderr = verification_result.get("raw_stderr", "")
        errors_list = verification_result.get("errors") or []
        combined_error = raw_stderr or "\n".join(errors_list)
        error_lines = _extract_error_line_numbers(combined_error)
        critical_fix_section = ""
        if error_lines:
            first_error_line = error_lines[0]
            first_three_lines = error_lines[:3]
            snippets: list[str] = []
            for ln in first_three_lines:
                focused_snippet = _get_code_snippet_by_line(code_to_verify, ln)
                if focused_snippet:
                    snippets.append(f"Line {ln}:\n{focused_snippet}")
            critical_fix_section = (
                "\n\nCRITICAL FIX REQUIRED: "
                f"The compiler reported an error at Line {first_error_line}. "
                "Look specifically at this logic:\n"
                f"{'\n\n'.join(snippets)}"
            )

        state["error_log"] = combined_error + critical_fix_section
        state["batched_target_functions"] = []
        logger.warning("Compile failed, keeping modernization for inspection.")
        logger.warning("Errors from g++:\n%s", state['error_log'])

    return state  # This return passes the updated state (including verification results) back into the LangGraph router.


def tester_node(state: ModernizationState) -> ModernizationState:
    """
    Node 4: Tester - Runs a differential test to check functional parity.

    This node:
      - Only runs when compilation succeeded.
      - Uses run_differential_test to compare outputs of the original and modernized code.
      - Sets is_parity_passed and, on failure, records a unified diff in error_log so
        the modernizer can perform a self-healing pass focused on logic parity.
    """
    logger.info("\n🧪 TESTER NODE")

    # Reset parity / equivalence flags and previous diff output before running a new test.
    state["is_parity_passed"] = True  # This line pessimistically assumes parity will pass; we flip it to False on failure.
    state["is_functionally_equivalent"] = True  # This line assumes functional equivalence until the tester proves otherwise.
    state["diff_output"] = ""  # This line clears any previous diff output.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++17", "c++20", "c++23"}  # This line checks whether we are working with C++ code.
    if not is_cpp:
        logger.info("Tester: skipping (non-C++ language).")
        return state  # This return leaves the state unchanged for non-C++ code.

    if not state["verification_result"].get("success"):  # This condition ensures we only run the tester when compilation succeeded.
        logger.info("Tester: skipping because compilation failed.")
        state["is_parity_passed"] = False  # This line records that parity has not been confirmed.
        state["is_functionally_equivalent"] = False
        return state  # This return leaves error_log with compiler errors.

    if not state["modernized_code"].strip():  # This condition checks that we have modernized code to test.
        logger.info("Tester: skipping because modernized_code is empty.")
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state

    # Differential testing compares the current final candidate against the original
    # source file path stored in state.
    original_cpp_path = state.get("source_file") or ""

    if not os.path.isfile(original_cpp_path):
        logger.warning("Tester: original C++ file not found at %s, skipping parity test.", original_cpp_path)
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state

    logger.info("🧪 Running differential test for functional parity...")
    parity_result = run_differential_test(original_cpp_path, state["modernized_code"])
    parity_ok = bool(parity_result.get("parity_ok"))
    diff_text = parity_result.get("diff_text", "")

    state["is_parity_passed"] = parity_ok  # This line records whether the latest parity test passed.
    state["is_functionally_equivalent"] = parity_ok  # This line mirrors the parity result into the functionally-equivalent flag.

    if parity_ok:
        logger.info("✅ Parity Test PASSED (outputs match).")
        return state

    # On parity failure, store the unified diff so the modernizer can fix logic.
    logger.warning("❌ Parity Test FAILED (outputs differ).")
    state["diff_output"] = diff_text  # This line stores the unified diff for the modernizer prompt.
    state["error_log"] = diff_text  # This line mirrors the diff into error_log for backward-compatible prompts and logging.

    return state  # This return passes the updated state (including parity info) back into the LangGraph router.


def verify_node(state: ModernizationState) -> ModernizationState:
    """Combined verification stage: compile, then run behavioral differential test."""
    compiled_state = verifier_node(state)
    if not compiled_state.get("verification_result", {}).get("success"):
        return compiled_state

    # Run parity once after all functions are modernized.
    current_index = int(compiled_state.get("current_function_index", 0))
    modernization_order = compiled_state.get("modernization_order") or []
    if current_index < len(modernization_order):
        logger.info("Tester: deferred until final function is modernized.")
        return compiled_state

    return tester_node(compiled_state)


def surgical_router(state: ModernizationState) -> str:
    """
    Router function: Decides whether to retry modernization or proceed to END.

    After a function is successfully modernized (compilation + parity), checks
    whether more functions remain in the modernization order.  If so, routes
    back to the modernizer node to process the next function; otherwise ends.
    """
    verification_success = bool(state["verification_result"].get("success"))
    attempt_count = int(state.get("attempt_count", 0))
    current_target_function = str(state.get("current_target_stable_key") or state.get("current_target_function") or "")
    stagnation_counts = state.get("function_stagnation_counts") or {}
    current_stagnation = int(stagnation_counts.get(current_target_function, 0)) if current_target_function else 0

    if verification_success:
        # Check if more functions remain to be modernized.
        current_index = int(state.get("current_function_index", 0))
        modernization_order = state.get("modernization_order") or []
        compliance_percent = _safe_score_percent(state.get("compliance_report") or {})

        if (
            current_index >= len(modernization_order)
            and modernization_order
            and compliance_percent < _WORKFLOW_MIN_FINAL_SCORE
        ):
            if attempt_count < _WORKFLOW_MAX_ATTEMPTS_PER_FUNCTION:
                refreshed_functions = list(state.get("functions_info") or [])
                if not refreshed_functions:
                    refreshed_functions = _parse_functions_from_source(str(state.get("modernized_code") or ""))
                retry_order = [
                    str(fn.get("name") or "")
                    for fn in refreshed_functions
                    if str(fn.get("name") or "")
                ]
                if retry_order:
                    state["modernization_order"] = retry_order
                    state["current_function_index"] = 0
                    state["current_target_function"] = str(retry_order[0])
                    state["functions_info"] = list(refreshed_functions)
                    state["functions_index"] = _index_functions_by_identifier(refreshed_functions)
                    state["modernized_functions"] = {}
                    state["batched_target_functions"] = []
                else:
                    state["partial_success"] = True
                    logger.warning(
                        "Cannot build retry order for low-score rerun; finishing with PARTIAL_SUCCESS."
                    )
                    logger.info("\n🏁 Routing to END (PARTIAL_SUCCESS: low-score rerun unavailable)")
                    return "end"
                state["error_log"] = (
                    f"Modernization depth score {compliance_percent}% is below required "
                    f"minimum {_WORKFLOW_MIN_FINAL_SCORE}% for final output."
                )
                logger.info(
                    "\n🔄 Routing back to MODERNIZER (final score %d%% < required %d%%)",
                    compliance_percent,
                    _WORKFLOW_MIN_FINAL_SCORE,
                )
                return "modernizer"

            state["partial_success"] = True
            logger.warning(
                "Final score %d%% is below required %d%% after max attempts; finishing with PARTIAL_SUCCESS.",
                compliance_percent,
                _WORKFLOW_MIN_FINAL_SCORE,
            )

        if current_index < len(modernization_order):
            logger.info("\n🔄 Routing back to MODERNIZER (next function: %s)", modernization_order[current_index])
            # Reset attempt counter and error log for the next function.
            state["attempt_count"] = 0
            state["error_log"] = ""
            return "modernizer"
        logger.info("\n🏁 Routing to END (SUCCESS: all functions modernized)")
        return "end"

    if current_target_function and current_stagnation >= _WORKFLOW_STAGNANT_SCORE_LIMIT:
        current_index = int(state.get("current_function_index", 0))
        modernization_order = state.get("modernization_order") or []
        state["current_function_index"] = current_index + 1
        state["partial_success"] = True
        state["attempt_count"] = 0
        state["error_log"] = ""
        logger.warning(
            "\n⏭️ Skipping stuck function '%s' after %d unchanged-score attempt(s).",
            current_target_function,
            current_stagnation,
        )
        if current_index + 1 < len(modernization_order):
            logger.info("\n🔄 Routing to MODERNIZER (next: %s)", modernization_order[current_index + 1])
            return "modernizer"
        logger.info("\n🏁 Routing to END (PARTIAL_SUCCESS: stuck function skipped)")
        return "end"

    if attempt_count >= _WORKFLOW_MAX_ATTEMPTS_PER_FUNCTION:
        # Current function exhausted retries — skip it and try the next one.
        current_index = int(state.get("current_function_index", 0))
        modernization_order = state.get("modernization_order") or []
        state["current_function_index"] = current_index + 1
        if current_index + 1 < len(modernization_order):
            state["partial_success"] = True
            state["attempt_count"] = 0
            state["error_log"] = ""
            logger.info("\n🔄 Skipping failed function after max attempts, routing to MODERNIZER (next: %s)", modernization_order[current_index + 1])
            return "modernizer"
        state["partial_success"] = True
        logger.info("\n🏁 Routing to END (PARTIAL_SUCCESS after max attempts=%d)", _WORKFLOW_MAX_ATTEMPTS_PER_FUNCTION)
        return "end"

    if attempt_count < _WORKFLOW_MAX_ATTEMPTS_PER_FUNCTION and not verification_success:
        logger.info("\n🔄 Routing back to MODERNIZER (compiler failure, surgical retry)")
        return "modernizer"

    logger.info("\n🏁 Routing to END")
    return "end"


def build_workflow():
    """
    Builds and returns the LangGraph workflow
    """
    workflow = StateGraph(ModernizationState)
    
    # Add nodes
    workflow.add_node("prune", pruner_node)
    workflow.add_node("analyze", analyzer_node)
    workflow.add_node("transform", modernizer_node)
    workflow.add_node("verify", verify_node)
    
    # Define edges for the graph-first, self-healing pipeline
    workflow.add_edge("prune", "analyze")
    workflow.add_edge("analyze", "transform")
    workflow.add_edge("transform", "verify")
    
    # Conditional edge from tester based on compilation and parity results
    workflow.add_conditional_edges(
        "verify",
        surgical_router,
        {
            "modernizer": "transform",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("prune")
    
    return workflow.compile()


def _warmup_workflow_runtime(use_llm: bool) -> tuple[bool, str]:
    """Warm parser/model readiness before the first graph invocation."""
    try:
        CppParser().parse_string("int __workflow_warmup__() { return 0; }")
        logger.info("Warmup: parser ready.")
    except Exception as exc:
        logger.warning("Warmup: parser preflight failed: %r", exc)

    if not use_llm:
        return False, "LLM explicitly disabled by workflow configuration"

    try:
        is_healthy, message = _MODEL_BRIDGE.check_health()
    except Exception as exc:
        return False, f"Model warmup health check failed: {exc!r}"

    if is_healthy:
        logger.info("Warmup: model provider '%s' is healthy (%s)", _MODEL_PROVIDER, message)
        return True, ""
    return False, _health_failure_guidance()


def run_modernization_workflow(code: str, language: str = "c++17", source_file: str = "", output_file_path: str = ""):
    """
    Executes the modernization workflow with provided code
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING CODE MODERNIZATION WORKFLOW")
    logger.info("=" * 60)

    source_abs = os.path.abspath(source_file) if source_file else ""
    normalized_output_path = os.path.abspath(output_file_path) if output_file_path else ""
    use_llm = _workflow_use_llm()
    llm_available, llm_fallback_reason = _warmup_workflow_runtime(use_llm)
    if use_llm and not llm_available:
        logger.warning("Warmup degraded to no-LLM fallback mode: %s", llm_fallback_reason)

    if normalized_output_path:
        output_dir = os.path.dirname(normalized_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Initialize state
    initial_state = ModernizationState(
        code=code,
        language=language,
        analysis="",
        dependency_map={},
        call_graph_data={},
        impact_map={},
        orphans=[],
        analysis_report="",
        modernized_code="",
        verification_result={},
        error_log="",
        attempt_count=0,
        is_parity_passed=False,
        is_functionally_equivalent=False,
        diff_output="",
        feedback_loop_count=0,
        modernization_order=[],
        modernized_functions={},
        current_function_index=0,
        partial_success=False,
        last_working_code=code,
        current_target_function="",
        functions_info=[],
        functions_index={},
        current_function_name="",
        current_function_span=(0, 0),
        project_map={},
        source_file=source_abs,
        output_file_path=normalized_output_path,
        legacy_findings=[],
        compliance_report={},
        use_llm=use_llm,
        llm_available=llm_available,
        llm_fallback_reason=llm_fallback_reason,
        batched_target_functions=[],
        current_target_stable_key="",
        last_function_scores={},
        function_stagnation_counts={},
    )
    
    # Build and run the workflow
    _MODEL_BRIDGE.start_modernization_trace(
        input_payload={
            "operation": "workflow_run",
            "language": language,
            "source_file": source_abs,
            "source_length": len(code),
        }
    )

    graph = build_workflow()
    final_state = graph.invoke(initial_state)

    if _STRICT_CPP23_MODE:
        compliance_percent = int(final_state.get("compliance_report", {}).get("percent", 0))
        strict_ok = (
            bool(final_state.get("verification_result", {}).get("success"))
            and bool(final_state.get("is_parity_passed", False))
            and compliance_percent >= _STRICT_CPP23_TARGET_PERCENT
        )
        if not strict_ok:
            strict_message = (
                "STRICT MODE FAILURE: final output did not satisfy required C++23 compliance target "
                f"({_STRICT_CPP23_TARGET_PERCENT}%). Final score: {compliance_percent}%."
            )
            final_state["verification_result"] = {
                "success": False,
                "errors": [strict_message],
                "warnings": [],
                "compilation_time_ms": final_state.get("verification_result", {}).get("compilation_time_ms", 0),
                "raw_stdout": final_state.get("verification_result", {}).get("raw_stdout", ""),
                "raw_stderr": final_state.get("verification_result", {}).get("raw_stderr", ""),
            }
            final_state["error_log"] = strict_message
            logger.error("❌ %s", strict_message)
        else:
            logger.info(
                "✅ STRICT MODE PASSED: C++23 compliance %d%% >= %d%%",
                compliance_percent, _STRICT_CPP23_TARGET_PERCENT
            )

    logger.info("\n" + "=" * 60)
    logger.info("📊 MODERNIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Language: %s", final_state['language'])
    logger.info("Total Attempts: %d", final_state['attempt_count'])
    logger.info("Verification Success: %s", final_state['verification_result'].get('success'))
    logger.info("\nFinal Modernized Code:\n%s", final_state['modernized_code'])

    # ------------------------------------------------------------------
    # Save modernized code to a separate output file
    # ------------------------------------------------------------------
    output_path: Optional[str] = final_state.get("output_file_path")
    if not output_path:
        # Derive output path from source_file in state, or fall back to cwd.
        source_path: str = final_state.get("source_file", "") or ""
        if source_path:
            base, _ext = os.path.splitext(source_path)
            output_path = f"{base}_modernized.cpp"
        else:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output_modernized.cpp",
            )

    try:
        with open(output_path, "w", encoding="utf-8") as _out:
            _out.write(final_state["modernized_code"])
        logger.info("💾 Modernized code saved to: %s", output_path)
    except OSError as _save_err:
        logger.warning("Could not save output file: %s", _save_err)

    return final_state


if __name__ == "__main__":  # This block runs only when this file is executed as a script (e.g. python -m agents.workflow), not when imported.
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This line gets the project root directory where test.cpp lives.
    _test_cpp_path = os.path.join(_base_dir, "test.cpp")  # This line builds the full path to test.cpp in the project root.

    if _workflow_use_llm() and not check_model_health():
        logger.warning("%s", _health_failure_guidance())
        logger.warning("Continuing with deterministic no-LLM fallback mode.")

    try:
        with open(_test_cpp_path, "r", encoding="utf-8") as f:
            cpp_code = f.read()
        logger.info("📄 Loaded test.cpp (%d characters)", len(cpp_code))
    except FileNotFoundError:
        logger.error("❌ test.cpp not found at %s", _test_cpp_path)
        exit(1)
        
    result = run_modernization_workflow(cpp_code, language="c++17", source_file=_test_cpp_path)  # This line invokes the full workflow (prune -> analyze -> transform -> verify with retries) on the loaded code.
    verification_ok = bool(result.get("verification_result", {}).get("success"))
    fail_on_verification = os.environ.get("WORKFLOW_FAIL_ON_VERIFICATION", "0").strip() == "1"
    if not verification_ok and fail_on_verification:
        exit(2)
    if not verification_ok:
        logger.warning("Run completed with verification warnings. Set WORKFLOW_FAIL_ON_VERIFICATION=1 to enforce non-zero exit.")