from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import logging
import os
import re
import threading
import tempfile
from typing import Any

from networkx.drawing.nx_pydot import write_dot

from core.ast_modernizer import ASTModernizationDetector
from core.graph import DependencyGraph
from core.local_ollama_bridge import CPP_MODERNIZATION_SYSTEM_PROMPT
from core.differential_tester import compile_cpp_source, run_differential_test
from core.inspect_parser import score_cpp23_compliance
from core.rule_modernizer import apply_modernization_rules


_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_SIMILARITY_THRESHOLD = 0.70
_MIN_CHANGE_LINES = 2
_MODERN_SKIP_THRESHOLD_PERCENT = 70
_MAX_FUNCTION_CHARS = 3000
_LLM_CACHE_FILENAME = ".local_llm_cache.json"
_FUNCTION_CACHE_DIR = "cache"
_FUNCTION_CACHE_FILENAME = "modernization_cache.json"
_RULE_ONLY_COMPLEXITY_THRESHOLD = 3
_CACHE_VERSION_SALT = os.environ.get("MODERNIZATION_CACHE_VERSION", "v2").strip() or "v2"
DEBUG_MODE = True

try:
    _langfuse_module = importlib.import_module("langfuse")
    Langfuse = getattr(_langfuse_module, "Langfuse", None)
except Exception:  # pragma: no cover - optional dependency at runtime
    Langfuse = None

_logger = logging.getLogger(__name__)


def _log(tag: str, message: str) -> None:
    _logger.debug("[%s] %s", tag, message)
    if DEBUG_MODE:
        print(f"[{tag}] {message}")


def code_similarity_ratio(old_code: str, new_code: str) -> float:
    return difflib.SequenceMatcher(None, old_code, new_code).ratio()


def has_meaningful_diff(old_code: str, new_code: str, min_lines: int = _MIN_CHANGE_LINES) -> bool:
    """Return True if at least min_lines non-header lines differ in the unified diff."""
    diff = list(difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm=""))
    changed = sum(
        1 for line in diff
        if line.startswith(("+", "-")) and not line.startswith(("++", "--"))
    )
    return changed >= min_lines


def is_similar_code(old_code: str, new_code: str, threshold: float = _SIMILARITY_THRESHOLD) -> bool:
    ratio = code_similarity_ratio(old_code, new_code)
    return ratio > threshold


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _extract_error_line_numbers(compiler_text: str) -> list[int]:
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


def _adaptive_similarity_threshold(function_body: str) -> float:
    line_count = max(1, len(function_body.splitlines()))
    if line_count <= 10:
        return 0.88
    if line_count <= 25:
        return 0.82
    if line_count <= 60:
        return 0.78
    return _SIMILARITY_THRESHOLD


class FunctionModernizer:
    def __init__(self, parser, llm):
        self.parser = parser
        self.llm = llm
        self.ast_detector = ASTModernizationDetector(parser)
        self._project_map: dict[str, Any] = {}
        self._modernized_fqns: set[str] = set()
        self._cache_path = os.path.join(os.getcwd(), _LLM_CACHE_FILENAME)
        self._disable_prompt_cache = _read_bool_env("MODERNIZER_DISABLE_PROMPT_CACHE", False)
        self._disable_function_cache = _read_bool_env("MODERNIZER_DISABLE_FUNCTION_CACHE", False)
        self._llm_cache: dict[str, str] = self._load_cache()
        self._function_cache_dir = os.path.join(os.getcwd(), _FUNCTION_CACHE_DIR)
        os.makedirs(self._function_cache_dir, exist_ok=True)
        self._function_cache_path = os.path.join(self._function_cache_dir, _FUNCTION_CACHE_FILENAME)
        self._function_cache: dict[str, str] = self._load_function_cache()
        if not os.path.isfile(self._function_cache_path):
            self._save_function_cache()
        self.file_lock = threading.Lock()
        self._llm_model_name = str(
            getattr(getattr(self.llm, "config", None), "model", "unknown-model")
            or "unknown-model"
        )
        self.langfuse = None
        if Langfuse is not None:
            try:
                self.langfuse = Langfuse()
            except Exception:
                self.langfuse = None
        self.stats = {
            "functions_analyzed": 0,
            "functions_modernized": 0,
            "rule_transformations": 0,
            "llm_transformations": 0,
            "legacy_constructs_detected": 0,
            "compile_retries": 0,
        }
        self.transformation_types: dict[str, int] = {}

    def _invoke_llm(self, prompt: str) -> str:
        generation = None
        if self.langfuse is not None:
            try:
                generation = self.langfuse.generation(
                    name="llm_modernization",
                    input=prompt,
                    model=self._llm_model_name,
                    metadata={"prompt_length": len(prompt), "cache_version": _CACHE_VERSION_SALT},
                )
            except Exception:
                generation = None
        try:
            result = self.llm.chat_completion(
                system_prompt=CPP_MODERNIZATION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            if generation is not None:
                try:
                    generation.end(output=result)
                except Exception:
                    pass
            return result
        except Exception as exc:
            message = str(exc)
            if generation is not None:
                try:
                    generation.end(level="ERROR", status_message=message)
                except Exception:
                    pass
            raise RuntimeError(f"LOCAL_LLM_FAILED: {message}") from exc

    def modernize_file(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"C++ file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            original_source = fh.read()

        self._project_map = self.parser.parse_file(file_path)
        functions = self._project_map.get("functions") or {}
        if not isinstance(functions, dict) or not functions:
            return original_source

        dep_graph = DependencyGraph(
            functions_info=list(functions.values()),
            types_info=self._project_map.get("types") or [],
        )
        try:
            write_dot(dep_graph.graph, "dependency_graph.dot")
        except Exception:
            pass
        order = dep_graph.get_modernization_order()

        modernization_fqns = self._resolve_fqn_order(order, functions)
        for fqn in modernization_fqns:
            if fqn in self._modernized_fqns:
                continue
            self.modernize_function(file_path, fqn)
            self._modernized_fqns.add(fqn)

        with open(file_path, "r", encoding="utf-8") as fh:
            modernized_source = fh.read()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cpp",
            delete=False,
            encoding="utf-8",
        ) as original_tmp:
            original_tmp.write(original_source)
            original_cpp_path = original_tmp.name

        try:
            parity_result = run_differential_test(
                original_cpp_path=original_cpp_path,
                modernized_code=modernized_source,
            )
        finally:
            if os.path.exists(original_cpp_path):
                os.remove(original_cpp_path)

        if not bool(parity_result.get("parity_ok")):
            diff_text = str(parity_result.get("diff_text") or "Differential test failed.")
            raise RuntimeError(
                "Final differential test failed after function-level modernization.\n"
                + diff_text
            )

        return modernized_source

    def modernize_function(self, file_path: str, fqn: str) -> None:
        max_attempts = 3
        min_modernization_score = 20
        compiler_feedback = ""
        self.stats["functions_analyzed"] += 1

        for _attempt in range(1, max_attempts + 1):
            self._project_map = self.parser.parse_file(file_path)
            functions = self._project_map.get("functions") or {}
            if fqn not in functions:
                return

            context = self.parser.get_context_for_function(fqn)
            function_meta = functions[fqn]
            with open(file_path, "r", encoding="utf-8") as fh:
                current_source = fh.read()

            function_body = self._extract_function_source(current_source, function_meta)
            if not function_body.strip():
                return

            if len(function_body) > _MAX_FUNCTION_CHARS:
                _log("SKIP", f"Function '{fqn}' is {len(function_body)} chars (>{_MAX_FUNCTION_CHARS}). Skipping.")
                return

            function_ast = self.ast_detector.get_function_ast_node(function_body)
            if function_ast is None:
                patterns: dict[str, int] = {}
                detected_types: list[str] = []
            else:
                pattern_result = self.ast_detector.detect_legacy_patterns(
                    function_ast,
                    function_body.encode("utf-8"),
                )
                if isinstance(pattern_result, dict) and isinstance(pattern_result.get("counts"), dict):
                    raw_counts = pattern_result.get("counts") or {}
                    patterns = {
                        str(key): int(value)
                        for key, value in raw_counts.items()
                        if isinstance(key, str)
                    }
                    raw_detected = pattern_result.get("detected") or []
                    detected_types = [str(item) for item in raw_detected]
                else:
                    patterns = {
                        str(key): int(value)
                        for key, value in (pattern_result or {}).items()
                        if isinstance(key, str)
                    }
                    detected_types = [name for name, count in patterns.items() if int(count) > 0]
            _log("AST", f"Detected patterns: {patterns} | active={detected_types}")
            detected_count = sum(int(v) for v in patterns.values())
            self.stats["legacy_constructs_detected"] += detected_count
            patterns_text = self._format_patterns(patterns, detected_types)

            function_score = score_cpp23_compliance(function_body)
            if int(function_score.get("percent", 0) or 0) > _MODERN_SKIP_THRESHOLD_PERCENT:
                _log("SKIP", "Function already modern. Skipping.")
                return

            raw_complexity = function_meta.get("complexity", 0)
            try:
                complexity = int(raw_complexity)
            except (TypeError, ValueError):
                complexity = 0

            _rule_preview, applied_rules = self._apply_rules_to_function_body(
                function_body,
                patterns,
            )
            function_for_llm = function_body
            _log("RULES", f"Applied {len(applied_rules)} rule(s): {applied_rules}")
            if applied_rules:
                self.stats["rule_transformations"] += len(applied_rules)
                for rule in applied_rules:
                    self.transformation_types[rule] = self.transformation_types.get(rule, 0) + 1

            # Skip LLM if no legacy constructs and no rules applied — nothing to modernize.
            if detected_count == 0 and not applied_rules:
                return

            if complexity < _RULE_ONLY_COMPLEXITY_THRESHOLD:
                _log(
                    "SKIP",
                    f"Rule-only path for '{fqn}' (complexity={complexity}).",
                )
                return

            function_hash = str(
                function_meta.get("function_hash")
                or hashlib.sha256(function_body.encode("utf-8")).hexdigest()
            )
            function_context_blob = json.dumps(
                {
                    "fqn": fqn,
                    "referenced_types": context.get("referenced_type_definitions")
                    or context.get("type_bundle")
                    or {},
                    "called_signatures": context.get("called_function_signatures") or {},
                    "cache_version": _CACHE_VERSION_SALT,
                },
                sort_keys=True,
                default=str,
            )
            function_cache_key = hashlib.sha256(
                f"{function_hash}|{function_context_blob}".encode("utf-8")
            ).hexdigest()
            cached_modernized_function = None
            if not self._disable_function_cache:
                cached_modernized_function = self._function_cache.get(function_cache_key)
            used_function_cache = False

            if cached_modernized_function is not None:
                _log("LLM", f"Function-hash cache hit for '{fqn}'. Reusing modernized code.")
                modernized_function = cached_modernized_function
                used_function_cache = True
            else:
                prompt = self._build_prompt(
                    function_body=function_for_llm,
                    referenced_types=context.get("referenced_type_definitions")
                    or context.get("type_bundle")
                    or {},
                    called_signatures=context.get("called_function_signatures") or {},
                    legacy_patterns=patterns_text,
                    applied_rules=applied_rules,
                    compiler_feedback=compiler_feedback,
                )

                cache_key = hashlib.sha256((prompt + f"|{_CACHE_VERSION_SALT}").encode("utf-8")).hexdigest()
                cached_response = None
                if not self._disable_prompt_cache:
                    cached_response = self._llm_cache.get(cache_key)
                if cached_response is not None:
                    _log("LLM", "Cache hit — reusing cached response.")
                    raw_response = cached_response
                else:
                    _log("LLM", f"Sending prompt ({len(prompt)} chars) to LLM (attempt {_attempt}).")
                    try:
                        raw_response = self._invoke_llm(prompt)
                    except Exception as exc:
                        error_text = str(exc)
                        _log("LLM", f"Local LLM unavailable while modernizing '{fqn}': {error_text}. Skipping function for this run.")
                        return
                    if not self._disable_prompt_cache:
                        self._llm_cache[cache_key] = raw_response
                        self._save_cache()
                modernized_function = self._clean_model_code(raw_response)
                _log("LLM", f"Response received ({len(modernized_function)} chars).")

            if not modernized_function.strip():
                _log("LLM", "Model returned empty function output.")
                compiler_feedback = "Model returned empty function output."
                continue

            diff_lines = list(difflib.unified_diff(
                function_body.splitlines(),
                modernized_function.splitlines(),
                fromfile="original",
                tofile="modernized",
                lineterm="",
            ))
            _log("DIFF", "\n".join(diff_lines) if diff_lines else "(no diff)")
            changed_lines = sum(
                1 for line in diff_lines
                if line.startswith(("+", "-")) and not line.startswith(("++", "--"))
            )
            similarity_ratio = code_similarity_ratio(function_body, modernized_function)
            similarity_threshold = _adaptive_similarity_threshold(function_body)
            _log("VERIFY", f"Similarity: {similarity_ratio:.2f}, changed lines: {changed_lines}")
            if similarity_ratio > similarity_threshold or changed_lines < _MIN_CHANGE_LINES:
                _log("VERIFY", "No meaningful modernization detected.")
                compiler_feedback = (
                    "LLM returned nearly identical code. You MUST rewrite with meaningful modern C++23 improvements "
                    "while preserving behavior. At least one legacy construct must be replaced."
                )
                if _attempt >= max_attempts:
                    _log("VERIFY", "No meaningful modernization after retries. Skipping function.")
                    return
                continue

            self.replace_function(file_path, fqn, modernized_function)
            self._project_map = self.parser.parse_file(file_path)

            with open(file_path, "r", encoding="utf-8") as fh:
                replaced_source = fh.read()
            compile_result = compile_cpp_source(replaced_source)
            _log("VERIFY", f"Compile: {'success' if compile_result.get('success') else 'failed'} for '{fqn}'.")
            if bool(compile_result.get("success")):
                # Hard guard: reject truly unchanged code even if compile passed.
                if is_similar_code(function_body, modernized_function):
                    _log("VERIFY", "Rejecting modernization — code unchanged after compile.")
                    with self.file_lock:
                        with open(file_path, "w", encoding="utf-8") as fh:
                            fh.write(current_source)
                    self._project_map = self.parser.parse_file(file_path)
                    compiler_feedback = (
                        "Code was unchanged. You MUST apply at least one meaningful modernization."
                    )
                    continue
                modernization_score = int(
                    score_cpp23_compliance(modernized_function).get("percent", 0) or 0
                )
                code_changed = not is_similar_code(
                    function_body,
                    modernized_function,
                    threshold=_SIMILARITY_THRESHOLD,
                )
                if modernization_score < min_modernization_score and code_changed:
                    # Avoid looping on quality-only retries when no meaningful change was made.
                    if _attempt >= max_attempts:
                        return
                    with self.file_lock:
                        with open(file_path, "w", encoding="utf-8") as fh:
                            fh.write(current_source)
                    self._project_map = self.parser.parse_file(file_path)
                    compiler_feedback = (
                        "Compilation passed but modernization quality score was low "
                        f"({modernization_score} < {min_modernization_score}). "
                        "Improve modernization while preserving behavior."
                    )
                    continue
                if code_changed:
                    self.stats["llm_transformations"] += 1
                    self.stats["functions_modernized"] += 1
                    self.transformation_types["llm_rewrite"] = (
                        self.transformation_types.get("llm_rewrite", 0) + 1
                    )
                    if not self._disable_function_cache:
                        self._function_cache[function_cache_key] = modernized_function
                        self._save_function_cache()
                return

            # Keep failed replacement so downstream verifier can analyze it.
            self.stats["compile_retries"] += 1
            _log("VERIFY", "Compile failed, keeping LLM modernization for analysis.")
            if used_function_cache and function_cache_key in self._function_cache:
                self._function_cache.pop(function_cache_key, None)
                self._save_function_cache()

            compiler_feedback = str(
                compile_result.get("raw_stderr")
                or "\n".join(compile_result.get("errors") or [])
                or "Compilation failed with unknown error."
            )
            error_lines = _extract_error_line_numbers(compiler_feedback)
            if error_lines:
                first_error_line = error_lines[0]
                snippet = _get_code_snippet_by_line(modernized_function, first_error_line)
                if snippet:
                    compiler_feedback += (
                        f"\n\nFocus on line {first_error_line} in the rewritten function:\n{snippet}"
                    )
            return

        raise RuntimeError(
            f"Failed to modernize function '{fqn}' after {max_attempts} attempts.\n"
            f"Last compiler errors:\n{compiler_feedback}"
        )

    def _cache_key(self, function_code: str) -> str:
        return hashlib.sha256(function_code.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, str]:
        if not os.path.isfile(self._cache_path):
            return {}
        try:
            with open(self._cache_path, "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            if isinstance(parsed, dict):
                return {
                    str(key): str(value)
                    for key, value in parsed.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
        except Exception:
            return {}
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._llm_cache, fh, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def _load_function_cache(self) -> dict[str, str]:
        if not os.path.isfile(self._function_cache_path):
            return {}
        try:
            with open(self._function_cache_path, "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            if isinstance(parsed, dict):
                return {
                    str(key): str(value)
                    for key, value in parsed.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
        except Exception:
            return {}
        return {}

    def _save_function_cache(self) -> None:
        try:
            with open(self._function_cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._function_cache, fh, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def replace_function(self, file_path: str, fqn: str, new_code: str) -> None:
        functions = self._project_map.get("functions") or {}
        function_meta = functions.get(fqn)
        if not isinstance(function_meta, dict):
            raise ValueError(f"Function metadata not found for FQN: {fqn}")

        start = function_meta.get("start_byte")
        end = function_meta.get("end_byte")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Missing byte offsets for function: {fqn}")

        with open(file_path, "r", encoding="utf-8") as fh:
            source_text = fh.read()

        source_bytes = source_text.encode("utf-8")
        if not (0 <= start <= end <= len(source_bytes)):
            raise ValueError(f"Invalid byte offsets for function: {fqn}")

        updated_bytes = source_bytes[:start] + new_code.strip().encode("utf-8") + source_bytes[end:]
        updated = updated_bytes.decode("utf-8", errors="strict")

        with self.file_lock:
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(updated)

    def _resolve_fqn_order(self, order: list[str], functions: dict[str, dict[str, Any]]) -> list[str]:
        name_to_fqns: dict[str, list[str]] = {}
        for fqn, meta in functions.items():
            simple_name = str(meta.get("name") or "")
            if simple_name:
                name_to_fqns.setdefault(simple_name, []).append(fqn)

        resolved: list[str] = []
        for item in order:
            if item in functions:
                resolved.append(item)
                continue
            for fqn in sorted(name_to_fqns.get(item, [])):
                resolved.append(fqn)

        # Preserve any functions not represented in dependency order.
        seen = set(resolved)
        for fqn in sorted(functions.keys()):
            if fqn not in seen:
                resolved.append(fqn)
        return resolved

    def _extract_function_source(self, source_text: str, function_meta: dict[str, Any]) -> str:
        start = function_meta.get("start_byte")
        end = function_meta.get("end_byte")
        if not isinstance(start, int) or not isinstance(end, int):
            return ""
        source_bytes = source_text.encode("utf-8")
        if not (0 <= start <= end <= len(source_bytes)):
            return ""
        return source_bytes[start:end].decode("utf-8", errors="strict")

    def _build_prompt(
        self,
        function_body: str,
        referenced_types: Any,
        called_signatures: Any,
        legacy_patterns: str,
        applied_rules: list[str],
        compiler_feedback: str,
    ) -> str:
        rules_text = self._format_applied_rules(applied_rules)
        prompt = (
            "You are a senior C++ modernization engineer.\n\n"
            "Your task is to rewrite the following legacy C++ function using C++23 features.\n\n"
            "STRICT REQUIREMENTS:\n"
            "- Preserve the exact behavior.\n"
            "- Improve the code using modern C++ features where appropriate.\n"
            "- Replace raw pointers with smart pointers if applicable.\n"
            "- Replace manual loops with ranges if possible.\n"
            "- Avoid NULL, use nullptr.\n"
            "- Avoid raw new/delete.\n"
            "- Prefer std::expected for error handling.\n"
            "- Prefer std::span for buffer handling.\n"
            "- Prefer std::format or std::print instead of printf.\n\n"
            "IMPORTANT:\n"
            "You MUST rewrite the function even if the improvement is small.\n"
            "Do not return the original code unchanged.\n"
            "If the code contains any legacy constructs listed below, you MUST replace them.\n"
            "Do NOT return the same code — at least one modernization MUST be applied.\n\n"
            "Return ONLY the modernized function.\n\n"
            "Function:\n"
            f"{function_body}\n\n"
            "Detected legacy constructs:\n"
            f"{legacy_patterns}\n\n"
            "Rewrite the function to eliminate these constructs using modern C++23 features.\n"
            "Do not rewrite unrelated logic.\n\n"
            "Deterministic modernization hints (apply only when semantically safe):\n"
            f"{rules_text}\n\n"
            "Referenced types:\n"
            f"{self._format_context_block(referenced_types)}\n\n"
            "Called functions:\n"
            f"{self._format_context_block(called_signatures)}\n\n"
            "Return ONLY the full modernized function."
        )

        if compiler_feedback.strip():
            prompt += (
                "\n\nThe previous modernization failed to compile. "
                "Fix the function based on these compiler errors:\n"
                f"{compiler_feedback.strip()}"
            )

        return prompt

    def _apply_rules_to_function_body(
        self,
        function_source: str,
        detected_patterns: dict[str, int] | None = None,
    ) -> tuple[str, list[str]]:
        """Apply regex modernization rules to function body only, preserving signature."""
        first_brace = function_source.find("{")
        last_brace = function_source.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return function_source, []

        signature_prefix = function_source[: first_brace + 1]
        body_text = function_source[first_brace + 1:last_brace]
        closing_suffix = function_source[last_brace:]

        updated_body, applied_rules = apply_modernization_rules(
            body_text,
            detected_patterns=detected_patterns,
        )
        _log("RULES", f"Applied rules: {applied_rules}")
        if not applied_rules:
            return function_source, []

        return signature_prefix + updated_body + closing_suffix, applied_rules

    def _format_applied_rules(self, applied_rules: list[str]) -> str:
        if not applied_rules:
            return "(none)"
        return "\n".join(f"- {rule}" for rule in applied_rules)

    def _format_patterns(self, patterns: dict[str, int], detected_types: list[str] | None = None) -> str:
        active = [f"{key}: {count}" for key, count in patterns.items() if int(count) > 0]
        if not active:
            return "(none)"
        if detected_types:
            return "Detected types: " + ", ".join(sorted(set(detected_types))) + "\n" + "\n".join(active)
        return "\n".join(active)

    def _format_context_block(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip() or "(none)"
        if isinstance(value, dict):
            if not value:
                return "(none)"
            parts: list[str] = []
            for key in sorted(value.keys()):
                parts.append(f"{key}: {value[key]}")
            return "\n".join(parts)
        if isinstance(value, list):
            if not value:
                return "(none)"
            return "\n".join(str(item) for item in value)
        return "(none)"

    def _clean_model_code(self, text: str) -> str:
        match = _FENCE_RE.search(text)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"```(?:[^\n]*)\n?", "", text)
        cleaned = re.sub(r"^\s*(assistant|model|ai)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def print_report(self) -> None:
        print("\n==== MODERNIZATION REPORT ====")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        if self.transformation_types:
            print("\n  Transformation types:")
            for transform, count in sorted(
                self.transformation_types.items(), key=lambda x: -x[1]
            ):
                print(f"    {transform}: {count}")
        print("==============================\n")
