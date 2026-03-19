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
from dataclasses import dataclass
from typing import Any

from networkx.drawing.nx_pydot import write_dot

from core.ast_modernizer import ASTModernizationDetector
from core.graph import DependencyGraph
from core.openai_bridge import CPP_MODERNIZATION_SYSTEM_PROMPT
from core.differential_tester import compile_cpp_source, run_differential_test
from core.inspect_parser import score_cpp23_compliance
from core.rule_modernizer import apply_modernization_rules
from core.similarity import code_similarity_ratio
from core.rag import get_global_rag


_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_SIMILARITY_THRESHOLD = 0.70
_MIN_CHANGE_LINES = 2
_MODERN_SKIP_THRESHOLD_PERCENT = 70
_MAX_FUNCTION_CHARS = 3000
_RULE_ONLY_COMPLEXITY_THRESHOLD = 3


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _default_cache_root() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return os.path.join(local_app_data, "cpp-modernizer")
    return os.path.join(os.path.expanduser("~"), ".cache", "cpp-modernizer")


@dataclass(frozen=True)
class ModernizerConfig:
    chunk_max_chars: int
    cache_version_salt: str
    enable_reflection: bool
    reflection_max_iters: int
    modern_retry_threshold_percent: int
    enable_rag: bool
    rag_top_k: int
    debug_mode: bool
    prompt_cache_path: str
    function_cache_path: str
    similarity_short_threshold: float
    similarity_medium_threshold: float
    similarity_long_threshold: float
    similarity_default_threshold: float
    llm_temperature: float
    enable_rule_based_rewrites: bool

    @classmethod
    def from_env(cls) -> "ModernizerConfig":
        cache_root = os.environ.get("MODERNIZER_CACHE_DIR", "").strip() or _default_cache_root()
        prompt_cache_path = os.path.join(cache_root, "prompt_cache.json")
        function_cache_path = os.path.join(cache_root, "modernization_cache.json")
        return cls(
            chunk_max_chars=_env_int("MODERNIZER_CHUNK_MAX_CHARS", 1800, minimum=200),
            cache_version_salt=os.environ.get("MODERNIZATION_CACHE_VERSION", "v2").strip() or "v2",
            enable_reflection=_read_bool_env("ENABLE_REFLECTION", True),
            reflection_max_iters=_env_int("REFLECTION_MAX_ITERS", 2, minimum=0),
            modern_retry_threshold_percent=_env_int("MODERN_RETRY_THRESHOLD_PERCENT", 50, minimum=0, maximum=100),
            enable_rag=_read_bool_env("ENABLE_RAG", False),
            rag_top_k=_env_int("RAG_TOP_K", 3, minimum=1),
            debug_mode=_read_bool_env("MODERNIZER_DEBUG", False),
            prompt_cache_path=prompt_cache_path,
            function_cache_path=function_cache_path,
            similarity_short_threshold=_env_float("SIMILARITY_THRESHOLD_SHORT", 0.88),
            similarity_medium_threshold=_env_float("SIMILARITY_THRESHOLD_MEDIUM", 0.82),
            similarity_long_threshold=_env_float("SIMILARITY_THRESHOLD_LONG", 0.78),
            similarity_default_threshold=_env_float("SIMILARITY_THRESHOLD_DEFAULT", _SIMILARITY_THRESHOLD),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
            # Rule-based rewrites are heuristic/regex-driven; keep disabled by default.
            enable_rule_based_rewrites=_read_bool_env("ENABLE_RULE_BASED_REWRITES", False),
        )


_CONFIG = ModernizerConfig.from_env()

_CHUNK_MAX_CHARS = _CONFIG.chunk_max_chars
_CACHE_VERSION_SALT = _CONFIG.cache_version_salt
_ENABLE_REFLECTION = _CONFIG.enable_reflection
_REFLECTION_MAX_ITERS = _CONFIG.reflection_max_iters
_MODERN_RETRY_THRESHOLD_PERCENT = _CONFIG.modern_retry_threshold_percent
_ENABLE_RAG = _CONFIG.enable_rag
_RAG_TOP_K = _CONFIG.rag_top_k
DEBUG_MODE = _CONFIG.debug_mode
_PROMPT_CACHE_PATH = _CONFIG.prompt_cache_path
_FUNCTION_CACHE_PATH = _CONFIG.function_cache_path
_ENABLE_RULE_BASED_REWRITES = _CONFIG.enable_rule_based_rewrites


_SIMILARITY_SHORT_THRESHOLD = _CONFIG.similarity_short_threshold
_SIMILARITY_MEDIUM_THRESHOLD = _CONFIG.similarity_medium_threshold
_SIMILARITY_LONG_THRESHOLD = _CONFIG.similarity_long_threshold
_SIMILARITY_DEFAULT_THRESHOLD = _CONFIG.similarity_default_threshold
_LLM_TEMPERATURE = _CONFIG.llm_temperature

try:
    _langfuse_module = importlib.import_module("langfuse")
    Langfuse = getattr(_langfuse_module, "Langfuse", None)
except Exception:  # pragma: no cover - optional dependency at runtime
    Langfuse = None

_logger = logging.getLogger(__name__)


def _log(tag: str, message: str) -> None:
    _logger.debug("[%s] %s", tag, message)


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
        return _SIMILARITY_SHORT_THRESHOLD
    if line_count <= 25:
        return _SIMILARITY_MEDIUM_THRESHOLD
    if line_count <= 60:
        return _SIMILARITY_LONG_THRESHOLD
    return _SIMILARITY_DEFAULT_THRESHOLD


class FunctionModernizer:
    _global_lock_guard = threading.Lock()
    _file_locks: dict[str, threading.Lock] = {}

    def __init__(self, parser, llm):
        self.parser = parser
        self.llm = llm
        self.ast_detector = ASTModernizationDetector(parser)
        self._project_map: dict[str, Any] = {}
        self._modernized_fqns: set[str] = set()
        self._cache_path = _PROMPT_CACHE_PATH
        self._disable_prompt_cache = _read_bool_env("MODERNIZER_DISABLE_PROMPT_CACHE", False)
        self._disable_function_cache = _read_bool_env("MODERNIZER_DISABLE_FUNCTION_CACHE", False)
        os.makedirs(os.path.dirname(self._cache_path) or os.getcwd(), exist_ok=True)
        self._llm_cache: dict[str, str] = self._load_cache()
        self._function_cache_path = _FUNCTION_CACHE_PATH
        os.makedirs(os.path.dirname(self._function_cache_path) or os.getcwd(), exist_ok=True)
        self._function_cache: dict[str, str] = self._load_function_cache()
        if not os.path.isfile(self._function_cache_path):
            self._save_function_cache()
        self.file_lock = threading.Lock()
        self.rag = get_global_rag(enabled=_ENABLE_RAG)
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
        self._dependency_graph_cache: dict[str, DependencyGraph] = {}

    def _dependency_graph_cache_key(
        self,
        functions: dict[str, Any],
        types_info: list[dict[str, Any]],
    ) -> str:
        """Return a stable cache key for dependency graph inputs.

        The key is based on parser metadata fields that affect call graph shape.
        """
        entries: list[dict[str, Any]] = []
        for unique_fqn, meta in sorted(functions.items(), key=lambda item: str(item[0])):
            if not isinstance(meta, dict):
                continue
            call_details = meta.get("call_details")
            entries.append(
                {
                    "unique_fqn": str(unique_fqn),
                    "name": str(meta.get("name") or ""),
                    "fqn": str(meta.get("fqn") or ""),
                    "signature_hash": str(meta.get("signature_hash") or ""),
                    "parameters": meta.get("parameters") if isinstance(meta.get("parameters"), list) else [],
                    "calls": meta.get("calls") if isinstance(meta.get("calls"), list) else [],
                    "call_details": call_details if isinstance(call_details, list) else [],
                }
            )

        payload = {
            "functions": entries,
            "types": types_info,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return digest

    def _get_or_build_dependency_graph(
        self,
        functions: dict[str, Any],
        types_info: list[dict[str, Any]],
    ) -> DependencyGraph:
        cache_key = self._dependency_graph_cache_key(functions, types_info)
        cached = self._dependency_graph_cache.get(cache_key)
        if cached is not None:
            return cached

        graph = DependencyGraph(
            functions_info=list(functions.values()),
            types_info=types_info,
        )
        self._dependency_graph_cache = {cache_key: graph}
        return graph

    @classmethod
    def _lock_for_file(cls, file_path: str) -> threading.Lock:
        normalized = os.path.abspath(file_path)
        with cls._global_lock_guard:
            lock = cls._file_locks.get(normalized)
            if lock is None:
                lock = threading.Lock()
                cls._file_locks[normalized] = lock
        return lock

    def _restore_source_snapshot(self, file_path: str, source_text: str) -> None:
        """Atomically restore file content from a known-good snapshot."""
        file_lock = self._lock_for_file(file_path)
        directory = os.path.dirname(os.path.abspath(file_path)) or os.getcwd()
        with file_lock:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=".restore_",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(source_text)
                tmp_path = tmp.name
            os.replace(tmp_path, file_path)

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
                temperature=_LLM_TEMPERATURE,
            )
            if generation is not None:
                try:
                    usage_details = self._latest_usage_details()
                    if usage_details:
                        generation.end(output=result, usage_details=usage_details)
                    else:
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

    def _latest_usage_details(self) -> dict[str, int] | None:
        """Best-effort extraction of token usage from bridge state, if exposed."""
        candidates: list[Any] = []
        getter = getattr(self.llm, "get_last_usage", None)
        if callable(getter):
            try:
                candidates.append(getter())
            except Exception:
                pass
        candidates.append(getattr(self.llm, "last_usage", None))
        candidates.append(getattr(self.llm, "_last_usage", None))

        for item in candidates:
            if not isinstance(item, dict):
                continue
            if any(key in item for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                return {
                    "prompt_tokens": int(item.get("prompt_tokens") or 0),
                    "completion_tokens": int(item.get("completion_tokens") or 0),
                    "total_tokens": int(item.get("total_tokens") or 0),
                }
        return None

    @staticmethod
    def _safe_percent(score_payload: dict[str, Any]) -> int:
        raw = score_payload.get("percent", 0)
        try:
            return int(float(str(raw)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _split_function_signature_and_body(function_source: str) -> tuple[str, str, str]:
        first_brace = function_source.find("{")
        last_brace = function_source.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return function_source, "", ""
        return (
            function_source[: first_brace + 1],
            function_source[first_brace + 1:last_brace],
            function_source[last_brace:],
        )

    def _chunk_function_ast(self, function_node: Any, source_bytes: bytes) -> list[str]:
        """Split a large function into top-level statement chunks using AST boundaries."""
        if function_node is None:
            return []

        body_node = function_node.child_by_field_name("body") if hasattr(function_node, "child_by_field_name") else None
        if body_node is None:
            return []

        chunks: list[str] = []
        current = ""
        split_types = {
            "if_statement",
            "for_statement",
            "while_statement",
            "switch_statement",
            "try_statement",
            "compound_statement",
            "expression_statement",
            "declaration",
            "return_statement",
        }

        for child in getattr(body_node, "children", []):
            child_type = str(getattr(child, "type", ""))
            if not child_type or child_type in {"{", "}"}:
                continue
            child_text = self.parser.node_text(child, source_bytes).strip()
            if not child_text:
                continue

            if current and (len(current) + len(child_text) + 1 > _CHUNK_MAX_CHARS or child_type in split_types):
                chunks.append(current.strip())
                current = ""

            current = (current + "\n" + child_text).strip() if current else child_text

        if current.strip():
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    def _reassemble_chunked_function(
        self,
        original_function_source: str,
        modernized_chunks: list[str],
    ) -> str:
        signature, _body, suffix = self._split_function_signature_and_body(original_function_source)
        if not modernized_chunks:
            return original_function_source
        assembled_body = "\n\n".join(chunk.strip() for chunk in modernized_chunks if chunk.strip())
        if not assembled_body.strip():
            return original_function_source
        return signature + "\n" + assembled_body + "\n" + suffix

    def _reflection_indicates_issues(self, reflection_text: str) -> bool:
        lowered = (reflection_text or "").lower()
        if not lowered.strip():
            return False
        if "no_issues" in lowered:
            return False
        no_issue_markers = (
            "no issues",
            "looks good",
            "no changes needed",
            "already modern",
            "correct as is",
        )
        if any(marker in lowered for marker in no_issue_markers):
            return False
        return any(marker in lowered for marker in ("issue", "error", "missing", "improve", "fix"))

    def _reflect_on_candidate(self, original: str, candidate: str) -> str:
        """Iteratively ask the model to critique and improve a candidate modernization."""
        if not _ENABLE_REFLECTION or _REFLECTION_MAX_ITERS <= 0:
            return candidate

        improved = candidate
        for reflection_iter in range(1, _REFLECTION_MAX_ITERS + 1):
            reflection_prompt = (
                "Review this modernized C++ function against the checklist below.\n"
                "Checklist:\n"
                "1) compile-safety risks (syntax/types/includes),\n"
                "2) behavior-preservation risks,\n"
                "3) missing C++17-safe modernizations,\n"
                "4) style/readability issues.\n"
                "If no problems exist, respond with EXACTLY: NO_ISSUES\n\n"
                "Original code:\n"
                f"{original}\n\n"
                "Modernized code:\n"
                f"{improved}\n\n"
                "Return concise bullet points with concrete fixes only."
            )
            try:
                reflection = self._invoke_llm(reflection_prompt)
            except Exception:
                return improved

            if not self._reflection_indicates_issues(reflection):
                return improved

            revise_prompt = (
                "Improve the modernized function using this review feedback.\n\n"
                "Original function:\n"
                f"{original}\n\n"
                "Current candidate:\n"
                f"{improved}\n\n"
                "Review feedback:\n"
                f"{reflection}\n\n"
                "Return ONLY valid C++ code for the improved full function. "
                "No explanation text, no markdown. Preserve behavior and signature."
            )
            try:
                revised = self._clean_model_code(self._invoke_llm(revise_prompt))
            except Exception:
                return improved
            if revised.strip():
                _log("REFLECT", f"Applied reflection iteration {reflection_iter}.")
                improved = revised

        return improved

    def _rag_examples(self, query: str) -> str:
        if self.rag is None or not _ENABLE_RAG:
            return ""
        results = self.rag.search(query=query, k=_RAG_TOP_K)
        if not results:
            return ""
        parts: list[str] = ["Here are similar functions for context:"]
        for idx, item in enumerate(results, start=1):
            code = str(item.get("code") or "").strip()
            metadata = item.get("metadata") or {}
            source = str(metadata.get("source") or metadata.get("fqn") or "unknown")
            if not code:
                continue
            parts.append(f"Example {idx} ({source}):\n{code}")
        return "\n\n".join(parts)

    def _index_project_functions_in_rag(self, project_map: dict[str, Any], source: str) -> None:
        if self.rag is None or not _ENABLE_RAG:
            return
        functions = project_map.get("functions") or {}
        if not isinstance(functions, dict):
            return
        for fqn, meta in functions.items():
            if not isinstance(meta, dict):
                continue
            body = str(meta.get("body") or "").strip()
            if not body:
                continue
            self.rag.add_document(
                code=body,
                metadata={
                    "source": source,
                    "fqn": str(fqn),
                    "name": str(meta.get("name") or ""),
                },
            )

    def modernize_file(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"C++ file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            original_source = fh.read()

        self._project_map = self.parser.parse_file(file_path)
        self._index_project_functions_in_rag(self._project_map, source=file_path)
        functions = self._project_map.get("functions") or {}
        if not isinstance(functions, dict) or not functions:
            return original_source

        dep_graph = self._get_or_build_dependency_graph(
            functions=functions,
            types_info=self._project_map.get("types") or [],
        )
        try:
            write_dot(dep_graph.graph, "dependency_graph.dot")
        except Exception:
            pass
        order = dep_graph.get_modernization_order()

        modernization_fqns = self._resolve_fqn_order(order, functions)
        for unique_fqn in modernization_fqns:
            if unique_fqn in self._modernized_fqns:
                continue
            self.modernize_function(file_path, unique_fqn)
            self._modernized_fqns.add(unique_fqn)

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

    def modernize_function(self, file_path: str, unique_fqn: str) -> None:
        max_attempts = 3
        min_modernization_score = 20
        compiler_feedback = ""
        self.stats["functions_analyzed"] += 1

        for _attempt in range(1, max_attempts + 1):
            self._project_map = self.parser.parse_file(file_path)
            self._index_project_functions_in_rag(self._project_map, source=file_path)
            functions = self._project_map.get("functions") or {}
            if unique_fqn not in functions:
                return

            context = self.parser.get_context_for_function(unique_fqn)
            function_meta = functions[unique_fqn]
            with open(file_path, "r", encoding="utf-8") as fh:
                current_source = fh.read()

            function_body = self._extract_function_source(current_source, function_meta)
            if not function_body.strip():
                return

            function_signature = str(function_meta.get("signature") or "")

            function_ast = self.ast_detector.get_function_ast_node(function_body)
            if function_ast is None:
                patterns: dict[str, int] = {}
                detected_types: list[str] = []
            else:
                pattern_result = self.ast_detector.detect_legacy_patterns(
                    function_ast,
                    function_body.encode("utf-8"),
                )
                # Handle DetectionResult dataclass (new format)
                if hasattr(pattern_result, "counts"):
                    patterns = dict(pattern_result.counts)
                    detected_types = list(pattern_result.detected)
                # Handle legacy dict format (for backward compatibility)
                elif isinstance(pattern_result, dict) and isinstance(pattern_result.get("counts"), dict):
                    raw_counts = pattern_result.get("counts") or {}
                    patterns = {
                        str(key): int(value)
                        for key, value in raw_counts.items()
                        if isinstance(key, str)
                    }
                    raw_detected = pattern_result.get("detected") or []
                    detected_types = [str(item) for item in raw_detected]
                else:
                    patterns = {}
                    detected_types = []
            _log("AST", f"Detected patterns: {patterns} | active={detected_types}")
            detected_count = sum(int(v) for v in patterns.values())
            self.stats["legacy_constructs_detected"] += detected_count
            patterns_text = self._format_patterns(patterns, detected_types)

            function_score = score_cpp23_compliance(function_body)
            if self._safe_percent(function_score) > _MODERN_SKIP_THRESHOLD_PERCENT:
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

            if complexity <= (_RULE_ONLY_COMPLEXITY_THRESHOLD + 1):
                _log(
                    "SKIP",
                    f"Rule-only path for '{unique_fqn}' (complexity={complexity}).",
                )
                return

            function_hash = str(
                function_meta.get("function_hash")
                or hashlib.sha256(function_body.encode("utf-8")).hexdigest()
            )
            normalized_signature = re.sub(r"/\*.*?\*/", " ", function_signature, flags=re.DOTALL)
            normalized_signature = re.sub(r"//[^\n]*", " ", normalized_signature)
            normalized_signature = re.sub(r"\s+", " ", normalized_signature).strip()
            signature_fingerprint = hashlib.sha256(normalized_signature.encode("utf-8")).hexdigest() if normalized_signature else ""
            function_context_blob = json.dumps(
                {
                    "fqn": unique_fqn,
                    "signature_fingerprint": signature_fingerprint,
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
                _log("LLM", f"Function-hash cache hit for '{unique_fqn}'. Reusing modernized code.")
                modernized_function = cached_modernized_function
                used_function_cache = True
            else:
                rag_context = self._rag_examples(function_signature or function_body)
                if len(function_body) > _MAX_FUNCTION_CHARS:
                    chunked_candidate = self._modernize_large_function_chunks(
                        function_body=function_body,
                        function_signature=function_signature,
                        referenced_types=context.get("referenced_type_definitions")
                        or context.get("type_bundle")
                        or {},
                        called_signatures=context.get("called_function_signatures") or {},
                        legacy_patterns=patterns_text,
                        applied_rules=applied_rules,
                        compiler_feedback=compiler_feedback,
                        rag_context=rag_context,
                    )
                    if chunked_candidate.strip():
                        modernized_function = chunked_candidate
                    else:
                        _log("CHUNK", "Chunked modernization failed; falling back to whole-function prompt.")
                        modernized_function = ""
                else:
                    modernized_function = ""

                if modernized_function.strip():
                    _log("CHUNK", f"Chunked response assembled ({len(modernized_function)} chars).")
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
                        rag_context=rag_context,
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
                            _log("LLM", f"Local LLM unavailable while modernizing '{unique_fqn}': {error_text}. Skipping function for this run.")
                            return
                        if not self._disable_prompt_cache:
                            self._llm_cache[cache_key] = raw_response
                            self._save_cache()
                    modernized_function = self._clean_model_code(raw_response)
                    _log("LLM", f"Response received ({len(modernized_function)} chars).")

            modernized_function = self._reflect_on_candidate(function_body, modernized_function)

            if not modernized_function.strip():
                _log("LLM", "Model returned empty function output.")
                compiler_feedback = "Model returned empty function output."
                continue

            validation_error = self._validate_model_function_output(
                fqn=unique_fqn,
                function_meta=function_meta,
                original_function_body=function_body,
                candidate_function_code=modernized_function,
            )
            if validation_error:
                _log("VERIFY", f"Rejecting model output: {validation_error}")
                compiler_feedback = (
                    "Model output was rejected before replacement. "
                    f"Reason: {validation_error}. Return only one compatible function body."
                )
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
                    "LLM returned nearly identical code. You MUST rewrite with meaningful modern C++17-safe improvements "
                    "while preserving behavior. At least one legacy construct must be replaced."
                )
                if _attempt >= max_attempts:
                    _log("VERIFY", "No meaningful modernization after retries. Skipping function.")
                    return
                continue

            self.replace_function(file_path, unique_fqn, modernized_function)
            self._project_map = self.parser.parse_file(file_path)

            with open(file_path, "r", encoding="utf-8") as fh:
                replaced_source = fh.read()
            compile_result = compile_cpp_source(replaced_source)
            _log("VERIFY", f"Compile: {'success' if compile_result.get('success') else 'failed'} for '{unique_fqn}'.")
            if bool(compile_result.get("success")):
                # Hard guard: reject truly unchanged code even if compile passed.
                if is_similar_code(function_body, modernized_function):
                    _log("VERIFY", "Rejecting modernization — code unchanged after compile.")
                    self._restore_source_snapshot(file_path, current_source)
                    self._project_map = self.parser.parse_file(file_path)
                    compiler_feedback = (
                        "Code was unchanged. You MUST apply at least one meaningful modernization."
                    )
                    continue
                modernization_score = int(
                    self._safe_percent(score_cpp23_compliance(modernized_function))
                )
                code_changed = not is_similar_code(
                    function_body,
                    modernized_function,
                    threshold=_SIMILARITY_THRESHOLD,
                )
                if modernization_score < min_modernization_score and code_changed:
                    # Keep retries deterministic: restore the pre-attempt snapshot before retrying.
                    self._restore_source_snapshot(file_path, current_source)
                    self._project_map = self.parser.parse_file(file_path)
                    if _attempt >= max_attempts:
                        return
                    compiler_feedback = (
                        "Compilation passed but modernization quality score was low "
                        f"({modernization_score} < {min_modernization_score}). "
                        "Improve modernization while preserving behavior."
                    )
                    continue
                if modernization_score < _MODERN_RETRY_THRESHOLD_PERCENT and code_changed:
                    enhanced = score_cpp23_compliance(modernized_function)
                    low_metric_feedback: list[str] = []
                    metrics_obj = enhanced.get("metrics")
                    metrics = metrics_obj if isinstance(metrics_obj, dict) else {}
                    metric_details = metrics.get("metric_details")
                    for metric in metric_details if isinstance(metric_details, list) else []:
                        if not isinstance(metric, dict):
                            continue
                        if int(metric.get("score", 0) or 0) < int(metric.get("max_score", 0) or 0) // 2:
                            feedback = str(metric.get("feedback") or "").strip()
                            if feedback:
                                low_metric_feedback.append(feedback)
                    if low_metric_feedback and _attempt < max_attempts:
                        self._restore_source_snapshot(file_path, current_source)
                        self._project_map = self.parser.parse_file(file_path)
                        compiler_feedback = (
                            "Compilation passed but enhanced modernization metrics are low. "
                            f"Current score={modernization_score}. Improve these aspects:\n"
                            + "\n".join(f"- {item}" for item in low_metric_feedback)
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
            _log("VERIFY", "Compile failed; restoring snapshot and retrying with compiler feedback.")
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
            self._restore_source_snapshot(file_path, current_source)
            self._project_map = self.parser.parse_file(file_path)
            continue

        raise RuntimeError(
            f"Failed to modernize function '{unique_fqn}' after {max_attempts} attempts.\n"
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
        except Exception as exc:
            _logger.warning("Failed to load prompt cache '%s': %s", self._cache_path, exc)
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
        except Exception as exc:
            _logger.warning("Failed to load function cache '%s': %s", self._function_cache_path, exc)
            return {}
        return {}

    def _save_function_cache(self) -> None:
        try:
            with open(self._function_cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._function_cache, fh, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def replace_function(self, file_path: str, fqn: str, new_code: str) -> None:
        # Byte offsets are valid only against a fresh parse of the current file content.
        # Callers must parse immediately before invoking this span-based replacement.
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

        if re.search(r"(?m)^\s*#\s*include\b", new_code):
            raise ValueError("Replacement contains '#include' directives; refusing span replacement.")

        parsed_candidate = self.parser.parse_string(new_code)
        parsed_candidate_functions = parsed_candidate.get("functions") or {}
        if isinstance(parsed_candidate_functions, dict):
            function_count = len(parsed_candidate_functions)
        elif isinstance(parsed_candidate_functions, list):
            function_count = len(parsed_candidate_functions)
        else:
            function_count = 0
        if function_count != 1:
            raise ValueError(
                f"Replacement must contain exactly one function definition, got {function_count}."
            )

        updated_bytes = source_bytes[:start] + new_code.strip().encode("utf-8") + source_bytes[end:]
        updated = updated_bytes.decode("utf-8", errors="strict")

        file_lock = self._lock_for_file(file_path)
        with file_lock:
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
        rag_context: str = "",
        chunk_context: str = "",
    ) -> str:
        rules_text = self._format_applied_rules(applied_rules)
        prompt = (
            "You are a senior C++ modernization engineer.\n\n"
            "Your task is to rewrite the following legacy C++ function using C++17-compatible modernization only.\n\n"
            "STRICT REQUIREMENTS:\n"
            "- Preserve the exact behavior.\n"
            "- Improve the code using modern C++17 features where appropriate.\n"
            "- Replace raw pointers with smart pointers if applicable.\n"
            "- Replace manual loops with range-based for loops where safe.\n"
            "- Avoid NULL, use nullptr.\n"
            "- Avoid raw new/delete.\n"
            "- Prefer std::optional or std::variant where appropriate.\n"
            "- Do not use concepts, std::ranges, coroutines, std::format, std::span, or modules.\n\n"
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
            "Rewrite the function to eliminate these constructs using modern C++17-compatible features.\n"
            "Do not rewrite unrelated logic.\n\n"
            f"{chunk_context}\n\n"
            "Deterministic modernization hints (apply only when semantically safe):\n"
            f"{rules_text}\n\n"
            f"{rag_context}\n\n"
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

    def _modernize_large_function_chunks(
        self,
        *,
        function_body: str,
        function_signature: str,
        referenced_types: Any,
        called_signatures: Any,
        legacy_patterns: str,
        applied_rules: list[str],
        compiler_feedback: str,
        rag_context: str,
    ) -> str:
        """Modernize very large functions by chunking top-level AST regions."""
        function_ast = self.ast_detector.get_function_ast_node(function_body)
        source_bytes = function_body.encode("utf-8")
        chunks = self._chunk_function_ast(function_ast, source_bytes)
        if len(chunks) <= 1:
            return ""

        _log("CHUNK", f"Chunking large function into {len(chunks)} parts.")
        modernized_chunks: list[str] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_prompt = self._build_prompt(
                function_body=function_body,
                referenced_types=referenced_types,
                called_signatures=called_signatures,
                legacy_patterns=legacy_patterns,
                applied_rules=applied_rules,
                compiler_feedback=compiler_feedback,
                rag_context=rag_context,
                chunk_context=(
                    "Chunked modernization mode:\n"
                    f"- Function signature: {function_signature}\n"
                    f"- Processing chunk {chunk_index}/{len(chunks)}\n"
                    "- Rewrite ONLY this chunk and preserve semantics/order:\n"
                    f"{chunk}"
                ),
            )
            try:
                chunk_response = self._invoke_llm(chunk_prompt)
            except Exception:
                return ""
            cleaned_chunk = self._clean_model_code(chunk_response)
            if not cleaned_chunk.strip():
                return ""
            modernized_chunks.append(cleaned_chunk.strip())

        reassembled = self._reassemble_chunked_function(function_body, modernized_chunks)
        if not reassembled.strip():
            return ""
        # Guard chunk mode with parser-level validation before returning a candidate.
        validation_error = self._validate_chunk_reassembly(function_body, reassembled)
        if validation_error:
            _log("CHUNK", f"Reassembly validation failed: {validation_error}")
            return ""
        return reassembled

    def _validate_chunk_reassembly(self, original_function: str, candidate_function: str) -> str | None:
        """Ensure chunk reassembly still yields exactly one compatible function."""
        original_parsed = self.parser.parse_string(original_function)
        candidate_parsed = self.parser.parse_string(candidate_function)

        original_functions_obj = original_parsed.get("functions") or {}
        candidate_functions_obj = candidate_parsed.get("functions") or {}

        if isinstance(original_functions_obj, dict):
            original_functions = list(original_functions_obj.values())
        else:
            original_functions = [item for item in original_functions_obj if isinstance(item, dict)] if isinstance(original_functions_obj, list) else []

        if isinstance(candidate_functions_obj, dict):
            candidate_functions = list(candidate_functions_obj.values())
        else:
            candidate_functions = [item for item in candidate_functions_obj if isinstance(item, dict)] if isinstance(candidate_functions_obj, list) else []

        if len(candidate_functions) != 1:
            return f"expected exactly one function after reassembly, got {len(candidate_functions)}"
        if len(original_functions) != 1:
            return "original function parse is ambiguous"

        original_signature = str(original_functions[0].get("signature") or "")
        candidate_signature = str(candidate_functions[0].get("signature") or "")
        normalize = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
        if normalize(original_signature) != normalize(candidate_signature):
            return "function signature changed during chunk reassembly"
        return None

    def _apply_rules_to_function_body(
        self,
        function_source: str,
        detected_patterns: dict[str, int] | None = None,
    ) -> tuple[str, list[str]]:
        """Apply regex modernization rules to function body only, preserving signature."""
        if not _ENABLE_RULE_BASED_REWRITES:
            _log("RULES", "Rule-based rewrites disabled (ENABLE_RULE_BASED_REWRITES=0).")
            return function_source, []

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

    def _validate_model_function_output(
        self,
        fqn: str,
        function_meta: dict[str, Any],
        original_function_body: str,
        candidate_function_code: str,
    ) -> str | None:
        """Return None when candidate is safe to span-replace; else rejection reason."""
        if re.search(r"(?m)^\s*#\s*include\b", candidate_function_code):
            return "output appears to be a whole file (#include found)"

        parsed = self.parser.parse_string(candidate_function_code)
        parsed_functions = parsed.get("functions") or {}
        if isinstance(parsed_functions, dict):
            fn_values = list(parsed_functions.values())
        elif isinstance(parsed_functions, list):
            fn_values = [item for item in parsed_functions if isinstance(item, dict)]
        else:
            fn_values = []

        if len(fn_values) != 1:
            return f"output must contain exactly one function definition (got {len(fn_values)})"

        expected_name = str(function_meta.get("name") or fqn.split("::")[-1])
        candidate_name = str(fn_values[0].get("name") or "")
        if candidate_name != expected_name:
            return f"function name mismatch (expected '{expected_name}', got '{candidate_name or 'unknown'}')"

        original_meta_params = function_meta.get("parameters") or []
        candidate_meta_params = fn_values[0].get("parameters") or []
        original_params = len([p for p in original_meta_params if isinstance(p, dict)])
        candidate_params = len([p for p in candidate_meta_params if isinstance(p, dict)])
        if original_params >= 0 and candidate_params >= 0 and original_params != candidate_params:
            return f"incompatible signature parameter count ({original_params} -> {candidate_params})"

        extra_types = parsed.get("types") or []
        extra_globals = parsed.get("global_variables") or []
        if extra_types or extra_globals:
            return "contains extra top-level declarations"

        return None

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
