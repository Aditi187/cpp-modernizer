from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import tempfile
import time
from typing import Any

from networkx.drawing.nx_pydot import write_dot

from core.ast_modernizer import ASTModernizationDetector
from core.graph import DependencyGraph
from core.gemini_bridge import CPP_MODERNIZATION_SYSTEM_PROMPT
from core.differential_tester import compile_cpp_source, run_differential_test
from core.inspect_parser import score_cpp23_compliance
from core.rule_modernizer import apply_modernization_rules


_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_LLM_CALL_DELAY_SECONDS = 1.5
_SIMILARITY_THRESHOLD = 0.95
_MODERN_SKIP_THRESHOLD_PERCENT = 70
_LLM_CACHE_FILENAME = ".llm_cache.json"
DEBUG_MODE = True


def code_similarity_ratio(old_code: str, new_code: str) -> float:
    return difflib.SequenceMatcher(None, old_code, new_code).ratio()


def similarity(a: str, b: str) -> float:
    return code_similarity_ratio(a, b)


def is_similar_code(old_code: str, new_code: str, threshold: float = 0.95) -> bool:
    ratio = code_similarity_ratio(old_code, new_code)
    return ratio > threshold


class FunctionModernizer:
    def __init__(self, parser, llm, fallback_llm=None):
        self.parser = parser
        self.llm = llm
        self.fallback_llm = fallback_llm
        self.ast_detector = ASTModernizationDetector(parser)
        self._project_map: dict[str, Any] = {}
        self._modernized_fqns: set[str] = set()
        self._cache_path = os.path.join(os.getcwd(), _LLM_CACHE_FILENAME)
        self._llm_cache: dict[str, str] = self._load_cache()
        self.file_lock = threading.Lock()
        self.stats = {
            "functions_analyzed": 0,
            "functions_modernized": 0,
            "rule_transformations": 0,
            "llm_transformations": 0,
            "legacy_constructs_detected": 0,
        }

    def _is_rate_limit_error(self, error_message: str) -> bool:
        lowered = (error_message or "").lower()
        return "429" in lowered or "rate limit" in lowered or "rate-limited" in lowered

    def _invoke_llm_with_fallback(self, prompt: str) -> str:
        try:
            return self.llm.chat_completion(
                system_prompt=CPP_MODERNIZATION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        except Exception as primary_exc:
            primary_message = str(primary_exc)
            if self.fallback_llm is None or not self._is_rate_limit_error(primary_message):
                raise

            print("Gemini rate-limited; trying OpenRouter fallback for this function.")
            try:
                return self.fallback_llm.chat_completion(
                    system_prompt=CPP_MODERNIZATION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Primary and fallback LLM calls failed. "
                    f"Primary: {primary_message} | Fallback: {fallback_exc}"
                ) from fallback_exc

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

            function_ast = self.ast_detector.get_function_ast_node(function_body)
            patterns = self.ast_detector.detect_legacy_patterns(function_ast)
            if DEBUG_MODE:
                print("\n===== AST DETECTION =====")
                print("Detected patterns:", patterns)
            detected_count = sum(int(v) for v in patterns.values())
            self.stats["legacy_constructs_detected"] += detected_count
            patterns_text = self._format_patterns(patterns)

            function_score = score_cpp23_compliance(function_body)
            if int(function_score.get("percent", 0) or 0) > _MODERN_SKIP_THRESHOLD_PERCENT:
                print("Function already modern. Skipping.")
                return

            function_for_llm, applied_rules = self._apply_rules_to_function_body(function_body)
            print(f"Applied rules: {applied_rules}")
            if applied_rules:
                self.stats["rule_transformations"] += len(applied_rules)
                print("Applied modernization rules:")
                for rule in applied_rules:
                    print(f"* {rule}")
                if function_for_llm != function_body:
                    self.replace_function(file_path, fqn, function_for_llm)
                    with open(file_path, "r", encoding="utf-8") as fh:
                        rules_updated_source = fh.read()
                    rules_compile_result = compile_cpp_source(rules_updated_source)
                    if bool(rules_compile_result.get("success")):
                        print("Rule modernization succeeded. Skipping LLM.")
                        self.stats["functions_modernized"] += 1
                        return

                    # Revert rule-only update if compile fails, then continue to LLM path.
                    with self.file_lock:
                        with open(file_path, "w", encoding="utf-8") as fh:
                            fh.write(current_source)
                    self._project_map = self.parser.parse_file(file_path)
                    compiler_feedback = str(
                        rules_compile_result.get("raw_stderr")
                        or "\n".join(rules_compile_result.get("errors") or [])
                        or "Compilation failed with unknown error."
                    )

            # If no legacy AST constructs were detected, avoid unnecessary LLM calls.
            if detected_count == 0:
                if function_for_llm != function_body:
                    self.replace_function(file_path, fqn, function_for_llm)
                    with open(file_path, "r", encoding="utf-8") as fh:
                        replaced_source = fh.read()
                    compile_result = compile_cpp_source(replaced_source)
                    if bool(compile_result.get("success")):
                        self.stats["functions_modernized"] += 1
                        return

                    # Revert rule-only update if compile fails.
                    with self.file_lock:
                        with open(file_path, "w", encoding="utf-8") as fh:
                            fh.write(current_source)
                    self._project_map = self.parser.parse_file(file_path)
                    compiler_feedback = str(
                        compile_result.get("raw_stderr")
                        or "\n".join(compile_result.get("errors") or [])
                        or "Compilation failed with unknown error."
                    )
                return

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

            cache_key = self._cache_key(function_for_llm)
            cached_response = self._llm_cache.get(cache_key)
            if cached_response is not None:
                raw_response = cached_response
            else:
                # Throttle requests slightly to reduce provider rate-limit pressure.
                time.sleep(_LLM_CALL_DELAY_SECONDS)
                if DEBUG_MODE:
                    print("\n===== LLM PROMPT =====")
                    print(prompt)
                raw_response = self._invoke_llm_with_fallback(prompt)
                self._llm_cache[cache_key] = raw_response
                self._save_cache()
            modernized_function = self._clean_model_code(raw_response)
            if DEBUG_MODE:
                print("\n===== LLM OUTPUT =====")
                print(modernized_function)
            if not modernized_function.strip():
                compiler_feedback = "Model returned empty function output."
                continue

            similarity_ratio = similarity(function_body, modernized_function)
            if DEBUG_MODE:
                print("\n===== SIMILARITY =====")
                print("Similarity:", similarity_ratio)
            print(f"Modernization similarity: {similarity_ratio:.2f}")
            if similarity_ratio > _SIMILARITY_THRESHOLD:
                print("No meaningful modernization detected.")
                compiler_feedback = (
                    "LLM returned nearly identical code. You MUST rewrite with meaningful modern C++23 improvements "
                    "while preserving behavior."
                )
                if _attempt >= max_attempts:
                    print("⚠️ No meaningful modernization after retries. Skipping function.")
                    return
                continue

            self.replace_function(file_path, fqn, modernized_function)
            self._project_map = self.parser.parse_file(file_path)

            with open(file_path, "r", encoding="utf-8") as fh:
                replaced_source = fh.read()
            compile_result = compile_cpp_source(replaced_source)
            if bool(compile_result.get("success")):
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
                return

            # Revert the failed replacement before retrying.
            with self.file_lock:
                with open(file_path, "w", encoding="utf-8") as fh:
                    fh.write(current_source)
            self._project_map = self.parser.parse_file(file_path)

            compiler_feedback = str(
                compile_result.get("raw_stderr")
                or "\n".join(compile_result.get("errors") or [])
                or "Compilation failed with unknown error."
            )

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

        replacement_bytes = new_code.strip().encode("utf-8")
        updated = source_bytes[:start] + replacement_bytes + source_bytes[end:]

        with self.file_lock:
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(updated.decode("utf-8", errors="replace"))

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
        return source_bytes[start:end].decode("utf-8", errors="replace")

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
            "Do not return the original code unchanged.\n\n"
            "Return ONLY the modernized function.\n\n"
            "Function:\n"
            f"{function_body}\n\n"
            "Detected legacy constructs:\n"
            f"{legacy_patterns}\n\n"
            "Rewrite the function to eliminate these constructs using modern C++23 features.\n"
            "Do not rewrite unrelated logic.\n\n"
            "Pre-applied modernization rules:\n"
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

    def _apply_rules_to_function_body(self, function_source: str) -> tuple[str, list[str]]:
        """Apply regex modernization rules to function body only, preserving signature."""
        first_brace = function_source.find("{")
        last_brace = function_source.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return function_source, []

        signature_prefix = function_source[: first_brace + 1]
        body_text = function_source[first_brace + 1:last_brace]
        closing_suffix = function_source[last_brace:]

        updated_body, applied_rules = apply_modernization_rules(body_text)
        if DEBUG_MODE:
            print("\n===== RULE ENGINE =====")
            print("Original code:\n", body_text)
            print("\nUpdated code:\n", updated_body)
            print("Applied rules:", applied_rules)
        if not applied_rules:
            return function_source, []

        return signature_prefix + updated_body + closing_suffix, applied_rules

    def _format_applied_rules(self, applied_rules: list[str]) -> str:
        if not applied_rules:
            return "(none)"
        return "\n".join(f"- {rule}" for rule in applied_rules)

    def _format_patterns(self, patterns: dict[str, int]) -> str:
        active = [f"{key}: {count}" for key, count in patterns.items() if int(count) > 0]
        if not active:
            return "(none)"
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
