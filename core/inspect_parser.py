from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplianceRule:
	"""Single weighted compliance rule used by score_cpp23_compliance.

	Each rule contributes up to weight points based on positive and negative
	pattern matches on masked source text.
	"""

	id: str
	weight: int
	positive_pattern: re.Pattern[str]
	negative_pattern: re.Pattern[str]
	recommendation: str


def _regex(pattern: str) -> re.Pattern[str]:
	return re.compile(pattern)


_RULES: List[ComplianceRule] = [
	ComplianceRule(
		id="expected_optional",
		weight=12,
		positive_pattern=_regex(r"\bstd::(expected|optional)\b"),
		negative_pattern=_regex(r"\b(return\s+-?\d+\s*;|errno\b)"),
		recommendation="Prefer std::expected/std::optional over manual return codes.",
	),
	ComplianceRule(
		id="format_print",
		weight=10,
		positive_pattern=_regex(r"\bstd::(print|println|format)\b"),
		negative_pattern=_regex(r"\b(printf|std::cout|std::cerr)\b"),
		recommendation="Prefer std::print/std::format for output formatting.",
	),
	ComplianceRule(
		id="span_mdspan",
		weight=10,
		positive_pattern=_regex(r"\bstd::(span|mdspan)\b"),
		negative_pattern=_regex(r"\b(char\s*\*|void\s*\*)\b"),
		recommendation="Use std::span/std::mdspan for buffer handling.",
	),
	ComplianceRule(
		id="ranges_views_pipelines",
		weight=10,
		positive_pattern=_regex(r"\bstd::(ranges|views)::"),
		negative_pattern=_regex(r"\bstd::(sort|transform|find|for_each|copy|count|any_of|all_of|none_of)\s*\("),
		recommendation="Prefer std::ranges/views pipelines where algorithm composition is needed.",
	),
	ComplianceRule(
		id="unique_ptr_stack",
		weight=12,
		positive_pattern=_regex(r"\bstd::unique_ptr\b|\bstd::make_unique\b"),
		negative_pattern=_regex(r"\b(new|delete)\b"),
		recommendation="Prefer std::unique_ptr and stack allocation over raw new/delete.",
	),
	ComplianceRule(
		id="string_view",
		weight=10,
		positive_pattern=_regex(r"\bstd::string_view\b"),
		negative_pattern=_regex(r"\b(const\s+std::string\s*&|const\s+char\s*\*)\b"),
		recommendation="Use std::string_view for non-owning string parameters and views.",
	),
	ComplianceRule(
		id="structured_bindings",
		weight=8,
		positive_pattern=_regex(r"\b(auto|const\s+auto|auto\s*&|const\s+auto\s*&)\s*\[[^\]]+\]"),
		negative_pattern=_regex(r"\bstd::(pair|tuple)\b"),
		recommendation="Use structured bindings (auto [x, y]) to unpack tuples/pairs clearly.",
	),
	ComplianceRule(
		id="filesystem",
		weight=8,
		positive_pattern=_regex(r"\bstd::filesystem\b|\bstd::fs\b"),
		negative_pattern=_regex(r"\b(opendir|readdir|closedir|stat\s*\()\b"),
		recommendation="Use std::filesystem instead of low-level directory APIs when possible.",
	),
	ComplianceRule(
		id="constexpr_usage",
		weight=6,
		positive_pattern=_regex(r"\bconstexpr\b"),
		negative_pattern=_regex(r"\bconst\s+(int|long|size_t|double|float|char)\s+[A-Za-z_]\w*\s*=\s*[^;]+;"),
		recommendation="Prefer constexpr for compile-time evaluable values and functions.",
	),
	ComplianceRule(
		id="consteval_constinit",
		weight=6,
		positive_pattern=_regex(r"\b(consteval|constinit)\b"),
		negative_pattern=_regex(r"\bconstexpr\b"),
		recommendation="Consider consteval/constinit where immediate evaluation or static initialization is required.",
	),
	ComplianceRule(
		id="concepts_requires",
		weight=10,
		positive_pattern=_regex(r"\b(concept\s+[A-Za-z_]\w*\s*=|requires\b)"),
		negative_pattern=_regex(r"\btemplate\s*<\s*typename\b"),
		recommendation="Use C++20 concepts/requires clauses for clearer template constraints.",
	),
	ComplianceRule(
		id="coroutines",
		weight=8,
		positive_pattern=_regex(r"\b(co_await|co_yield|co_return)\b"),
		negative_pattern=_regex(r"\b(std::async|std::thread)\b"),
		recommendation="Use coroutine primitives when asynchronous control flow benefits from suspension/resumption.",
	),
	ComplianceRule(
		id="noexcept_specifier",
		weight=6,
		positive_pattern=_regex(r"\bnoexcept\b"),
		negative_pattern=_regex(r"\bthrow\s*\(\s*\)"),
		recommendation="Prefer noexcept over legacy dynamic exception specifications.",
	),
	ComplianceRule(
		id="attributes",
		weight=6,
		positive_pattern=_regex(r"\[\[(?:nodiscard|maybe_unused|likely|unlikely)\]\]"),
		negative_pattern=_regex(r"\b__attribute__\b|\bdeclspec\b"),
		recommendation="Use standard C++ attributes such as [[nodiscard]], [[likely]], and [[maybe_unused]].",
	),
	ComplianceRule(
		id="three_way_comparison",
		weight=8,
		positive_pattern=_regex(r"operator\s*<=>"),
		negative_pattern=_regex(r"operator\s*==|operator\s*<|operator\s*>"),
		recommendation="Consider operator<=> to simplify and unify comparison operators.",
	),
	ComplianceRule(
		id="designated_initializers",
		weight=6,
		positive_pattern=_regex(r"\{\s*\.[A-Za-z_]\w*\s*="),
		negative_pattern=_regex(r"\bmemset\s*\("),
		recommendation="Use designated initializers for clearer aggregate initialization where supported.",
	),
	ComplianceRule(
		id="source_location",
		weight=6,
		positive_pattern=_regex(r"\bstd::source_location\b"),
		negative_pattern=_regex(r"\b(__FILE__|__LINE__|__func__)\b"),
		recommendation="Use std::source_location instead of preprocessor location macros for diagnostics.",
	),
	ComplianceRule(
		id="constant_evaluated",
		weight=6,
		positive_pattern=_regex(r"\bstd::is_constant_evaluated\s*\("),
		negative_pattern=_regex(r"\b#if\s+defined\("),
		recommendation="Use std::is_constant_evaluated for constexpr-aware branching when appropriate.",
	),
	ComplianceRule(
		id="variant",
		weight=8,
		positive_pattern=_regex(r"\bstd::variant\b"),
		negative_pattern=_regex(r"\bunion\b|\bvoid\s*\*\b"),
		recommendation="Use std::variant for type-safe tagged unions instead of raw unions/void*.",
	),
	ComplianceRule(
		id="visit",
		weight=8,
		positive_pattern=_regex(r"\bstd::visit\b"),
		negative_pattern=_regex(r"\bstd::get\s*<"),
		recommendation="Use std::visit for variant dispatch instead of manual std::get branching.",
	),
	ComplianceRule(
		id="modern_concurrency",
		weight=10,
		positive_pattern=_regex(r"\bstd::(thread|jthread|future|async|mutex|scoped_lock|atomic|condition_variable)\b"),
		negative_pattern=_regex(r"\b(pthread_|CreateThread\b|WaitForSingleObject\b)"),
		recommendation="Prefer std::thread/jthread and standard concurrency primitives over platform-specific threading APIs.",
	),
]


_DEFAULT_RULES: List[ComplianceRule] = list(_RULES)

_SCORE_CACHE: OrderedDict[str, Dict[str, object]] = OrderedDict()
_SCORE_CACHE_LOCK = threading.Lock()
_CACHE_MAX_SIZE = 100

_DEFAULT_TARGET_STD = "c++23"
_MODERN_HEADERS_BY_STD: Dict[str, set[str]] = {
	"c++17": {"<memory>", "<optional>", "<variant>", "<filesystem>", "<string_view>"},
	"c++20": {"<memory>", "<span>", "<ranges>", "<optional>", "<expected>", "<string_view>"},
	"c++23": {"<memory>", "<span>", "<print>", "<ranges>", "<optional>", "<expected>"},
}

_SCORE_RULE_WEIGHT_DEFAULT = 0.6
_SCORE_METRIC_WEIGHT_DEFAULT = 0.4

_RE_INCLUDE = re.compile(r"#\s*include\s*([<\"][^>\"]+[>\"])")
_RE_RAW_POINTER = re.compile(
	r"\b(?:const\s+)?(?:[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:\s*<[^;\n{}()]*>)?\s+)+\*+\s*[A-Za-z_]\w*\b"
)
_RE_SMART_POINTER = re.compile(r"\bstd::(?:unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared)\b")
_RE_PRINTF = re.compile(r"\bprintf\s*\(")
_RE_STD_PRINT = re.compile(r"\bstd::(?:print|println)\s*\(")
_RE_MANUAL_LOOP = re.compile(r"\b(?:for|while)\s*\(")
_RE_RANGE_LOOP = re.compile(
	r"for\s*\(\s*(?:const\s+)?auto(?:\s*[&]{1,2})?\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*[^\)]+\)"
)
_RE_CYCLO_DECISIONS = re.compile(r"\b(if|for|while|case|catch)\b")
_RE_CYCLO_BOOL = re.compile(r"&&|\|\|")
_RE_CYCLO_TERNARY = re.compile(r"\?")


def _normalize_target_std(target_std: str) -> str:
	normalized = str(target_std or _DEFAULT_TARGET_STD).strip().lower()
	if normalized in _MODERN_HEADERS_BY_STD:
		return normalized
	return _DEFAULT_TARGET_STD


def _rules_config_path() -> str:
	return str(os.environ.get("CPP_COMPLIANCE_RULES_PATH", "")).strip()


def _safe_regex(pattern: str) -> re.Pattern[str]:
	"""Compile regex defensively for configurable rule loading."""
	return re.compile(pattern)


def _load_rules_from_config(path: str) -> Optional[List[ComplianceRule]]:
	"""Load rule definitions from JSON file.

	Expected shape:
	{
	  "rules": [
	    {
	      "id": "...",
	      "weight": 10,
	      "positive_pattern": "...",
	      "negative_pattern": "...",
	      "recommendation": "..."
	    }
	  ]
	}
	"""
	if not path:
		return None
	try:
		with open(path, "r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except Exception as exc:
		_LOG.warning("Failed to load compliance rules from %s: %r", path, exc)
		return None

	raw_rules = payload.get("rules") if isinstance(payload, dict) else None
	if not isinstance(raw_rules, list) or not raw_rules:
		_LOG.warning("Rules config %s missing non-empty 'rules' array", path)
		return None

	rules: List[ComplianceRule] = []
	for item in raw_rules:
		if not isinstance(item, dict):
			continue
		try:
			rule = ComplianceRule(
				id=str(item.get("id") or "").strip(),
				weight=max(0, int(item.get("weight") or 0)),
				positive_pattern=_safe_regex(str(item.get("positive_pattern") or "")),
				negative_pattern=_safe_regex(str(item.get("negative_pattern") or "")),
				recommendation=str(item.get("recommendation") or "").strip(),
			)
		except Exception as exc:
			_LOG.warning("Skipping invalid configured rule item: %r", exc)
			continue
		if not rule.id or rule.weight <= 0:
			continue
		rules.append(rule)

	if not rules:
		_LOG.warning("No valid compliance rules loaded from %s", path)
		return None
	return rules


def _get_rules() -> List[ComplianceRule]:
	config_rules = _load_rules_from_config(_rules_config_path())
	if config_rules:
		return config_rules
	return list(_DEFAULT_RULES)


def _validate_rule_weights(rules: List[ComplianceRule]) -> int:
	total_weight = sum(rule.weight for rule in rules)
	if total_weight < 40 or total_weight > 250:
		_LOG.warning(
			"Compliance rules total weight is %d (outside recommended range 40-250); scoring remains heuristic.",
			total_weight,
		)
	return total_weight


def _cache_get(key: str) -> Optional[Dict[str, object]]:
	with _SCORE_CACHE_LOCK:
		cached = _SCORE_CACHE.get(key)
		if cached is None:
			return None
		_SCORE_CACHE.move_to_end(key)
		return dict(cached)


def _cache_set(key: str, value: Dict[str, object]) -> None:
	with _SCORE_CACHE_LOCK:
		_SCORE_CACHE[key] = dict(value)
		_SCORE_CACHE.move_to_end(key)
		if len(_SCORE_CACHE) > _CACHE_MAX_SIZE:
			_SCORE_CACHE.popitem(last=False)


def _read_score_weight_env(name: str, default: float) -> float:
	raw = str(os.environ.get(name, "")).strip()
	if not raw:
		return default
	try:
		parsed = float(raw)
	except ValueError:
		return default
	return max(0.0, parsed)


def _resolve_component_weights() -> Tuple[float, float]:
	rule_weight = _read_score_weight_env("CPP_COMPLIANCE_RULE_WEIGHT", _SCORE_RULE_WEIGHT_DEFAULT)
	metric_weight = _read_score_weight_env("CPP_COMPLIANCE_METRIC_WEIGHT", _SCORE_METRIC_WEIGHT_DEFAULT)
	total = rule_weight + metric_weight
	if total <= 0:
		return _SCORE_RULE_WEIGHT_DEFAULT, _SCORE_METRIC_WEIGHT_DEFAULT
	return rule_weight / total, metric_weight / total


def _ast_assisted_signals(source_code: str) -> Dict[str, int]:
	"""Best-effort AST signals used to supplement critical regex metrics.

	If AST parsing is unavailable or fails, this returns an empty mapping and
	the score remains purely regex-driven.
	"""
	enabled = str(os.environ.get("CPP_COMPLIANCE_ENABLE_AST_ASSIST", "1")).strip().lower() in {
		"1",
		"true",
		"yes",
		"on",
	}
	if not enabled:
		return {}
	try:
		from core.ast_modernizer import ASTModernizationDetector
		detector = ASTModernizationDetector()
		node = detector.get_function_ast_node(source_code)
		result = detector.detect_legacy_patterns(node, source_code.encode("utf-8"))
		counts = result.counts if hasattr(result, "counts") else {}
		if not isinstance(counts, dict):
			return {}
		return {
			"raw_pointer_count": int(counts.get("raw_pointer", 0) or 0),
			"printf_count": int(counts.get("printf_usage", 0) or 0),
		}
	except Exception:
		return {}


def _mask_non_code(source_code: str) -> str:
	"""Mask comments and literals to reduce regex false positives.

	This routine is intentionally lexer-like but still heuristic (not a full C++
	parser). It preserves original length and newline positions.

	Handled:
	- line comments (//...)
	- block comments (/* ... */)
	- normal string and char literals with escape handling
	- raw string literals such as R"(...)", u8R"tag(... )tag"

	Limitations:
	- nested block comments are not fully modeled (non-standard extension)
	- trigraphs and exotic preprocessor corner cases are not fully modeled
	- escaped newlines in unusual literal forms may still produce edge misses
	"""
	chars = list(source_code)
	length = len(chars)
	i = 0

	def _blank_range(start: int, end: int) -> None:
		for idx in range(start, min(end, length)):
			if chars[idx] != "\n":
				chars[idx] = " "

	while i < length:
		ch = chars[i]
		nxt = chars[i + 1] if i + 1 < length else ""

		# Line comment.
		if ch == "/" and nxt == "/":
			start = i
			i += 2
			while i < length and chars[i] != "\n":
				i += 1
			_blank_range(start, i)
			continue

		# Block comment.
		if ch == "/" and nxt == "*":
			start = i
			i += 2
			while i + 1 < length and not (chars[i] == "*" and chars[i + 1] == "/"):
				i += 1
			i = min(length, i + 2)
			_blank_range(start, i)
			continue

		# Raw string literal prefixes: R"...", uR"...", UR"...", LR"...", u8R"...".
		if ch in {"R", "u", "U", "L"}:
			prefix_end = i
			if ch == "u" and i + 2 < length and chars[i + 1] == "8" and chars[i + 2] == "R":
				prefix_end = i + 2
			elif i + 1 < length and chars[i + 1] == "R":
				prefix_end = i + 1
			if chars[prefix_end] == "R" and prefix_end + 1 < length and chars[prefix_end + 1] == '"':
				start = i
				j = prefix_end + 2
				delim_chars: List[str] = []
				while j < length and chars[j] != "(":
					delim_chars.append(chars[j])
					j += 1
				if j >= length:
					_blank_range(start, length)
					break
				delim = "".join(delim_chars)
				j += 1  # Skip '('
				end_marker = ")" + delim + '"'
				marker_len = len(end_marker)
				while j + marker_len <= length and "".join(chars[j:j + marker_len]) != end_marker:
					j += 1
				j = min(length, j + marker_len)
				_blank_range(start, j)
				i = j
				continue

		# Normal string or character literal.
		if ch in {'"', "'"}:
			quote = ch
			start = i
			i += 1
			escaped = False
			while i < length:
				current = chars[i]
				if escaped:
					escaped = False
					i += 1
					continue
				if current == "\\":
					escaped = True
					i += 1
					continue
				if current == quote:
					i += 1
					break
				i += 1
			_blank_range(start, i)
			continue

		i += 1

	return "".join(chars)


def _compute_grade(percent: int) -> str:
	if percent >= 90:
		return "A"
	if percent >= 75:
		return "B"
	if percent >= 50:
		return "C"
	if percent >= 30:
		return "D"
	return "F"


def _build_modernization_suggestions(details: List[Dict[str, object]]) -> List[str]:
	seen: set[str] = set()
	suggestions: List[str] = []

	for item in details:
		recommendation = str(item.get("recommendation") or "").strip()
		if not recommendation:
			continue
		legacy = bool(item.get("legacy_detected", False))

		# Suggestions are shown only when legacy usage was actually detected.
		if legacy:
			if recommendation not in seen:
				seen.add(recommendation)
				suggestions.append(recommendation)

	return suggestions


def _compute_cyclomatic_complexity(code: str) -> int:
	"""Return a heuristic cyclomatic complexity estimate.

	This approximation counts control-flow keywords and boolean operators. It is
	not AST-precise and should be treated as a directional metric only.
	"""
	decision_tokens = len(_RE_CYCLO_DECISIONS.findall(code))
	bool_ops = len(_RE_CYCLO_BOOL.findall(code))
	ternary_ops = len(_RE_CYCLO_TERNARY.findall(code))
	return 1 + decision_tokens + bool_ops + ternary_ops


def _extract_modernization_metrics(
	source_code: str,
	masked_code: str,
	target_std: str = _DEFAULT_TARGET_STD,
) -> Dict[str, object]:
	"""Extract modernization metrics from masked code.

	This function intentionally keeps regex-based scoring for speed and
	portability. For higher-fidelity critical metrics, AST-assisted signals are
	optionally merged when available.
	"""
	headers = set(_RE_INCLUDE.findall(source_code))
	std_key = _normalize_target_std(target_std)
	modern_headers = _MODERN_HEADERS_BY_STD.get(std_key, _MODERN_HEADERS_BY_STD[_DEFAULT_TARGET_STD])

	raw_pointer_count = len(_RE_RAW_POINTER.findall(masked_code))
	smart_pointer_count = len(_RE_SMART_POINTER.findall(masked_code))
	printf_count = len(_RE_PRINTF.findall(masked_code))
	std_print_count = len(_RE_STD_PRINT.findall(masked_code))
	manual_loop_count = len(_RE_MANUAL_LOOP.findall(masked_code))
	range_loop_count = len(_RE_RANGE_LOOP.findall(masked_code))

	# Supplement a few high-value metrics with AST signals when available.
	ast_signals = _ast_assisted_signals(source_code)
	raw_pointer_count = max(raw_pointer_count, int(ast_signals.get("raw_pointer_count", 0) or 0))
	printf_count = max(printf_count, int(ast_signals.get("printf_count", 0) or 0))

	cyclomatic_complexity = _compute_cyclomatic_complexity(masked_code)
	present_headers = sorted(h for h in modern_headers if h in headers)

	metric_breakdown: List[Dict[str, object]] = []
	metric_score = 0
	metric_max = 100

	if smart_pointer_count > 0 or raw_pointer_count == 0:
		pointer_score = 20
	elif raw_pointer_count > smart_pointer_count:
		pointer_score = 6
	else:
		pointer_score = 12
	metric_score += pointer_score
	metric_breakdown.append(
		{
			"id": "pointer_modernity",
			"score": pointer_score,
			"max_score": 20,
			"raw_pointers": raw_pointer_count,
			"smart_pointers": smart_pointer_count,
			"feedback": "Reduce mutable raw pointers and prefer std::unique_ptr/std::shared_ptr.",
		}
	)

	if printf_count == 0 and std_print_count > 0:
		print_score = 15
	elif printf_count == 0:
		print_score = 10
	else:
		print_score = 2
	metric_score += print_score
	metric_breakdown.append(
		{
			"id": "print_modernity",
			"score": print_score,
			"max_score": 15,
			"printf": printf_count,
			"std_print": std_print_count,
			"feedback": "Prefer std::print/std::println over printf where available.",
		}
	)

	if manual_loop_count == 0:
		loop_score = 15
	elif range_loop_count >= manual_loop_count:
		loop_score = 15
	elif range_loop_count > 0:
		loop_score = 9
	else:
		loop_score = 3
	metric_score += loop_score
	metric_breakdown.append(
		{
			"id": "loop_modernity",
			"score": loop_score,
			"max_score": 15,
			"manual_loops": manual_loop_count,
			"range_loops": range_loop_count,
			"feedback": "Prefer range-based loops and std::ranges algorithms when safe.",
		}
	)

	if cyclomatic_complexity <= 6:
		complexity_score = 20
	elif cyclomatic_complexity <= 12:
		complexity_score = 14
	elif cyclomatic_complexity <= 20:
		complexity_score = 8
	else:
		complexity_score = 3
	metric_score += complexity_score
	metric_breakdown.append(
		{
			"id": "cyclomatic_complexity",
			"score": complexity_score,
			"max_score": 20,
			"cyclomatic_complexity": cyclomatic_complexity,
			"feedback": "Reduce branching complexity where possible by decomposing logic.",
		}
	)

	header_score = min(30, len(present_headers) * 5)
	metric_score += header_score
	metric_breakdown.append(
		{
			"id": "modern_headers",
			"score": header_score,
			"max_score": 30,
			"present": present_headers,
			"target_std": std_key,
			"feedback": "Include modern headers such as <memory>, <span>, <ranges>, and <print> when required.",
		}
	)

	return {
		"raw_pointer_count": raw_pointer_count,
		"smart_pointer_count": smart_pointer_count,
		"printf_count": printf_count,
		"std_print_count": std_print_count,
		"manual_loop_count": manual_loop_count,
		"range_loop_count": range_loop_count,
		"cyclomatic_complexity": cyclomatic_complexity,
		"modern_headers_present": present_headers,
		"metric_score": metric_score,
		"metric_max_score": metric_max,
		"metric_details": metric_breakdown,
		"target_std": std_key,
		"ast_assisted": bool(ast_signals),
	}


def score_cpp23_compliance(source_code: str, target_std: str = _DEFAULT_TARGET_STD) -> Dict[str, object]:
	"""Score C++ source code compliance using weighted heuristic rules.

	Limitations:
	- Pattern matching is regex-based and not AST/lexer perfect.
	- Comments and string literals are masked heuristically to reduce false
	  positives, but edge cases may still exist.
	- The final score is a heuristic blend of rule and metric components; it is
	  not a mathematically exact quality measure.
	- Cyclomatic complexity is approximate and keyword-based, not AST exact.

	Returns:
	- score: total weighted points achieved
	- max_score: maximum possible points
	- percent: integer percentage
	- grade: A-F bucket
	- suggestions: deduplicated recommendations for detected legacy patterns
	- details: per-rule detection and scoring breakdown
	"""
	std_key = _normalize_target_std(target_std)
	cache_key = hashlib.sha256(f"{std_key}\n{source_code}".encode("utf-8")).hexdigest()
	cached = _cache_get(cache_key)
	if cached is not None:
		return cached

	rules = _get_rules()
	code_for_matching = _mask_non_code(source_code)
	total_weight = _validate_rule_weights(rules)
	score = 0
	details: List[Dict[str, object]] = []

	for rule in rules:
		positive = bool(rule.positive_pattern.search(code_for_matching))
		negative = bool(rule.negative_pattern.search(code_for_matching))

		rule_score = 0
		if positive and not negative:
			rule_score = rule.weight
		elif positive and negative:
			rule_score = rule.weight // 2

		score += rule_score
		details.append(
			{
				"id": rule.id,
				"score": rule_score,
				"max_score": rule.weight,
				"positive_detected": positive,
				"legacy_detected": negative,
				"recommendation": rule.recommendation,
			}
		)

	metrics = _extract_modernization_metrics(source_code, code_for_matching, target_std=std_key)
	metric_score = int(metrics.get("metric_score", 0) or 0)
	metric_max_score = int(metrics.get("metric_max_score", 0) or 0)
	rule_percent = int(round((score / total_weight) * 100)) if total_weight > 0 else 0
	metric_percent = int(round((metric_score / metric_max_score) * 100)) if metric_max_score > 0 else 0
	rule_weight, metric_weight = _resolve_component_weights()

	combined_score = int(round(rule_percent * rule_weight + metric_percent * metric_weight))
	combined_max = 100
	percent = int(round((combined_score / combined_max) * 100)) if combined_max > 0 else 0
	grade = _compute_grade(percent)
	suggestions = _build_modernization_suggestions(details)

	for metric in metrics.get("metric_details", []):
		if not isinstance(metric, dict):
			continue
		if int(metric.get("score", 0) or 0) < int(metric.get("max_score", 0) or 0) // 2:
			feedback = str(metric.get("feedback") or "").strip()
			if feedback and feedback not in suggestions:
				suggestions.append(feedback)

	result = {
		"score": combined_score,
		"max_score": combined_max,
		"percent": percent,
		"grade": grade,
		"suggestions": suggestions,
		"details": details,
		"rule_score": score,
		"rule_max_score": total_weight,
		"rule_percent": rule_percent,
		"metric_percent": metric_percent,
		"rule_component_weight": rule_weight,
		"metric_component_weight": metric_weight,
		"metrics": metrics,
		"target_std": std_key,
		"heuristic_note": "Score is a weighted heuristic blend of rule and metric components.",
	}
	_cache_set(cache_key, result)
	return result
