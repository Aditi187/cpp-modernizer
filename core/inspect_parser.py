from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Dict, List


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


_MASK_NON_CODE_RE = re.compile(
	r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
	re.DOTALL,
)

_SCORE_CACHE: Dict[str, Dict[str, object]] = {}


def _mask_non_code(source_code: str) -> str:
	"""Mask comments and string literals to reduce regex false positives.

	This is a heuristic and not a full lexer. It intentionally preserves text
	length and newlines so match offsets remain stable if needed.
	"""

	def _blank(match: re.Match[str]) -> str:
		return re.sub(r"[^\n]", " ", match.group(0))

	return _MASK_NON_CODE_RE.sub(_blank, source_code)


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


def score_cpp23_compliance(source_code: str) -> Dict[str, object]:
	"""Score C++ source code for C++23 compliance using weighted heuristic rules.

	Limitations:
	- Pattern matching is regex-based and not AST/lexer perfect.
	- Comments and string literals are masked heuristically to reduce false
	  positives, but edge cases may still exist.

	Returns:
	- score: total weighted points achieved
	- max_score: maximum possible points
	- percent: integer percentage
	- grade: A-F bucket
	- suggestions: deduplicated recommendations for detected legacy patterns
	- details: per-rule detection and scoring breakdown
	"""
	cache_key = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
	cached = _SCORE_CACHE.get(cache_key)
	if cached is not None:
		return dict(cached)

	code_for_matching = _mask_non_code(source_code)
	total_weight = sum(rule.weight for rule in _RULES)
	score = 0
	details: List[Dict[str, object]] = []

	for rule in _RULES:
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

	percent = int(round((score / total_weight) * 100)) if total_weight > 0 else 0
	grade = _compute_grade(percent)
	suggestions = _build_modernization_suggestions(details)
	result = {
		"score": score,
		"max_score": total_weight,
		"percent": percent,
		"grade": grade,
		"suggestions": suggestions,
		"details": details,
	}
	_SCORE_CACHE[cache_key] = dict(result)
	return result
