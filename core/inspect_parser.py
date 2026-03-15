from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass(frozen=True)
class ComplianceRule:
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
		weight=8,
		positive_pattern=_regex(r"\bstd::(ranges|views)::"),
		negative_pattern=_regex(r"\bstd::(sort|transform|find|for_each|copy|count|any_of|all_of|none_of)\s*\("),
		recommendation="Prefer std::ranges/views pipelines where algorithm composition is needed.",
	),
	ComplianceRule(
		id="ranges_algorithms",
		weight=8,
		positive_pattern=_regex(r"\bstd::ranges::(sort|transform|find|for_each|copy|count|any_of|all_of|none_of)\b"),
		negative_pattern=_regex(r"\bstd::(sort|transform|find|for_each|copy|count|any_of|all_of|none_of)\s*\("),
		recommendation="Use std::ranges algorithms (e.g., std::ranges::sort) when modern alternatives exist.",
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
		positive = bool(item.get("positive_detected", False))
		legacy = bool(item.get("legacy_detected", False))
		raw_score = item.get("score", 0)
		if isinstance(raw_score, (int, float)):
			rule_score = int(raw_score)
		else:
			rule_score = 0

		# Suggest improvements when legacy patterns exist or the rule has no score yet.
		if legacy or (not positive and rule_score == 0):
			if recommendation not in seen:
				seen.add(recommendation)
				suggestions.append(recommendation)

	return suggestions


def score_cpp23_compliance(source_code: str) -> Dict[str, object]:
	total_weight = sum(rule.weight for rule in _RULES)
	score = 0
	details: List[Dict[str, object]] = []

	for rule in _RULES:
		positive = bool(rule.positive_pattern.search(source_code))
		negative = bool(rule.negative_pattern.search(source_code))

		rule_score = 0
		if positive and not negative:
			rule_score = rule.weight
		elif positive and negative:
			rule_score = max(1, rule.weight // 2)

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
	return {
		"score": score,
		"max_score": total_weight,
		"percent": percent,
		"grade": grade,
		"suggestions": suggestions,
		"details": details,
	}
