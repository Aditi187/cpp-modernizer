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


_RULES: List[ComplianceRule] = [
	ComplianceRule(
		id="expected_optional",
		weight=20,
		positive_pattern=re.compile(r"\bstd::(expected|optional)\b"),
		negative_pattern=re.compile(r"\b(return\s+-?\d+\s*;|errno\b)"),
		recommendation="Prefer std::expected/std::optional over manual return codes.",
	),
	ComplianceRule(
		id="format_print",
		weight=20,
		positive_pattern=re.compile(r"\bstd::(print|println|format)\b"),
		negative_pattern=re.compile(r"\b(printf|std::cout|std::cerr)\b"),
		recommendation="Prefer std::print/std::format for output formatting.",
	),
	ComplianceRule(
		id="span_mdspan",
		weight=20,
		positive_pattern=re.compile(r"\bstd::(span|mdspan)\b"),
		negative_pattern=re.compile(r"\b(char\s*\*|void\s*\*)\b"),
		recommendation="Use std::span/std::mdspan for buffer handling.",
	),
	ComplianceRule(
		id="ranges_views",
		weight=20,
		positive_pattern=re.compile(r"\bstd::(ranges|views)::"),
		negative_pattern=re.compile(r"for\s*\([^\)]*;[^\)]*;[^\)]*\)"),
		recommendation="Use std::ranges/views where container iteration is modernized.",
	),
	ComplianceRule(
		id="unique_ptr_stack",
		weight=20,
		positive_pattern=re.compile(r"\bstd::unique_ptr\b|\bstd::make_unique\b"),
		negative_pattern=re.compile(r"\b(new|delete)\b"),
		recommendation="Prefer std::unique_ptr and stack allocation over raw new/delete.",
	),
]


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
	return {
		"score": score,
		"max_score": total_weight,
		"percent": percent,
		"details": details,
	}
