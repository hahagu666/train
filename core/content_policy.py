"""Centralized eligibility policy for adult interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


ELIGIBILITY_VERSION = 2
MIN_ADULT_AGE = 18


@dataclass(frozen=True)
class EligibilityDecision:
    allowed: bool
    code: str
    reason: str
    version: int = ELIGIBILITY_VERSION


def evaluate_adult_eligibility(
    age: Optional[int],
    adult_verified: bool,
    identity_text: str = "",
) -> EligibilityDecision:
    """Return the authoritative, fail-closed adult eligibility decision."""
    if age is None:
        return EligibilityDecision(False, "AGE_MISSING", "角色年龄未填写，成人互动已暂停")
    if age < MIN_ADULT_AGE:
        return EligibilityDecision(False, "UNDER_18", "角色未满18岁，不能进入成人互动")
    if not adult_verified:
        return EligibilityDecision(
            False,
            "ADULT_CONFIRMATION_REQUIRED",
            "角色尚未完成成年人确认，成人互动已暂停",
        )
    return EligibilityDecision(True, "ADULT_VERIFIED", "已确认角色年满18岁")


def evaluate_character(character) -> EligibilityDecision:
    if character is None:
        return EligibilityDecision(False, "CHARACTER_MISSING", "角色不存在")
    identity_text = " ".join(
        str(value or "")
        for value in (
            getattr(character, "character_description", ""),
            getattr(character, "backstory", ""),
            getattr(character, "relationship_type", ""),
        )
    )
    return evaluate_adult_eligibility(
        getattr(character, "age", None),
        bool(getattr(character, "adult_verified", False)),
        identity_text,
    )


def refresh_character_eligibility(character) -> EligibilityDecision:
    decision = evaluate_character(character)
    character.sexual_interaction_allowed = decision.allowed
    character.eligibility_reason = decision.reason
    character.eligibility_version = decision.version
    return decision


def can_execute_adult_action(character) -> bool:
    return evaluate_character(character).allowed
