import unittest

from app.models import CharacterBase
from core.content_policy import (
    can_execute_adult_action,
    evaluate_character,
    refresh_character_eligibility,
)
from core.parser import ParsedAction
from app.chat_orchestrator import ChatOrchestrator


class EligibilityTests(unittest.TestCase):
    def test_missing_age_is_not_adult_eligible(self):
        character = CharacterBase(id="missing", name="角色")
        decision = evaluate_character(character)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "AGE_MISSING")

    def test_under_eighteen_is_not_adult_eligible(self):
        character = CharacterBase(id="minor", name="角色", age=17)
        decision = evaluate_character(character)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "UNDER_18")

    def test_unverified_adult_is_not_adult_eligible(self):
        character = CharacterBase(id="unverified", name="角色", age=18)
        decision = refresh_character_eligibility(character)
        self.assertFalse(decision.allowed)
        self.assertFalse(character.sexual_interaction_allowed)
        self.assertFalse(can_execute_adult_action(character))

    def test_stale_derived_flag_does_not_override_policy(self):
        character = CharacterBase(
            id="stale",
            name="角色",
            age=17,
            sexual_interaction_allowed=True,
        )
        self.assertFalse(can_execute_adult_action(character))
        refresh_character_eligibility(character)
        self.assertFalse(character.sexual_interaction_allowed)

    def test_missing_character_fails_closed(self):
        decision = evaluate_character(None)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "CHARACTER_MISSING")

    def test_verified_eighteen_year_old_high_school_student_is_allowed(self):
        character = CharacterBase(
            id="school",
            name="角色",
            age=18,
            adult_verified=True,
            character_description="18岁的高中同班同学",
        )
        decision = evaluate_character(character)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.code, "ADULT_VERIFIED")

    def test_verified_adult_with_young_appearance_is_allowed(self):
        character = CharacterBase(
            id="adult",
            name="角色",
            age=18,
            adult_verified=True,
            character_description="成年后仍然身材娇小，外观偏幼态",
        )
        decision = evaluate_character(character)
        self.assertTrue(decision.allowed)

    def test_sexual_action_gate_is_separate_from_nonsexual_action(self):
        character = CharacterBase(id="minor", name="角色", age=17)
        orchestrator = ChatOrchestrator(None, None, None)
        sexual = ParsedAction("touch_breast")
        nonsexual = ParsedAction("hug")
        self.assertTrue(orchestrator._blocked_by_adult_eligibility(character, sexual))
        self.assertFalse(orchestrator._blocked_by_adult_eligibility(character, nonsexual))


if __name__ == "__main__":
    unittest.main()
