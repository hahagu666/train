import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.character_manager import CharacterManager
from app.models import CreateCharacterRequest, UpdateAppSettingsRequest, UpdateCharacterRequest
from world.scenario_engine import get_scenario_engine


class UIContractTests(unittest.TestCase):
    def test_structured_character_creation_derives_server_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.character_manager.CHARACTERS_DIR", directory), patch(
                "app.character_manager.MediaService"
            ) as media_type:
                media_type.return_value.apply_to_character.return_value = None
                manager = CharacterManager()
                request = CreateCharacterRequest(
                    name=" 测试角色 ",
                    age=20,
                    adult_confirmed=True,
                    character_description="结构化创建",
                    likes=["音乐", "音乐", " 电影 "],
                )
                character = manager.create_character(request)
                self.assertTrue(character.id.startswith("custom_"))
                self.assertFalse(character.is_preset)
                self.assertTrue(character.sexual_interaction_allowed)
                self.assertEqual(character.likes, ["音乐", "电影"])
                self.assertTrue(Path(directory, character.id + ".json").is_file())

    def test_character_creation_rejects_minor_confirmation_and_extra_fields(self):
        with self.assertRaises(ValidationError):
            CreateCharacterRequest(
                name="角色", age=17, adult_confirmed=True,
                character_description="测试",
            )
        with self.assertRaises(ValidationError):
            CreateCharacterRequest(
                name="角色", age=20, character_description="测试",
                sexual_interaction_allowed=True,
            )

    def test_character_update_rejects_minor_verified_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.character_manager.CHARACTERS_DIR", directory), patch(
                "app.character_manager.MediaService"
            ) as media_type:
                media_type.return_value.apply_to_character.return_value = None
                manager = CharacterManager()
                character = manager.create_character(CreateCharacterRequest(
                    name="角色",
                    age=20,
                    adult_confirmed=True,
                    character_description="测试",
                ))
                with self.assertRaises(ValueError):
                    manager.update_character(character.id, UpdateCharacterRequest(age=17))
                self.assertEqual(manager.get_character(character.id).age, 20)
                with self.assertRaises(ValueError):
                    manager.update_character(
                        character.id,
                        UpdateCharacterRequest(age=17, adult_verified=True),
                    )

    def test_character_update_can_clear_verification_before_lowering_age(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.character_manager.CHARACTERS_DIR", directory), patch(
                "app.character_manager.MediaService"
            ) as media_type:
                media_type.return_value.apply_to_character.return_value = None
                manager = CharacterManager()
                character = manager.create_character(CreateCharacterRequest(
                    name="角色",
                    age=20,
                    adult_confirmed=True,
                    character_description="测试",
                ))
                updated = manager.update_character(
                    character.id,
                    UpdateCharacterRequest(age=17, adult_verified=False),
                )
                self.assertFalse(updated.adult_verified)
                self.assertFalse(updated.sexual_interaction_allowed)

    def test_scenario_projection_does_not_expose_source(self):
        cards = get_scenario_engine().list_intro_summaries()
        self.assertTrue(cards)
        self.assertNotIn("raw_content", cards[0])
        self.assertEqual(len({card["id"] for card in cards}), len(cards))

    def test_settings_reject_unknown_fields(self):
        with self.assertRaises(ValidationError):
            UpdateAppSettingsRequest(stream_response=True)


if __name__ == "__main__":
    unittest.main()
