"""
用户人设管理
"""
import os
import json
import uuid
from copy import deepcopy
from typing import Dict, Optional

from .models import SessionUserProfileInput, UserProfileResponse, UserProfileBase
from .config import USER_PROFILE_PATH


class UserProfileManager:
    def __init__(self, small_llm_fn=None):
        self.profile: Optional[UserProfileResponse] = None
        self.small_llm_fn = small_llm_fn
        self._load()

    def _load(self):
        if os.path.exists(USER_PROFILE_PATH):
            try:
                with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 兼容旧数据：extracted_traits可能是dict
                if "extracted_traits" in data and isinstance(data["extracted_traits"], dict):
                    data["extracted_traits"] = list(data["extracted_traits"].values()) if data["extracted_traits"] else []
                self.profile = UserProfileResponse(**data)
            except Exception:
                self.profile = self._default()
        else:
            self.profile = self._default()

    def _default(self) -> UserProfileResponse:
        return UserProfileResponse(
            name="你",
            gender="unknown",
            age=0,
            personality_description="",
            speaking_style_hint="",
            appearance_hint="",
            relation_to_characters={},
            custom_name_preference={},
            extracted_traits=[],
        )

    def _save(self):
        os.makedirs(os.path.dirname(USER_PROFILE_PATH), exist_ok=True)
        temp_path = f"{USER_PROFILE_PATH}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.profile.model_dump(), f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, USER_PROFILE_PATH)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_profile(self) -> UserProfileResponse:
        return self.profile

    def build_session_snapshot(
        self,
        char_id: str,
        overrides: Optional[SessionUserProfileInput] = None,
    ) -> dict:
        """Resolve an independent user identity snapshot for a new session."""
        profile = self.profile or self._default()
        snapshot = {
            "name": profile.name or "你",
            "gender": profile.gender or "unknown",
            "age": max(int(profile.age or 0), 0),
            "personality_description": profile.personality_description or "",
            "speaking_style_hint": profile.speaking_style_hint or "",
            "appearance_hint": profile.appearance_hint or "",
            "traits": list(profile.extracted_traits),
            "relation": profile.relation_to_characters.get(char_id, ""),
            "preferred_address": self.get_address_form(char_id),
            "additional_context": "",
        }
        if overrides is not None:
            for key, value in overrides.model_dump(exclude_none=True).items():
                snapshot[key] = deepcopy(value)
        return snapshot

    async def set_profile_from_text(self, text: str) -> UserProfileResponse:
        """用自然语言设置/更新用户人设"""
        extracted = {}
        if self.small_llm_fn:
            try:
                extracted = await self._extract_with_small_model(text)
            except Exception as e:
                print(f"[UserProfile] 小模型提取失败: {e}")

        if not extracted:
            extracted = self._extract_with_rules(text)

        # 更新profile
        self.profile.personality_description = text
        if "name" in extracted and extracted["name"]:
            self.profile.name = extracted["name"]
        if "gender" in extracted:
            self.profile.gender = extracted["gender"]
        if "age" in extracted:
            self.profile.age = int(extracted.get("age", 0))
        if "speaking_style" in extracted:
            self.profile.speaking_style_hint = extracted["speaking_style"]
        if "appearance" in extracted:
            self.profile.appearance_hint = extracted["appearance"]
        if "traits" in extracted:
            self.profile.extracted_traits = extracted["traits"]
        if "relations" in extracted:
            self.profile.relation_to_characters.update(extracted["relations"])
        if "addresses" in extracted:
            self.profile.custom_name_preference.update(extracted["addresses"])

        self._save()
        return self.profile

    async def _extract_with_small_model(self, text: str) -> dict:
        prompt = f"""从下面这段用户自我描述中提取JSON，只输出JSON：
{{"name":"名字或称呼","gender":"male/female/unknown","age":数字,"relations":{{"char_id":"关系"}},"addresses":{{"char_id":"她怎么称呼我"}},"traits":["性格特质"],"speaking_style":"说话风格","appearance":"外貌"}}
用户描述：{text}
JSON:"""
        resp = await self.small_llm_fn(prompt, max_new_tokens=300, temperature=0.3)
        import re
        m = re.search(r'\{[\s\S]+\}', resp)
        if m:
            return json.loads(m.group(0))
        return {}

    def _extract_with_rules(self, text: str) -> dict:
        result = {"traits": [], "relations": {}, "addresses": {}}
        t = text
        # 性别
        if any(k in t for k in ["男", "哥哥", "他"]):
            result["gender"] = "male"
        elif any(k in t for k in ["女", "姐姐", "她"]):
            result["gender"] = "female"
        # 关系
        if "哥哥" in t or "哥" in t:
            result["relations"]["imouto"] = "step_brother"
            result["addresses"]["imouto"] = "哥哥"
        if "老师" in t:
            for cid in ["imouto", "desk_mate", "childhood_friend"]:
                result["relations"][cid] = "teacher"
                result["addresses"][cid] = "老师"
        if "同学" in t or "同桌" in t:
            result["relations"]["desk_mate"] = "classmate"
        if "青梅竹马" in t or "发小" in t or "一起长大" in t:
            result["relations"]["childhood_friend"] = "childhood_friend"
            result["addresses"]["childhood_friend"] = ""
        return result

    def update_name(self, name: str):
        self.profile.name = name
        self._save()

    def set_relation(self, char_id: str, relation: str, custom_address: Optional[str] = None):
        self.profile.relation_to_characters[char_id] = relation
        if custom_address:
            self.profile.custom_name_preference[char_id] = custom_address
        self._save()

    def get_address_form(self, char_id: str) -> str:
        """获取该角色对用户的称呼"""
        if char_id in self.profile.custom_name_preference:
            return self.profile.custom_name_preference[char_id]
        # 默认称呼根据关系
        rel = self.profile.relation_to_characters.get(char_id, "")
        default_map = {
            "step_brother": "哥哥",
            "brother": "哥哥",
            "teacher": "老师",
            "classmate": "你",
            "childhood_friend": "",
            "senior": "学长",
            "junior": "学弟",
            "neighbor": "喂",
        }
        return default_map.get(rel, self.profile.name or "你")

    def reset(self):
        self.profile = self._default()
        self._save()
