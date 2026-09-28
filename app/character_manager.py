"""
角色管理服务
"""
import os
import json
import uuid
import re
from typing import Dict, List, Optional
from pathlib import Path

from .models import CharacterBase, CharacterSummary, CreateCharacterRequest, UpdateCharacterRequest
from .preset_characters import get_all_presets, get_preset_by_id
from .config import CHARACTERS_DIR
from .media_service import MediaService
from core.content_policy import refresh_character_eligibility


class CharacterManager:
    def __init__(self, small_llm_fn=None):
        self.characters: Dict[str, CharacterBase] = {}
        self.small_llm_fn = small_llm_fn  # 小模型函数，用于从文本提取角色
        self.media_service = MediaService()
        self._init_characters()

    def _init_characters(self):
        """初始化：加载预设 + 加载用户自定义角色"""
        os.makedirs(CHARACTERS_DIR, exist_ok=True)
        # 1. 加载预设
        for preset in get_all_presets():
            refresh_character_eligibility(preset)
            self.media_service.apply_to_character(preset)
            self.characters[preset.id] = preset
        # 2. 加载自定义
        for fname in os.listdir(CHARACTERS_DIR):
            if fname.startswith("custom_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(CHARACTERS_DIR, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    char = CharacterBase(**data)
                    refresh_character_eligibility(char)
                    self.media_service.apply_to_character(char)
                    self.characters[char.id] = char
                except Exception as e:
                    print(f"[CharacterManager] 加载自定义角色失败 {fname}: {e}")

    def list_presets(self) -> List[CharacterBase]:
        """返回预设角色列表"""
        return [c for c in self.characters.values() if c.is_preset]

    def list_characters(self) -> List[CharacterSummary]:
        """返回所有角色的摘要列表"""
        result = []
        for c in self.characters.values():
            preview = c.character_description[:60] + "..." if len(c.character_description) > 60 else c.character_description
            result.append(CharacterSummary(
                id=c.id,
                name=c.name,
                avatar=c.avatar,
                background=c.background,
                avatar_revision=c.avatar_revision,
                background_revision=c.background_revision,
                is_preset=c.is_preset,
                relationship_type=c.relationship_type,
                description_preview=preview,
            ))
        return result

    def get_character(self, char_id: str) -> Optional[CharacterBase]:
        return self.characters.get(char_id)

    def create_character(self, request: CreateCharacterRequest) -> CharacterBase:
        char_id = f"custom_{uuid.uuid4().hex[:8]}"
        data = request.model_dump(exclude={"adult_confirmed"})
        data["limits"] = request.limits.model_dump()
        char = CharacterBase(
            id=char_id,
            avatar="",
            background="",
            is_preset=False,
            adult_verified=request.adult_confirmed,
            **data,
        )
        refresh_character_eligibility(char)
        self.characters[char_id] = char
        self._save_custom_character(char)
        return char

    async def create_character_from_text(self, text: str) -> CharacterBase:
        """从自然语言描述创建自定义角色（优先用小模型提取，失败用关键词规则）"""
        char_id = f"custom_{uuid.uuid4().hex[:8]}"

        # 尝试用小模型提取
        extracted = None
        if self.small_llm_fn:
            try:
                extracted = await self._extract_with_small_model(text)
            except Exception as e:
                print(f"[CharacterManager] 小模型提取失败: {e}, 使用规则提取")

        if extracted is None:
            extracted = self._extract_with_rules(text)

        char = CharacterBase(
            id=char_id,
            name=extracted.get("name", "她"),
            avatar="",
            is_preset=False,
            age=extracted.get("age"),
            adult_verified=False,
            relationship_type=extracted.get("relationship", "classmate"),
            character_description=text,
            initial_outfit=extracted.get("outfit", "casual"),
            initial_closeness=extracted.get("closeness", 0.3),
            initial_trust=extracted.get("trust", 0.3),
        )
        refresh_character_eligibility(char)

        # 保存
        self.characters[char_id] = char
        self._save_custom_character(char)
        return char

    async def _extract_with_small_model(self, text: str) -> dict:
        """用小模型从文本提取结构化信息"""
        prompt = f"""从下面这段角色描述中提取JSON信息，只输出JSON：
{{"name":"名字","age":年龄整数或null,"relationship":"关系(classmate/childhood_friend/senior/junior/teacher/neighbor/stranger/other)","appearance":"外貌描述","personality":"性格关键词"}}
描述：{text}
JSON:"""
        resp = await self.small_llm_fn(prompt, max_new_tokens=200, temperature=0.3)
        # 简单解析JSON
        import re
        m = re.search(r'\{[^}]+\}', resp)
        if m:
            return json.loads(m.group(0))
        return {}

    def _extract_with_rules(self, text: str) -> dict:
        """关键词规则提取（降级方案）"""
        result = {"name": "她", "age": None, "relationship": "classmate", "closeness": 0.3, "trust": 0.3, "outfit": "casual"}
        t = text
        age_match = re.search(r"(?:年龄|年纪|岁数)\s*[:：]?\s*(\d{1,3})\s*岁?|(?<!\d)(\d{1,3})\s*岁", t)
        if age_match:
            result["age"] = int(age_match.group(1) or age_match.group(2))
        # 关系判断
        if any(k in t for k in ["妹妹", "妹", "继妹", "义妹"]):
            result["relationship"] = "step_sister"
            result["closeness"] = 0.5
            result["trust"] = 0.55
        elif any(k in t for k in ["青梅", "一起长大", "隔壁", "发小", "从小认识"]):
            result["relationship"] = "childhood_friend"
            result["closeness"] = 0.65
            result["trust"] = 0.7
        elif any(k in t for k in ["同桌", "同班", "同学"]):
            result["relationship"] = "classmate"
        elif any(k in t for k in ["学姐", "高年级"]):
            result["relationship"] = "senior"
        elif any(k in t for k in ["老师", "班主任"]):
            result["relationship"] = "teacher"
        elif any(k in t for k in ["邻居"]):
            result["relationship"] = "neighbor"
        return result

    def update_character(self, char_id: str, updates: UpdateCharacterRequest) -> Optional[CharacterBase]:
        char = self.characters.get(char_id)
        if not char:
            return None
        data = updates.model_dump(exclude_unset=True)
        next_age = data.get("age", char.age)
        next_verified = data.get("adult_verified", char.adult_verified)
        if next_verified and (next_age is None or next_age < 18):
            raise ValueError("未满 18 岁或年龄缺失时不能确认成年资格")
        for k, v in data.items():
            if v is not None and hasattr(char, k):
                setattr(char, k, v)
        self.media_service.apply_to_character(char)
        refresh_character_eligibility(char)
        if not char.is_preset:
            self._save_custom_character(char)
        return char

    def delete_character(self, char_id: str) -> bool:
        if char_id not in self.characters:
            return False
        if self.characters[char_id].is_preset:
            return False  # 预设不能删除
        path = os.path.join(CHARACTERS_DIR, f"{char_id}.json")
        if os.path.exists(path):
            os.remove(path)
        del self.characters[char_id]
        self.media_service.delete_character_media(char_id)
        return True

    def _save_custom_character(self, char: CharacterBase):
        os.makedirs(CHARACTERS_DIR, exist_ok=True)
        path = os.path.join(CHARACTERS_DIR, f"{char.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(char.model_dump(), f, ensure_ascii=False, indent=2)
