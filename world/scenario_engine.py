"""
剧情卡引擎 - 扫描JB/目录，智能选择开场和剧情事件
剧情阶段分级：
  A:日常学习  B:同学校园  C:家庭  D:日常亲密生活
  E:感情互动  F:暧昧边缘  G:初阶性接触  H:口交  I:性爱  J:节日  K:争吵

开场时会根据剧情卡自动设置世界状态（位置、时间、穿着、情绪、隐私度等）。
小模型可用时用小模型智能提取状态，不可用时用关键词规则匹配。
"""
import os
import re
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from core.output_format import normalize_character_output

JB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "JB")

RELATIONSHIP_STAGES = tuple("ABCDEFGHI")
SCENARIO_CATEGORY_STAGES = ("J", "K")

# 阶段前缀 -> 阶段名/所需亲密度门槛/所需私密度
STAGE_INFO: Dict[str, dict] = {
    "A": {"name": "日常学习", "min_trust": 0.0, "min_privacy": 0.0, "min_arousal": 0.0, "intro_ok": True},
    "B": {"name": "同学校园", "min_trust": 0.2, "min_privacy": 0.1, "min_arousal": 0.0, "intro_ok": True},
    "C": {"name": "家庭", "min_trust": 0.35, "min_privacy": 0.25, "min_arousal": 0.0, "intro_ok": True},
    "D": {"name": "日常亲密", "min_trust": 0.5, "min_privacy": 0.4, "min_arousal": 0.0, "intro_ok": True},
    "E": {"name": "感情互动", "min_trust": 0.6, "min_privacy": 0.55, "min_arousal": 0.0, "intro_ok": True},
    "F": {"name": "暧昧边缘", "min_trust": 0.7, "min_privacy": 0.65, "min_arousal": 0.15, "intro_ok": False, "adult_only": True},
    "G": {"name": "初阶性接触", "min_trust": 0.75, "min_privacy": 0.7, "min_arousal": 0.3, "intro_ok": False, "adult_only": True},
    "H": {"name": "口交", "min_trust": 0.82, "min_privacy": 0.78, "min_arousal": 0.5, "intro_ok": False, "adult_only": True},
    "I": {"name": "性爱", "min_trust": 0.88, "min_privacy": 0.85, "min_arousal": 0.6, "intro_ok": False, "adult_only": True},
    "J": {"name": "节日特殊", "min_trust": 0.2, "min_privacy": 0.2, "min_arousal": 0.0, "intro_ok": True},
    "K": {"name": "争吵小情绪", "min_trust": 0.0, "min_privacy": 0.0, "min_arousal": 0.0, "intro_ok": True},
}


@dataclass
class ScenarioCard:
    """一张剧情卡"""
    card_id: str           # e.g. "D-C05"
    title: str             # e.g. "偷偷进你房间"
    location: str          # e.g. "你房间，深夜一点多"
    context: str           # e.g. "你已经睡着了，门轻轻被推开..."
    raw_content: str       # 完整原文
    stage: str             # e.g. "C"
    first_line_bro: str = ""  # 【哥】开头的第一句（用于开场初始化）
    first_line_character: str = ""  # 第一条【妹】回应（权威第0轮文本）

    @property
    def intro_text(self) -> str:
        """给模型的开场描述（地点+情境）"""
        return f"地点：{self.location}\n情境：{self.context}"

    def apply_initial_state(self, engine) -> None:
        """Apply this card's inferred state without generating opening text."""
        s, w = engine.state, engine.world

        # 用小模型提取状态（可用时）；每个字段单独校验，非法字段逐项回退规则值
        init = self._validated_initial_state(self._extract_initial_state())

        # === 位置 ===
        # Explicit card prose is authoritative. The small model only fills gaps and
        # must not relocate a clearly named dinner table, classroom, or bathroom.
        explicit_loc = self._infer_explicit_location()
        loc = explicit_loc or init.get("location") or "bedroom"
        w.set_location(loc)
        w.position_detail = self._infer_position_detail(loc)
        # 用卡片显式地点作为中文显示名（如"妹妹房间"），避免通用"卧室"违和
        label = self._card_location_label()
        if label:
            w.location_label = label

        # === 时间 ===
        time_hour = init.get("time_hour", self._infer_time_hour())
        weekday = init.get("weekday", self._infer_weekday())
        w.game_time.set_time(time_hour, 0)
        w.weekday = weekday
        is_weekend = weekday in ("Saturday", "Sunday")

        # === 隐私度/门锁 ===
        w.door_locked = loc in ("bedroom", "bathroom")
        w.set_base_privacy(init.get("privacy", self._infer_privacy(loc, init)))
        w.danger_level = 0.0

        # === 人物在场 ===
        explicit_people = self._infer_explicit_people()
        if explicit_people is not None:
            w.set_scenario_people(explicit_people)
        elif "people_present" in init:
            w.set_scenario_people(init["people_present"])
        else:
            w.scenario_people_override = False
            w.people_present = {
                "user": {"location": "current", "activity": None},
                "sister": {"location": "current", "activity": None},
            }
            w._update_parents_presence()
        if w.has_people:
            w.door_locked = False
            presence_cap = 0.3 if w.others_present else 0.6
            w.privacy_level = min(w.base_privacy_level, presence_cap)
            w.danger_level = max(w.danger_level, 0.3)

        # === 信任度（根据阶段+情境）===
        base_trust = {
            "A": 0.5, "B": 0.5, "C": 0.55, "D": 0.6,
            "E": 0.65, "F": 0.7, "G": 0.75, "H": 0.8,
            "I": 0.85, "J": 0.6, "K": 0.45,
        }.get(self.stage, 0.55)
        s.relationship_trust = init.get("trust", base_trust + random.uniform(-0.05, 0.05))
        # 亲密度是会话元数据字段，暂存于实例供 SessionManager 建立 meta
        engine.initial_closeness = init.get("closeness", max(0.0, min(1.0, s.relationship_trust - 0.15)))

        # === 情绪初始化 ===
        s.emotion.reset_scenario_baseline()
        init_emotions = init.get("emotions", self._infer_emotions())
        for emo, val in init_emotions.items():
            if emo in s.emotion.blend:
                s.emotion.blend[emo] = max(0.0, min(1.0, float(val)))
        s.emotion.compute_conflict()
        s.emotion.update_primary()

        # 争吵类开场降低信任度
        if self.stage == "K":
            s.relationship_trust = min(s.relationship_trust, 0.45)
            s.mind.active_resistance_will = 0.3

        # === 穿着 ===
        # Explicit garment words in the authored card outrank a contradictory
        # small-model guess, just like location and physical presence.
        outfit = self._infer_explicit_outfit() or init.get("outfit") or self._infer_outfit(loc, time_hour)
        self._apply_outfit(s, outfit)

        # === 生理初始 ===
        s.global_arousal = 0.0
        s.ans.arousal_global = 0.0
        s.ans.heart_rate = getattr(s.ans, "heart_rate_base", 75.0)
        s.orgasm.phase = "excitement"
        s.orgasm.orgasm_count = 0
        # Keep authored/template acceptance unless the card explicitly overrides it.
        # A scenario reset must not turn every sexual interaction into a hard refusal.
        if "acceptance" in init:
            s.mind.acceptance_level = max(0.0, min(1.0, float(init["acceptance"])))
        else:
            s.mind.acceptance_level = max(0.0, min(1.0, float(getattr(s.mind, "acceptance_level", 0.1))))
        s.mind.active_resistance_will = 0.0
        s.mind.refusal_sincerity = 0.0
        s.mind.token_resistance = 0.0

        # Opening text is generated separately after callers finish manual overrides.

    def generate_turn_zero(self, engine, character=None) -> str:
        """Generate the pre-user turn from the final initial state."""
        s, w = engine.state, engine.world
        scene_desc = w.get_scene_description(s)
        clothing = s.clothing.clothing_description(verbose=False)
        deterministic = normalize_character_output(self._build_intro_narrative(scene_desc, clothing))

        # Cards already contain the author's immediate character response. It is
        # safer and more faithful than asking a model to infer who performed the
        # trigger action, which can invert roles (for example, who is shaving).
        curated = normalize_character_output(self.first_line_character)
        if curated:
            return curated

        character_name = getattr(character, "name", "角色") or "角色"
        speech_style = getattr(character, "speech_style", None)
        first_person = getattr(speech_style, "first_person", "我") or "我"
        address_options = getattr(speech_style, "address_user", {}) or {}
        user_address = address_options.get("default", "你") if isinstance(address_options, dict) else "你"
        try:
            from llm_small import is_available, generate
            if not is_available():
                return deterministic
            sys_p = (
                f"你负责生成角色扮演会话的第0轮开场。你扮演{character_name}，"
                f"只能用‘{first_person}’或省略主语指代自己，用‘你’指代对方；"
                f"‘{user_address}’只能作为对话中的称呼。不要用她/他指代{character_name}，"
                "不得替用户追加动作或台词。台词直接写，动作、神态、声音及其他非台词内容"
                "必须用全角中文括号（ ）括起来。1-3句话，直接输出内容，不要角色名前缀、解释或引号。"
            )
            setup = self.first_line_bro or "对方刚刚出现在场景中，尚未说话或行动"
            user_p = (
                f"剧情卡：{self.title}\n{self.intro_text}\n最终场景：{scene_desc}\n"
                f"最终衣着：{clothing}\n卡片给定的开场触发：{setup}\n"
                f"这是用户输入前的第0轮。请写{character_name}紧接着的第一人称开场反应："
            )
            generated = normalize_character_output(
                generate(user_p, system_prompt=sys_p, max_new_tokens=96, temperature=0.45)
            )
            # 开场协议禁止角色名前缀；仅在已知的开场生成路径移除，避免影响普通台词中的合法冒号。
            generated = re.sub(r"^[^：:\n]{1,20}[：:]\s*", "", generated)
            return generated or deterministic
        except Exception:
            return deterministic

    # ---------- 内部推理方法（关键词规则，小模型不可用时使用） ----------

    def _infer_explicit_location(self) -> Optional[str]:
        """Return a location only when the card prose names one unambiguously."""
        text = self.location + self.context
        if any(k in text for k in ("浴室", "卫生间", "厕所", "洗澡", "淋浴")):
            return "bathroom"
        if any(k in text for k in ("学校", "教室", "操场", "图书馆", "校园", "晚自习", "上课")):
            return "school"
        if any(k in text for k in ("客厅", "沙发", "电视", "餐厅", "餐桌", "饭桌")):
            return "living_room"
        if any(k in text for k in ("厨房", "灶台")):
            return "kitchen"
        if any(k in text for k in ("玄关", "门口", "门外")):
            return "entrance"
        if any(k in text for k in ("公园", "商场", "超市", "街上", "电影院", "外面")):
            return "outside"
        if any(k in text for k in ("卧室", "房间", "床上", "被窝", "书房")):
            return "bedroom"
        return None

    def _infer_explicit_people(self) -> Optional[List[str]]:
        """Infer people who are physically present in the current scene."""
        location_text = self.location
        context_text = self.context
        text = location_text + context_text
        # Explicit absence applies to the current scene and must suppress
        # an unreliable roster returned by the small extraction model.
        if any(k in text for k in ("爸妈不在", "父母不在", "家里没人", "独处", "只有我们")):
            return []

        labels: List[str] = []
        # The location line describes the current physical setup. A person
        # merely mentioned in the context may belong to an earlier event.
        if any(k in location_text for k in ("爸妈", "父母", "妈妈", "爸爸")):
            labels.append("爸妈")
        if any(k in location_text for k in ("老师", "班主任")):
            labels.append("老师")
        if any(k in location_text for k in ("同学", "同桌", "班长", "同班")):
            labels.append("同学")
        if any(k in location_text for k in ("闺蜜", "朋友")):
            labels.append("朋友")

        loc = self._infer_explicit_location()
        # A school scene inherently has other students around unless the card
        # explicitly says otherwise. Household inference never applies there.
        if loc == "school" and "同学" not in labels:
            labels.append("同学")

        current_presence_patterns = {
            "爸妈": ("和爸妈一起", "跟爸妈一起", "爸妈都在", "爸妈在旁边", "爸妈在场", "一家人一起"),
            "老师": ("和老师一起", "跟老师一起", "老师在旁边", "老师在场", "老师当面", "老师走过来"),
            "同学": ("和同学一起", "跟同学一起", "同学在旁边", "同学在场", "同学当面", "同学走过来"),
            "朋友": ("和朋友一起", "跟朋友一起", "和闺蜜一起", "跟闺蜜一起", "朋友在旁边", "闺蜜在旁边", "朋友来家里", "闺蜜来家里"),
        }
        for label, patterns in current_presence_patterns.items():
            if label not in labels and any(pattern in context_text for pattern in patterns):
                labels.append(label)

        if labels:
            return labels

        # Returning home after school while recounting a classmate/teacher
        # interaction is an explicit empty current roster, not missing data.
        school_people = ("同学", "同桌", "班长", "老师", "班主任", "后座")
        returned_home = loc in {"bedroom", "living_room", "kitchen", "entrance"} and any(
            marker in location_text for marker in ("放学回来", "放学回家", "回到家", "刚回家")
        )
        if returned_home and any(person in context_text for person in school_people):
            return []
        return None


    def _card_location_label(self) -> str:
        """提取卡片显式地点的中文显示名（去掉时间/星期等），如'妹妹房间'。"""
        import re as _re
        if not self.location:
            return ""
        seg = self.location.split("，")[0].split(",")[0].strip()
        seg = _re.sub(r"\d+[:：]?\d*", "", seg)
        seg = _re.sub(r"(周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六日]|早晨|早上|清晨|晚上|下午|中午|深夜|凌晨)", "", seg).strip()
        return seg.strip("，。；、 ")

    def _infer_position_detail(self, loc: str) -> str:
        ctx = self.context
        if "躺" in ctx or "睡" in ctx or "被窝" in ctx or "床" in ctx:
            return "on_bed"
        if "坐" in ctx and "沙发" in ctx:
            return "on_sofa"
        if loc == "bathroom":
            return "in_bath"
        return "standing" if "站" in ctx else "sitting"

    def _infer_time_hour(self) -> int:
        text = self.location + self.context + self.title
        _CN = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
               '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        def cn2num(word):
            if word == '十':
                return 10
            if word.startswith('十'):
                return 10 + _CN.get(word[1], 0)
            if word.endswith('十') and len(word) > 1:
                return _CN.get(word[0], 1) * 10
            return sum(_CN.get(c, 0) for c in word)
        m = re.search(r"(\d{1,2})\s*[:：点]\s*(\d{1,2})?(?:半|多|左右)?", text)
        explicit = int(m.group(1)) if m else None
        if explicit is None:
            cm = re.search(r"([一二两三四五六七八九十]{1,3})\s*点", text)
            if cm:
                explicit = cn2num(cm.group(1))
        # 深夜X点：显式数字 <=3 为凌晨，>=7 为晚间（深夜十一点=23，深夜一点=1）
        if "深夜" in text and explicit is not None:
            return explicit if explicit <= 3 else (explicit + 12 if explicit < 12 else explicit)
        # 时段词表：(关键词, 合理区间, 是否12小时制下午→+12)
        DAY_PARTS = [
            (("凌晨", "午夜", "半夜", "起夜", "半夜三更"), 0, 4, False),
            (("早晨", "早上", "清晨", "早起", "赖床", "晨跑"), 5, 8, False),
            (("上午", "早自习", "早读"), 7, 11, False),
            (("中午", "午休", "午饭", "午餐"), 11, 13, False),
            (("下午", "午后", "午睡"), 13, 17, True),
            (("傍晚", "黄昏", "放学", "下课", "放学回家", "晚饭", "晚餐", "吃晚饭"), 16, 20, True),
            (("晚上", "晚间", "夜里", "夜晚", "晚自习", "晚修", "睡前"), 19, 22, True),
        ]
        for keys, lo, hi, twelve in DAY_PARTS:
            if any(k in text for k in keys):
                if explicit is not None:
                    h = explicit + 12 if (twelve and explicit < 12) else explicit
                    return max(lo, min(hi, h))
                return random.randint(lo, hi)
        if explicit is not None:
            return max(8, min(22, explicit))
        # 白天活动：运动会/义卖/毕业照/扫墓/逛街等
        if any(k in text for k in ("运动会", "体育会考", "体育课", "课间操", "义卖", "毕业照",
                                   "拍毕业照", "扫墓", "清明", "逛街", "商场", "游乐场", "看电影",
                                   "早操", "大扫除", "社会实践")):
            return random.randint(9, 17)
        # 校园/学习场景默认放学或白天时段
        if self._infer_explicit_location() in ("school",):
            return random.randint(9, 17)
        # 现实 fallback：白天/傍晚，避免大半夜活跃对话
        return random.randint(16, 20)

    def _infer_weekday(self) -> str:
        text = self.location + self.context + self.title
        if any(k in text for k in ["周末", "周六", "周日", "放假", "暑假", "寒假"]):
            return "Saturday"
        if any(k in text for k in ["上学", "上课", "周一", "早读", "课间操", "晚自习", "高考", "考试", "课间"]):
            return random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        # 校园/学习类更可能是工作日
        if self.stage in ("A", "B"):
            return random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        # 亲密/家庭/感情类更可能是周末
        if self.stage in ("D", "E", "F", "C", "K"):
            return "Saturday"
        return random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])

    def _infer_privacy(self, loc: str, init: dict) -> float:
        people = init.get("people_present", [])
        if any("爸妈" in p or "同学" in p or "闺蜜" in p for p in people):
            return 0.1
        if loc in ("school", "outside"):
            return 0.1
        if loc == "living_room":
            return 0.4
        if loc in ("bathroom", "bedroom"):
            return 0.8
        return 0.5

    def _infer_emotions(self) -> Dict[str, float]:
        text = self.title + self.context
        emos = {}
        if any(k in text for k in ["害羞", "脸红", "不好意思", "心跳", "脸发烫"]):
            emos["shyness"] = 0.4
        if any(k in text for k in ["开心", "笑", "快乐", "好玩", "笑眯眯", "眼睛亮"]):
            emos["happiness"] = 0.5
            emos["playfulness"] = 0.3
        if any(k in text for k in ["难过", "哭", "委屈", "红了眼", "伤心", "眼眶红", "哭出来", "想哭"]):
            emos["hurt"] = 0.5
            emos["frustration"] = 0.2
        if any(k in text for k in ["害怕", "怕", "噩梦", "紧张", "慌"]):
            emos["fear"] = 0.4
        if any(k in text for k in ["生气", "闹别扭", "吃醋", "鼓脸", "赌气", "气鼓鼓"]):
            emos["frustration"] = 0.4
            if "吃醋" in text:
                emos["jealousy"] = 0.3
        if any(k in text for k in ["惊喜", "没想到", "愣住", "僵住"]):
            emos["shock"] = 0.4
        if any(k in text for k in ["疼", "肚子痛", "例假", "不舒服", "酸痛"]):
            emos["pain"] = 0.3
        if any(k in text for k in ["紧张", "攥紧", "手心出汗"]):
            emos["anxiety"] = 0.3
        return emos

    def _infer_outfit(self, loc: str, hour: int) -> str:
        text = self.location + self.context
        if any(k in text for k in ["校服", "制服", "上学", "放学", "上课"]):
            return "school_uniform"
        if any(k in text for k in ["浴巾", "裹浴", "洗完澡", "刚洗"]):
            return "bath_towel"
        if any(k in text for k in ["睡衣", "小熊", "睡着", "被窝", "做噩梦"]):
            return "pajamas"
        if any(k in text for k in ["短裙", "裙子", "连衣裙"]):
            return "summer_dress"
        if hour >= 22 or hour < 6:
            return "pajamas" if loc in ("bedroom", "living_room") else "casual"
        if loc in ("school",):
            return "school_uniform"
        return "casual"

    def _infer_explicit_outfit(self) -> Optional[str]:
        """Return an outfit explicitly established for the opening scene."""
        text = self.location + self.context + self.first_line_character
        if any(k in text for k in ("浴巾", "裹浴", "洗完澡", "刚洗完", "刚洗澡")):
            return "bath_towel"
        if any(k in text for k in ("睡衣", "睡裤", "小熊图案", "睡袍", "被窝")):
            return "pajamas"
        if any(k in text for k in ("校服", "制服")):
            return "school_uniform"
        if any(k in text for k in ("内衣", "内裤", "文胸", "胸罩", "蕾丝")) and not any(
            k in text for k in ("白衬衫", "百褶裙")
        ):
            return "lingerie"
        if any(k in text for k in ("连衣裙", "夏日裙", "夏装裙")):
            return "summer_dress"
        return None

    def _apply_outfit(self, s, outfit: str):
        """根据outfit设置衣物层状态（重置为对应套装）"""
        outfit_map = {
            "pajamas": "pajamas",
            "casual": "casual",
            "summer_dress": "summer_dress",
            "school_uniform": "school_uniform",
            "bath_towel": "bath_towel",
            "lingerie": "lingerie",
        }
        mapped = outfit_map.get(outfit, "summer_home")
        s.clothing.reset(mapped)

    def _build_intro_narrative(self, scene_desc: str, clothing: str = "") -> str:
        """Build a deterministic opening from the final objective state."""
        clothing_line = f"，穿着{clothing}" if clothing else ""
        return f"（{scene_desc}{clothing_line}）{self.context}"

    def _validated_initial_state(self, raw: dict) -> dict:
        """逐字段校验小模型结果；缺失或非法值不污染场景状态。"""
        if not isinstance(raw, dict):
            return {}
        valid_locations = {"bedroom", "living_room", "kitchen", "bathroom", "school", "entrance", "outside"}
        valid_weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        valid_outfits = {"pajamas", "casual", "school_uniform", "bath_towel", "summer_dress", "lingerie"}
        result = {}
        if raw.get("location") in valid_locations:
            result["location"] = raw["location"]
        if isinstance(raw.get("time_hour"), (int, float)) and not isinstance(raw["time_hour"], bool) and 0 <= raw["time_hour"] <= 23:
            result["time_hour"] = int(raw["time_hour"])
        if raw.get("weekday") in valid_weekdays:
            result["weekday"] = raw["weekday"]
        people = raw.get("people_present")
        if isinstance(people, list) and all(isinstance(item, str) for item in people):
            result["people_present"] = people[:8]
        emotions = raw.get("emotions")
        if isinstance(emotions, dict):
            allowed_emotions = {"pleasure", "shame", "trust", "fear", "hurt", "jealousy", "anxiety", "happiness", "love", "sleepy", "frustration", "satisfaction", "anticipation", "embarrassment", "overwhelm", "loss_of_control", "inevitability", "pain", "shyness", "longing", "playfulness", "shock"}
            result["emotions"] = {key: float(value) for key, value in emotions.items() if key in allowed_emotions and isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1}
        if raw.get("outfit") in valid_outfits:
            result["outfit"] = raw["outfit"]
        for key in ("trust", "closeness", "privacy", "acceptance"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
                result[key] = float(value)
        return result

    def _extract_initial_state(self) -> dict:
        """用小模型提取初始状态，失败返回空dict。"""
        try:
            from llm_small import extract_structured, is_available
            if not is_available():
                return {}
        except Exception:
            return {}

        import json
        sys_prompt = "你是场景分析器。根据剧情卡描述提取初始状态信息，严格输出JSON，不要其他文字。"
        user_prompt = f"""剧情卡标题：{self.title}
地点：{self.location}
情境：{self.context}

输出JSON：
{{
  "location": "bedroom/living_room/kitchen/bathroom/school/entrance/outside",
  "time_hour": 0-23的整数,
  "weekday": "Monday/Tuesday/.../Sunday",
  "people_present": ["爸妈"或"同学"或"闺蜜"或""],
  "emotions": {{"emotion_name": 0.0-1.0}},
  "outfit": "pajamas/casual/school_uniform/bath_towel/summer_dress/lingerie",
  "trust": 0.0-1.0,
  "closeness": 0.0-1.0,
  "privacy": 0.0-1.0
}}"""
        data = extract_structured(user_prompt, system_prompt=sys_prompt)
        return data if isinstance(data, dict) else {}


def _parse_card(filepath: str) -> Optional[ScenarioCard]:
    """解析一个.md剧情卡文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    lines = [l.rstrip() for l in content.strip().split("\n") if l.strip()]
    if not lines:
        return None

    # 第一行 # D-Xxx：标题
    title = ""
    m = re.match(r"#\s*D-([A-K])(\d+)[：:]\s*(.+)", lines[0])
    if m:
        stage = m.group(1)
        number = m.group(2)
        title = m.group(3).strip()
        card_id = f"D-{stage}{number}"
    else:
        # fallback：从文件名解析
        fname = os.path.basename(filepath).replace(".md", "")
        m2 = re.match(r"D-([A-K])\d+_(.+)", fname)
        if m2:
            stage = m2.group(1)
            number = re.match(r"\d+", m2.group(0).split("_", 1)[0].split(stage, 1)[-1]).group(0)
            title = m2.group(2)
            card_id = f"D-{stage}{number}"
        else:
            return None

    # 找地点和情境
    location = ""
    context = ""
    for line in lines[1:5]:
        ml = re.match(r"#\s*地点[：:]\s*(.+)", line)
        if ml:
            location = ml.group(1).strip()
        mc = re.match(r"#\s*情境[：:]\s*(.+)", line)
        if mc:
            context = mc.group(1).strip()

    # 保留首个用户触发和紧随其后的角色回应，避免生成时重新猜测动作归属。
    first_bro = ""
    first_character = ""
    for line in lines:
        if not first_bro and line.startswith("【哥】"):
            first_bro = line.replace("【哥】", "", 1).strip()
            continue
        if first_bro and line.startswith("【妹】"):
            first_character = line.replace("【妹】", "", 1).strip()
            break

    return ScenarioCard(
        card_id=card_id,
        title=title,
        location=location,
        context=context,
        raw_content=content,
        stage=stage,
        first_line_bro=first_bro,
        first_line_character=first_character,
    )


def scan_scenarios() -> Dict[str, List[ScenarioCard]]:
    """扫描JB/目录，返回按阶段分组的剧情卡"""
    result: Dict[str, List[ScenarioCard]] = {k: [] for k in STAGE_INFO.keys()}
    if not os.path.isdir(JB_DIR):
        return result
    for fname in os.listdir(JB_DIR):
        if not fname.endswith(".md") or not fname.startswith("D-"):
            continue
        card = _parse_card(os.path.join(JB_DIR, fname))
        if card and card.stage in result:
            result[card.stage].append(card)
    return result


class ScenarioEngine:
    """剧情卡引擎"""
    def __init__(self):
        self.scenarios = scan_scenarios()
        self.cards_by_id: Dict[str, ScenarioCard] = {}
        for stage_cards in self.scenarios.values():
            for card in stage_cards:
                if card.card_id in self.cards_by_id:
                    raise ValueError(f"剧情卡ID重复: {card.card_id}")
                self.cards_by_id[card.card_id] = card
        self.used_intro_ids: set = set()  # 已经用过的开场，避免重复
        self._total = sum(len(v) for v in self.scenarios.values())
        # 尝试导入小模型（不强制）
        self._use_small = False
        try:
            from llm_small import is_available
            self._use_small = is_available()
        except Exception:
            pass

    def list_intro_summaries(self) -> List[dict]:
        result = []
        for card_id in sorted(self.cards_by_id):
            card = self.cards_by_id[card_id]
            info = STAGE_INFO.get(card.stage, {})
            if not info.get("intro_ok"):
                continue
            result.append({
                "id": card.card_id,
                "title": card.title,
                "stage": card.stage,
                "stage_name": info.get("name", card.stage),
                "location": card.location,
                "context_preview": card.context[:160],
            })
        return result

    def get_card(self, card_id: str) -> Optional[ScenarioCard]:
        return self.cards_by_id.get(card_id)

    @property
    def total_cards(self) -> int:
        return self._total

    def pick_intro(self, trust: float = 0.6) -> ScenarioCard:
        """选一个开场剧情卡（从intro_ok的阶段中选）"""
        candidates: List[ScenarioCard] = []
        weights: List[float] = []
        for stage_code, info in STAGE_INFO.items():
            if not info["intro_ok"]:
                continue
            if trust < info["min_trust"] - 0.1:
                continue
            cards = [c for c in self.scenarios.get(stage_code, []) if c.card_id not in self.used_intro_ids]
            if not cards:
                # 都用过了就允许重复
                cards = self.scenarios.get(stage_code, [])
            for c in cards:
                candidates.append(c)
                # 初始信任0.6，更倾向于日常/感情互动
                w = 1.0
                if stage_code in ("A", "B", "C"):
                    w = 0.6
                elif stage_code == "D":
                    w = 1.0
                elif stage_code == "E":
                    w = 1.2
                elif stage_code == "J":
                    w = 0.3
                elif stage_code == "K":
                    w = 0.2
                weights.append(w)

        if not candidates:
            return self._fallback_intro(trust=trust)

        # Opening selection is genuinely stochastic. The small model is used
        # later for state extraction and prose generation, not as a deterministic
        # "best card" classifier.
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        self.used_intro_ids.add(chosen.card_id)
        return chosen

    def pick_next_event(self, state, world) -> Optional[ScenarioCard]:
        """根据当前状态选择下一个随机事件（可选，用于插入剧情）"""
        trust = state.relationship_trust
        privacy = world.get_privacy_level(state) if world else 0.5
        arousal = state.global_arousal

        candidates: List[ScenarioCard] = []
        weights: List[float] = []
        for stage_code, info in STAGE_INFO.items():
            if trust < info["min_trust"]:
                continue
            if privacy < info["min_privacy"] * 0.7:
                continue
            if arousal < info["min_arousal"] * 0.5:
                continue
            for c in self.scenarios.get(stage_code, [])[:]:  # 每个阶段最多取前10个做候选
                candidates.append(c)
                weights.append(1.0)

        if not candidates:
            return None

        if self._use_small and len(candidates) >= 2:
            return self._smart_pick(candidates, weights, trust, privacy, arousal, for_intro=False)

        return random.choices(candidates, weights=weights, k=1)[0] if candidates else None

    def _smart_pick(self, candidates, weights, trust, privacy=0.5, arousal=0.0,
                    for_intro=True) -> Optional[ScenarioCard]:
        """用小模型智能选择最合适的剧情卡"""
        try:
            from llm_small import classify
            # 构造候选列表：取权重最高的前8个
            indexed = list(enumerate(candidates))
            indexed.sort(key=lambda x: weights[x[0]], reverse=True)
            top = indexed[:8]
            labels = [f"{i+1}" for i in range(len(top))]
            desc_lines = []
            for i, (_, card) in enumerate(top):
                desc_lines.append(f"{i+1}. {card.title}（{card.location}，{card.context[:30]}...）")
            if for_intro:
                prompt = (
                    f"我们在玩一个角色扮演游戏，哥哥和妹妹的关系信任度{trust:.1f}。"
                    f"请选一个最适合开场的剧情，只输出数字。\n" + "\n".join(desc_lines)
                )
            else:
                prompt = (
                    f"当前状态：信任度{trust:.2f}，私密度{privacy:.2f}，唤起度{arousal:.2f}。"
                    f"请选一个最适合当前情境的随机事件剧情，只输出数字。\n" + "\n".join(desc_lines)
                )
            result = classify(prompt, labels)
            idx = int(result) - 1 if result.isdigit() else 0
            if 0 <= idx < len(top):
                return top[idx][1]
        except Exception:
            pass
        return None

    def _fallback_intro(self, trust: float = 0.6) -> ScenarioCard:
        """Return a safe fallback whose stage is compatible with current trust."""
        compatible_stages = [
            stage for stage, info in STAGE_INFO.items()
            if info.get("intro_ok") and trust >= info["min_trust"] - 0.1
        ]
        stage = "D" if "D" in compatible_stages else (
            "A" if "A" in compatible_stages else compatible_stages[0]
        )
        if stage == "D":
            title = "深夜卧室"
            location = "卧室，深夜"
            context = "父母已经睡了，角色坐在床上抱着抱枕刷手机，听到你进门后抬头看了一眼。"
            first_line = "你推开房门走进卧室。"
        else:
            title = "一起学习"
            location = "卧室，傍晚"
            context = "角色把课本和练习册摊在桌上，安静地等你一起复习。"
            first_line = "你在书桌旁坐下。"
        return ScenarioCard(
            card_id=f"D-{stage}00",
            title=title,
            location=location,
            context=context,
            raw_content="",
            stage=stage,
            first_line_bro=first_line,
        )


# 单例
_engine_instance: Optional[ScenarioEngine] = None


def get_scenario_engine() -> ScenarioEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ScenarioEngine()
    return _engine_instance


if __name__ == "__main__":
    eng = get_scenario_engine()
    print(f"扫描到 {eng.total_cards} 张剧情卡")
    for stage, cards in eng.scenarios.items():
        info = STAGE_INFO[stage]
        print(f"  {stage}({info['name']}): {len(cards)}张, 开场={'✓' if info['intro_ok'] else '✗'}")
    print()
    for _ in range(3):
        c = eng.pick_intro(trust=0.6)
        print(f"随机开场: {c.card_id} - {c.title}")
        print(f"  {c.intro_text}")
        print()
