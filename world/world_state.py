"""
世界状态系统（整合时间系统+事件引擎）
"""
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional
from .time_system import GameTime
from .events import EventEngine

# 星期序列：weekday 随 game_time.day 推进而滚动
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# === 日志补接线（详细排障） ===
try:
    from app.logger import debug, info, success, warning, error, trace
except Exception:
    debug = info = success = warning = error = trace = lambda *a, **k: None
@dataclass
class WorldState:
    # 时间 - 默认深夜23点，父母已睡，安全私密
    game_time: GameTime = field(default_factory=lambda: GameTime(start_hour=23))
    weekday: str = "Saturday"  # 周末深夜更安全
    season: str = "summer"
    is_raining: bool = False

    # 位置
    location_type: str = "bedroom"  # bedroom/bathroom/living_room/kitchen/outside/school
    privacy_level: float = 0.9
    lighting: str = "dark"  # bright/dim/dark/very_dark
    noise_level: float = 0.1
    temperature: float = 26.0
    door_locked: bool = True  # 默认锁门
    curtains_drawn: bool = True

    # 在场人物
    people_present: Dict[str, Dict] = field(default_factory=lambda: {
        "user": {"location": "bed", "activity": None},
        "sister": {"location": "bed", "activity": None},
    })
    has_people: bool = False  # 父母/其他人在场
    parents_present: bool = False
    others_present: bool = False
    scenario_people_override: bool = False
    nearby_sounds: List[str] = field(default_factory=lambda: ["AC_hum"])
    objects_in_reach: List[str] = field(default_factory=lambda: ["lamp", "tissue", "phone", "blanket"])
    condoms_available: bool = False
    danger_level: float = 0.1
    base_privacy_level: Optional[float] = None

    # 事件引擎
    events: EventEngine = field(default_factory=EventEngine)
    event_tick_counter: float = 0.0
    last_interrupt: Optional[str] = None
    last_interrupt_desc: str = ""
    detected: bool = False  # 是否被发现/撞见

    # 位置内具体位置
    position_detail: str = "on_bed"  # on_bed/edge_of_bed/standing/against_wall/on_floor/kneeling/sitting
    mattress_softness: float = 0.7
    blanket_on: bool = True
    pillow_count: int = 2

    LOCATION_ALIASES: ClassVar[Dict[str, str]] = {
        "classroom": "school",
        "教室": "school",
        "卧室": "bedroom",
        "浴室": "bathroom",
        "客厅": "living_room",
        "厨房": "kitchen",
        "室外": "outside",
        "外面": "outside",
    }

    LOCATION_NAMES: ClassVar[Dict[str, str]] = {
        "bedroom": "卧室",
        "bathroom": "浴室",
        "living_room": "客厅",
        "kitchen": "厨房",
        "school": "学校",
        "outside": "外面",
        "public": "公共场所",
        "entrance": "玄关",
    }

    def __post_init__(self):
        if self.base_privacy_level is None:
            self.base_privacy_level = max(0.0, min(1.0, float(self.privacy_level)))


    @classmethod
    def normalize_location(cls, loc: str) -> str:
        return cls.LOCATION_ALIASES.get(loc, loc)

    @property
    def location_name(self) -> str:
        return self.LOCATION_NAMES.get(self.location_type, self.location_type)

    @property
    def hour(self) -> int:
        return self.game_time.hour

    @property
    def minute(self) -> int:
        return self.game_time.minute

    def set_base_privacy(self, level: float):
        """Set the scene baseline; household risk is applied separately."""
        self.base_privacy_level = max(0.0, min(1.0, float(level)))
        self.privacy_level = self.base_privacy_level

    def set_location(self, loc: str):
        """设置位置，自动更新隐私度等参数"""
        loc = self.normalize_location(loc)
        self.location_type = loc
        loc_params = {
            "bedroom": {"privacy": 0.9, "lighting": "dim", "noise": 0.2, "danger": 0.1,
                       "objects": ["lamp", "tissue", "phone", "blanket", "pillow", "alarm_clock", "stuffed_toy"]},
            "bathroom": {"privacy": 0.95, "lighting": "bright", "noise": 0.3, "danger": 0.15,
                        "objects": ["towel", "toilet_paper", "soap", "shower_head", "mirror"]},
            "living_room": {"privacy": 0.4, "lighting": "bright", "noise": 0.3, "danger": 0.3,
                          "objects": ["remote", "couch_cushion", "magazine"]},
            "kitchen": {"privacy": 0.3, "lighting": "bright", "noise": 0.2, "danger": 0.3,
                       "objects": ["counter", "chair", "fridge_handle"]},
            "school": {"privacy": 0.05, "lighting": "bright", "noise": 0.8, "danger": 0.05,
                      "objects": ["desk", "chair", "textbook"]},
            "outside": {"privacy": 0.1, "lighting": "bright", "noise": 0.5, "danger": 0.05,
                       "objects": []},
            "public": {"privacy": 0.02, "lighting": "bright", "noise": 0.9, "danger": 0.02,
                      "objects": []},
        }
        p = loc_params.get(loc, {"privacy": 0.7, "lighting": "dim", "noise": 0.2, "danger": 0.2,
                                 "objects": []})
        self.privacy_level = p["privacy"]
        self.base_privacy_level = p["privacy"]
        self.lighting = p["lighting"]
        self.noise_level = p["noise"]
        self.danger_level = p["danger"]
        self.objects_in_reach = p["objects"]
        self.nearby_sounds = {
            "bedroom": ["AC_hum", "clock_ticking"],
            "bathroom": ["water_heater_hum"],
            "living_room": ["TV_faint", "AC_hum"],
            "kitchen": ["fridge_hum"],
            "school": ["students_talking", "bell_distant"],
            "outside": ["traffic_distant", "wind"],
        }.get(loc, [])
        self._update_parents_presence()
        debug("世界", f"位置变更: {loc}, 隐私={self.privacy_level:.2f}, 光线={self.lighting}, 危险={self.danger_level:.2f}")

    def set_scenario_people(self, people: List[str]):
        """Apply an explicit scenario roster and keep all presence flags aligned."""
        roster = {"user": {"location": "current", "activity": None},
                  "sister": {"location": "current", "activity": None}}
        for index, label in enumerate(people or []):
            if not label:
                continue
            key = f"scenario_{index}"
            roster[key] = {"label": label, "location": "current", "activity": None}
        self.people_present = roster
        self.scenario_people_override = True
        self._sync_presence_flags()

    def _sync_presence_flags(self):
        """Derive presence booleans from the concrete roster."""
        labels = [str(item.get("label", key)) for key, item in self.people_present.items()]
        parent_tokens = ("爸妈", "父母", "妈妈", "爸爸", "mom", "dad", "parent")
        base_people = {"user", "sister", "character", "self"}
        self.parents_present = any(any(token in label for token in parent_tokens) for label in labels)
        self.others_present = any(
            key not in base_people and not any(token in label for token in parent_tokens)
            for key, label in zip(self.people_present, labels)
        )
        self.has_people = self.parents_present or self.others_present

    def _update_parents_presence(self):
        """Update household presence unless the scenario supplied an explicit roster."""
        if self.scenario_people_override:
            self._sync_presence_flags()
            return
        if self.base_privacy_level is not None:
            self.privacy_level = self.base_privacy_level
        h = self.game_time.hour
        weekend = self.weekday in ["Saturday", "Sunday"]
        parents_home = (8 <= h < 23) and (weekend or h >= 18)
        household_location = self.location_type in {
            "bedroom", "bathroom", "living_room", "kitchen", "entrance"
        }
        if parents_home and household_location:
            self.parents_present = True
            self.people_present["mom"] = {
                "label": "妈妈", "location": "living_room", "activity": "watching_tv",
                "can_hear_bedroom": True,
            }
        else:
            self.people_present.pop("mom", None)
            self.people_present.pop("dad", None)
        self._sync_presence_flags()

        if (
            self.parents_present
            and self.location_type in {"bedroom", "living_room", "kitchen"}
        ):
            self.privacy_level = min(
                self.base_privacy_level
                if self.base_privacy_level is not None
                else self.privacy_level,
                0.6,
            )
            self.danger_level = max(self.danger_level, 0.3)

    def advance_time(self, dt_seconds: float):
        """推进时间"""
        prev_day = self.game_time.day
        self.game_time.advance(dt_seconds)
        if self.game_time.day != prev_day:
            self._advance_weekday(self.game_time.day - prev_day)
        self._update_parents_presence()
        debug("世界", f"时间推进: {dt_seconds:.0f}s -> {self.game_time.get_time_str()}, 位置={self.location_type}")

    def _advance_weekday(self, days: int):
        """天数推进时同步星期，否则周末判断永远停留在初始化值。"""
        if self.weekday in WEEKDAYS:
            idx = (WEEKDAYS.index(self.weekday) + days) % 7
            self.weekday = WEEKDAYS[idx]

    def tick(self, dt: float, state=None):
        """世界状态tick"""
        self.advance_time(dt)
        # 事件引擎负责累计 event_tick_counter 并推进事件。
        self.events.tick(dt, self, state)
        # 中断结束后的隐私恢复
        if not self.events.is_interrupted():
            self.last_interrupt = None
            self.last_interrupt_desc = ""

    def get_privacy_level(self, state) -> float:
        """获取实际隐私度（考虑锁门、窗帘、情绪抑制等）"""
        priv = self.privacy_level
        if self.door_locked:
            priv = min(1.0, priv + 0.15)
        if self.curtains_drawn:
            priv = min(1.0, priv + 0.05)
        if self.blanket_on and state and state.global_arousal < 0.5:
            priv = min(1.0, priv + 0.1)
        return priv

    def get_scene_description(self, state=None) -> str:
        """生成场景自然语言描述（给模型用）"""
        parts = []
        # 时间描述
        tod = self.game_time.get_time_of_day()
        time_str = self.game_time.get_time_str()
        parts.append(time_str)
        season_str = ""
        if self.season == "summer":
            season_str = "夏天，房间里有点热"
        elif self.season == "winter":
            season_str = "冬天，被子里很暖"
        if season_str:
            parts.append(season_str)
        # 位置描述
        loc_desc = {
            "bedroom": f"在卧室里，{'门关着，' if self.door_locked else '门没锁，'}"
                      f"{'窗帘拉着' if self.curtains_drawn else '窗帘没拉'}，灯光暖黄",
            "bathroom": "在浴室里，镜子蒙着一层水汽",
            "living_room": "在客厅，电视开着但声音不大",
            "kitchen": "在厨房",
            "school": "在学校里",
            "outside": "在外面",
            "entrance": "在玄关门口",
        }
        parts.append(loc_desc.get(self.location_type, f"在{self.location_type}"))
        # 人物
        if self.parents_present:
            parts.append("爸妈在家，可能听见动静")
        if self.others_present:
            other_labels = [
                str(item.get("label", key))
                for key, item in self.people_present.items()
                if key not in {"user", "sister", "character", "self", "mom", "dad"}
            ]
            if other_labels:
                parts.append(f"附近还有{'、'.join(other_labels[:3])}")
            else:
                parts.append("附近还有其他人")
        # 中断
        if self.last_interrupt_desc:
            parts.append(self.last_interrupt_desc)
        # 姿势
        pos_desc = {
            "on_bed": "躺在床上",
            "on_sofa": "坐在沙发上",
            "edge_of_bed": "坐在床沿",
            "standing": "站着",
            "against_wall": "靠在墙上",
            "on_floor": "在地板上",
            "kneeling": "跪坐着",
            "sitting": "坐着",
            "in_bath": "在浴缸里",
        }
        pd = pos_desc.get(self.position_detail, "")
        if pd:
            parts.append(pd)
        return "，".join(p for p in parts if p) + "。"

    def to_dict(self) -> dict:
        return {
            "game_time": self.game_time.to_dict(),
            "weekday": self.weekday, "season": self.season, "is_raining": self.is_raining,
            "location_type": self.location_type, "privacy_level": self.privacy_level,
            "base_privacy_level": self.base_privacy_level,
            "lighting": self.lighting, "noise_level": self.noise_level,
            "temperature": self.temperature, "door_locked": self.door_locked,
            "curtains_drawn": self.curtains_drawn,
            "has_people": self.has_people,
            "parents_present": self.parents_present,
            "others_present": self.others_present,
            "scenario_people_override": self.scenario_people_override,
            "people_present": self.people_present,
            "danger_level": self.danger_level,
            "position_detail": self.position_detail,
            "blanket_on": self.blanket_on, "detected": self.detected,
            "condoms_available": self.condoms_available,
            "events": self.events.to_dict(),
        }

    def load_from_dict(self, d: dict):
        """从dict加载世界状态"""
        from .time_system import GameTime
        if "game_time" in d:
            self.game_time = GameTime.from_dict(d["game_time"])
        location = d.get("location_type", d.get("location"))
        if location is not None:
            self.set_location(location)
        for k in ["weekday", "season", "is_raining", "lighting", "noise_level", "temperature", "door_locked",
                  "curtains_drawn", "has_people", "parents_present", "others_present", "scenario_people_override",
                  "danger_level", "position_detail", "blanket_on", "detected",
                  "condoms_available"]:
            if k in d:
                setattr(self, k, d[k])
        if isinstance(d.get("people_present"), dict):
            self.people_present = d["people_present"]
            self._sync_presence_flags()
        if "privacy_level" in d:
            self.privacy_level = d["privacy_level"]
        if "base_privacy_level" in d:
            self.base_privacy_level = d["base_privacy_level"]
        elif "privacy_level" in d:
            self.base_privacy_level = d["privacy_level"]
        if "events" in d:
            self.events.load_from_dict(d["events"], world=self)
