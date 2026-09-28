"""
Layer 0d: 经验-技能-新鲜感系统
每个行为独立mastery值，含艾宾浩斯遗忘曲线
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class Skill:
    mastery: float = 0.0        # 熟练度 0-1
    last_practiced: Optional[str] = None
    times: int = 0
    half_life_days: int = 30

    def to_dict(self) -> dict:
        return {
            "mastery": self.mastery,
            "last_practiced": self.last_practiced,
            "times": self.times,
            "half_life_days": self.half_life_days,
        }


def default_skills() -> Dict[str, Skill]:
    """初始技能（妹妹默认设定：有接吻经验，性经验少）"""
    return {
        "kissing": Skill(mastery=0.5, times=10, half_life_days=45),
        "hugging": Skill(mastery=0.8, times=50, half_life_days=60),
        "hand_holding": Skill(mastery=0.9, times=100, half_life_days=90),
        "breast_touch": Skill(mastery=0.15, times=0, half_life_days=14),
        "handjob_giving": Skill(mastery=0.0, times=0, half_life_days=14),
        "blowjob": Skill(mastery=0.0, times=0, half_life_days=10),
        "cunnilingus_receiving": Skill(mastery=0.0, times=0, half_life_days=10),
        "missionary": Skill(mastery=0.0, times=0, half_life_days=14),
        "cowgirl": Skill(mastery=0.0, times=0, half_life_days=10),
        "doggy": Skill(mastery=0.0, times=0, half_life_days=7),
        "deepthroat": Skill(mastery=0.0, times=0, half_life_days=7),
        "dirty_talk": Skill(mastery=0.0, times=0, half_life_days=7),
        "initiating": Skill(mastery=0.05, times=0, half_life_days=7),
        "self_pleasure": Skill(mastery=0.3, times=5, half_life_days=30),
        "clit_stimulation_receiving": Skill(mastery=0.2, times=2, half_life_days=14),
        "gspot_stimulation_receiving": Skill(mastery=0.0, times=0, half_life_days=10),
        "pillow_talk": Skill(mastery=0.6, times=20, half_life_days=45),
        "cuddling": Skill(mastery=0.85, times=80, half_life_days=60),
    }


def calc_effective_mastery(skill: Skill, now: datetime = None) -> float:
    """计算当前有效熟练度（考虑遗忘）"""
    if skill.times == 0:
        return 0.0
    if now is None:
        now = datetime.now()
    if not skill.last_practiced:
        return skill.mastery * 0.3

    last = datetime.fromisoformat(skill.last_practiced)
    days = max(0.0, (now - last).total_seconds() / 86400)
    half_life = skill.half_life_days * (0.5 + skill.mastery * 0.5)
    retention = math.exp(-days / max(half_life, 1))
    effective = skill.mastery * retention
    minimum = 0.3 if skill.mastery > 0.3 else skill.mastery
    return max(effective, minimum)


def practice_skill(skills: Dict[str, Skill], skill_name: str, intensity: float = 0.5):
    """练习后提升技能"""
    if skill_name not in skills:
        skills[skill_name] = Skill(mastery=0.1, times=1, half_life_days=14)
        skills[skill_name].last_practiced = datetime.now().isoformat()
        return
    s = skills[skill_name]
    s.times += 1
    s.last_practiced = datetime.now().isoformat()
    # 提升幅度：越接近1提升越慢（边际递减）
    gain = (1.0 - s.mastery) * intensity * 0.15
    s.mastery = min(1.0, s.mastery + gain)


@dataclass
class NoveltySystem:
    global_novelty: float = 1.0
    per_act: Dict[str, float] = field(default_factory=dict)
    per_location: Dict[str, float] = field(default_factory=dict)
    session_fatigue: float = 0.0
    interrupts_seen: Dict[str, int] = field(default_factory=dict)

    def update_after_action(self, act_name: str, location: str, arousal_at_time: float):
        cur = self.per_act.get(act_name, 1.0)
        decay = 0.05 * (0.5 + arousal_at_time * 0.5)
        self.per_act[act_name] = cur * (1 - decay)
        loc_cur = self.per_location.get(location, 1.0)
        self.per_location[location] = loc_cur * 0.95
        self.session_fatigue = min(1.0, self.session_fatigue + 0.05)

    def get_sensitivity_bonus(self, act_name: str, location: str) -> float:
        act_nov = self.per_act.get(act_name, 1.0)
        loc_nov = self.per_location.get(location, 1.0)
        return 0.5 + 0.5 * (act_nov * 0.4 + loc_nov * 0.3 + self.global_novelty * 0.3)

    def register_interrupt(self, interrupt_type: str):
        self.interrupts_seen[interrupt_type] = self.interrupts_seen.get(interrupt_type, 0) + 1

    def reset_session(self):
        self.session_fatigue = 0.0

    def to_dict(self) -> dict:
        return {
            "global_novelty": self.global_novelty,
            "per_act": dict(self.per_act),
            "per_location": dict(self.per_location),
            "session_fatigue": self.session_fatigue,
            "interrupts_seen": dict(self.interrupts_seen),
        }


@dataclass
class BodyAwareness:
    """内感受能力：感知自己身体的能力"""
    value: float = 0.4

    def increase_from_experience(self, delta: float = 0.02):
        self.value = min(1.0, self.value + delta)

    def to_dict(self) -> dict:
        return {"value": self.value}
