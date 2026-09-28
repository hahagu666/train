"""
情感记忆系统
- hurt_episodes: 受伤/吃醋事件，随时间愈合，但会留下创伤痕迹
- trauma_sensitivity: 特定触发词/动作会让旧伤复发
- positive_memories: 正面体验（高潮、温柔时刻等）提升信任
- memory_consolidation: 睡眠/时间跳跃时记忆巩固
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class HurtEpisode:
    """受伤事件记录"""
    type: str  # hurt/jealousy/rejection/forced/betrayal
    intensity: float  # 0-1
    time_stamp: float  # sim_time
    cause: str = ""
    description: str = ""
    healed: float = 0.0  # 0=未愈合, 1=完全愈合
    impact_trust: float = 0.0
    trigger_keywords: List[str] = field(default_factory=list)


@dataclass
class PositiveMemory:
    """正面记忆"""
    type: str  # orgasm/tenderness/confession/aftercare/trust_moment
    intensity: float
    time_stamp: float
    description: str = ""
    impact_trust: float = 0.0


@dataclass
class MemorySystem:
    """完整情感记忆"""
    hurt_episodes: List[HurtEpisode] = field(default_factory=list)
    positive_memories: List[PositiveMemory] = field(default_factory=list)
    trauma_sensitivity: Dict[str, float] = field(default_factory=dict)
    # 关键词 -> 敏感度 0-1，超过0.3会被触发
    total_positive_bond: float = 0.0
    total_hurt_unhealed: float = 0.0

    # 最近一次互动的正负分数（用于更新关系）
    recent_valence: float = 0.0
    valence_window: List[float] = field(default_factory=list)

    def record_hurt(self, hurt_type: str, intensity: float, sim_time: float,
                    cause: str = ""):
        """记录一次受伤事件"""
        episode = HurtEpisode(
            type=hurt_type, intensity=intensity, time_stamp=sim_time,
            cause=cause, impact_trust=intensity * 0.1,
            trigger_keywords=_keywords_for_hurt_type(hurt_type, cause),
        )
        self.hurt_episodes.append(episode)
        self.recent_valence -= intensity * 0.5
        # 创伤敏感度累积
        for kw in episode.trigger_keywords:
            self.trauma_sensitivity[kw] = min(1.0,
                self.trauma_sensitivity.get(kw, 0) + intensity * 0.3)
        self._recalculate_totals()

    def record_positive(self, pos_type: str, intensity: float, sim_time: float,
                        description: str = ""):
        """记录一次正面体验"""
        mem = PositiveMemory(
            type=pos_type, intensity=intensity, time_stamp=sim_time,
            description=description,
            impact_trust=intensity * 0.05 if pos_type != "orgasm" else intensity * 0.02,
        )
        self.positive_memories.append(mem)
        self.recent_valence += intensity * 0.3
        # 正面记忆也能部分修复创伤
        if self.hurt_episodes:
            for ep in self.hurt_episodes:
                ep.healed = min(1.0, ep.healed + intensity * 0.05)
        self._recalculate_totals()

    def check_trigger(self, text: str, action_type: str = "") -> Optional[str]:
        """检查当前输入是否触发创伤反应"""
        if not self.trauma_sensitivity:
            return None
        words = text + " " + action_type
        triggered = []
        for kw, sens in self.trauma_sensitivity.items():
            if sens < 0.2:
                continue
            if kw in words:
                triggered.append((kw, sens))
        if not triggered:
            return None
        triggered.sort(key=lambda x: -x[1])
        return triggered[0][0]

    def get_trigger_modifier(self, trigger_kw: str) -> float:
        """获取创伤触发带来的情绪修正"""
        sens = self.trauma_sensitivity.get(trigger_kw, 0)
        return sens  # 0-1，越高越负面

    def maintenance(self, dt_hours: float = 0):
        """
        记忆维护（在时间跳跃/长时间休息后调用）
        - hurt愈合
        - 过旧的记忆淡化
        - 创伤敏感度自然衰减（非常慢）
        """
        # 受伤事件愈合
        for ep in self.hurt_episodes:
            # 每小时愈合一点，强度低的愈合快
            heal_rate = 0.02 / (ep.intensity + 0.3)  # 每小时愈合率
            ep.healed = min(1.0, ep.healed + heal_rate * dt_hours)
        # 删除完全愈合且很久了的
        self.hurt_episodes = [
            ep for ep in self.hurt_episodes
            if not (ep.healed > 0.95 and ep.time_stamp < 100 * 3600)
        ]
        # 创伤敏感度缓慢衰减
        for kw in list(self.trauma_sensitivity.keys()):
            self.trauma_sensitivity[kw] *= math.exp(-0.01 * dt_hours / 24)
            if self.trauma_sensitivity[kw] < 0.05:
                del self.trauma_sensitivity[kw]
        # valence窗口清空（新的一天）
        self.valence_window.append(self.recent_valence)
        self.valence_window = self.valence_window[-10:]
        self.recent_valence = 0.0
        self._recalculate_totals()

    def _recalculate_totals(self):
        """重算总体bond/hurt"""
        self.total_hurt_unhealed = sum(
            ep.intensity * (1 - ep.healed) for ep in self.hurt_episodes
        )
        self.total_positive_bond = sum(
            m.intensity for m in self.positive_memories[-50:]  # 最近50次
        )

    def get_trust_modifier(self) -> float:
        """获取记忆系统对信任的影响"""
        # 正面累积+受伤扣分
        pos = min(0.3, self.total_positive_bond * 0.01)
        neg = -min(0.4, self.total_hurt_unhealed * 0.1)
        return pos + neg

    def get_current_mood_modifier(self) -> Dict[str, float]:
        """获取当前记忆带来的情绪倾向"""
        mods = {}
        # 未愈的伤带来base受伤感
        if self.total_hurt_unhealed > 0.5:
            mods["hurt"] = min(0.5, self.total_hurt_unhealed * 0.3)
            mods["trust_baseline"] = -self.total_hurt_unhealed * 0.2
        # 大量正面记忆带来base安全感
        if self.total_positive_bond > 5.0:
            mods["safety"] = min(0.3, self.total_positive_bond * 0.02)
        # 最近valence
        avg_valence = (sum(self.valence_window[-5:]) / max(1, len(self.valence_window[-5:])))
        if abs(avg_valence) > 0.2:
            mods["recent_valence"] = avg_valence
        return mods

    def to_dict(self) -> dict:
        return {
            "hurt_episodes": [ep.__dict__ for ep in self.hurt_episodes],
            "positive_memories": [m.__dict__ for m in self.positive_memories[-100:]],
            "trauma_sensitivity": dict(self.trauma_sensitivity),
            "total_positive_bond": self.total_positive_bond,
            "total_hurt_unhealed": self.total_hurt_unhealed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemorySystem':
        m = cls()
        for ep_data in data.get("hurt_episodes", []):
            m.hurt_episodes.append(HurtEpisode(**ep_data))
        for pm_data in data.get("positive_memories", []):
            m.positive_memories.append(PositiveMemory(**pm_data))
        m.trauma_sensitivity = data.get("trauma_sensitivity", {})
        m.total_positive_bond = data.get("total_positive_bond", 0)
        m.total_hurt_unhealed = data.get("total_hurt_unhealed", 0)
        return m


def _keywords_for_hurt_type(hurt_type: str, cause: str) -> List[str]:
    """根据伤害类型生成触发关键词"""
    base = []
    if hurt_type == "jealousy":
        base = ["other_girl", "别的女人", "别人", "她", "前女友"]
    elif hurt_type == "rejection":
        base = ["不喜欢", "讨厌", "嫌弃"]
    elif hurt_type == "betrayal":
        base = ["骗我", "撒谎", "骗"]
    elif hurt_type == "forced":
        base = ["用力", "硬来", "强行"]
    elif hurt_type == "hurt":
        base = ["痛", "疼", "粗鲁"]
    # 加入cause里的名词
    if cause:
        for w in cause.split():
            if len(w) >= 2:
                base.append(w)
    return list(set(base))


def record_hurt(memory: MemorySystem, hurt_type: str, intensity: float, sim_time: float,
                cause: str = ""):
    """便捷函数记录伤害"""
    if memory:
        memory.record_hurt(hurt_type, intensity, sim_time, cause)


def record_orgasm_memory(memory: MemorySystem, intensity: float, sim_time: float,
                        orgasm_type: str = "", is_mutual: bool = False):
    """记录高潮正面记忆"""
    if memory:
        desc = f"{orgasm_type}高潮，强度{intensity:.2f}"
        if is_mutual:
            desc += "，同时到达"
        memory.record_positive("orgasm", intensity, sim_time, desc)
