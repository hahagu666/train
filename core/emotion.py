"""
Layer 0c: 认知情绪门控
情绪是连续混合向量，不是离散标签
支持矛盾情绪同时存在（既羞耻又渴望）
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class EmotionState:
    primary: str = "calm"
    blend: Dict[str, float] = field(default_factory=lambda: {
        "pleasure": 0.0,
        "shame": 0.2,
        "trust": 0.7,
        "fear": 0.0,
        "hurt": 0.0,
        "jealousy": 0.0,
        "anxiety": 0.0,
        "happiness": 0.3,
        "love": 0.5,
        "sleepy": 0.0,
        "frustration": 0.0,
        "satisfaction": 0.0,
        "anticipation": 0.0,
        "embarrassment": 0.2,
        "overwhelm": 0.0,
        "loss_of_control": 0.0,
        "inevitability": 0.0,
        "pain": 0.0,
        "shyness": 0.3,
        "longing": 0.0,
        "playfulness": 0.2,
        "shock": 0.0,
    })
    conflict_level: float = 0.0
    cognitive_gate: float = 0.8
    forbidden_arousal_bonus: float = 0.0  # 禁果效应唤起加成
    attention_focus: List[str] = field(default_factory=lambda: ["partner"])
    inhibited: bool = False
    suppressing_sounds: bool = False
    suppressing_movements: bool = False
    interrupt_triggered: bool = False
    interrupt_type: str = ""
    last_interrupt_time: float = -100.0

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "blend": dict(self.blend),
            "conflict_level": self.conflict_level,
            "cognitive_gate": self.cognitive_gate,
            "forbidden_arousal_bonus": self.forbidden_arousal_bonus,
            "attention_focus": list(self.attention_focus),
            "inhibited": self.inhibited,
            "suppressing_sounds": self.suppressing_sounds,
            "suppressing_movements": self.suppressing_movements,
            "interrupt_triggered": self.interrupt_triggered,
            "interrupt_type": self.interrupt_type,
            "last_interrupt_time": self.last_interrupt_time,
        }

    def reset(self):
        """重置情绪到默认初始状态（平静/日常）"""
        self.primary = "calm"
        for k in self.blend:
            self.blend[k] = 0.0
        self.blend["happiness"] = 0.3
        self.blend["shyness"] = 0.3
        self.blend["trust"] = 0.7
        self.blend["love"] = 0.5
        self.blend["shame"] = 0.2
        self.blend["embarrassment"] = 0.2
        self.blend["playfulness"] = 0.2
        self.conflict_level = 0.0
        self.cognitive_gate = 0.8
        self.forbidden_arousal_bonus = 0.0
        self.attention_focus = ["partner"]
        self.inhibited = False
        self.suppressing_sounds = False
        self.suppressing_movements = False
        self.interrupt_triggered = False
        self.interrupt_type = ""
        self.last_interrupt_time = -100.0

    def reset_scenario_baseline(self):
        """Reset to a neutral scene baseline before applying card-specific values."""
        self.reset()
        for key in self.blend:
            self.blend[key] = 0.0
        self.primary = "calm"
        self.conflict_level = 0.0

    def update_primary(self):
        """根据混合向量更新主导情绪"""
        # 优先级：高 arousal 冲击性情绪优先
        priority = ["shock", "fear", "hurt", "overwhelm", "jealousy", "pleasure",
                    "love", "happiness", "frustration", "satisfaction", "anticipation"]
        max_val = 0
        max_e = "calm"
        for e in priority:
            v = self.blend.get(e, 0)
            if v > max_val:
                max_val = v
                max_e = e
        if max_val < 0.2:
            max_e = "calm"
        self.primary = max_e

    def get_dominant_emotions(self) -> str:
        """返回主导情绪+伴随情绪的简短描述"""
        self.update_primary()
        parts = [self.primary]
        # 找出其他超过0.3的情绪
        secondary = []
        for e, v in self.blend.items():
            if e != self.primary and v > 0.3:
                secondary.append(e)
        if secondary:
            parts.append("+".join(secondary[:2]))
        return "/".join(parts)

    def compute_conflict(self) -> float:
        """
        计算情绪冲突度：矛盾情绪共存时的张力
        冲突高→意识模糊、容易失控、高潮更强烈
        """
        # 羞耻/害怕 vs 快感/期待
        inhibit = max(self.blend.get("shame", 0), self.blend.get("fear", 0),
                      self.blend.get("anxiety", 0))
        excite = max(self.blend.get("pleasure", 0), self.blend.get("anticipation", 0),
                     self.blend.get("longing", 0))
        # 两个都高→冲突强
        conflict = min(inhibit, excite) * 2.0
        # 加上禁果效应
        conflict += self.forbidden_arousal_bonus * 0.5
        self.conflict_level = min(1.0, conflict)
        return self.conflict_level

    def shift(self, emotion: str, delta: float, dt: float = 1.0):
        """情绪渐变（不是突变）"""
        if emotion in self.blend:
            target = self.blend[emotion] + delta
            rate = 0.3 * dt
            self.blend[emotion] += (target - self.blend[emotion]) * rate
            self.blend[emotion] = max(0.0, min(1.0, self.blend[emotion]))
        self.compute_conflict()

    def trigger_interrupt(self, interrupt_type: str, current_time: float = 0.0):
        """情绪冲击中断 - 身体还在反应但意识被打断"""
        self.interrupt_triggered = True
        self.interrupt_type = interrupt_type
        self.last_interrupt_time = current_time
        self.suppressing_sounds = True

        if interrupt_type == "mention_other_girl":
            self.shift("hurt", 0.8)
            self.shift("jealousy", 0.9)
            self.shift("shock", 0.7)
            self.shift("pleasure", -0.5)  # 不直接清零，快感余韵还在
            self.cognitive_gate = 0.1
            self.attention_focus = ["partner_face"]
        elif interrupt_type == "parents_come_home":
            self.shift("fear", 0.9)
            self.shift("anxiety", 0.8)
            self.shift("shock", 0.6)
            self.suppressing_sounds = True
            self.suppressing_movements = True
            self.cognitive_gate = 0.15
            self.attention_focus = ["door", "sound"]
        elif interrupt_type == "door_knock":
            self.shift("fear", 0.6)
            self.shift("anxiety", 0.7)
            self.shift("shock", 0.5)
            self.suppressing_sounds = True
            self.suppressing_movements = True
            self.attention_focus = ["door"]
        elif interrupt_type == "phone_ring":
            self.shift("anxiety", 0.5)
            self.shift("shock", 0.4)
            self.suppressing_sounds = True
            self.attention_focus = ["phone"]
        elif interrupt_type == "heard_sound":
            self.shift("anxiety", 0.4)
            self.shift("fear", 0.3)
            self.suppressing_sounds = True
            self.attention_focus = ["sound_source"]
        self.update_primary()
        self.compute_conflict()

    def decay(self, dt: float):
        """情绪自然衰减"""
        for e in list(self.blend.keys()):
            # 高 arousal 情绪衰减快，信任/爱衰减很慢
            fast_decay = {"pleasure", "fear", "overwhelm", "frustration",
                         "anxiety", "pain", "anticipation", "jealousy", "hurt",
                         "shock"}
            slow_decay = {"shame", "embarrassment", "shyness", "playfulness",
                         "forbidden_arousal_bonus"}
            very_slow = {"trust", "love", "happiness", "satisfaction"}
            if e in fast_decay:
                self.blend[e] *= math.exp(-0.03 * dt)
            elif e in slow_decay:
                self.blend[e] *= math.exp(-0.008 * dt)
            elif e in very_slow:
                # Scenario cards may intentionally start these emotions at zero.
                # Decay must preserve that authored baseline instead of recreating
                # the generic relationship defaults on the first tick.
                self.blend[e] *= math.exp(-0.001 * dt)
            else:
                self.blend[e] *= math.exp(-0.015 * dt)
            self.blend[e] = max(0.0, min(1.0, self.blend[e]))
        # 禁果效应随隐私恢复而消失
        self.forbidden_arousal_bonus *= math.exp(-0.02 * dt)
        # 抑制状态随时间解除
        if self.interrupt_triggered:
            if dt > 30:
                self.suppressing_sounds = False
                self.suppressing_movements = False
                self.interrupt_triggered = False
        self.update_primary()
        self.compute_conflict()

    def compute_cognitive_gate(self, relationship_trust: float, privacy_level: float) -> float:
        """计算认知闸门开放度 0-1，同时返回禁果效应唤起加成"""
        gate = 1.0
        gate *= (0.3 + 0.7 * relationship_trust)
        gate *= privacy_level
        if self.blend["fear"] > 0.5:
            gate *= (1 - self.blend["fear"] * 0.7)
        if self.blend["hurt"] > 0.4:
            gate *= (1 - self.blend["hurt"] * 0.5)
        if self.blend["shame"] > 0.6:
            gate *= (1 - self.blend["shame"] * 0.2)
        if self.blend["trust"] > 0.8 and self.blend["pleasure"] > 0.5:
            gate *= 1.2
        gate = min(gate, 1.0)
        # === 禁果效应：不完全私密(privacy 0.3-0.7)时反而加兴奋 ===
        forbidden_bonus = 0.0
        if 0.25 < privacy_level < 0.75:
            forbidden_bonus = (1.0 - abs(privacy_level - 0.5) * 2.0) * 0.15
            # 信任越高禁果效应越明显（因为平时安全所以刺激）
            forbidden_bonus *= (0.5 + relationship_trust * 0.5)
        self.forbidden_arousal_bonus = forbidden_bonus
        # 期待/渴望时闸门保持开放
        if self.blend["anticipation"] > 0.7:
            gate = max(gate, 0.7)
        # 震惊时闸门暂时关闭
        if self.blend["shock"] > 0.5:
            gate *= (1 - self.blend["shock"] * 0.6)
        self.cognitive_gate = max(0.05, gate)
        self.compute_conflict()
        return self.cognitive_gate

    def get_cognitive_mod(self, part_name: str) -> float:
        """获取对指定部位的认知调制系数"""
        mod = self.cognitive_gate
        # 胸部/生殖区受羞耻抑制更多
        if part_name in ["nipple_left", "nipple_right", "vaginal_canal",
                         "clitoris", "inner_labia", "g_spot"]:
            if self.blend["shame"] > 0.5:
                mod *= (1 - self.blend["shame"] * 0.3)
        if self.blend["trust"] > 0.8 and self.blend["pleasure"] > 0.4:
            mod *= 1.1
        # 正在抑制声音/动作时，性反应被压制但不会消失
        if self.suppressing_movements:
            mod *= 0.7
        return max(0.05, mod)

    def get_forbidden_bonus(self) -> float:
        """获取禁果效应的唤起加成"""
        return self.forbidden_arousal_bonus
