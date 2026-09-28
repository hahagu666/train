"""
事件引擎
- 日程事件：固定时间触发（晚餐/睡觉/父母回家等）
- 随机事件：条件概率触发（摸太久妈妈敲门/手机响等）
- 条件事件：状态触发（连续高潮2次后困倦等）
- 完整中断系统：中断效果表
"""
import random
import math
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field


@dataclass
class GameEvent:
    """游戏事件"""
    name: str
    event_type: str  # scheduled/random/conditional/interrupt
    trigger_fn: Callable  # (world, state) -> bool 是否触发
    effect_fn: Callable   # (world, state) -> None 应用效果
    description: str = ""
    fired: bool = False
    one_shot: bool = True


# === 中断效果定义 ===
INTERRUPT_EFFECTS = {
    "mention_other_girl": {
        "emotion_shift": {"hurt": 0.8, "jealousy": 0.9, "shock": 0.7, "pleasure": -0.4},
        "arousal_mult": 0.2,
        "suppress_sounds": False,
        "suppress_movements": False,
        "attention_focus": ["partner_face"],
        "muscle_tension": 0.8,
        "description": "她脸上的表情瞬间僵住了，快感像退潮一样消失，取而代之的是一种被刺痛的感觉",
        "body_response": "她的身体僵硬着，原本环在你脖子上的手慢慢垂了下来",
        "reaction_window": 15,  # 秒，需要安抚
        "event_desc": "对方刚才提到了其他女生",
    },
    "parents_come_home": {
        "emotion_shift": {"fear": 0.95, "anxiety": 0.9, "shock": 0.8},
        "arousal_mult": 0.4,
        "suppress_sounds": True,
        "suppress_movements": True,
        "attention_focus": ["door", "sound"],
        "muscle_tension": 1.0,
        "freeze_duration": 10,
        "description": "玄关传来钥匙转动的声音——爸妈回来了！她整个人瞬间僵住，眼睛瞪得大大的",
        "body_response": "她的手猛地捂住自己的嘴，身子绷紧不敢动一下，连呼吸都屏住了",
        "reaction_window": 20,
        "event_desc": "玄关传来钥匙转动的声音--爸妈回来了",
        "detection_risk": 0.3,  # 被发现的基础概率
    },
    "door_knock": {
        "emotion_shift": {"fear": 0.7, "anxiety": 0.8, "shock": 0.6},
        "arousal_mult": 0.5,
        "suppress_sounds": True,
        "suppress_movements": True,
        "attention_focus": ["door"],
        "muscle_tension": 0.9,
        "freeze_duration": 5,
        "description": "突然传来敲门声，她吓得整个人一缩",
        "body_response": "她咬着嘴唇死死盯着门，身体不受控制地发抖",
        "reaction_window": 15,
        "event_desc": "突然传来敲门声",
        "detection_risk": 0.15,
    },
    "phone_ring": {
        "emotion_shift": {"anxiety": 0.6, "shock": 0.5},
        "arousal_mult": 0.7,
        "suppress_sounds": True,
        "suppress_movements": False,
        "attention_focus": ["phone"],
        "muscle_tension": 0.5,
        "description": "手机突然响了，她吓了一跳",
        "body_response": "她慌忙摸索着想按掉铃声，脸涨得通红",
        "reaction_window": 10,
        "event_desc": "手机突然响了",
    },
    "heard_sound": {
        "emotion_shift": {"anxiety": 0.5, "fear": 0.3},
        "arousal_mult": 0.8,
        "suppress_sounds": True,
        "suppress_movements": True,
        "attention_focus": ["sound_source"],
        "muscle_tension": 0.4,
        "description": "她突然停下动作，侧耳听着外面的动静",
        "body_response": "「……你听到了吗？」她小声问，身体绷紧着",
        "reaction_window": 10,
        "event_desc": "外面传来一阵响动",
    },
    "someone_passing_by": {
        "emotion_shift": {"fear": 0.6, "anxiety": 0.7, "shock": 0.5},
        "arousal_mult": 0.6,
        "suppress_sounds": True,
        "suppress_movements": True,
        "attention_focus": ["window", "door"],
        "description": "窗外传来脚步声，有人路过！她立刻捂住你的嘴",
        "body_response": "她把你按在怀里不让你动，心跳得像要撞出来",
        "reaction_window": 10,
        "event_desc": "窗外传来脚步声，有人路过",
        "detection_risk": 0.1,
    },
    "pet_interrupt": {
        "emotion_shift": {"embarrassment": 0.8, "shock": 0.3},
        "arousal_mult": 0.8,
        "suppress_sounds": True,
        "suppress_movements": True,
        "attention_focus": ["pet"],
        "description": "猫跳上床蹭她的脚，她吓得差点叫出来",
        "body_response": "她红着脸把猫赶下去，又羞又恼地瞪了你一眼",
        "reaction_window": 10,
        "event_desc": "猫跳上了床",
    },
}

# 亲密阈值：普通聊天（低唤起）不触发中断事件。整套中断机制只服务于亲密情境，
# 否则"爸妈回来了"这类事件会在日常对话里凭空出现并反复发作。
INTERRUPT_INTIMACY_THRESHOLD = 0.25


class EventEngine:
    """事件引擎"""
    def __init__(self):
        self.events: List[GameEvent] = []
        self.active_interrupt: Optional[str] = None
        self.interrupt_timer: float = 0.0
        self.detection_risk: float = 0.0
        self._register_default_events()

    def _register_default_events(self):
        """注册默认事件"""
        # === 日程事件 ===
        self.events.append(GameEvent(
            name="dinner_time",
            event_type="scheduled",
            trigger_fn=lambda w, s: w.game_time.hour == 18 and w.game_time.minute < 5,
            effect_fn=lambda w, s: self._apply_interrupt("parents_come_home", w, s)
            if w.location_type in ["bedroom", "living_room"] else None,
            description="晚饭时间到了",
            one_shot=True
        ))
        self.events.append(GameEvent(
            name="parents_return",
            event_type="scheduled",
            trigger_fn=lambda w, s: (w.game_time.hour >= 22 or w.game_time.hour <= 5) == False
                and w.has_people and w.privacy_level < 0.8
                and random.random() < 0.001 * w.event_tick_counter,
            effect_fn=lambda w, s: self._apply_interrupt("parents_come_home", w, s),
            description="父母随时可能回来",
            one_shot=True
        ))
        self.events.append(GameEvent(
            name="bedtime",
            event_type="scheduled",
            trigger_fn=lambda w, s: w.game_time.hour == 23 and w.game_time.minute < 5
                and w.location_type == "bedroom" and s.global_arousal < 0.3,
            effect_fn=lambda w, s: s.emotion.shift("sleepy", 0.6),
            description="该睡觉了",
            one_shot=False
        ))

        # === 随机事件 ===
        self.events.append(GameEvent(
            name="random_door_knock",
            event_type="random",
            trigger_fn=lambda w, s: (w.privacy_level < 0.8 and s.global_arousal > 0.5
                and random.random() < 0.0005 * (1 + s.emotion.blend.get("anxiety", 0))),
            effect_fn=lambda w, s: self._apply_interrupt("door_knock", w, s),
            description="突然有人敲门"
        ))
        self.events.append(GameEvent(
            name="random_phone_ring",
            event_type="random",
            trigger_fn=lambda w, s: (s.global_arousal > 0.6 and random.random() < 0.0003),
            effect_fn=lambda w, s: self._apply_interrupt("phone_ring", w, s),
            description="手机突然响了"
        ))
        self.events.append(GameEvent(
            name="heard_sound_outside",
            event_type="random",
            trigger_fn=lambda w, s: (w.privacy_level < 0.7 and s.global_arousal > 0.4
                and random.random() < 0.0008),
            effect_fn=lambda w, s: self._apply_interrupt("heard_sound", w, s),
            description="听到外面有声音"
        ))
        self.events.append(GameEvent(
            name="someone_outside_window",
            event_type="random",
            trigger_fn=lambda w, s: (w.location_type in ["bedroom", "bathroom"]
                and s.global_arousal > 0.7 and s.emotion.suppressing_sounds
                and random.random() < 0.0002),
            effect_fn=lambda w, s: self._apply_interrupt("someone_passing_by", w, s),
            description="有人经过"
        ))

        # === 条件事件 ===
        self.events.append(GameEvent(
            name="post_orgasm_sleepy",
            event_type="conditional",
            trigger_fn=lambda w, s: (s.orgasm.orgasms_so_far >= 2 and s.orgasm.phase == "resolution"
                and s.orgasm.time_since_last > 60 and random.random() < 0.01),
            effect_fn=lambda w, s: s.emotion.shift("sleepy", 0.3),
            description="多次高潮后困倦"
        ))
        self.events.append(GameEvent(
            name="prolonged_stimulation_discovery_risk",
            event_type="conditional",
            trigger_fn=lambda w, s: (s.global_arousal > 0.7 and w.privacy_level < 0.5
                and s.ans.breathing_rate > 25 and not s.emotion.suppressing_sounds
                and random.random() < 0.001),
            effect_fn=lambda w, s: self._apply_interrupt("heard_sound", w, s),
            description="太大声可能被听到"
        ))
        self.events.append(GameEvent(
            name="tension_release_cry",
            event_type="conditional",
            trigger_fn=lambda w, s: (s.orgasm.phase == "orgasm" and s.orgasm.orgasm_type in ["blended", "multiple_chain"]
                and s.relationship_trust > 0.8 and not s.emotion.suppressing_sounds
                and random.random() < 0.3),
            effect_fn=lambda w, s: s.emotion.blend.__setitem__("overwhelm",
                min(1.0, s.emotion.blend.get("overwhelm",0)+0.3)),
            description="高潮时忍不住哭出来"
        ))

    def _apply_interrupt(self, interrupt_type: str, world, state):
        """应用中断效果"""
        if interrupt_type not in INTERRUPT_EFFECTS:
            return
        # 普通聊天（低唤起）不触发中断：中断机制只服务于亲密情境，
        # 日常对话里"爸妈回来了"这类事件毫无逻辑且会劫持后续所有回合
        if getattr(state, "global_arousal", 0.0) < INTERRUPT_INTIMACY_THRESHOLD:
            return
        # 爸妈已经在家就不可能再"回来"，避免与场景描述自相矛盾
        if interrupt_type == "parents_come_home" and getattr(world, "parents_present", False):
            return
        effect = INTERRUPT_EFFECTS[interrupt_type]
        self.active_interrupt = interrupt_type
        self.interrupt_timer = effect["reaction_window"]
        # 情绪
        for emo, delta in effect.get("emotion_shift", {}).items():
            state.emotion.shift(emo, delta, dt=1.0)
        # 唤起降低
        state.global_arousal *= effect.get("arousal_mult", 1.0)
        state.ans.arousal_global *= effect.get("arousal_mult", 1.0)
        # 抑制
        if effect.get("suppress_sounds"):
            state.emotion.suppressing_sounds = True
        if effect.get("suppress_movements"):
            state.emotion.suppressing_movements = True
        # 注意力
        state.emotion.attention_focus = effect.get("attention_focus", ["partner"])
        # 被发现风险
        self.detection_risk = effect.get("detection_risk", 0.0)
        # 触发emotion的interrupt
        state.emotion.trigger_interrupt(interrupt_type, state.sim_time)
        world.last_interrupt = interrupt_type
        # 场景描述只保留事件本身，不携带第三人称反应（避免污染模型人称视角）
        world.last_interrupt_desc = effect.get("event_desc", effect["description"])

    def trigger_interrupt_manually(self, interrupt_type: str, world, state):
        """外部触发中断（如parser检测到用户输入触发）"""
        self._apply_interrupt(interrupt_type, world, state)

    def tick(self, dt: float, world, state):
        """每时间步检查事件"""
        world.event_tick_counter += dt
        # 中断计时
        if self.active_interrupt:
            self.interrupt_timer -= dt
            # 中断期间被发现的概率累积
            if not state.emotion.suppressing_sounds and self.detection_risk > 0:
                if random.random() < self.detection_risk * dt * 0.01:
                    # 被发现了！
                    world.detected = True
            if self.interrupt_timer <= 0:
                # 中断结束
                self._end_interrupt(state)
        # 检查事件
        for event in self.events:
            if event.fired and event.one_shot:
                continue
            try:
                if event.trigger_fn(world, state):
                    event.effect_fn(world, state)
                    event.fired = True
            except:
                pass

    def _end_interrupt(self, state):
        """中断结束"""
        self.active_interrupt = None
        self.detection_risk = 0.0
        state.emotion.suppressing_movements = False
        # 声音抑制慢慢解除
        state.emotion.attention_focus = ["partner"]

    def is_interrupted(self) -> bool:
        return self.active_interrupt is not None

    def get_interrupt_description(self) -> str:
        if not self.active_interrupt:
            return ""
        return INTERRUPT_EFFECTS.get(self.active_interrupt, {}).get("description", "")

    def to_dict(self) -> dict:
        return {
            "active_interrupt": self.active_interrupt,
            "interrupt_timer": self.interrupt_timer,
            "detection_risk": self.detection_risk,
            "events": {
                event.name: event.fired for event in self.events
            },
        }

    def load_from_dict(self, data: dict, world=None):
        """恢复事件运行时状态，并清理旧存档中的非法/过期中断。"""
        active = data.get("active_interrupt")
        timer = float(data.get("interrupt_timer", 0.0) or 0.0)
        # 旧版本可能把中断保存为负数、未知类型，或保存了超过当前事件
        # 反应窗口的旧计时器；这些状态不应在恢复后继续劫持后续回合。
        effect = INTERRUPT_EFFECTS.get(active) if active else None
        if not effect or timer <= 0:
            active = None
            timer = 0.0
        else:
            timer = min(timer, float(effect.get("reaction_window", timer)))
            if active == "parents_come_home" and world is not None \
                    and getattr(world, "parents_present", False):
                active = None
                timer = 0.0

        self.active_interrupt = active
        self.interrupt_timer = timer
        self.detection_risk = float(data.get("detection_risk", 0.0) or 0.0) if active else 0.0
        fired = data.get("events", {})
        for event in self.events:
            if event.name in fired:
                event.fired = bool(fired[event.name])
