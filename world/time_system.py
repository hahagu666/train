"""
世界时间系统
- 每个动作有base时长，根据当前状态修正
- 支持时间跳跃推断（"第二天"等）
- 时间跳跃后的状态衰减
"""
import re
import math
from typing import Dict, Tuple, Optional, List


# === 动作基础时长（秒）===
ACTION_BASE_DURATION: Dict[str, float] = {
    # 亲吻/抚摸类
    "kiss": 8.0, "deep_kiss": 12.0, "hug": 6.0, "hold": 5.0,
    "touch_cheek": 3.0, "caress_hair": 4.0, "neck_kiss": 6.0,
    "ear_kiss": 4.0, "lick_neck": 5.0, "suck_neck": 8.0,
    # 胸部
    "touch_breast": 6.0, "fondle_breast": 10.0, "suck_nipple": 12.0,
    "pinch_nipple": 3.0, "twist_nipple": 4.0,
    # 下体外部
    "touch_over_clothes": 6.0, "rub_clit_through_panties": 8.0,
    "rub_clit": 10.0, "finger_clit": 12.0, "lick_clit": 15.0,
    "suck_clit": 12.0, "cunnilingus": 20.0,
    # 插入
    "finger_entrance": 5.0, "finger_insert_one": 8.0,
    "finger_insert_two": 10.0, "finger_gspot": 15.0,
    "rub_penis_against": 8.0, "guide_penis": 6.0,
    "penetrate": 5.0, "thrust": 12.0, "slow_thrust": 15.0,
    "deep_thrust": 10.0, "quick_thrust": 8.0,
    # 姿势相关
    "missionary": 30.0, "cowgirl": 30.0, "doggy": 30.0,
    "spoon": 25.0, "legs_on_shoulders": 20.0,
    # 手交/口交
    "handjob": 20.0, "blowjob": 25.0, "deepthroat": 15.0,
    "sixty_nine": 25.0,
    # 脱衣服
    "remove_outerwear": 15.0, "remove_shirt": 12.0, "unhook_bra": 10.0,
    "remove_bra": 8.0, "remove_panties": 10.0, "pull_up_skirt": 5.0,
    "unzip_pants": 8.0, "remove_pants": 12.0,
    # 情感/对话
    "compliment": 3.0, "tease": 5.0, "whisper": 3.0,
    "say_name": 2.0, "confess_love": 8.0, "dirty_talk": 6.0,
    "moan": 2.0, "look_in_eyes": 4.0, "bite_lip": 2.0,
    # 拥抱/事后
    "cuddle": 30.0, "pillow_talk": 60.0, "fall_asleep_together": 600.0,
    "rest": 30.0, "aftercare": 120.0,
    # 默认
    "wait": 3.0, "look": 2.0, "talk": 8.0, "default": 5.0,
}

# === 时间跳跃关键词推断 ===
TIME_JUMP_PATTERNS = [
    (r"(?:第二天|次日|隔天|早上醒来|醒来)", 8 * 3600),
    (r"(?:过了一?小时|一小时后|一小时过去)", 3600),
    (r"(?:过了半?小时|半小时后)", 1800),
    (r"(?:过了几分钟|几分钟后|一会儿后|等一下)", 300),
    (r"(?:过了一会|过了会儿|稍后|不久)", 120),
    (r"(?:第二天早上|早晨|第二天一早)", 10 * 3600),
    (r"(?:晚上|夜幕降临|天黑了)", 6 * 3600),
    (r"(?:下午|午后)", 4 * 3600),
    (r"(?:放学|下课|下班后)", 5 * 3600),
    (r"(?:洗澡|洗个澡|去洗澡)", 900),
    (r"(?:吃饭|吃晚饭|吃午饭|吃早饭)", 1800),
]


def estimate_action_duration(action_type: str, state, world) -> float:
    """
    根据当前状态计算动作实际耗时
    返回：秒数
    """
    base = ACTION_BASE_DURATION.get(action_type, ACTION_BASE_DURATION["default"])
    mod = 1.0
    emotion = state.emotion
    arousal = state.global_arousal
    trust = state.relationship_trust

    # 兴奋/着急时动作变快
    if arousal > 0.7:
        mod *= 0.7
    elif arousal > 0.5:
        mod *= 0.85

    # 害羞/犹豫时动作变慢、停顿
    if emotion.blend.get("shyness", 0) > 0.6:
        mod *= 1.3
    if emotion.blend.get("fear", 0) > 0.4:
        mod *= 1.5
    if emotion.blend.get("anticipation", 0) > 0.7:
        mod *= 1.2

    # 新手动作笨拙慢
    from core.skills import calc_effective_mastery
    if action_type in state.skills:
        mast = calc_effective_mastery(state.skills[action_type])
        mod *= (1.3 - mast * 0.5)  # 新手慢30%，熟练快20%

    # 信任高放松慢享受
    if trust > 0.8 and arousal < 0.6:
        mod *= 1.2

    return base * mod


def resolve_target_time(text: str, game_time) -> Optional[float]:
    """Resolve a future natural-language target to a positive second delta."""
    period_hours = {
        "早上": 8, "早晨": 8, "上午": 10, "中午": 12,
        "下午": 15, "傍晚": 18, "晚上": 20, "夜里": 22,
    }
    day_offset = None
    if re.search(r"第二天|次日|隔天|明天|明日", text):
        day_offset = 1
    elif re.search(r"今天|今日", text):
        day_offset = 0

    target_hour = None
    for token, hour in period_hours.items():
        if token in text:
            target_hour = hour
            break

    clock = re.search(r"(\d{1,2})\s*[点时](?:\s*(\d{1,2})\s*分?)?", text)
    if clock:
        target_hour = int(clock.group(1))
        minute = int(clock.group(2) or 0)
        if any(token in text for token in ("下午", "傍晚", "晚上")) and target_hour < 12:
            target_hour += 12
    else:
        minute = 0

    if day_offset is None and target_hour is None:
        return None
    if target_hour is None:
        target_hour = game_time.hour
        minute = game_time.minute

    target_day = game_time.day + (day_offset or 0)
    current_absolute = (game_time.day - 1) * 86400 + game_time.hour * 3600 + game_time.minute * 60
    target_absolute = (target_day - 1) * 86400 + target_hour * 3600 + minute * 60
    if target_absolute <= current_absolute:
        if day_offset is None:
            target_absolute += 86400
        else:
            return None
    return float(target_absolute - current_absolute)


def detect_time_jump(text: str, game_time=None) -> Optional[float]:
    """
    从用户输入中检测时间跳跃
    返回：跳跃秒数，None表示无跳跃
    """
    if game_time is not None:
        target_delta = resolve_target_time(text, game_time)
        if target_delta is not None:
            return target_delta
    for pattern, seconds in TIME_JUMP_PATTERNS:
        if re.search(pattern, text):
            return seconds
    # 数字+时间单位
    m = re.search(r"(\d+)\s*(分钟|分|小时|秒|天)", text)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        mult = {"秒": 1, "分钟": 60, "分": 60, "小时": 3600, "天": 86400}[unit]
        return num * mult
    return None


def apply_time_jump(state, world, jump_seconds: float):
    """
    应用时间跳跃后的状态衰减
    长时间跳跃→身体状态重置，情绪保留长期部分，记忆巩固
    """
    dt = max(0.0, jump_seconds)
    if dt <= 0:
        return
    hours = dt / 3600.0
    from core.physical_engine import recover_stamina

    # 短时间跳跃（几分钟）只衰减生理状态
    if dt < 300:
        remaining = dt
        while remaining > 0:
            step = min(10, remaining)
            state.tick(step, world)
            remaining -= step
        world.advance_time(dt)
        return

    # 中等跳跃（几分钟到几小时）
    if dt < 7200:  # 2小时内
        # 快速衰减生理状态
        decay_factor = math.exp(-hours * 2.0)
        state.global_arousal *= decay_factor * 0.3
        state.ans.arousal_global *= decay_factor * 0.2
        state.ans.heart_rate = 75 + (state.ans.heart_rate - 75) * decay_factor
        state.ans.breathing_rate = 14 + (state.ans.breathing_rate - 14) * decay_factor
        state.ans.skin_flush *= decay_factor
        state.ans.tremor *= decay_factor
        state.ans.muscle_tone = 0.2 + (state.ans.muscle_tone - 0.2) * decay_factor
        # 情绪中高arousal的部分衰减
        for e in ["pleasure", "fear", "overwhelm", "frustration",
                  "anxiety", "pain", "shock"]:
            state.emotion.blend[e] *= math.exp(-hours * 1.5)
        # 润滑/充血快速消退
        gf = state.body["genital_female"]
        for p in ["vaginal_canal", "vaginal_vestibule"]:
            sp = gf.get(p)
            if "wetness" in sp.extra:
                sp.extra["wetness"] *= math.exp(-hours * 3)
            sp.extra["tent_lub"] = 0
        for p in ["nipple_left", "nipple_right", "clitoris"]:
            for region in state.body.values():
                from core.body.parts import BodyRegion
                if isinstance(region, BodyRegion) and p in region.sub_parts:
                    sp = region.get(p)
                    if "congestion" in sp.extra:
                        sp.extra["congestion"] *= math.exp(-hours * 2)
                    if "erection" in sp.extra:
                        sp.extra["erection"] *= math.exp(-hours * 2)
        # 高潮系统重置
        if state.orgasm.phase != "excitement":
            state.orgasm.phase = "resolution"
            state.orgasm.time_since_last = hours * 3600
        recover_stamina(state.stamina, dt, sleeping=False)
        state.sim_time += dt
        world.advance_time(dt)
        return

    # 长时间跳跃（几小时以上/过夜）→ 身体完全重置
    # 身体部位重置
    for region in state.body.values():
        from core.body.parts import BodyRegion
        if not isinstance(region, BodyRegion):
            continue
        for sp in region.all_parts():
            sp.arousal = 0.0
            for k in list(sp.extra.keys()):
                if k in ["wetness", "congestion", "erection", "tremor", "swelling",
                         "contraction", "tension", "tingling", "tent_lub", "ejac_vol",
                         "oversensitive_post"]:
                    if k == "oversensitive_post" and hours < 4:
                        sp.extra[k] = False
                    else:
                        sp.extra[k] = 0.0 if isinstance(sp.extra[k], (int, float)) else False
                elif k in ["hymen_intact", "first_time", "first_touch"]:
                    pass  # 保留第一次状态
                elif isinstance(sp.extra[k], (int, float)) and sp.extra[k] < 1.0:
                    sp.extra[k] *= math.exp(-hours * 0.5)
    # ANS重置到静息
    state.ans.__init__()
    # 高arousal情绪完全消退，留下长期情绪
    for e in list(state.emotion.blend.keys()):
        if e in ["pleasure", "fear", "overwhelm", "frustration",
                 "anxiety", "pain", "shock", "loss_of_control", "inevitability"]:
            state.emotion.blend[e] = 0.0
        elif e in ["shame", "embarrassment", "shyness", "satisfaction"]:
            state.emotion.blend[e] *= math.exp(-hours * 0.3)
        elif e in ["hurt", "jealousy"]:
            state.emotion.blend[e] *= math.exp(-hours * 0.1)  # 受伤/吃醋消退慢
    state.emotion.suppressing_sounds = False
    state.emotion.suppressing_movements = False
    state.emotion.interrupt_triggered = False
    state.emotion.attention_focus = ["partner"]
    state.emotion.compute_cognitive_gate(state.relationship_trust, world.privacy_level)
    # 高潮完全重置
    state.orgasm.__init__()
    # 润滑重置
    gf = state.body["genital_female"]
    gf.get("clitoris").set("oversensitive_post", False)
    gf.get("clitoris").set("pain_overstim", 0.0)
    # 全局状态
    state.global_arousal = 0.0
    recover_stamina(state.stamina, dt, sleeping=dt >= 6 * 3600)
    state.sim_time += dt
    # 记忆巩固
    if hasattr(state, 'memory') and state.memory:
        state.memory.maintenance()
    # 时间更新后更新世界时间
    world.advance_time(dt)


class GameTime:
    """游戏内时间"""
    def __init__(self, start_hour: int = 20, start_day: int = 1):
        self.day: int = start_day
        self.hour: int = start_hour
        self.minute: int = 0
        self.sim_seconds: float = 0.0

    def advance(self, seconds: float):
        previous_seconds = self.sim_seconds
        self.sim_seconds += seconds
        previous_minute = int(previous_seconds // 60)
        current_minute = int(self.sim_seconds // 60)
        elapsed_minutes = current_minute - previous_minute
        if elapsed_minutes <= 0:
            return

        total_minutes = self.day * 24 * 60 + self.hour * 60 + self.minute + elapsed_minutes
        self.day, day_minutes = divmod(total_minutes, 24 * 60)
        self.hour, self.minute = divmod(day_minutes, 60)

    def set_time(self, hour: int, minute: int = 0, day: int = None):
        """直接设置时间（用于开场初始化）"""
        self.hour = max(0, min(23, hour))
        self.minute = max(0, min(59, minute))
        if day is not None:
            self.day = day
        self.sim_seconds = (self.day - 1) * 86400 + self.hour * 3600 + self.minute * 60

    def get_time_of_day(self) -> str:
        if 5 <= self.hour < 9:
            return "early_morning"
        if 9 <= self.hour < 12:
            return "morning"
        if 12 <= self.hour < 14:
            return "noon"
        if 14 <= self.hour < 17:
            return "afternoon"
        if 17 <= self.hour < 19:
            return "evening"
        if 19 <= self.hour < 22:
            return "night"
        return "late_night"

    def get_time_str(self) -> str:
        period = "早上" if self.hour < 12 else ("下午" if self.hour < 18 else "晚上")
        h = self.hour if self.hour <= 12 else self.hour - 12
        if h == 0:
            h = 12
        return f"第{self.day}天 {period}{h}点{self.minute:02d}分"

    def to_string(self) -> str:
        return self.get_time_str()

    def to_dict(self) -> dict:
        return {
            "day": self.day, "hour": self.hour, "minute": self.minute,
            "sim_seconds": self.sim_seconds
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GameTime':
        gt = cls(start_hour=d.get("hour", 23))
        gt.day = d.get("day", 1)
        gt.hour = d.get("hour", 23)
        gt.minute = d.get("minute", 0)
        gt.sim_seconds = d.get("sim_seconds", 0)
        return gt
