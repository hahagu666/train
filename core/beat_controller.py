"""
Beat Controller - 节奏控制器
将一个动作分解为多个时间片（beat），每个beat内：
- 控制刺激的节奏变化（渐强/渐弱/稳定/快慢交替）
- 控制插入动作的深度、速度
- 决定是否需要中断（声音/事件等）
- 每个beat推进state.tick，收集身体反应
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple


@dataclass
class Beat:
    """单个时间片"""
    t_start: float  # 开始时间（动作内相对时间，秒）
    t_end: float
    intensity: float  # 该beat的平均刺激强度
    speed: float = 1.0  # 速度倍率
    depth: float = 1.0  # 深度（插入类）
    rhythm: str = "steady"  # steady/accelerating/decelerating/bursting/paused
    tension_change: float = 0.0  # 性紧张增量
    note: str = ""  # 该beat的特殊标记


@dataclass
class BeatPattern:
    """一个动作的beat序列"""
    beats: List[Beat] = field(default_factory=list)
    total_duration: float = 0.0
    action_type: str = ""
    peak_time: float = 0.0  # 快感峰值在第几秒
    build_type: str = "gradual"  # gradual/sudden/wave

    def total_beats(self) -> int:
        return len(self.beats)


# === 预定义节奏模式 ===

def _make_thrust_pattern(duration: float, base_intensity: float,
                          variant: str = "normal") -> BeatPattern:
    """抽插节奏"""
    beats = []
    dt = 0.8  # 每个beat约0.8秒
    t = 0.0
    if variant == "slow_build":
        # 慢慢加速变深
        while t < duration:
            progress = t / duration
            inten = base_intensity * (0.3 + progress * 0.7)
            speed = 0.6 + progress * 0.8
            depth = 0.4 + progress * 0.6
            rhythm = "accelerating" if progress > 0.6 else "steady"
            beats.append(Beat(t, min(t + dt, duration), inten, speed, depth, rhythm))
            t += dt
    elif variant == "fast":
        dt = 0.5
        while t < duration:
            beats.append(Beat(t, min(t + dt, duration),
                             base_intensity * 1.2, 1.5, 0.9 + random.random()*0.1,
                             "bursting"))
            t += dt
    elif variant == "slow_deep":
        dt = 1.2
        while t < duration:
            beats.append(Beat(t, min(t + dt, duration),
                             base_intensity * 0.9, 0.5, 1.0,
                             "steady"))
            t += dt
    elif variant == "tease":
        # 快慢交替，浅浅深深
        deep = False
        while t < duration:
            beat_dur = 1.5 if deep else 0.6
            inten = base_intensity * (1.0 if deep else 0.4)
            depth = 1.0 if deep else 0.3
            speed = 0.4 if deep else 1.3
            rhythm = "paused" if t > duration*0.7 and random.random() < 0.3 else "steady"
            beats.append(Beat(t, min(t + beat_dur, duration),
                             inten, speed, depth, rhythm,
                             note="deep_thrust" if deep else "shallow"))
            deep = not deep
            t += beat_dur
    else:  # normal
        while t < duration:
            progress = t / duration
            # 中段最舒服
            inten = base_intensity * (0.6 + 0.4 * math.sin(progress * math.pi))
            speed = 0.8 + 0.4 * math.sin(progress * math.pi * 0.8)
            depth = 0.6 + 0.3 * math.sin(progress * math.pi)
            beats.append(Beat(t, min(t + dt, duration), inten, speed, depth, "steady"))
            t += dt
    bp = BeatPattern(beats=beats, total_duration=duration, action_type="thrust",
                     peak_time=duration * 0.7, build_type="wave")
    return bp


def _make_rub_pattern(duration: float, base_intensity: float,
                       variant: str = "circular") -> BeatPattern:
    """抚摸/揉/摩擦节奏"""
    beats = []
    dt = 1.0
    t = 0.0
    if variant == "circular":
        while t < duration:
            progress = t / duration
            inten = base_intensity * (0.5 + 0.5 * (1 - abs(progress - 0.5)*2))
            beats.append(Beat(t, min(t + dt, duration), inten,
                             rhythm="steady", note="circular"))
            t += dt
    elif variant == "building":
        while t < duration:
            progress = t / duration
            inten = base_intensity * (0.4 + progress * 0.8)
            speed = 0.7 + progress * 0.8
            beats.append(Beat(t, min(t + dt, duration), inten, speed,
                             rhythm="accelerating" if progress > 0.5 else "steady"))
            t += dt
    elif variant == "tease":
        # 快碰到又离开
        on = True
        while t < duration:
            beat_dur = 1.2 if on else 0.8
            inten = base_intensity * (1.0 if on else 0.1)
            rhythm = "paused" if not on else "steady"
            beats.append(Beat(t, min(t + beat_dur, duration), inten,
                             1.0 if on else 0.0, rhythm=rhythm,
                             note="contact" if on else "hover"))
            on = not on
            t += beat_dur
    else:
        while t < duration:
            beats.append(Beat(t, min(t + dt, duration), base_intensity, rhythm="steady"))
            t += dt
    bp = BeatPattern(beats=beats, total_duration=duration, action_type="rub",
                     peak_time=duration * 0.6)
    return bp


def _make_kiss_pattern(duration: float, base_intensity: float,
                       is_deep: bool = False) -> BeatPattern:
    """亲吻节奏"""
    beats = []
    dt = 1.5
    t = 0.0
    while t < duration:
        progress = t / duration
        inten = base_intensity * (0.7 + 0.3 * math.sin(progress * math.pi * 2))
        note = "deepen" if is_deep and progress > 0.3 and random.random() < 0.4 else "soft"
        beats.append(Beat(t, min(t + dt, duration), inten, rhythm="steady", note=note))
        t += dt
    bp = BeatPattern(beats=beats, total_duration=duration,
                     action_type="deep_kiss" if is_deep else "kiss")
    return bp


def _make_orgasm_beat(duration: float = 8.0) -> BeatPattern:
    """高潮时的beat（强烈的节律收缩）"""
    beats = []
    dt = 0.6
    t = 0.0
    while t < duration:
        progress = t / duration
        # 前半段最强烈，后半逐渐减弱
        if progress < 0.4:
            inten = 1.0
            rhythm = "bursting"
        elif progress < 0.7:
            inten = 0.7 - (progress - 0.4)
            rhythm = "bursting"
        else:
            inten = 0.3 * (1 - (progress - 0.7)/0.3)
            rhythm = "decelerating"
        beats.append(Beat(t, min(t + dt, duration), inten, 1.0, 1.0, rhythm,
                         tension_change=-0.1, note="contraction"))
        t += dt
    bp = BeatPattern(beats=beats, total_duration=duration, action_type="orgasm",
                     peak_time=2.0, build_type="sudden")
    return bp


def select_pattern(action_type: str, duration: float, intensity: float,
                   state, world=None) -> BeatPattern:
    """根据当前状态选择最合适的节奏模式"""
    arousal = state.global_arousal
    orgasm_phase = state.orgasm.phase
    trust = state.relationship_trust

    # 高潮中用高潮节奏
    if orgasm_phase == "orgasm":
        return _make_orgasm_beat(6.0 + state.orgasm.intensity * 6.0)

    # 根据动作类型选模式
    variant = "normal"
    if action_type in ["thrust", "slow_thrust", "deep_thrust", "quick_thrust",
                        "missionary", "cowgirl", "doggy", "spoon"]:
        if arousal > 0.8:
            variant = random.choice(["fast", "fast", "normal"])
        elif arousal > 0.5:
            variant = random.choice(["normal", "slow_build", "tease"])
        else:
            variant = random.choice(["slow_build", "slow_deep", "tease"])
        if action_type == "slow_thrust":
            variant = "slow_deep"
        if action_type == "quick_thrust":
            variant = "fast"
        if action_type == "deep_thrust":
            variant = "slow_deep"
        return _make_thrust_pattern(duration, intensity, variant)

    if action_type in ["rub_clit", "finger_clit", "rub_clit_through_panties",
                        "fondle_breast", "touch_breast"]:
        if arousal > 0.7:
            variant = random.choice(["building", "circular", "tease"])
        elif arousal > 0.4:
            variant = random.choice(["building", "circular"])
        else:
            variant = random.choice(["tease", "circular"])
        return _make_rub_pattern(duration, intensity, variant)

    if action_type in ["cunnilingus", "lick_clit", "suck_clit"]:
        if arousal > 0.6:
            variant = random.choice(["building", "circular"])
        else:
            variant = random.choice(["tease", "circular"])
        return _make_rub_pattern(duration, intensity * 1.1, variant)

    if action_type in ["suck_nipple", "pinch_nipple", "twist_nipple"]:
        return _make_rub_pattern(duration, intensity * 0.8, "circular")

    if action_type in ["kiss"]:
        return _make_kiss_pattern(duration, intensity, is_deep=False)
    if action_type in ["deep_kiss"]:
        return _make_kiss_pattern(duration, intensity, is_deep=True)

    if action_type in ["neck_kiss", "ear_kiss", "suck_neck", "lick_neck",
                        "caress_hair", "touch_cheek", "hold", "cuddle",
                        "hug", "aftercare"]:
        return _make_rub_pattern(duration, intensity * 0.6, "steady")

    # 默认：稳定节奏
    return _make_rub_pattern(duration, intensity, "steady")


def apply_beat_stimulation(state, beat: Beat, target_parts: List[str],
                           action_type: str, through_clothes: bool = None,
                           world=None):
    """应用单个beat的刺激到state"""
    # 该beat的强度受速度和深度影响
    eff_intensity = beat.intensity
    if action_type in ["thrust", "deep_thrust", "quick_thrust", "missionary",
                        "cowgirl", "doggy"]:
        # 抽插类：深度和速度都影响
        eff_intensity *= (0.4 + beat.depth * 0.4 + beat.speed * 0.2)
    else:
        eff_intensity *= (0.5 + beat.speed * 0.5)

    # 暂停beat不施加刺激
    if beat.rhythm == "paused" and "contact" not in beat.note:
        # 暂停但保持接触，仍有轻微刺激
        eff_intensity *= 0.1

    beat_dt = beat.t_end - beat.t_start
    # 对每个目标部位施加刺激
    primary_part = target_parts[0] if target_parts else None
    for part in target_parts:
        is_primary = (part == primary_part)
        part_intensity = eff_intensity if is_primary else eff_intensity * 0.4
        state.apply_stimulation(part, part_intensity, beat_dt, action_type,
                               through_clothes=through_clothes)
    # 特殊动作的附加刺激
    if action_type in ["thrust", "missionary", "cowgirl", "doggy", "spoon",
                        "deep_thrust", "quick_thrust", "penetrate",
                        "slow_thrust", "legs_on_shoulders"]:
        # 抽插时G点和阴道口都会受刺激
        if "g_spot" not in target_parts and beat.depth > 0.7:
            state.apply_stimulation("g_spot", eff_intensity * 0.6, beat_dt,
                                   "deep_thrust", through_clothes=False)
        if beat.note == "deep_thrust" and random.random() < 0.3:
            state.apply_stimulation("cervix", eff_intensity * 0.2, beat_dt * 0.3,
                                   "deep_thrust", through_clothes=False)
        # 间接刺激阴蒂
        if "clitoris" not in target_parts and eff_intensity > 0.4:
            state.apply_stimulation("clitoris", eff_intensity * 0.15, beat_dt,
                                   action_type, through_clothes=False)

    state.tick(beat_dt, world)
    return state.orgasm.phase == "orgasm"  # 返回是否达到高潮（用于打断节奏）
