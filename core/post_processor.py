"""
Post Processor - 模型输出解析器
从LLM生成的回应文本中提取隐含的状态变化信号，反哺Layer 0状态
让模型的创意发挥能影响状态机，而不是单向的状态→提示词

优先级：小模型结构化提取（如果可用） > 正则关键词匹配
"""
import re
import math
from typing import Dict, List, Tuple, Optional


SMALL_MODEL_EMOTIONS = {
    "happiness", "playfulness", "shy", "shyness", "embarrassment", "pleasure", "fear", "pain",
    "surprise", "jealousy", "hurt", "overwhelm", "satisfaction",
    "shock", "longing",
}
SMALL_MODEL_ACTIONS = {
    "pull_closer", "arch_into", "grab", "push_away_real", "cover_mouth",
    "hide_face", "guide_hand", "wrap_legs", "clench", "none",
}
SMALL_MODEL_SOUNDS = {
    "silent", "suppressed", "moan_soft", "moan_loud", "gasp",
    "broken_cry", "none",
}
SMALL_MODEL_ORGASM = {"approaching", "orgasm", "post", "none"}
SMALL_MODEL_INTERRUPTS = {
    "heard_sound", "door_knock", "phone_ring", "parents_come_home", "none",
}


def _finite_number(value, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _validate_small_model_data(data) -> Optional[Dict]:
    """Validate and sanitize structured extraction before it can affect state."""
    if not isinstance(data, dict):
        return None

    emotions = data.get("emotions", {})
    if not isinstance(emotions, dict):
        return None
    clean_emotions = {}
    for key, value in emotions.items():
        if key not in SMALL_MODEL_EMOTIONS or not _finite_number(value, 0.0, 1.0):
            continue
        clean_emotions[key] = float(value)

    def enum(name, allowed):
        value = data.get(name, "none")
        return value if isinstance(value, str) and value in allowed else "none"

    numeric_ranges = {
        "refusal": (0.0, 1.0),
        "acceptance": (0.0, 1.0),
        "arousal_delta": (-0.2, 0.2),
    }
    clean = {"emotions": clean_emotions}
    for name, (minimum, maximum) in numeric_ranges.items():
        value = data.get(name, 0.0)
        if not _finite_number(value, minimum, maximum):
            value = 0.0
        clean[name] = float(value)

    clean["active_action"] = enum("active_action", SMALL_MODEL_ACTIONS)
    clean["sound"] = enum("sound", SMALL_MODEL_SOUNDS)
    clean["orgasm_signal"] = enum("orgasm_signal", SMALL_MODEL_ORGASM)
    clean["interrupt_trigger"] = enum("interrupt_trigger", SMALL_MODEL_INTERRUPTS)
    for name in ("hurt_trigger", "clothing_self_remove"):
        value = data.get(name, False)
        clean[name] = value if isinstance(value, bool) else False
    return clean


def _merge_small_adjustments(adjustments: Dict, small: Dict):
    """Merge validated model signals while retaining regex-only supplemental cues."""
    adjustments["emotion_shift"].update(small.get("emotion_shift", {}))
    adjustments["mind_shift"].update(small.get("mind_shift", {}))
    for key in ("arousal_delta", "trust_delta", "acceptance_delta",
                "active_resistance_delta", "token_resistance_delta"):
        value = small.get(key, 0.0)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            adjustments[key] += value
    for key in ("sound_state", "refusal_sincerity", "orgasm_signal",
                "interrupt_trigger", "clothing_change", "hurt_episode"):
        value = small.get(key)
        if value not in (None, "", {}, []):
            adjustments[key] = value


NON_INTIMATE_ACTIONS = {
    "talk", "wait", "continue", "change_location", "change_position",
    "compliment", "pillow_talk", "say_name", "look_in_eyes",
}


def _allows_arousal_inference(action_type: str) -> bool:
    action_types = {part for part in (action_type or "").split("+") if part}
    return not action_types or not action_types.issubset(NON_INTIMATE_ACTIONS)


def _suppress_positive_intimate_inference(adjustments: Dict):
    """保留拒绝等防御信号，丢弃纯非亲密动作中的正向亲密推断。"""
    adjustments["arousal_delta"] = min(0.0, adjustments["arousal_delta"])
    adjustments["acceptance_delta"] = min(0.0, adjustments["acceptance_delta"])
    adjustments["trust_delta"] = min(0.0, adjustments["trust_delta"])
    adjustments["mind_shift"].pop("acceptance_level", None)
    adjustments["mind_shift"].pop("immersion", None)
    adjustments["orgasm_signal"] = None
    adjustments["clothing_change"] = None


def _try_small_model_extract(text: str, state, action_type: str) -> Optional[Dict]:
    """尝试用小模型做结构化提取，不可用/失败返回None"""
    allow_arousal_inference = _allows_arousal_inference(action_type)
    try:
        from llm_small import extract_structured, is_available
        if not is_available():
            return None
    except Exception:
        return None

    arousal = state.global_arousal
    phase = state.orgasm.phase
    prompt = f"""分析以下角色回应文本，提取其中的情绪、动作、态度信号，输出JSON。

当前角色状态参考：唤起度{arousal:.2f}，阶段{phase}，动作类型{action_type or 'unknown'}。

回应文本：
{text}

输出JSON格式：
{{
  "emotions": {{"emotion_name": 0.0-1.0}}, // 出现的情绪: shy/pleasure/fear/pain/surprise/jealousy/hurt/overwhelm/satisfaction/shock/longing
  "active_action": "", // 她做了什么主动动作: pull_closer/arch_into/grab/push_away_real/cover_mouth/hide_face/guide_hand/wrap_legs/clench/none
  "sound": "", // 声音: silent/suppressed/moan_soft/moan_loud/gasp/broken_cry/none
  "refusal": 0.0, // 拒绝真诚度0-1：0=接受/半推半就，1=坚决拒绝
  "acceptance": 0.0, // 接受度增量0-1
  "arousal_delta": 0.0, // 唤起度增量-0.2~0.2
  "orgasm_signal": "", // approaching/orgasm/post/none
  "hurt_trigger": false, // 是否触发受伤/吃醋
  "interrupt_trigger": "", // heard_sound/door_knock/phone_ring/parents_come_home/none
  "clothing_self_remove": false // 她是否自己脱衣服
}}

只输出JSON，不要其他文字。"""

    data = extract_structured(prompt)
    data = _validate_small_model_data(data)
    if not data:
        return None

    # 将小模型输出转换成和正则一致的adjustments格式
    adj = {
        "emotion_shift": data.get("emotions", {}) or {},
        "mind_shift": {},
        "arousal_delta": (
            float(data.get("arousal_delta", 0) or 0)
            if allow_arousal_inference
            else 0.0
        ),
        "trust_delta": 0.0,
        "sound_state": None,
        "refusal_sincerity": None,
        "acceptance_delta": (
            float(data.get("acceptance", 0) or 0)
            if allow_arousal_inference
            else 0.0
        ),
        "orgasm_signal": None,
        "active_resistance_delta": 0.0,
        "token_resistance_delta": 0.0,
        "interrupt_trigger": None,
        "clothing_change": None,
        "skill_practice": [],
        "hurt_episode": None,
        "dialogue_tags": [],
        "_from_small_model": True,
        "_active_action": data.get("active_action", "none"),
    }

    sound = data.get("sound", "")
    if sound and sound != "none":
        adj["sound_state"] = sound

    refusal = float(data.get("refusal", 0) or 0)
    if refusal > 0.6:
        adj["refusal_sincerity"] = refusal
        adj["active_resistance_delta"] = refusal * 0.4
    elif refusal > 0.2:
        adj["refusal_sincerity"] = refusal * 0.4
        adj["token_resistance_delta"] = refusal * 0.3

    action = data.get("active_action", "")
    if action and action != "none":
        if action in ("pull_closer", "arch_into", "guide_hand", "wrap_legs", "clench"):
            if allow_arousal_inference:
                adj["mind_shift"]["acceptance_level"] = {
                    "pull_closer": 0.15, "arch_into": 0.2,
                    "guide_hand": 0.3, "wrap_legs": 0.25,
                    "clench": 0.05,
                }.get(action, 0.1)
        elif action == "push_away_real":
            adj["active_resistance_delta"] += 0.4
            adj["refusal_sincerity"] = 0.9
            adj["arousal_delta"] -= 0.15
        elif action == "cover_mouth":
            adj["sound_state"] = "suppressed"
        elif action == "hide_face":
            adj["emotion_shift"]["embarrassment"] = 0.2

    orgasm = data.get("orgasm_signal", "")
    if orgasm in ("approaching", "orgasm", "post"):
        adj["orgasm_signal"] = orgasm
        if orgasm == "orgasm":
            adj["arousal_delta"] += 0.15

    itype = data.get("interrupt_trigger", "")
    if itype and itype != "none":
        adj["interrupt_trigger"] = itype

    if data.get("clothing_self_remove") and allow_arousal_inference:
        adj["clothing_change"] = "self_remove"

    if data.get("hurt_trigger"):
        adj["hurt_episode"] = {"type": "hurt", "intensity": 0.4, "cause": action_type or "partner_action"}

    return adj


# === 关键词模式 ===

# 主动动作（她做了什么 → 调整mind状态）
ACTIVE_ACTION_PATTERNS = {
    "pull_closer": [r"贴近", r"贴上来", r"靠过来", r"往.*怀里", r"主动.*抱", r"环住.*脖子",
                    r"缠上", r"腿勾住", r"搂住.*腰", r"往.*怀里蹭"],
    "arch_into": [r"弓.*身", r"迎上去", r"腰.*弓", r"挺.*胸", r"往.*手.*蹭", r"主动贴"],
    "grab": [r"抓紧", r"攥紧.*床单", r"抓.*背", r"抓.*手臂", r"手指.*抠"],
    "hide_face": [r"埋.*脸", r"捂住.*脸", r"别过脸", r"转过头.*不看"],
    "cover_mouth": [r"捂住.*嘴", r"掩住.*唇", r"咬.*唇", r"咬住.*嘴唇"],
    "push_away_token": [r"推.*一下.*没用力", r"手抵.*胸.*推", r"象征性.*推", r"轻轻推开"],
    "push_away_real": [r"用力推开", r"使劲推", r"挣扎", r"躲开", r"退开", r"缩.*角落"],
    "pull_clothing": [r"自己.*脱", r"拉起.*衣服", r"解开.*扣子", r"脱掉.*内衣", r"撩起.*裙"],
    "guide_hand": [r"拉着.*手", r"引导.*手", r"把.*手放"],
    "brace": [r"撑住", r"扶住.*肩", r"手撑.*床"],
    "wrap_legs": [r"腿.*缠.*腰", r"腿.*勾住"],
    "clench": [r"夹.*紧", r"缩.*紧", r"绞紧"],
}

# 情绪表现 → 情绪值调整
EMOTION_SIGNAL_PATTERNS = {
    "shy": [r"脸红", r"脸颊.*红", r"耳尖.*红", r"红透.*脸", r"害羞", r"娇羞", r"脸.*发烫"],
    "embarrassment": [r"羞死人", r"丢人", r"不敢看", r"脸.*烧", r"好羞"],
    "pleasure": [r"舒服", r"好棒", r"太.*了", r"爽", r"快感", r"妙", r"销魂"],
    "fear": [r"害怕", r"紧张", r"不安", r"慌", r"担心.*被.*听见", r"被.*发现"],
    "longing": [r"想要", r"还要", r"别停", r"继续", r"不要.*停", r"给我"],
    "satisfaction": [r"满足", r"安心", r"松.*口气", r"瘫软", r"软在.*怀里"],
    "pain": [r"痛", r"疼", r"嘶.*", r"皱眉", r"不适", r"撑大.*疼"],
    "overwhelm": [r"受不了", r"不行了", r"要.*了", r"到.*了", r"要死了", r"坏掉了"],
    "surprise": [r"吓.*跳", r"没想到", r"愣.*住", r"睁大眼睛"],
    "jealousy": [r"别人", r"她.*是谁", r"你和.*她", r"不.*别的.*女人"],
    "hurt": [r"心.*揪", r"难过", r"受伤", r"你怎么能"],
    "shock": [r"僵住", r"瞪大眼睛", r"难以置信", r"脑子.*空白"],
}

# 声音信号 → 更新suppress状态和ANS
SOUND_SIGNAL_PATTERNS = {
    "suppressed": [r"忍住.*声", r"咬.*唇.*不发出", r"捂.*嘴.*没叫", r"憋住", r"死死咬住"],
    "moan_soft": [r"嘤咛", r"闷哼", r"小声.*呻吟", r"鼻息", r"轻轻.*哼"],
    "moan_loud": [r"呻吟", r"叫出声", r"喊出声", r"叫.*出来", r"呜咽", r"哭腔"],
    "gasp": [r"倒吸", r"喘息", r"喘气", r"吸.*凉气", r"啊.*一声"],
    "silent": [r"没出声", r"咬着.*没发声", r"不发出声"],
    "broken_cry": [r"哭了出来", r"哭着", r"眼泪.*流", r"抽泣"],
}

# 拒绝/接受信号（判断真诚度）
REFUSAL_SIGNALS = {
    "token": [r"不要…", r"不可以…", r"别这样…", r"嗯…不要", r"不行…", r"停下…"],
    # 带省略号或语气词，不坚决
    "sincere": [r"不要！", r"停下来！", r"别碰我", r"放开我", r"我不要", r"你别这样", r"求你.*停"],
    # 感叹号/明确/恳求
    "mixed": [r"不行…但是", r"不可以…可是", r"别…嗯…"],
}

ACCEPTANCE_SIGNALS = {
    "implicit": [r"嗯…", r"啊…", r"随你", r"交给你了", r"你来吧"],
    "explicit": [r"我想要", r"给我", r"快点", r"进来", r"全部给我", r"我要你"],
}

# 高潮/接近高潮信号
ORGASM_SIGNALS = {
    "approaching": [r"要.*到了", r"快.*了", r"不行了.*要", r"要去了"],
    "orgasm": [r"来了", r"——！", r"啊——", r"到了.*", r"高潮", r"去了"],
    "post": [r"瘫软", r"脱力", r"没力气", r"晕过去", r"失神", r"虚脱"],
}

# 中断触发信号（她说听到/看到什么）
INTERRUPT_TRIGGERS = {
    "heard_sound": [r"什么声音", r"你听", r"外面有", r"好像有.*声", r"谁来了"],
    "door_knock": [r"敲门", r"有人敲门", r"门口有人"],
    "phone_ring": [r"手机.*响", r"电话.*响", r"有电话"],
    "parents_come_home": [r"爸妈回来了", r"门口有声音", r"钥匙.*声", r"妈妈回来了"],
    "pet_interrupt": [r"猫", r"狗", r"猫跳", r"狗叫"],
}

# 衣物变化信号（她主动做了什么）
CLOTHING_CHANGE_SIGNALS = {
    "removed_top": [r"脱掉.*上衣", r"自己.*脱.*衣服", r"T恤.*脱", r"上衣.*脱了"],
    "removed_bra": [r"解开.*内衣", r"脱掉.*胸罩", r"内衣.*掉"],
    "removed_panties": [r"内裤.*脱", r"自己.*褪下.*内裤", r"内裤.*滑下"],
    "pushed_aside_top": [r"撩起.*衣服", r"掀起.*T恤", r"衣服.*拉.*胸口下"],
    "pulled_up_skirt": [r"裙摆.*撩起", r"裙子.*掀"],
}


def _match_any(patterns: List[str], text: str) -> int:
    """返回匹配到的数量"""
    cnt = 0
    for p in patterns:
        if re.search(p, text):
            cnt += 1
    return cnt


def post_process(model_response: str, state, world=None,
                 action_type: str = "") -> Dict:
    """
    解析模型回应，返回状态调整建议
    返回：一个包含各种状态变化的dict，调用方决定如何应用

    策略：优先用小模型结构化提取；小模型不可用/失败时降级为正则关键词匹配。
    """
    text = model_response
    adjustments = {
        "emotion_shift": {},
        "mind_shift": {},
        "arousal_delta": 0.0,
        "trust_delta": 0.0,
        "sound_state": None,
        "refusal_sincerity": None,
        "acceptance_delta": 0.0,
        "orgasm_signal": None,
        "active_resistance_delta": 0.0,
        "token_resistance_delta": 0.0,
        "interrupt_trigger": None,
        "clothing_change": None,
        "skill_practice": [],
        "hurt_episode": None,
        "dialogue_tags": [],
    }

    # === 0. 尝试小模型结构化提取（作为主路径）===
    small_result = _try_small_model_extract(text, state, action_type)
    used_small = small_result is not None
    if used_small:
        # 小模型只提供经过校验的基础信号，随后仍运行正则补充，避免
        # 小模型成功时直接 return 导致明确文本信号丢失。
        _merge_small_adjustments(adjustments, small_result)

    # === 1. 情绪信号 ===
    for emo, patterns in EMOTION_SIGNAL_PATTERNS.items():
        matches = _match_any(patterns, text)
        if matches > 0:
            base_delta = 0.15 * min(matches, 2)
            adjustments["emotion_shift"][emo] = adjustments["emotion_shift"].get(emo, 0) + base_delta

    # === 2. 声音信号 ===
    max_sound = None
    max_sound_priority = 0
    sound_priority = {"silent": 0, "suppressed": 1, "moan_soft": 2, "gasp": 3,
                      "moan_loud": 4, "broken_cry": 5}
    for stype, patterns in SOUND_SIGNAL_PATTERNS.items():
        if _match_any(patterns, text) > 0:
            if sound_priority.get(stype, 0) > max_sound_priority:
                max_sound = stype
                max_sound_priority = sound_priority.get(stype, 0)
    if max_sound:
        adjustments["sound_state"] = max_sound

    # === 3. 拒绝/接受信号 ===
    sincere_refuse = _match_any(REFUSAL_SIGNALS["sincere"], text)
    token_refuse = _match_any(REFUSAL_SIGNALS["token"], text)
    mixed_refuse = _match_any(REFUSAL_SIGNALS["mixed"], text)
    explicit_accept = _match_any(ACCEPTANCE_SIGNALS["explicit"], text)
    implicit_accept = _match_any(ACCEPTANCE_SIGNALS["implicit"], text)

    if sincere_refuse > 0 and token_refuse == 0:
        adjustments["refusal_sincerity"] = 0.8
        adjustments["active_resistance_delta"] += 0.3
    elif token_refuse > 0 and sincere_refuse == 0:
        adjustments["refusal_sincerity"] = 0.2
        adjustments["token_resistance_delta"] += 0.2
    elif mixed_refuse > 0:
        adjustments["refusal_sincerity"] = 0.4
        adjustments["token_resistance_delta"] += 0.15

    if explicit_accept > 0:
        adjustments["acceptance_delta"] += 0.3
    elif implicit_accept > 0:
        adjustments["acceptance_delta"] += 0.1

    # 说"不要"但她主动贴上来=半推半就
    if token_refuse > 0 and _match_any(ACTIVE_ACTION_PATTERNS["arch_into"], text) > 0:
        adjustments["token_resistance_delta"] += 0.2
        adjustments["refusal_sincerity"] = 0.1

    # === 4. 主动动作信号（影响mind）===
    if _match_any(ACTIVE_ACTION_PATTERNS["pull_closer"], text) > 0:
        adjustments["mind_shift"]["acceptance_level"] = 0.15
        adjustments["arousal_delta"] += 0.05
    if _match_any(ACTIVE_ACTION_PATTERNS["arch_into"], text) > 0:
        adjustments["mind_shift"]["acceptance_level"] = 0.2
        adjustments["arousal_delta"] += 0.08
    if _match_any(ACTIVE_ACTION_PATTERNS["grab"], text) > 0:
        adjustments["arousal_delta"] += 0.05
        adjustments["mind_shift"]["immersion"] = 0.1
    if _match_any(ACTIVE_ACTION_PATTERNS["cover_mouth"], text) > 0:
        adjustments["sound_state"] = "suppressed"
    if _match_any(ACTIVE_ACTION_PATTERNS["hide_face"], text) > 0:
        adjustments["emotion_shift"]["embarrassment"] = adjustments["emotion_shift"].get("embarrassment", 0) + 0.2
    if _match_any(ACTIVE_ACTION_PATTERNS["push_away_real"], text) > 0:
        adjustments["active_resistance_delta"] += 0.4
        adjustments["refusal_sincerity"] = 0.9
        adjustments["arousal_delta"] -= 0.15
    if _match_any(ACTIVE_ACTION_PATTERNS["wrap_legs"], text) > 0:
        adjustments["mind_shift"]["acceptance_level"] = 0.25
        adjustments["arousal_delta"] += 0.1
    if _match_any(ACTIVE_ACTION_PATTERNS["clench"], text) > 0:
        adjustments["arousal_delta"] += 0.06  # 夹紧本身增加刺激
    if _match_any(ACTIVE_ACTION_PATTERNS["guide_hand"], text) > 0:
        adjustments["mind_shift"]["acceptance_level"] = 0.3
        adjustments["trust_delta"] += 0.03
    if _match_any(ACTIVE_ACTION_PATTERNS["pull_clothing"], text) > 0:
        adjustments["mind_shift"]["acceptance_level"] = 0.2
        adjustments["clothing_change"] = "self_remove"

    # === 5. 高潮信号 ===
    if _match_any(ORGASM_SIGNALS["orgasm"], text) > 0 and state.global_arousal > 0.65:
        adjustments["orgasm_signal"] = "orgasm"
        # 加速高潮触发
        adjustments["arousal_delta"] += 0.15
    elif _match_any(ORGASM_SIGNALS["approaching"], text) > 0 and state.global_arousal > 0.5:
        adjustments["orgasm_signal"] = "approaching"
        adjustments["arousal_delta"] += 0.05
    elif _match_any(ORGASM_SIGNALS["post"], text) > 0 and state.orgasm.phase in ["orgasm", "resolution"]:
        adjustments["orgasm_signal"] = "post"

    # === 6. 中断信号 ===
    for itype, patterns in INTERRUPT_TRIGGERS.items():
        if _match_any(patterns, text) > 0:
            adjustments["interrupt_trigger"] = itype
            break

    # === 7. 衣物变化 ===
    for ctype, patterns in CLOTHING_CHANGE_SIGNALS.items():
        if _match_any(patterns, text) > 0:
            adjustments["clothing_change"] = ctype
            break

    # === 8. 受伤/吃醋（需要hurt episode）===
    if _match_any(EMOTION_SIGNAL_PATTERNS["hurt"], text) > 0 or \
       _match_any(EMOTION_SIGNAL_PATTERNS["jealousy"], text) > 0:
        hurt_level = 0.3
        if _match_any(EMOTION_SIGNAL_PATTERNS["jealousy"], text) > 0:
            hurt_level = 0.5
        adjustments["hurt_episode"] = {
            "type": "jealousy" if _match_any(EMOTION_SIGNAL_PATTERNS["jealousy"], text) else "hurt",
            "intensity": hurt_level,
            "cause": action_type or "partner_action",
        }

    if not _allows_arousal_inference(action_type):
        _suppress_positive_intimate_inference(adjustments)

    return adjustments


def apply_post_adjustments(adjustments: Dict, state, world=None):
    """将post processor的调整应用到state"""
    # 情绪调整
    for emo, delta in adjustments.get("emotion_shift", {}).items():
        state.emotion.shift(emo, delta, dt=1.0)

    # mind调整
    for attr, delta in adjustments.get("mind_shift", {}).items():
        if hasattr(state.mind, attr):
            cur = getattr(state.mind, attr)
            setattr(state.mind, attr, max(0.0, min(1.0, cur + delta)))

    # 声音状态
    sound = adjustments.get("sound_state")
    if sound == "suppressed" or sound == "silent":
        state.emotion.suppressing_sounds = True
    elif sound in ["moan_loud", "gasp", "broken_cry", "moan_soft"]:
        state.emotion.suppressing_sounds = False
        # 不抑制时隐私风险
        if world and world.privacy_level < 0.5 and sound in ["moan_loud", "broken_cry"]:
            world.danger_level = min(1.0, world.danger_level + 0.2)

    # 拒绝/抵抗
    # 互斥校验：强拒绝（refusal_sincerity >= 0.7，对应 explicit_refusal 档）与
    # 接受度/信任增长在语义上矛盾。小模型可能从同一句害羞回避的台词里同时
    # 提取两者，此时以拒绝信号为准，丢弃正向接受/信任增量，避免状态机
    # 在“她明确说不”的同时把接受度一路推高。
    rs = adjustments.get("refusal_sincerity")
    acceptance_delta = float(adjustments.get("acceptance_delta", 0) or 0)
    trust_delta = float(adjustments.get("trust_delta", 0) or 0)
    arousal_delta = float(adjustments.get("arousal_delta", 0) or 0)
    if rs is not None and float(rs) >= 0.7:
        acceptance_delta = min(acceptance_delta, 0.0)
        trust_delta = min(trust_delta, 0.0)
    # 防推脱螺旋：模型用"喝杯茶/坐下来聊聊"式回避时，小模型每轮都提取负向
    # acceptance/arousal，状态被单轮大幅拖低后，下一轮模型看到更低的接受度
    # 反而更回避，形成下行螺旋。真实拒绝（refusal_sincerity >= 0.5，对应
    # gentle_boundary 及以上）不受此限；微弱拒绝信号下限制单轮负向下调幅度。
    if rs is None or float(rs) < 0.5:
        acceptance_delta = max(acceptance_delta, -0.05)
        arousal_delta = max(arousal_delta, -0.1)
    if rs is not None:
        state.mind.refusal_sincerity = max(state.mind.refusal_sincerity, rs)
    state.mind.active_resistance_will = min(1.0,
        state.mind.active_resistance_will + adjustments.get("active_resistance_delta", 0))
    state.mind.token_resistance = min(1.0,
        state.mind.token_resistance + adjustments.get("token_resistance_delta", 0))
    state.mind.acceptance_level = min(1.0,
        state.mind.acceptance_level + acceptance_delta)

    # arousal调整（小幅）
    state.global_arousal = max(0.0, min(1.0,
        state.global_arousal + arousal_delta))
    state.ans.arousal_global = max(state.ans.arousal_global, state.global_arousal * 0.8)

    # trust调整
    state.relationship_trust = max(0.1, min(1.0,
        state.relationship_trust + trust_delta))

    # 环境中断不能由角色自己的生成文本“确认”。否则模型幻觉一个声音，
    # 下一回合就会被写入权威世界状态并持续污染上下文；真实中断只能来自
    # EventEngine 或显式的用户/外部事件输入。保留字段以兼容旧调整数据，
    # 但这里不把 assistant 输出升级成事实。
    # interrupt = adjustments.get("interrupt_trigger")
    # if interrupt and world and not world.events.is_interrupted():
    #     world.events.trigger_interrupt_manually(interrupt, world, state)

    # 高潮信号加速
    if adjustments.get("orgasm_signal") == "orgasm" and state.orgasm.phase == "plateau":
        state.global_arousal = max(state.global_arousal, state.orgasm.orgasm_threshold)

    return state
