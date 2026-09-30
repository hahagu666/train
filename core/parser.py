"""
动作解析器 Parser
- 从用户自然语言输入中提取动作
- 拆分复合动作（一边吻一边摸 = kiss + touch_breast）
- 检测时间跳跃
- 映射到具体target部位、动作类型、强度
- 歧义检测（需要澄清的情况）
- 衣物操作识别
- 自动expose_part（手伸进衣服摸）
"""
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# === 日志补接线（详细排障） ===
try:
    from app.logger import debug, info, success, warning, error, trace
except Exception:
    debug = info = success = warning = error = trace = lambda *a, **k: None

@dataclass
class ParsedAction:
    """解析后的单个动作"""
    action_type: str  # 动作类型标识
    target_parts: List[str] = field(default_factory=list)  # 目标部位列表
    intensity: float = 0.7  # 强度 0.2-1.0
    duration_mod: float = 1.0  # 时长修正
    through_clothes: Optional[bool] = None  # 是否隔着衣物（None=自动判断）
    clothing_action: Optional[str] = None  # 衣物操作类型（None/remove/push_aside/unfasten/pull_down）
    clothing_target: Optional[str] = None  # 目标衣物key
    position_change: Optional[str] = None  # 姿势变化
    location_change: Optional[str] = None  # 位置变化
    verbal: Optional[str] = None  # 说的话
    emotional_action: Optional[str] = None  # 情感动作（look_in_eyes等）
    sub_actions: List['ParsedAction'] = field(default_factory=list)  # 同时进行的子动作
    ambiguity: List[str] = field(default_factory=list)  # 歧义点
    time_jump_seconds: Optional[float] = None  # 时间跳跃（秒）
    is_wait: bool = False
    is_meta: bool = False  # 元指令（如"继续"、"不要停"）
    source_text: str = ""  # 该动作对应的原文分句
    relation: str = "single"  # single/sequential/simultaneous
    sequence_index: int = 0  # 按原文出现顺序编号
    clothing_barriers: List[str] = field(default_factory=list)  # 触达目标前隔着的衣物
    intent_only: bool = False  # 仅表达/询问意图，不表示身体动作已经发生

    @property
    def is_sexual(self) -> bool:
        return (
            self.action_type in SEXUAL_ACTIONS
            or self.action_type == "sexual_intent"
            or (
                self.action_type == "handle_clothing"
                and self.clothing_target in SEXUAL_CLOTHING_TARGETS
            )
        )


SEXUAL_CLOTHING_TARGETS = {"bra", "panties", "pants", "shorts", "skirt"}

SEXUAL_ACTIONS = {
    "touch_breast", "fondle_breast", "suck_nipple", "pinch_nipple", "twist_nipple",
    "touch_over_clothes", "rub_clit_through_panties", "rub_clit", "finger_clit",
    "lick_clit", "suck_clit", "cunnilingus", "finger_entrance", "finger_insert_one",
    "finger_insert_two", "finger_gspot", "rub_penis_against", "guide_penis", "penetrate",
    "thrust", "slow_thrust", "deep_thrust", "quick_thrust", "missionary", "cowgirl",
    "doggy", "spoon", "legs_on_shoulders", "dirty_talk", "remove_shirt", "unhook_bra",
    "remove_bra", "remove_panties", "pull_up_skirt", "remove_pants", "remove_outerwear",
}

# 成人语境/请求不等同于已经发生的具体身体动作。该集合只在没有更具体
# 动作命中时使用，避免把用户的询问直接转换成刺激或衣物变化。
SEXUAL_INTENT_PATTERNS = (
    r"肉棒", r"阴茎", r"阳具", r"生殖器", r"阴部", r"私处",
    r"口交", r"做爱", r"性爱", r"性交", r"色情", r"性行为",
    r"自慰", r"手淫", r"射精", r"勃起",
    r"想上你", r"操你", r"干你", r"用嘴含", r"帮我含", r"含住",
)


def _is_sexual_intent(text: str) -> bool:
    """Detect explicit adult context without claiming a physical action occurred."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SEXUAL_INTENT_PATTERNS)


# === 动作关键词映射 ===
ACTION_KEYWORDS = {
    # 亲吻类
    "kiss": ["吻", "亲", "啵", "亲一下", "亲亲", "kiss", "亲吻"],
    "deep_kiss": ["深吻", "舌吻", "法式接吻", "伸进.*嘴", "舌头.*交缠", "deep kiss", "french kiss"],
    "neck_kiss": ["吻.*脖子", "亲.*脖子", "亲.*颈", "吻.*颈", "neck kiss", "脖子.*亲"],
    "ear_kiss": ["亲.*耳朵", "吻.*耳朵", "咬.*耳朵", "舔.*耳朵", "ear kiss"],
    "suck_neck": ["吸.*脖子", "种草莓", "吻痕", "咬.*脖子"],
    "lick_neck": ["舔.*脖子"],

    # 拥抱/抚摸类
    "hug": ["抱", "拥抱", "搂", "hug", "抱住"],
    "hold": ["握着", "拉着.*手", "牵.*手", "握住"],
    "caress_hair": ["摸.*头发", "抚.*头发", "揉.*头发", "顺.*头发", "摸头"],
    "touch_cheek": ["摸.*脸", "抚.*脸", "捏.*脸", "捧.*脸"],
    "look_in_eyes": ["看着.*眼睛", "对视", "凝视", "盯着.*看"],

    # 胸部类
    "touch_breast": ["摸.*胸", "抚.*胸", "碰.*胸", "揉.*胸", "按.*胸",
                     "握.*乳", "抓.*胸", "握.*胸", "揉.*乳房",
                     "touch.*breast", "feel.*chest"],
    "fondle_breast": ["揉捏.*胸", "搓揉.*乳房", "把玩.*乳", "爱抚.*胸", "玩弄.*乳"],
    "suck_nipple": ["含.*乳头", "吮.*乳头", "舔.*乳头", "吸.*乳", "吻.*乳首",
                    "suck.*nipple", "含着.*乳", "舔.*奶"],
    "pinch_nipple": ["捏.*乳头", "夹.*乳头", "掐.*乳头"],
    "twist_nipple": ["捻.*乳头", "拧.*乳头"],

    # 下体类（外部）
    "touch_over_clothes": ["隔着.*摸", "隔.*裤.*摸", "放在.*腿间", "手放在.*下"],
    "rub_clit_through_panties": ["隔着.*内裤.*揉", "隔.*内裤.*摸", "内裤.*摩擦",
                                 "揉.*下面.*内裤", "内裤上.*摸"],
    "rub_clit": ["揉.*蒂", "搓.*阴蒂", "按.*阴蒂", "摸.*阴蒂",
                 "揉.*豆豆", "揉.*核", "抚弄.*小豆豆", "rub.*clit"],
    "finger_clit": ["用手指.*揉.*蒂", "指尖.*按.*蒂", "拨弄.*蒂", "弹.*蒂"],
    "lick_clit": ["舔.*蒂", "舌头.*蒂", "舔.*小豆豆"],
    "suck_clit": ["吮.*蒂", "含.*蒂", "吸.*蒂"],
    "cunnilingus": ["舔.*下", "口交", "吻.*私处", "亲.*下面", "舔.*穴", "舔.*私处"],

    # 插入类
    "finger_entrance": ["手指.*入口", "摸.*阴道口", "在.*口.*徘徊"],
    "finger_insert_one": ["一根手指.*插", "一根手指.*进", "一指.*插入", "中指.*进去",
                          "手指.*探入", "手指.*伸进"],
    "finger_insert_two": ["两根手指.*插", "两指.*插入", "两根手指.*进"],
    "finger_gspot": ["抠.*G点", "按.*G点", "顶.*G点", "刮.*G点", "找.*G点"],
    "rub_penis_against": ["肉棒.*蹭", "阴茎.*抵", "顶着.*口", "蹭.*穴口", "磨.*入口"],
    "guide_penis": ["扶着.*对准", "引导.*进来", "拿着.*对准"],
    "penetrate": ["插进去", "插进", "进入", "插入", "顶进去", "顶进", "送进去", "捅进去", " penetrat", "进入.*身体"],
    "thrust": ["抽插", "抽送", " thrust", "动起来", "开始动", "继续动"],
    "slow_thrust": ["慢慢.*抽插", "缓缓.*进入", "慢慢.*动", "浅插",
                   "缓缓插进", "慢慢插进", "轻轻推进", "慢慢送进去", "轻轻送进去", "缓缓推进"],
    "deep_thrust": ["深深.*插", "深深插进", "顶到底", "整根.*没入", "直捣", "捅到底", "整个顶进"],
    "quick_thrust": ["快速.*抽", "快速.*动", "猛.*插", "快速抽送"],

    # 姿势
    "missionary": ["男上", " missionary", "传教士", "压在.*身上", "在.*上面"],
    "cowgirl": ["骑在.*身上", "女上", " cowgirl", "让.*坐上来", "上来自己动"],
    # "从背后"单独出现可能只是从背后拥抱，不构成后入；必须伴随插入类动词。
    "doggy": ["后入", "狗爬", "跪.*屁股", "撅起", "从后面(?:插|进|入|干|操|顶|来)"],
    "spoon": ["侧躺", "从身后抱着.*做", " spoon"],
    "legs_on_shoulders": ["腿.*架.*肩", "腿抬到肩上", "把腿.*举起来"],

    # 脱衣类
    "remove_shirt": ["脱掉.*上衣", "脱下.*T恤", "脱.*衣服", "脱掉.*衣服", "掀.*衣服",
                     "衣服.*脱", "把.*衣服.*脱"],
    "unhook_bra": ["解开.*内衣扣", "解开.*胸罩", "解开.*内衣"],
    "remove_bra": ["脱掉.*内衣", "脱下.*胸罩", "把.*内衣.*脱"],
    "remove_panties": ["脱掉.*内裤", "脱下.*内裤", "把.*内裤.*脱", "拉下.*内裤"],
    "pull_up_skirt": ["掀起.*裙子", "把裙子.*撩", "撩起.*裙"],
    "remove_pants": ["脱掉.*裤子", "脱下.*裤"],
    "remove_outerwear": ["脱掉.*外套", "脱下.*开衫"],

    # 情感/对话
    "compliment": ["夸", "说.*漂亮", "说.*可爱", "赞美"],
    "tease": ["调戏", "逗", "捉弄", "坏笑.*说", "tease"],
    "whisper": ["耳语", "小声说", "在耳边说", "低声说"],
    "say_name": ["叫.*名字", "喊.*名字"],
    "confess_love": ["说.*爱", "告白", "表白", "喜欢.*你"],
    "dirty_talk": ["说.*淫荡", "说.*色", "dirty talk", "说.*坏"],
    "bite_lip": ["咬.*嘴唇", "咬唇"],

    # 事后
    "cuddle": ["抱着", "搂住", "依偎", "cuddle", "抱在怀里"],
    "pillow_talk": ["聊天", "说话", "聊", "pillow talk"],
    "aftercare": ["安抚", "aftercare", "轻抚.*背", "吻.*额头"],
}

# === 部位关键词映射 ===
PART_KEYWORDS = {
    "lips": ["唇", "嘴", "嘴唇", "lips", "mouth"],
    "neck": ["脖子", "颈", "neck"],
    "ear": ["耳朵", "耳", "ear"],
    "cheek": ["脸", "脸颊", "cheek", "face"],
    "hair": ["头发", "发", "hair"],
    "shoulders": ["肩膀", "肩", "shoulder"],
    "breast_left": ["左胸", "左乳"],
    "breast_right": ["右胸", "右乳"],
    "breast": ["胸", "乳房", "胸部", "breast", "boob", "奶"],
    "nipple_left": ["左乳头", "左乳首"],
    "nipple_right": ["右乳头", "右乳首"],
    "nipple": ["乳头", "乳首", "nipple", "奶头"],
    "belly": ["肚子", "腹部", "belly", "stomach"],
    "back": ["背", "后背", "back"],
    "waist": ["腰", "waist"],
    "hips": ["胯", "腰胯", "hip"],
    "buttocks": ["屁股", "臀", "butt", "ass"],
    "thighs": ["大腿", "腿", "thigh", "腿内侧"],
    "crotch": ["腿间", "胯下", "crotch"],
    "genital_female": ["私处", "下面", "那里", "阴部", "私密处"],
    "clitoris": ["阴蒂", "豆豆", "小豆豆", "蒂", "clit"],
    "vaginal_vestibule": ["阴道口", "穴口", "入口"],
    "anus": ["肛门", "后庭", "后穴", "菊穴", "屁眼", "后门", "菊花", "后面那", "后面这个", "后面那个"],
    "vaginal_canal": ["里面", "体内", "深处", "里面"],
    "g_spot": ["G点", "g点"],
    "hands": ["手", "手指", "hand", "finger"],
}

# === 强度修饰词 ===
INTENSITY_MODS = {
    "gently": ["轻轻", "缓缓", "温柔", "轻柔", "慢慢", "softly", "gently"],
    "normal": [],
    "hard": ["用力", "使劲", "狠狠", "大力", "rough", "hard"],
    "fast": ["快速", "急促", "飞快", "加快", "fast", "quick"],
    "slow": ["慢慢", "缓缓", "徐徐", "slow"],
}

# === 衣物关键词映射 ===
CLOTHING_KEYWORDS = {
    "tshirt": ["T恤", "t恤", "上衣", "衣服", "shirt"],
    "shirt": ["衬衫", "衬衣"],
    "bra": ["内衣", "胸罩", "文胸", "bra"],
    "shorts": ["短裤", "热裤", "shorts"],
    "skirt": ["裙子", "短裙", "百褶裙", "skirt"],
    "panties": ["内裤", "小裤裤", "panties"],
    "pants": ["裤子", "长裤", "pants"],
    "blazer": ["外套", "西装外套", "blazer"],
    "pajama_top": ["睡衣"],
    "pajama_bottom": ["睡裤"],
    "socks": ["袜子", "socks"],
}

CLOTHING_ACTIONS = {
    "remove": ["脱掉", "脱下", "褪去", "除去", "remove"],
    "push_aside": ["掀起", "撩起", "推到一边", "拨开", "掀开"],
    "unfasten": ["解开", "解", "松开", "unhook", "unbutton", "unzip"],
    "pull_down": ["拉下", "往下拉", "褪下"],
    "pull_up": ["拉起", "往上推"],
}

# === 姿势变化 ===
POSITION_CHANGES = {
    "missionary": ["压在.*身上", "把.*按在床上", "push down", "男上"],
    "cowgirl": ["让.*坐.*腿上", "让.*骑上来", "sit on"],
    "doggy": ["让.*跪", "让.*趴", "turn over", "撅起.*屁股", "从后面(?:来|干|操|插|顶|趴)"],
    "on_bed": ["推倒在床上", "推到床上", "按到床上"],
    "kneeling": ["让.*跪", "跪下来"],
}

LOCATION_CHANGES = {
    "bedroom": ["去卧室", "回房间", "到房间里"],
    "bathroom": ["去浴室", "去洗澡", "去卫生间"],
    "living_room": ["去客厅", "到沙发"],
}

# === 时间跳跃（从time_system导入，这里做本地检测）===
TIME_JUMP_PATTERNS = [
    (r"第二天|次日|隔天|早上醒来|醒来", 8 * 3600),
    (r"过了一?小时|一小时后", 3600),
    (r"过了半?小时|半小时后", 1800),
    (r"过了几分钟|几分钟后|一会儿后", 300),
    (r"过了一会|稍后|不久", 120),
]

META_ACTIONS = ["继续", "不要停", "继续做", "别停", "接着", "more", "continue"]

# 只暴露经过解析器验证、不会绕过动作/状态处理的快捷命令。
# group 是客户端分组契约；min_stage 是命令可用的最早关系阶段。
COMMAND_CATALOG = (
    {"id": "continue", "label": "继续", "text": "继续", "group": "flow", "category": "flow", "min_stage": "A"},
    {"id": "wait_short", "label": "休息一会儿", "text": "过了一会", "group": "time", "category": "time", "min_stage": "A"},
    {"id": "wait_hour", "label": "一小时后", "text": "一小时后", "group": "time", "category": "time", "min_stage": "A"},
    {"id": "next_morning", "label": "第二天早上", "text": "第二天早上", "group": "time", "category": "time", "min_stage": "A"},
    {"id": "go_bedroom", "label": "去卧室", "text": "去卧室", "group": "location", "category": "location", "min_stage": "A"},
    {"id": "go_living_room", "label": "去客厅", "text": "去客厅", "group": "location", "category": "location", "min_stage": "A"},
    {"id": "go_bathroom", "label": "去浴室", "text": "去浴室", "group": "location", "category": "location", "min_stage": "A"},
)


def get_command_catalog() -> List[Dict[str, str]]:
    """Return a copy of the authoritative parser-supported command catalog."""
    return [dict(command) for command in COMMAND_CATALOG]


def _match_any(patterns: list, text: str) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def _detect_intensity(text: str) -> float:
    base = 0.7
    if _match_any(INTENSITY_MODS["gently"], text):
        base *= 0.6
    if _match_any(INTENSITY_MODS["hard"], text):
        base *= 1.4
    if _match_any(INTENSITY_MODS["fast"], text):
        base *= 1.2
    if _match_any(INTENSITY_MODS["slow"], text):
        base *= 0.8
    # 温柔地用力？矛盾时偏向平均
    return max(0.2, min(1.2, base))


def _detect_parts(text: str) -> List[str]:
    found = []
    for part, kws in PART_KEYWORDS.items():
        for kw in kws:
            match = re.search(kw, text)
            if match:
                found.append((match.start(), part))
                break
    found.sort(key=lambda item: item[0])
    # 去重并映射泛化部位到具体神经节点，同时保持原文顺序。
    concrete = []
    for _, part in found:
        if part == "breast" and any(p.startswith("nipple_") for p in concrete):
            continue
        if part == "nipple" and any(p.startswith("nipple_") for p in concrete):
            continue
        mapped = {
            "breast": ["breast_left", "breast_right", "nipple_left", "nipple_right"],
            "nipple": ["nipple_left", "nipple_right"],
            "genital_female": ["clitoris", "vaginal_vestibule", "labia_majora", "labia_minora"],
            "anus": ["anus"],
        }.get(part, [part])
        for item in mapped:
            if item not in concrete:
                concrete.append(item)
    return concrete


def _detect_clothing_barriers(text: str) -> List[str]:
    """识别明确描述的衣物阻隔层，按外到内的原文顺序返回。"""
    if not re.search(r"隔着|隔了|透过|下的|下面的", text):
        return []
    matches = []
    for garment, keywords in CLOTHING_KEYWORDS.items():
        for keyword in keywords:
            match = re.search(keyword, text)
            if match:
                matches.append((match.start(), garment))
                break
    return list(dict.fromkeys(garment for _, garment in sorted(matches)))


def _split_action_clauses(text: str) -> List[Tuple[str, str]]:
    """按分句及连续/同时连接词切分，隔离每个动作的目标作用域。"""
    clauses = []
    for chunk in (part.strip() for part in re.split(r"[，。；,;]+", text) if part.strip()):
        simultaneous = [part.strip() for part in re.split(r"一边|同时|并且|并", chunk) if part.strip()]
        if len(simultaneous) > 1:
            clauses.extend((part, "simultaneous") for part in simultaneous)
            continue
        sequential = [part.strip() for part in re.split(r"然后|接着|随后|再", chunk) if part.strip()]
        for index, part in enumerate(sequential):
            relation = "sequential" if clauses or index else "single"
            clauses.append((part.removeprefix("先").strip(), relation))
    return clauses or [(text, "single")]


def _matched_actions_in_order(text: str) -> List[str]:
    matches = []
    for action_type, patterns in ACTION_KEYWORDS.items():
        positions = [m.start() for pattern in patterns if (m := re.search(pattern, text))]
        if positions:
            matches.append((min(positions), action_type))
    return [action_type for _, action_type in sorted(matches)]


def _detect_clothing_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """检测是否是脱衣动作，返回(action, garment_key)"""
    for c_action, patterns in CLOTHING_ACTIONS.items():
        for p in patterns:
            if p in text:
                # 找目标衣物
                for g_key, g_kws in CLOTHING_KEYWORDS.items():
                    for g_kw in g_kws:
                        if g_kw in text:
                            return c_action, g_key
    # 单独出现衣物词也可能是摸上面
    return None, None


def _detect_position(text: str) -> Optional[str]:
    for pos, patterns in POSITION_CHANGES.items():
        for p in patterns:
            if re.search(p, text):
                return pos
    return None


def _detect_location(text: str) -> Optional[str]:
    for loc, patterns in LOCATION_CHANGES.items():
        for p in patterns:
            if p in text:
                return loc
    return None


def _detect_time_jump(text: str, world=None) -> Optional[float]:
    from world.time_system import detect_time_jump

    game_time = getattr(world, "game_time", None) if world is not None else None
    return detect_time_jump(text, game_time)


def parse_input(user_text: str, state=None, world=None) -> List[ParsedAction]:
    """
    解析用户输入为一个或多个动作序列
    复合动作用sub_actions表示同时进行
    """
    text = user_text.strip()
    actions: List[ParsedAction] = []

    # 1. 时间跳跃检测
    jump = _detect_time_jump(text, world)
    if jump and ("后" in text or "第二天" in text or "明天" in text or "次日" in text or "醒来" in text or "过了" in text):
        act = ParsedAction(
            action_type="wait",
            time_jump_seconds=jump,
            is_wait=True,
        )
        actions.append(act)
        text = re.sub(
            r"(?:休息|等待|等)?\s*(?:\d+\s*(?:秒|分钟|分|小时|天)后|第二天(?:早上|早晨|上午|中午|下午|傍晚|晚上)?|明天(?:早上|早晨|上午|中午|下午|傍晚|晚上)?|次日(?:早上|中午|下午|晚上)?|过了[^，。；,;]*)\s*(?:再|然后|后)?",
            "",
            text,
            count=1,
        ).strip("，。；,; ")
        if not text:
            return actions

    # 2. 元指令
    for meta in META_ACTIONS:
        if meta in text:
            act = ParsedAction(action_type="continue", is_meta=True)
            actions.append(act)
            return actions

    # 3. 位置变化
    loc_change = _detect_location(text)
    if loc_change:
        act = ParsedAction(action_type="change_location", location_change=loc_change)
        actions.append(act)

    # 4. 姿势变化
    pos_change = _detect_position(text)
    if pos_change:
        act = ParsedAction(action_type="change_position", position_change=pos_change)
        actions.append(act)

    # 5. 衣物操作
    cloth_action, cloth_target = _detect_clothing_action(text)
    if cloth_action:
        act = ParsedAction(
            action_type="handle_clothing",
            clothing_action=cloth_action,
            clothing_target=cloth_target,
        )
        actions.append(act)

    # 6. 分句解析动作，目标只绑定在所在分句，动作顺序按原文保留。
    sequence_start = len(actions)
    parsed_clause_actions = []
    for clause, relation in _split_action_clauses(text):
        matched = _matched_actions_in_order(clause)
        if not matched:
            continue
        parts = _detect_parts(clause)
        barriers = _detect_clothing_barriers(clause)
        for action_type in matched[:3]:
            targets = [part for part in parts if _part_belongs_to(part, action_type)] or parts
            parsed_clause_actions.append(ParsedAction(
                action_type=action_type,
                target_parts=targets,
                intensity=_detect_intensity(clause),
                through_clothes=True if barriers else None,
                source_text=clause,
                relation=relation,
                sequence_index=sequence_start + len(parsed_clause_actions),
                clothing_barriers=barriers,
            ))

    if parsed_clause_actions:
        actions.extend(parsed_clause_actions)
        debug("解析器", f"分句解析命中: 动作数={len(parsed_clause_actions)}, 类型=[{', '.join(a.action_type for a in parsed_clause_actions)}]")
        return actions

    if not actions:
        # 明确成人语境但没有具体动作动词时，保留为意图层信号；不能静默
        # 降级为普通对话，否则编排器会跳过成人资格与安全边界判断。
        if _is_sexual_intent(text):
            actions.append(ParsedAction(
                action_type="sexual_intent",
                verbal=text,
                source_text=text,
                intent_only=True,
            ))
        elif any(kw in text for kw in ["说", "问", "告诉", "回答", "？", "?", "…", "呢", "吗"]):
            actions.append(ParsedAction(action_type="talk", verbal=text, source_text=text))
        else:
            actions.append(ParsedAction(action_type="wait", source_text=text))
    debug("解析器", f"parse_input完成: 动作数={len(actions)}, 类型=[{', '.join(a.action_type for a in actions)}]")
    return actions


def _part_belongs_to(part: str, action_type: str) -> bool:
    """判断部位是否属于这个动作的合理目标"""
    part_groups = {
        "kiss": ["lips", "cheek", "neck", "ear"],
        "deep_kiss": ["lips"],
        "neck_kiss": ["neck"],
        "ear_kiss": ["ear"],
        "suck_neck": ["neck"],
        "lick_neck": ["neck"],
        "touch_cheek": ["cheek"],
        "caress_hair": ["hair"],
        "hold": ["hands"],
        "touch_breast": ["breast_left", "breast_right", "nipple_left", "nipple_right", "chest"],
        "fondle_breast": ["breast_left", "breast_right", "nipple_left", "nipple_right"],
        "suck_nipple": ["nipple_left", "nipple_right"],
        "pinch_nipple": ["nipple_left", "nipple_right"],
        "twist_nipple": ["nipple_left", "nipple_right"],
        "touch_over_clothes": ["crotch", "genital_female", "vaginal_vestibule", "clitoris", "thighs", "buttocks"],
        "rub_clit_through_panties": ["clitoris", "crotch"],
        "rub_clit": ["clitoris"],
        "finger_clit": ["clitoris"],
        "lick_clit": ["clitoris"],
        "suck_clit": ["clitoris"],
        "cunnilingus": ["clitoris", "vaginal_vestibule", "labia_minora", "labia_majora"],
        "finger_entrance": ["vaginal_vestibule"],
        "finger_insert_one": ["vaginal_canal", "g_spot"],
        "finger_insert_two": ["vaginal_canal", "g_spot"],
        "finger_gspot": ["g_spot", "vaginal_canal"],
        "penetrate": ["vaginal_canal", "anus"],
        "thrust": ["vaginal_canal", "g_spot", "anus"],
        "slow_thrust": ["vaginal_canal", "anus"],
        "deep_thrust": ["vaginal_canal", "cervix", "anus"],
        "quick_thrust": ["vaginal_canal", "anus"],
    }
    valid = part_groups.get(action_type, [])
    if not valid:
        return True  # 不限制的动作
    return part in valid


def resolve_action_targets(actions: List[ParsedAction], state) -> List[ParsedAction]:
    """
    后处理：根据当前状态推断目标可达性，不修改衣物状态。
    """
    for act in actions:
        if act.action_type in ["rub_clit", "finger_clit", "suck_clit", "lick_clit",
                                "touch_breast", "fondle_breast", "suck_nipple",
                                "finger_insert_one", "finger_insert_two", "penetrate",
                                "cunnilingus"]:
            target_part = None
            if act.target_parts:
                target_part = act.target_parts[0]
            elif act.action_type in ["touch_breast", "fondle_breast", "suck_nipple"]:
                target_part = "nipple_left"
            elif act.action_type in ["rub_clit", "finger_clit"]:
                target_part = "clitoris"
            elif act.action_type in ["penetrate", "finger_insert_one", "finger_insert_two"]:
                target_part = "vaginal_vestibule"
            if target_part:
                access = state.clothing.get_accessibility(target_part)
                act.through_clothes = access < 0.5

        for sub in act.sub_actions:
            resolve_action_targets([sub], state)
    return actions
