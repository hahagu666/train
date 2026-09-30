"""
Layer 0a: 身体部位节点网络
按功能区域组织，每个子部位有独立的连续参数
"""
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SubPart:
    """子部位参数"""
    name: str
    arousal: float = 0.0
    sensitivity: float = 1.0
    # 通用参数（部位不同有不同字段，用extra存储）
    extra: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=0.0):
        return self.extra.get(key, default)

    def set(self, key: str, value):
        self.extra[key] = value

    def add(self, key: str, value, max_val=1.0, min_val=0.0):
        cur = self.extra.get(key, 0.0)
        self.extra[key] = max(min_val, min(max_val, cur + value))

    def decay(self, key: str, rate: float, dt: float):
        """指数衰减"""
        if key in self.extra:
            self.extra[key] *= math.exp(-rate * dt)

    def tick_decay(self, dt: float):
        """每时间步自然衰减"""
        self.arousal *= math.exp(-0.06 * dt)
        # 对常见参数衰减
        for k in ["swelling", "tremor", "congestion", "tingling", "tension",
                  "contraction", "erection", "wetness", "goosebumps", "shiver",
                  "arch", "curling", "quivering", "weakness", "butterflies"]:
            if k in self.extra:
                self.extra[k] *= math.exp(-0.05 * dt)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arousal": self.arousal,
            "sensitivity": self.sensitivity,
            "extra": dict(self.extra),
        }


@dataclass
class BodyRegion:
    """身体功能区域"""
    name: str
    sub_parts: Dict[str, SubPart] = field(default_factory=dict)
    daily: bool = True
    sexual: bool = False

    def get(self, part_name: str) -> SubPart:
        return self.sub_parts[part_name]

    def all_parts(self):
        return self.sub_parts.values()

    def tick_decay(self, dt: float):
        for sp in self.sub_parts.values():
            sp.tick_decay(dt)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "daily": self.daily,
            "sexual": self.sexual,
            "sub_parts": {n: sp.to_dict() for n, sp in self.sub_parts.items()},
        }


def build_default_body() -> Dict[str, BodyRegion]:
    """构建默认身体状态"""
    body = {}

    # === 头部区域 ===
    body["head"] = BodyRegion("head", daily=True, sexual=True, sub_parts={
        "lips": SubPart("lips", sensitivity=0.8, extra={"swollen": 0.0, "tremor": 0.0, "parted": 0.1}),
        "tongue": SubPart("tongue", sensitivity=0.9, extra={"active": False}),
        "throat": SubPart("throat", sensitivity=0.5, extra={"gag_reflex": 1.0, "contraction": 0.0, "relaxed": 0.1}),
        "mouth_interior": SubPart("mouth_interior", sensitivity=0.7, extra={"saliva": 0.5, "wetness": 0.5}),
        "earlobes": SubPart("earlobes", sensitivity=0.9, extra={"tingling": 0.0}),
        "neck": SubPart("neck", sensitivity=0.85, extra={"pulse_tension": 0.0, "bruising": 0.0}),
    })

    # === 胸部区域 ===
    def breast_side(side: str):
        return {
            f"breast_{side}": SubPart(f"breast_{side}", sensitivity=0.9,
                extra={"congestion": 0.0, "swelling": 0.0, "heaviness": 0.0, "fondled_dur": 0.0}),
            f"nipple_{side}": SubPart(f"nipple_{side}", sensitivity=1.2,
                extra={"congestion": 0.0, "erection": 0.0, "pain": 0.0, "first_touch": True}),
            f"areola_{side}": SubPart(f"areola_{side}", sensitivity=1.0,
                extra={"puffiness": 0.0}),
        }
    chest_parts = {}
    chest_parts.update(breast_side("left"))
    chest_parts.update(breast_side("right"))
    body["chest"] = BodyRegion("chest", daily=True, sexual=True, sub_parts=chest_parts)

    # === 手臂/手部 ===
    body["arms_hands"] = BodyRegion("arms_hands", daily=True, sexual=True, sub_parts={
        "fingers": SubPart("fingers", sensitivity=0.5, extra={"grip": 0.5, "tremor": 0.0, "curled": 0.0}),
        "palms": SubPart("palms", sensitivity=0.6, extra={"sweat": 0.0}),
        "wrists": SubPart("wrists", sensitivity=0.5, extra={"held": False}),
        "upper_arms": SubPart("upper_arms", sensitivity=0.4, extra={"goosebumps": 0.0}),
    })

    # === 核心躯干/腰腹 ===
    body["torso_core"] = BodyRegion("torso_core", daily=True, sexual=True, sub_parts={
        "waist": SubPart("waist", sensitivity=0.7, extra={"weakness": 0.0, "ticklish": 0.5, "grip_marks": 0.0}),
        "abdomen": SubPart("abdomen", sensitivity=0.6, extra={"muscle_tension": 0.0, "butterflies": 0.0, "flinching": 0.0}),
        "back": SubPart("back", sensitivity=0.6, extra={"arch": 0.0, "goosebumps": 0.0, "scratch_marks": 0.0}),
        "buttocks": SubPart("buttocks", sensitivity=0.8, extra={"tension": 0.0, "tingling": 0.0, "redness": 0.0, "sting": 0.0}),
        "navel": SubPart("navel", sensitivity=0.5, extra={"ticklish": 0.8}),
    })

    # === 女性生殖区域（最复杂）===
    body["genital_female"] = BodyRegion("genital_female", daily=False, sexual=True, sub_parts={
        "mons_pubis": SubPart("mons_pubis", sensitivity=0.4),
        "outer_labia": SubPart("outer_labia", sensitivity=0.6, extra={"swelling": 0.0, "parted": 0.0}),
        "inner_labia": SubPart("inner_labia", sensitivity=0.9,
            extra={"swelling": 0.0, "engorgement": 0.0, "color_change": 0.0}),
        "clitoral_hood": SubPart("clitoral_hood", sensitivity=0.8, extra={"retracted": 0.0}),
        "clitoris": SubPart("clitoris", sensitivity=1.5,
            extra={"engorgement": 0.0, "erection": 0.0,
                   "oversensitive_post": False, "pain_overstim": 0.0}),
        "vaginal_vestibule": SubPart("vaginal_vestibule", sensitivity=1.0,
            extra={"wetness": 0.0, "tingling": 0.0}),
        "vaginal_opening": SubPart("vaginal_opening", sensitivity=0.9,
            extra={"relaxation": 0.1, "stretch": 0.0, "first_time": True,
                   "hymen_intact": True, "burning": 0.0}),
        "vaginal_canal": SubPart("vaginal_canal", sensitivity=0.8,
            extra={"wetness": 0.0, "contraction": 0.0, "depth_pen": 0.0,
                   "girth_stretch": 0.0, "tent_lub": 0.0, "muscle_tone": 0.5,
                   "ejac_vol": 0.0, "tent_effect": 0.0}),
        "g_spot": SubPart("g_spot", sensitivity=0.8,
            extra={"swelling": 0.0, "stim": 0.0, "engorged": False}),
        "cervix": SubPart("cervix", sensitivity=0.3,
            extra={"position": "high", "bump_pain": 0.0, "reached": False,
                   "dip": False, "sens_increased": False}),
        "anus": SubPart("anus", sensitivity=0.7,
            extra={"contraction": 0.0, "relaxation": 0.1, "fullness": 0.0,
                   "lubrication": 0.05, "adaptation": 0.05,
                   "tightness": 1.0, "anal_pain": 0.0, "anal_pleasure": 0.0}),
        "perineum": SubPart("perineum", sensitivity=0.6),
    })

    # === 腿/脚 ===
    body["legs_feet"] = BodyRegion("legs_feet", daily=True, sexual=True, sub_parts={
        "thighs": SubPart("thighs", sensitivity=0.7,
            extra={"tension": 0.0, "tremor": 0.0, "openness": 0.2, "quivering": 0.0}),
        "inner_thighs": SubPart("inner_thighs", sensitivity=0.9,
            extra={"tingling": 0.0, "goosebumps": 0.0}),
        "calves": SubPart("calves", sensitivity=0.4, extra={"tension": 0.0, "cramping": 0.0}),
        "knees": SubPart("knees", sensitivity=0.3, extra={"weakness": 0.0, "buckling": 0.0}),
        "feet": SubPart("feet", sensitivity=0.4, extra={"curling": 0.0}),
    })

    # === 全局皮肤 ===
    body["skin_global"] = BodyRegion("skin_global", daily=True, sexual=True, sub_parts={
        "skin": SubPart("skin", sensitivity=0.5, extra={
            "flush": 0.0, "goosebumps": 0.0, "sweat": 0.0,
            "temperature": 36.5, "shiver": 0.0, "cold_flash": 0.0, "hot_flash": 0.0
        }),
    })

    return body


def compute_global_arousal(body: Dict[str, BodyRegion], ans) -> float:
    """计算全局性兴奋度（ANS驱动为主，局部身体反应加权）"""
    weights = {
        "genital_female": 0.45,
        "chest": 0.15,
        "head": 0.08,
        "torso_core": 0.10,
        "legs_feet": 0.10,
        "arms_hands": 0.02,
        "skin_global": 0.10,
    }
    local_total = 0.0
    for region_name, w in weights.items():
        region = body.get(region_name)
        if region:
            region_arousal = sum(sp.arousal for sp in region.all_parts()) / len(region.sub_parts)
            local_total += region_arousal * w
    # 全局唤起 = 85% ANS全局信号 + 15% 局部加权平均（放大）
    # 平台期ANS接近1时，全局唤起能到0.85+触发高潮
    combined = 0.85 * ans.arousal_global + 0.15 * local_total * 3.0
    return max(0.0, min(1.0, combined))


def compute_readiness(body: Dict[str, BodyRegion]) -> float:
    """计算是否足够润滑/放松可以进入 0-1"""
    gf = body["genital_female"]
    wet = gf.get("vaginal_canal").get("wetness") * 0.4 + gf.get("vaginal_vestibule").get("wetness") * 0.3 + gf.get("vaginal_canal").get("tent_lub") * 0.3
    relax = gf.get("vaginal_opening").get("relaxation")
    return wet * 0.6 + relax * 0.4


def compute_total_wetness(body: Dict[str, BodyRegion]) -> float:
    """综合润滑度"""
    gf = body["genital_female"]
    return (gf.get("vaginal_canal").get("wetness") * 0.5 +
            gf.get("vaginal_vestibule").get("wetness") * 0.3 +
            gf.get("vaginal_canal").get("tent_lub") * 0.2)

def compute_anal_readiness(body: Dict[str, BodyRegion]) -> float:
    """后穴可进入度 0-1：润滑 + 适应"""
    anus = body["genital_female"].get("anus")
    return max(0.0, min(1.0, anus.get("lubrication") * 0.6 + anus.get("adaptation") * 0.4))


# === 伙伴状态感知（男性）===
@dataclass
class PartnerState:
    """妹妹感知到的男性伙伴状态"""
    penis_erection: float = 0.0
    penis_throbbing: float = 0.0
    pre_cum: float = 0.0
    ejaculation_phase: Optional[str] = None  # None/imminent/happening/recent
    ejaculate_volume: float = 0.0
    inside_her: bool = False
    position: str = "outside"
    sens_post_orgasm: float = 0.0
    temp_inside: float = 0.0
    # 反应信号
    breathing_heavy: bool = False
    gripping: bool = False
    thrusting: bool = False
    thrust_speed: float = 0.0
    thrust_depth: float = 0.0
    making_sounds: bool = False
    tensed: bool = False
    pulling_out: bool = False
    pushing_deeper: bool = False
    pulsing_inside: bool = False

    def to_dict(self) -> dict:
        return {
            "penis_erection": self.penis_erection,
            "penis_throbbing": self.penis_throbbing,
            "pre_cum": self.pre_cum,
            "ejaculation_phase": self.ejaculation_phase,
            "ejaculate_volume": self.ejaculate_volume,
            "inside_her": self.inside_her,
            "position": self.position,
            "sens_post_orgasm": self.sens_post_orgasm,
            "temp_inside": self.temp_inside,
            "breathing_heavy": self.breathing_heavy,
            "gripping": self.gripping,
            "thrusting": self.thrusting,
            "thrust_speed": self.thrust_speed,
            "thrust_depth": self.thrust_depth,
            "making_sounds": self.making_sounds,
            "tensed": self.tensed,
            "pulling_out": self.pulling_out,
            "pushing_deeper": self.pushing_deeper,
            "pulsing_inside": self.pulsing_inside,
        }
