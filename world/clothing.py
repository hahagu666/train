"""
衣物系统 V2 - 6层精细模型
层序（从外到内）：
1. outerwear: 外套/开衫/卫衣
2. top_outer: 上衣外层（T恤/衬衫/连衣裙上半）
3. bottom_outer: 下装外层（短裤/裙子/裤子）
4. top_inner: 内衣上装（文胸/吊带）
5. bottom_inner: 内衣下装（内裤）
6. accessories: 袜子/鞋子/发饰等

中间状态（progress 0→1）：
- worn: 0.0-0.2 正常穿着
- unfastened: 0.2-0.4 解开了（扣子/拉链/挂钩）
- pushed_up/pushed_aside: 0.4-0.7 掀起来/推到一边（胸部可接触）
- pulled_down: 0.5-0.8 拉下来（肩膀/臀部可接触）
- partially_removed: 0.7-0.9 脱了一半挂在身上
- removed: 1.0 完全脱掉

每个部位的accessibility根据覆盖该部位的所有衣物综合计算
"""
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Garment:
    key: str
    name: str
    layer: int  # 1-6，越小越外层
    covers: Dict[str, float] = field(default_factory=dict)
    # covers: {部位: 覆盖程度(1=完全覆盖, 0.5=部分覆盖)}
    state_progress: float = 0.0  # 0=穿着, 1=完全脱掉
    is_fastened: bool = True  # 扣子/拉链/挂钩是否扣着
    removal_method: str = "pull_over_head"
    # pull_over_head / unbutton_and_remove / pull_down / unclasp_back / slide_off
    requires_assistance: bool = False  # 需要对方帮忙（背后的挂钩等）
    friction_mod: float = 0.0  # 隔着摩擦的额外质感
    fabric_thickness: float = 0.5  # 面料厚度（薄纱0.1 vs 厚毛衣0.9）
    softness: float = 0.5  # 面料柔软度

    def get_coverage(self, part: str) -> float:
        """获取当前状态下对某部位的实际覆盖度"""
        base = self.covers.get(part, 0.0)
        if base == 0:
            return 0.0
        # 根据脱衣进度减少覆盖
        return base * (1.0 - self.state_progress)

    def get_accessibility(self, part: str) -> float:
        """获取对某部位的可接触度（1=完全可接触）"""
        cov = self.get_coverage(part)
        # 厚面料阻碍更多
        block = cov * (0.6 + self.fabric_thickness * 0.4)
        return 1.0 - block

    def is_fully_worn(self) -> bool:
        return self.state_progress < 0.1

    def is_removed(self) -> bool:
        return self.state_progress > 0.95

    def advance_removal(self, amount: float) -> Tuple[str, float]:
        """推进脱衣，返回(状态描述, 实际进度增量)"""
        if self.is_removed():
            return "removed", 0.0
        # 如果还扣着，需要先解开
        if self.is_fastened and self.state_progress < 0.3:
            self.is_fastened = False
            self.state_progress = max(self.state_progress, 0.25)
            return "unfastened", 0.25
        old = self.state_progress
        self.state_progress = min(1.0, self.state_progress + amount)
        delta = self.state_progress - old
        return self._describe_state(), delta

    def push_aside(self, part: str = None) -> str:
        """把衣服掀起来/推到一边（不完全脱）"""
        if self.is_removed():
            return "removed"
        # 推到一边最多到0.7
        target = 0.6
        if part in ["chest", "nipple_left", "nipple_right", "breast_left", "breast_right", "belly"]:
            target = 0.65  # 上衣推起来露出胸
        if part in ["crotch", "genital_female", "clitoris"]:
            target = 0.7  # 内裤/短裤拉到一边
        self.state_progress = max(self.state_progress, target)
        return self._describe_state()

    def pull_down(self, part: str = None) -> str:
        """拉下来（露肩膀/臀部）"""
        if self.is_removed():
            return "removed"
        target = 0.6
        if part in ["shoulders", "back"]:
            target = 0.5
        self.state_progress = max(self.state_progress, target)
        return self._describe_state()

    def _describe_state(self) -> str:
        p = self.state_progress
        if p < 0.15:
            return "worn"
        if p < 0.35:
            return "unfastened"
        if p < 0.55:
            return "pushed_aside"
        if p < 0.75:
            return "pulled_down"
        if p < 0.95:
            return "partially_removed"
        return "removed"

    def state_label(self) -> str:
        return self._describe_state()

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "layer": self.layer,
            "covers": self.covers, "state_progress": self.state_progress,
            "is_fastened": self.is_fastened, "friction_mod": self.friction_mod,
            "fabric_thickness": self.fabric_thickness, "softness": self.softness,
        }


# ========== 默认服装 ==========

def _summer_home_garments() -> Dict[str, Garment]:
    """夏天在家：宽松T恤+内衣+短裤+内裤+袜子"""
    return {
        # layer 2: 上衣外层
        "tshirt": Garment(
            key="tshirt", name="宽大粉色T恤", layer=2,
            covers={
                "shoulders": 1.0, "chest": 1.0, "nipple_left": 1.0, "nipple_right": 1.0,
                "breast_left": 1.0, "breast_right": 1.0, "belly": 0.8, "back": 0.9,
                "upper_arm_left": 0.5, "upper_arm_right": 0.5,
            },
            removal_method="pull_over_head",
            friction_mod=0.2, fabric_thickness=0.2, softness=0.8,
        ),
        # layer 3: 下装外层
        "shorts": Garment(
            key="shorts", name="浅色棉质短裤", layer=3,
            covers={
                "thighs": 0.7, "hips": 1.0, "buttocks": 1.0, "waist": 1.0,
                "crotch": 0.9, "genital_female": 0.8,
            },
            removal_method="pull_down",
            is_fastened=False,  # 松紧带不需要解扣
            friction_mod=0.2, fabric_thickness=0.3, softness=0.7,
        ),
        # layer 4: 内衣上装
        "bra": Garment(
            key="bra", name="白色棉质内衣", layer=4,
            covers={
                "nipple_left": 1.0, "nipple_right": 1.0,
                "breast_left": 0.8, "breast_right": 0.8, "chest": 0.5,
            },
            removal_method="unclasp_back",
            requires_assistance=True,
            friction_mod=0.3, fabric_thickness=0.2, softness=0.6,
        ),
        # layer 5: 内衣下装
        "panties": Garment(
            key="panties", name="白色棉质内裤", layer=5,
            covers={
                "crotch": 1.0, "genital_female": 1.0, "clitoris": 1.0,
                "vaginal_vestibule": 1.0, "buttocks": 0.8, "hips": 0.5,
            },
            removal_method="pull_down",
            is_fastened=False,  # 松紧带
            friction_mod=0.4, fabric_thickness=0.15, softness=0.9,
        ),
        # layer 6: 配饰
        "socks": Garment(
            key="socks", name="白色短袜", layer=6,
            covers={"feet": 1.0, "ankles": 0.8},
            removal_method="slide_off",
            is_fastened=False,
            fabric_thickness=0.2, softness=0.6,
        ),
    }


def _school_uniform_garments() -> Dict[str, Garment]:
    """校服"""
    return {
        # layer 1: 外套
        "blazer": Garment(
            key="blazer", name="制服西装外套", layer=1,
            covers={"shoulders": 1.0, "chest": 0.9, "back": 0.9, "upper_arm_left": 0.7,
                   "upper_arm_right": 0.7},
            removal_method="unbutton_and_remove",
            fabric_thickness=0.6, softness=0.3,
        ),
        # layer 2: 衬衫
        "shirt": Garment(
            key="shirt", name="白衬衫", layer=2,
            covers={"shoulders": 1.0, "chest": 1.0, "belly": 0.8, "back": 1.0,
                   "upper_arm_left": 0.8, "upper_arm_right": 0.8},
            removal_method="unbutton_and_remove",
            friction_mod=0.1, fabric_thickness=0.2, softness=0.5,
        ),
        # layer 3: 裙子
        "skirt": Garment(
            key="skirt", name="百褶裙", layer=3,
            covers={"thighs": 0.6, "hips": 1.0, "buttocks": 0.8, "waist": 1.0},
            removal_method="pull_up",  # 裙子是掀起来不是脱下来
            friction_mod=0.1, fabric_thickness=0.3, softness=0.4,
        ),
        # layer 4: 内衣
        "bra": Garment(
            key="bra", name="白色内衣", layer=4,
            covers={"nipple_left": 1.0, "nipple_right": 1.0, "breast_left": 0.8,
                   "breast_right": 0.8},
            removal_method="unclasp_back",
            requires_assistance=True,
            friction_mod=0.3, fabric_thickness=0.2, softness=0.6,
        ),
        # layer 5: 内裤
        "panties": Garment(
            key="panties", name="白色内裤", layer=5,
            covers={"crotch": 1.0, "genital_female": 1.0, "clitoris": 1.0,
                   "buttocks": 0.8},
            removal_method="pull_down",
            is_fastened=False,
            friction_mod=0.4, fabric_thickness=0.15, softness=0.9,
        ),
        # layer 6: 鞋袜
        "socks": Garment(
            key="socks", name="及膝白袜", layer=6,
            covers={"calves": 0.9, "feet": 1.0, "ankles": 1.0},
            removal_method="slide_off",
            is_fastened=False,
            fabric_thickness=0.3, softness=0.5,
        ),
        "shoes": Garment(
            key="shoes", name="制服鞋", layer=6,
            covers={"feet": 1.0},
            removal_method="slide_off",
            is_fastened=False,
            fabric_thickness=0.9, softness=0.1,
        ),
    }


def _pajamas_garments() -> Dict[str, Garment]:
    """睡衣"""
    return {
        "pajama_top": Garment(
            key="pajama_top", name="小熊图案睡衣上衣", layer=2,
            covers={"shoulders": 1.0, "chest": 1.0, "nipple_left": 1.0, "nipple_right": 1.0,
                   "belly": 0.9, "back": 1.0, "upper_arm_left": 0.5, "upper_arm_right": 0.5},
            removal_method="unbutton_and_remove",
            friction_mod=0.2, fabric_thickness=0.3, softness=0.9,
        ),
        "pajama_bottom": Garment(
            key="pajama_bottom", name="小熊图案睡裤", layer=3,
            covers={"thighs": 1.0, "hips": 1.0, "buttocks": 1.0, "waist": 1.0,
                   "crotch": 0.9, "genital_female": 0.7, "calves": 0.5},
            removal_method="pull_down",
            is_fastened=False,
            friction_mod=0.2, fabric_thickness=0.3, softness=0.9,
        ),
        "panties": Garment(
            key="panties", name="白色棉质内裤", layer=5,
            covers={"crotch": 1.0, "genital_female": 1.0, "clitoris": 1.0, "buttocks": 0.8},
            removal_method="pull_down",
            is_fastened=False,
            friction_mod=0.4, fabric_thickness=0.15, softness=0.9,
        ),
    }


def _casual_garments() -> Dict[str, Garment]:
    """日常便服（T恤+短裤+内衣+内裤+袜子）"""
    return {
        "tshirt": Garment(
            key="tshirt", name="白色T恤", layer=2,
            covers={"shoulders": 1.0, "chest": 0.9, "nipple_left": 0.9, "nipple_right": 0.9,
                   "belly": 0.7, "back": 1.0, "upper_arm_left": 0.4, "upper_arm_right": 0.4},
            removal_method="pull_up",
            is_fastened=False,
            friction_mod=0.3, fabric_thickness=0.25, softness=0.8,
        ),
        "shorts": Garment(
            key="shorts", name="牛仔短裤", layer=3,
            covers={"thighs": 0.4, "hips": 1.0, "buttocks": 1.0, "waist": 1.0, "crotch": 0.7},
            removal_method="unbutton_and_remove",
            friction_mod=0.5, fabric_thickness=0.6, softness=0.3,
        ),
        "bra": Garment(
            key="bra", name="浅色内衣", layer=4,
            covers={"chest": 1.0, "nipple_left": 1.0, "nipple_right": 1.0, "back": 0.8},
            removal_method="unhook_and_remove",
            friction_mod=0.3, fabric_thickness=0.2, softness=0.7,
        ),
        "panties": Garment(
            key="panties", name="棉质内裤", layer=5,
            covers={"crotch": 1.0, "genital_female": 1.0, "clitoris": 1.0, "buttocks": 0.8},
            removal_method="pull_down",
            is_fastened=False,
            friction_mod=0.4, fabric_thickness=0.15, softness=0.9,
        ),
        "socks": Garment(
            key="socks", name="白色短袜", layer=1,
            covers={"feet": 1.0, "calves": 0.3},
            removal_method="pull_off",
            is_fastened=False,
            friction_mod=0.3, fabric_thickness=0.4, softness=0.5,
        ),
    }


def _summer_dress_garments() -> Dict[str, Garment]:
    """夏季连衣裙。"""
    garments = _casual_garments()
    garments.pop("tshirt", None)
    garments.pop("shorts", None)
    garments["dress"] = Garment(
        key="dress", name="轻薄的夏季连衣裙", layer=2,
        covers={
            "shoulders": 0.7, "chest": 1.0, "nipple_left": 1.0,
            "nipple_right": 1.0, "belly": 0.9, "back": 0.8,
            "waist": 1.0, "hips": 0.8, "buttocks": 0.7, "thighs": 0.5,
        },
        removal_method="pull_over_head", is_fastened=False,
        friction_mod=0.15, fabric_thickness=0.2, softness=0.8,
    )
    return garments


def _bath_towel_garments() -> Dict[str, Garment]:
    """洗浴后的浴巾。"""
    return {
        "bath_towel": Garment(
            key="bath_towel", name="裹在身上的浴巾", layer=2,
            covers={
                "chest": 1.0, "nipple_left": 1.0, "nipple_right": 1.0,
                "breast_left": 1.0, "breast_right": 1.0, "belly": 0.8,
                "back": 0.7, "waist": 1.0, "hips": 0.8,
                "buttocks": 0.8, "thighs": 0.4,
            },
            removal_method="unfasten_and_remove", is_fastened=False,
            friction_mod=0.15, fabric_thickness=0.35, softness=0.9,
        ),
    }


def _lingerie_garments() -> Dict[str, Garment]:
    """内衣套装。"""
    casual = _casual_garments()
    return {key: casual[key] for key in ("bra", "panties")}


class ClothingSystem:
    def __init__(self, outfit: str = "summer_home", custom_description: str = ""):
        self.garments: Dict[str, Garment] = {}
        self.base_outfit = outfit
        self.custom_description = ""
        self.reset(outfit, custom_description)

    def reset(self, outfit: str = "summer_home", custom_description: str = None):
        outfits = {
            "summer_home": _summer_home_garments,
            "school": _school_uniform_garments,
            "school_uniform": _school_uniform_garments,
            "pajamas": _pajamas_garments,
            "casual": _casual_garments,
            "summer_dress": _summer_dress_garments,
            "bath_towel": _bath_towel_garments,
            "lingerie": _lingerie_garments,
            "nude": lambda: {},
        }
        self.base_outfit = outfit if outfit in outfits else "summer_home"
        if custom_description is not None:
            self.custom_description = (custom_description or "").strip()
        else:
            self.custom_description = ""
        self.garments = outfits[self.base_outfit]()

    def get_accessibility(self, part_name: str) -> float:
        """
        获取某部位的综合可接触度（0-1）
        考虑所有覆盖该部位的衣物，外层没推开时内层碰不到
        """
        max_block = 0.0
        # 从外到内检查，外层未脱的话内层即使脱了也接触不到
        garments_sorted = sorted(self.garments.values(), key=lambda g: g.layer)
        for g in garments_sorted:
            cov = g.get_coverage(part_name)
            if cov > 0:
                # 外层覆盖度 > 0.5 时，内层衣物保护该部位
                effective_block = cov * (0.6 + g.fabric_thickness * 0.4)
                max_block = max(max_block, effective_block)
        return max(0.05, 1.0 - max_block)

    def get_friction_mod(self, part_name: str) -> float:
        """隔着衣物摩擦时的质感加成"""
        total = 0.0
        for g in self.garments.values():
            if not g.is_removed() and part_name in g.covers:
                cov = g.get_coverage(part_name)
                total += g.friction_mod * cov
        return min(0.8, total)

    def get_clothing_feeling(self, part_name: str) -> str:
        """获取某部位的衣物触感描述"""
        garms = [g for g in self.garments.values()
                 if not g.is_removed() and g.get_coverage(part_name) > 0.3]
        if not garms:
            return "skin"
        g = garms[0]  # 最外层
        if g.fabric_thickness > 0.6:
            return "thick_cloth"
        if g.softness > 0.7:
            return "soft_cloth"
        if "cotton" in g.name:
            return "cotton"
        return "cloth"

    def remove_garment(self, garment_key: str, amount: float = 0.5) -> str:
        """尝试脱掉一件衣物，返回新状态"""
        if garment_key not in self.garments:
            return "worn"
        g = self.garments[garment_key]
        # 检查是否有外层挡着（外层衣物必须覆盖了目标衣物的主要区域才阻挡）
        g_covers = set(k for k, v in g.covers.items() if v > 0.3)
        for og in self.garments.values():
            if og.layer < g.layer and not og.is_removed() and og.state_progress < 0.5:
                og_covers = set(k for k, v in og.covers.items() if v > 0.3)
                # 外层必须实际覆盖了目标衣物的关键区域才阻挡
                if g_covers & og_covers:
                    return "blocked_by_outer"
        state, _ = g.advance_removal(amount)
        return state

    def push_aside(self, garment_key: str, part: str = None) -> str:
        """把衣物掀起来/推到一边（不完全脱）"""
        if garment_key not in self.garments:
            return "worn"
        g = self.garments[garment_key]
        # 检查外层
        g_covers = set(k for k, v in g.covers.items() if v > 0.3)
        for og in self.garments.values():
            if og.layer < g.layer and not og.is_removed() and og.state_progress < 0.3:
                og_covers = set(k for k, v in og.covers.items() if v > 0.3)
                if g_covers & og_covers:
                    return "blocked_by_outer"
        return g.push_aside(part)

    def expose_part(self, part_name: str) -> List[str]:
        """
        尝试暴露某部位（自动推开/脱下必要衣物），返回被操作的衣物列表
        这模拟"伸手进去摸"的动作
        """
        operated = []
        # 从外到内
        garments_sorted = sorted(self.garments.values(), key=lambda g: g.layer)
        for g in garments_sorted:
            cov = g.get_coverage(part_name)
            if cov > 0.3:
                # 把这件衣物推到一边
                if g.layer <= 3:  # 外层衣物可以推起来
                    old = g.state_progress
                    g.push_aside(part_name)
                    if g.state_progress > old:
                        operated.append(g.key)
                elif part_name in ["clitoris", "vaginal_vestibule", "crotch",
                                    "genital_female", "nipple_left", "nipple_right"]:
                    # 内衣也可以推到一边（手伸进去）
                    old = g.state_progress
                    g.push_aside(part_name)
                    if g.state_progress > old:
                        operated.append(g.key)
        return operated

    def is_completely_nude(self) -> bool:
        for g in self.garments.values():
            if g.key in ["socks", "shoes"]:
                continue
            if not g.is_removed():
                return False
        return True

    def is_topless(self) -> bool:
        for g in self.garments.values():
            if g.layer in [2, 4] and any(p in g.covers
                    for p in ["chest", "nipple_left", "breast_left"]):
                if not g.is_removed() and g.state_progress < 0.7:
                    return False
        return True

    def is_bottomless(self) -> bool:
        for g in self.garments.values():
            if g.layer in [3, 5] and any(p in g.covers
                    for p in ["crotch", "genital_female", "buttocks"]):
                if not g.is_removed() and g.state_progress < 0.7:
                    return False
        return True

    def clothing_description(self, verbose: bool = False) -> str:
        """生成衣物状态描述；自定义描述仅代表初始穿着，衣物变化后仍展示动态状态。"""
        if self.custom_description and all(
            garment.state_label() == "worn" for garment in self.garments.values()
        ):
            return self.custom_description
        if self.is_completely_nude():
            socks = any(g.key == "socks" and not g.is_removed() for g in self.garments.values())
            return "全身赤裸，只有脚上还穿着袜子" if socks else "全身赤裸"
        worn = []
        pushed = []
        removed = []
        unfastened = []
        for g in self.garments.values():
            state = g.state_label()
            if state == "worn":
                worn.append(g.name)
            elif state == "unfastened":
                unfastened.append(g.name + "解开了")
            elif state in ["pushed_aside", "pulled_down"]:
                if any(p in g.covers for p in ["nipple_left", "breast_left", "chest"]):
                    pushed.append(g.name + "被撩起来露出胸口")
                elif any(p in g.covers for p in ["crotch", "genital_female"]):
                    pushed.append(g.name + "被拉到一边")
                else:
                    pushed.append(g.name + "凌乱地掀着")
            elif state == "partially_removed":
                pushed.append(g.name + "脱了一半挂在身上")
            elif state == "removed":
                removed.append(g.name)
        parts = []
        if worn:
            parts.append("穿着" + "、".join(worn))
        if unfastened:
            parts.append("，".join(unfastened))
        if pushed:
            parts.append("，".join(pushed))
        return "".join(parts) if parts else "衣着整齐"

    def to_dict(self) -> dict:
        return {
            "base_outfit": self.base_outfit,
            "custom_description": self.custom_description,
            "garments": {k: g.to_dict() for k, g in self.garments.items()},
        }
