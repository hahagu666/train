"""
Layer 0e: 动态敏感度调制
8个因素动态调制每个部位的有效敏感度：
1. 当前唤起水平（唤起越高越敏感）
2. 新鲜感加成
3. 生理状态（经期/疲惫/晨勃等）
4. 高潮后状态（阴蒂过度敏感/内部敏感）
5. 适应（持续同样刺激→敏感度下降）
6. 注意力过滤
7. 情绪门控
8. 刺激变化（换手法重新激活）
"""
import math
from typing import Dict, Any


class SensitivityModulators:
    """生理状态调制器"""
    def __init__(self):
        self.chest_tightness: float = 0.0      # 胸部紧绷（经期前）
        self.menstrual_cramps: float = 0.0     # 痛经
        self.sore_genitals: float = 0.0        # 下体酸痛
        self.tired: float = 0.0                # 疲惫
        self.horny_morning: float = 0.0        # 晨欲
        self.stimulation_changed: bool = False # 最近10秒是否换了刺激方式
        self.last_stimulation_target: str = ""
        self.stimulation_same_count: int = 0   # 连续同样刺激计数
        self.adaptation: Dict[str, float] = {} # 各部位适应度
        self.post_orgasm_suppression: float = 0.0  # 高潮后抑制

    def to_dict(self) -> dict:
        return {
            "chest_tightness": self.chest_tightness,
            "menstrual_cramps": self.menstrual_cramps,
            "sore_genitals": self.sore_genitals,
            "tired": self.tired,
            "horny_morning": self.horny_morning,
            "stimulation_changed": self.stimulation_changed,
            "last_stimulation_target": self.last_stimulation_target,
            "stimulation_same_count": self.stimulation_same_count,
            "adaptation": dict(self.adaptation),
            "post_orgasm_suppression": self.post_orgasm_suppression,
        }

    def on_stimulation_change(self, target: str):
        """刺激方式/部位变化时调用"""
        if target != self.last_stimulation_target:
            self.stimulation_changed = True
            self.last_stimulation_target = target
            self.stimulation_same_count = 0
            # 换部位时旧部位适应清零
            for k in self.adaptation:
                self.adaptation[k] *= 0.5
        else:
            self.stimulation_changed = False
            self.stimulation_same_count += 1

    def tick(self, dt: float):
        """每时间步更新"""
        # 适应度累积（持续刺激同一部位→适应）
        if self.last_stimulation_target and not self.stimulation_changed:
            t = self.last_stimulation_target
            cur = self.adaptation.get(t, 0.0)
            self.adaptation[t] = min(1.0, cur + 0.02 * dt)
        # 自然恢复
        for k in list(self.adaptation.keys()):
            self.adaptation[k] *= math.exp(-0.03 * dt)
            if self.adaptation[k] < 0.01:
                del self.adaptation[k]
        # 状态衰减
        self.tired *= math.exp(-0.005 * dt)
        self.sore_genitals *= math.exp(-0.01 * dt)
        self.horny_morning *= math.exp(-0.02 * dt)
        self.stimulation_changed = False


def effective_sensitivity(part_name: str, base_sensitivity: float,
                          state, world, novelty_system=None,
                          current_action: str = "") -> float:
    """
    计算某部位当前的有效敏感度
    part_name: 子部位名称
    base_sensitivity: 基础敏感度
    state: CharacterState
    world: WorldState
    novelty_system: NoveltySystem
    current_action: 当前动作类型
    """
    sens = base_sensitivity
    sm = getattr(state, 'sensitivity_mods', None)

    # 1. 当前唤起水平（唤起越高越敏感）
    arousal = state.ans.arousal_global
    sens *= (0.5 + arousal * 1.0)

    # 2. 新鲜感加成
    if novelty_system:
        loc = world.location_type if world else "bedroom"
        bonus = novelty_system.get_sensitivity_bonus(current_action or "touch", loc)
        sens *= bonus

    # 3. 生理状态
    if sm:
        if part_name in ["nipple_left", "nipple_right", "breast_left", "breast_right"]:
            sens *= (1 - sm.chest_tightness * 0.4)
        if part_name in ["vaginal_canal", "clitoris", "inner_labia", "g_spot",
                         "vaginal_opening", "vaginal_vestibule"]:
            sens *= (1 - sm.menstrual_cramps * 0.3)
            if sm.sore_genitals > 0.3:
                sens *= (1 - sm.sore_genitals * 0.5)
        if sm.tired > 0.5:
            sens *= (1 - sm.tired * 0.3)
        if sm.horny_morning > 0.5:
            sens *= 1.3

    # 4. 高潮后状态
    orgasm = state.orgasm
    if orgasm.phase == "resolution":
        t = orgasm.time_since_last
        if t < 30:
            if part_name == "clitoris":
                sens *= 0.3  # 刚结束阴蒂触碰不舒服
            elif part_name in ["vaginal_canal", "g_spot", "cervix"]:
                sens *= 1.2  # 内部还是敏感
        elif t < 120 and orgasm.orgasms_so_far == 1:
            # 多重高潮窗口
            sens *= 1.1

    # 5. 适应（持续同样刺激→敏感度下降）
    if sm and part_name in sm.adaptation:
        sens *= (1 - sm.adaptation[part_name] * 0.5)

    # 6. 注意力过滤
    attention = state.emotion.attention_focus
    if attention and len(attention) > 0 and part_name not in attention:
        # 注意力不在这个部位→感受被弱化（但性器官不会完全过滤）
        is_sexual = part_name in ["clitoris", "vaginal_canal", "g_spot", "nipple_left",
                                   "nipple_right", "inner_labia", "cervix"]
        if not is_sexual:
            sens *= 0.6
        else:
            sens *= 0.85  # 性器官只是弱化不是过滤

    # 7. 情绪门控
    sens *= (0.3 + 0.7 * state.emotion.cognitive_gate)

    # 8. 刺激变化（换手法/换部位→重新激活）
    if sm and sm.stimulation_changed:
        sens *= 1.2

    return max(sens, 0.05)  # 最低5%
