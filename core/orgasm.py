"""
Layer 0 高潮系统
以女性为中心，4阶段（兴奋/平台/高潮/消退）+ 多类型 + 多次高潮
"""
import random
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OrgasmSystem:
    # 平台期阈值
    plateau_arousal: float = 0.65
    plateau_duration_base: float = 8.0  # 秒
    plateau_duration_required: float = 8.0
    plateau_timer: float = 0.0

    # 高潮临界点
    orgasm_threshold_base: float = 0.78
    orgasm_threshold: float = 0.78
    point_of_no_return: bool = False

    # 高潮类型概率
    orgasm_type_probs: Dict[str, float] = field(default_factory=lambda: {
        "clitoral": 0.40,
        "vaginal": 0.15,
        "blended": 0.35,
        "cervical": 0.05,
        "multiple_chain": 0.05,
    })

    # 当前状态
    phase: str = "excitement"  # excitement/plateau/orgasm/resolution
    orgasm_duration: float = 0.0
    orgasm_intensity: float = 0.0
    orgasm_type: Optional[str] = None
    contraction_wave: Dict = field(default_factory=dict)

    # 多次高潮
    can_have_multiple: bool = True
    orgasms_so_far: int = 0
    time_since_last: float = 999.0
    refractory_type: str = "partial"

    def _select_orgasm_type(self, active_stims: Dict[str, float] = None,
                            skills: Dict = None, emotion=None) -> str:
        """
        根据刺激方式、经验、情绪状态智能选择高潮类型
        active_stims: {部位名: 近期刺激强度}
        """
        probs = {
            "clitoral": 0.28,
            "vaginal": 0.10,
            "blended": 0.25,
            "cervical": 0.03,
            "anal": 0.05,
            "multiple_chain": 0.02,
        }

        # === 根据当前主要刺激部位调整概率 ===
        if active_stims:
            clit_stim = active_stims.get("clitoris", 0)
            vag_stim = active_stims.get("vaginal_canal", 0) + active_stims.get("g_spot", 0)
            cervix_stim = active_stims.get("cervix", 0)
            anal_stim = active_stims.get("anus", 0)

            if anal_stim > 0.5 and vag_stim < 0.4 and clit_stim < 0.4:
                probs["anal"] += 0.45
                probs["vaginal"] -= 0.15
                probs["clitoral"] -= 0.15
                probs["blended"] -= 0.1
            elif clit_stim > 0.5 and vag_stim < 0.3:
                probs["clitoral"] += 0.35
                probs["blended"] -= 0.1
            elif vag_stim > 0.5 and clit_stim < 0.3:
                probs["vaginal"] += 0.35
                probs["blended"] -= 0.1
                if cervix_stim > 0.3:
                    probs["cervical"] += 0.15
            elif clit_stim > 0.4 and vag_stim > 0.4:
                probs["blended"] += 0.40
                probs["clitoral"] -= 0.15
                probs["vaginal"] -= 0.1

        # === 根据经验调整概率 ===
        if skills:
            from .skills import calc_effective_mastery
            # 有插入经验→阴道高潮/混合高潮概率上升
            m_miss = calc_effective_mastery(skills.get("missionary", type('S',(),{'times':0,'mastery':0,'last_practiced':None,'half_life_days':14})()))
            m_cow = calc_effective_mastery(skills.get("cowgirl", type('S',(),{'times':0,'mastery':0,'last_practiced':None,'half_life_days':14})()))
            m_dogg = calc_effective_mastery(skills.get("doggy", type('S',(),{'times':0,'mastery':0,'last_practiced':None,'half_life_days':14})()))
            insert_exp = max(m_miss, m_cow, m_dogg)
            if insert_exp > 0.5:
                probs["vaginal"] += insert_exp * 0.2
                probs["blended"] += insert_exp * 0.15
                probs["clitoral"] -= insert_exp * 0.1
            # 经验多了更容易多重高潮链
            total_exp = sum(calc_effective_mastery(s) for s in skills.values() if hasattr(s, 'times'))
            if total_exp > 3.0:
                probs["multiple_chain"] += min(0.15, total_exp * 0.03)
            # 后入式经验→宫颈高潮概率上升
            if m_dogg > 0.4:
                probs["cervical"] += m_dogg * 0.1

        # === 根据情绪冲突调整===
        if emotion:
            conflict = emotion.conflict_level
            # 情绪冲突高（又怕又想要）→混合高潮/失控感高潮概率上升
            if conflict > 0.5:
                probs["blended"] += conflict * 0.15
                probs["multiple_chain"] += conflict * 0.1
            # 完全放松信任→更容易深阴道高潮
            if emotion.blend.get("trust", 0) > 0.8 and conflict < 0.2:
                probs["vaginal"] += 0.1
                probs["cervical"] += 0.05
            # 禁果效应刺激→高潮更强烈/混合
            if emotion.forbidden_arousal_bonus > 0.05:
                probs["blended"] += emotion.forbidden_arousal_bonus
                probs["multiple_chain"] += emotion.forbidden_arousal_bonus * 0.5

        # === 多重高潮历史===
        if self.orgasms_so_far >= 1:
            probs["multiple_chain"] += 0.1 + self.orgasms_so_far * 0.05
            probs["clitoral"] -= 0.1  # 第二次起阴蒂过度敏感

        # 归一化
        probs = {k: max(0.01, v) for k, v in probs.items()}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # 采样
        r = random.random()
        cum = 0.0
        for t, p in probs.items():
            cum += p
            if r < cum:
                return t
        return "blended"

    def _init_contraction_wave(self, intensity: float):
        """初始化收缩波"""
        base_count = 5 + int(intensity * 10)
        if self.orgasm_type in ["blended", "multiple_chain"]:
            base_count += 3
        if self.orgasm_type == "cervical":
            base_count += 5  # 宫颈高潮收缩更久
        self.contraction_wave = {
            "interval": 0.8,
            "count": 0,
            "max_count": base_count,
            "strength": min(1.0, intensity * 1.1),
        }

    def update(self, global_arousal: float, ans, emotion, stim_active: bool, stim_adequate: bool,
               dt: float, skills, active_stims: Dict[str, float] = None,
               on_clit_pain: bool = False) -> Optional[str]:
        """
        每时间步更新
        global_arousal: 全局唤起值0-1（传入而非从body dict取）
        active_stims: 近期刺激部位强度映射
        返回: 事件字符串（如 "orgasm_start"/None）
        """
        event = None

        # === 高潮中 ===
        if self.phase == "orgasm":
            self.orgasm_duration += dt
            self.time_since_last = 0.0
            self._process_contractions(ans, emotion, dt)
            total_dur = self._current_duration()
            if self.orgasm_duration > total_dur:
                self._to_resolution(emotion, global_arousal)
                event = "orgasm_end"
            return event

        # === 消退期 ===
        if self.phase == "resolution":
            self.time_since_last += dt
            self._process_resolution(ans, emotion, stim_active, stim_adequate, dt, global_arousal, on_clit_pain)
            return event

        # === 兴奋期→平台期 ===
        if global_arousal > self.plateau_arousal:
            if self.phase == "excitement":
                self.phase = "plateau"
                self.plateau_timer = 0.0
                event = "enter_plateau"

        # === 平台期累积 ===
        if self.phase == "plateau":
            if stim_active and stim_adequate and not on_clit_pain:
                self.plateau_timer += dt
            else:
                # 刺激中断/阴蒂痛→平台期回落
                self.plateau_timer -= dt * 0.5
                if self.plateau_timer < 0:
                    self.phase = "excitement"
                    self.plateau_timer = 0.0

            # 达到临界点
            if (self.plateau_timer >= self.plateau_duration_required and
                    global_arousal >= self.orgasm_threshold):
                self.point_of_no_return = True
                self.phase = "orgasm"
                self.orgasm_type = self._select_orgasm_type(active_stims, skills, emotion)
                self.orgasm_intensity = min(1.0, global_arousal +
                                            emotion.conflict_level * 0.15 +
                                            random.uniform(-0.05, 0.15))
                self.orgasm_duration = 0.0
                self.orgasms_so_far += 1
                self._init_contraction_wave(self.orgasm_intensity)
                event = "orgasm_start"

        return event

    def _process_contractions(self, ans, emotion, dt):
        cw = self.contraction_wave
        cw["interval"] -= dt

        if cw["interval"] <= 0 and cw["count"] < cw["max_count"]:
            cw["count"] += 1
            # 前几次更强，涉及范围更大
            if cw["count"] <= 3:
                ans.muscle_tone = 1.0
                ans.tremor = min(1.0, ans.tremor + 0.8)
            if cw["count"] <= 2:
                ans.muscle_tone = 1.0
            # 宫颈dipping
            if self.orgasm_type in ["vaginal", "blended", "cervical"]:
                pass  # 由narrative层处理
            # 射液概率
            ejac_prob = 0.05 + self.orgasm_intensity * 0.15
            if self.orgasm_type in ["vaginal", "blended"]:
                ejac_prob += 0.1
            self.last_ejaculation_prob = ejac_prob
            # 声音不自主
            if cw["count"] <= 3:
                ans.voice_breathiness = 1.0
                if random.random() < self.orgasm_intensity * 0.6:
                    emotion.suppressing_sounds = False
            # 间隔变长、强度递减
            cw["interval"] = 0.8 + cw["count"] * 0.12
            cw["strength"] *= 0.85

    def _to_resolution(self, emotion, global_arousal: float):
        self.phase = "resolution"
        self.point_of_no_return = False
        self.plateau_timer = 0.0

        if (self.can_have_multiple and self.orgasms_so_far < 3 and
                self.orgasm_intensity > 0.7):
            # 多重高潮窗口
            self._post_arousal = 0.65
            self.orgasm_threshold = 0.82
            self.plateau_duration_required = 3.0
            emotion.blend["satisfaction"] = 0.6
            emotion.blend["overwhelm"] = min(1.0, emotion.blend.get("overwhelm", 0) + 0.4)
        else:
            # 完全消退
            self._post_arousal = 0.3
            self.orgasm_threshold = self.orgasm_threshold_base
            self.plateau_duration_required = self.plateau_duration_base
            emotion.blend["satisfaction"] = 0.9
            emotion.blend["sleepy"] = min(1.0,
                emotion.blend.get("sleepy", 0) + 0.3 + self.orgasms_so_far * 0.2)
            emotion.blend["shame"] = min(1.0, emotion.blend.get("shame", 0) + 0.1)

    def _process_resolution(self, ans, emotion, stim_active, stim_adequate, dt,
                            global_arousal: float, on_clit_pain: bool):
        new_arousal = getattr(self, '_post_arousal', 0.3)
        new_arousal = max(0.1, new_arousal - 0.008 * dt)
        self._post_arousal = new_arousal
        ans.heart_rate = max(75, ans.heart_rate - 0.8 * dt)
        ans.breathing_rate = max(14, ans.breathing_rate - 0.4 * dt)
        ans.muscle_tone = max(0.2, ans.muscle_tone - 0.02 * dt)
        ans.skin_flush = max(0, ans.skin_flush - 0.005 * dt)
        ans.tremor = max(0, ans.tremor - 0.02 * dt)

        emotion.blend["satisfaction"] = min(1.0, emotion.blend.get("satisfaction", 0) + 0.005 * dt)
        if self.orgasms_so_far >= 2:
            emotion.blend["sleepy"] = min(1.0, emotion.blend.get("sleepy", 0) + 0.005 * dt)

        # 足够刺激可重回平台期（多重高潮）
        if (stim_active and stim_adequate and
                new_arousal > 0.5 and not on_clit_pain and
                self.orgasms_so_far < 3 and
                self.time_since_last < 60):
            self.phase = "plateau"
            self.plateau_timer = self.plateau_duration_required * 0.5

    def _current_duration(self) -> float:
        base = 8 + self.orgasm_intensity * 12
        if self.orgasms_so_far > 1:
            base *= 0.8
        return base

    def reset(self):
        self.__init__()

    @property
    def intensity(self) -> float:
        return self.orgasm_intensity

    def to_dict(self) -> dict:
        return {
            "plateau_arousal": self.plateau_arousal,
            "plateau_duration_base": self.plateau_duration_base,
            "plateau_duration_required": self.plateau_duration_required,
            "plateau_timer": self.plateau_timer,
            "orgasm_threshold_base": self.orgasm_threshold_base,
            "orgasm_threshold": self.orgasm_threshold,
            "point_of_no_return": self.point_of_no_return,
            "phase": self.phase,
            "orgasm_duration": self.orgasm_duration,
            "orgasm_intensity": self.orgasm_intensity,
            "orgasm_type": self.orgasm_type,
            "contraction_wave": dict(self.contraction_wave),
            "can_have_multiple": self.can_have_multiple,
            "orgasms_so_far": self.orgasms_so_far,
            "time_since_last": self.time_since_last,
            "refractory_type": self.refractory_type,
            "orgasm_type_probs": dict(self.orgasm_type_probs),
        }
