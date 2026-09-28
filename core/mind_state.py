"""
MindState - 主观认知层（Layer 0.5）
她"认为"正在发生什么，而不是客观上正在发生什么
- 情境解读：她如何理解当前的互动
- 意图感知：她认为你想做什么
- 抵抗意愿：主观上的抵抗/接受程度（区别于身体的无力）
- 自我认知：她怎么看自己现在的样子
- 边界意识：她意识到哪些边界正在被跨越
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class MindState:
    # === 情境解读 ===
    # 她认为当前是什么情况：casual/affectionate/intimate/sexual/forced/dreaming
    context_interpretation: str = "casual"
    context_confidence: float = 0.9  # 她对自己判断的确定度

    # 她认为当前关系亲密度（主观感受，不等于客观trust值）
    perceived_intimacy: float = 0.3

    # === 意图感知 ===
    # 她认为你想做什么（一个不断更新的猜测）
    perceived_intent: str = "hanging_out"
    # hanging_out/cuddling/kissing/going_further/just_teasing/sex/want_sex_but_shy/confused
    intent_suspicion_level: float = 0.0  # 怀疑你想更进一步的程度
    intent_certainty: float = 0.0  # 多确定你的意图

    # === 抵抗/顺从意愿（主观层）===
    # 主动抵抗的意愿（0-1）：即使身体有反应也想推开
    active_resistance_will: float = 0.0
    # 口头拒绝的真诚度（0=装装样子，1=真的想停）
    refusal_sincerity: float = 0.0
    # 接受程度（主观上愿意继续到下一步）
    acceptance_level: float = 0.1
    # 犹豫度：想继续又怕的矛盾
    hesitation: float = 0.0
    # 半推半就的程度：嘴上说不要但身体很诚实
    token_resistance: float = 0.0

    # === 自我认知 ===
    # 她觉得自己现在有多淫荡（羞耻的自我感知）
    perceived_sluttiness: float = 0.0
    # 她觉得自己表现得怎么样（笨拙/自然/放荡）
    self_perceived_skill: float = 0.1
    # 她认为你怎么看她
    perceived_partner_view: str = "as_sister"  # as_sister/as_woman/as_sex_object/loved
    # 自我意识程度（意识到自己的反应/脸红/声音）
    self_consciousness: float = 0.3
    # 是否意识到自己在发出声音
    aware_of_moans: bool = False

    # === 边界意识 ===
    # 已经跨越的边界（她意识到的）
    boundaries_crossed: List[str] = field(default_factory=list)
    # 正在逼近的边界
    boundaries_approaching: List[str] = field(default_factory=list)
    # 禁止/禁忌标签（她认为这是错的）
    taboo_labels: List[str] = field(default_factory=lambda: ["siblings"])
    # 禁忌违背的刺激感（禁果效应的主观层）
    taboo_thrill: float = 0.0

    # === 沉浸/解离 ===
    # 沉浸程度（0=完全清醒，1=沉入快感无法思考）
    immersion: float = 0.0
    # 现实解离（快感太强不真实感）
    dissociation: float = 0.0
    # 大脑空白度（0=能思考，1=什么都想不了）
    mind_blank: float = 0.0
    # 是否在"飞"（高潮前的恍惚状态）
    is_flying: bool = False

    # === 期望/预测 ===
    # 她预测接下来会发生什么
    expected_next: str = ""
    # 她害怕发生的事
    fear_next: str = ""
    # 她暗中期待的事
    secretly_want: str = ""
    # 是否感到意外
    surprised: bool = False
    surprise_level: float = 0.0

    # === 元认知 ===
    # 她意识到自己身体有反应了吗
    aware_of_arousal: bool = False
    # 她在纠结（理不清自己的感受）
    conflicted_about: str = ""
    # 她在对自己说什么（内心OS标签）
    inner_voice: str = ""

    # 上次更新时间
    last_update_time: float = 0.0

    def update(self, state, world, action_type: str, dt: float):
        """根据新动作更新主观认知"""
        emotion = state.emotion
        arousal = state.global_arousal
        ans = state.ans
        trust = state.relationship_trust
        shame = emotion.blend.get("shame", 0) + emotion.blend.get("embarrassment", 0)
        pleasure = emotion.blend.get("pleasure", 0)
        fear = emotion.blend.get("fear", 0)
        longing = emotion.blend.get("longing", 0)
        conflict = emotion.conflict_level
        # world可以为None（简化更新路径）
        privacy = 0.9
        if world:
            privacy = world.privacy_level
        self._current_privacy = privacy

        # === 1. 更新情境解读 ===
        self._update_context_interpretation(action_type, arousal, trust, shame, fear, pleasure, dt)

        # === 2. 更新意图感知 ===
        self._update_intent_perception(action_type, state, world, dt)

        # === 3. 更新抵抗/接受意愿 ===
        self._update_resistance_will(arousal, pleasure, shame, fear, trust, conflict, dt)

        # === 4. 更新自我认知 ===
        self._update_self_perception(arousal, shame, pleasure, state, world, dt)

        # === 5. 更新边界意识 ===
        self._update_boundaries(action_type, state, dt)

        # === 6. 更新沉浸/解离 ===
        self._update_immersion(arousal, pleasure, ans, state, dt)

        self.last_update_time = state.sim_time

    def _update_context_interpretation(self, action_type: str, arousal: float,
                                        trust: float, shame: float, fear: float,
                                        pleasure: float, dt: float):
        """更新她对当前情境的解读"""
        # 动作类型权重
        intimate_actions = {"kiss", "deep_kiss", "hug", "caress_hair", "neck_kiss",
                           "hold", "cuddle", "look_in_eyes", "whisper"}
        sexual_actions = {"touch_breast", "fondle_breast", "suck_nipple", "rub_clit",
                         "finger_clit", "cunnilingus", "thrust", "missionary",
                         "cowgirl", "doggy", "suck_clit", "rub_clit_through_panties",
                         "touch_over_clothes", "finger_insert_one", "finger_insert_two"}
        forced_cues = fear > 0.5 and trust < 0.5

        # 根据动作逐渐改变认知
        target = self.context_interpretation
        weight = 0.0

        if action_type in sexual_actions or arousal > 0.5:
            if forced_cues:
                target = "forced"
                weight = 0.4
            else:
                target = "sexual"
                weight = 0.3
        elif action_type in intimate_actions:
            target = "intimate"
            weight = 0.2
        elif arousal > 0.3:
            target = "intimate"
            weight = 0.1
        else:
            target = "casual"
            weight = 0.15

        # 羞耻感让她倾向于否认sexual情境
        if shame > 0.7 and self.context_interpretation == "sexual":
            weight *= 0.5
            if trust < 0.7:
                target = "intimate"  # 告诉自己只是亲密不是色情

        # 平滑过渡：只有足够确定时才切换标签，避免单次噪声抖动。
        confidence_delta = weight * dt * 0.1
        self.context_confidence = max(0.3, min(1.0, self.context_confidence + confidence_delta))
        if target != self.context_interpretation:
            transition = min(1.0, max(0.0, confidence_delta))
            if transition > 0:
                self.context_interpretation = target
        # 主观亲密度
        intimacy_target = min(1.0, trust * 0.5 + arousal * 0.3 + pleasure * 0.3)
        self.perceived_intimacy += (intimacy_target - self.perceived_intimacy) * 0.05 * dt

    def _update_intent_perception(self, action_type: str, state, world, dt: float):
        """更新她对你意图的感知"""
        sexual_actions = {"touch_breast", "fondle_breast", "suck_nipple", "rub_clit",
                         "finger_clit", "cunnilingus", "rub_clit_through_panties",
                         "finger_insert_one", "finger_insert_two", "penetrate"}
        intimate_actions = {"kiss", "deep_kiss", "neck_kiss", "hug", "cuddle"}

        target_suspicion = self.intent_suspicion_level
        target_certainty = self.intent_certainty

        if action_type in sexual_actions:
            target_suspicion = 1.0
            target_certainty = min(1.0, self.intent_certainty + 0.3)
            self.perceived_intent = "sex"
            self.surprised = self.intent_suspicion_level < 0.3
            if self.surprised:
                self.surprise_level = 0.7
        elif action_type in intimate_actions and state.global_arousal > 0.3:
            target_suspicion = 0.7
            target_certainty = 0.5
            if self.intent_suspicion_level < 0.3:
                self.perceived_intent = "going_further"
        elif action_type == "tease" and state.global_arousal > 0.4:
            target_suspicion = 0.6
            if self.intent_suspicion_level < 0.2:
                self.perceived_intent = "just_teasing"

        # 信任高的话她更早察觉你的意图（你也不需要隐瞒）
        if state.relationship_trust > 0.8:
            target_certainty = min(1.0, target_certainty + 0.2)

        self.intent_suspicion_level += (target_suspicion - self.intent_suspicion_level) * 0.1 * dt
        self.intent_certainty += (target_certainty - self.intent_certainty) * 0.08 * dt
        self.surprise_level *= math.exp(-0.05 * dt)

    def _update_resistance_will(self, arousal: float, pleasure: float, shame: float,
                                 fear: float, trust: float, conflict: float, dt: float):
        """更新抵抗/接受意愿"""
        # 接受程度：信任+快感+arousal提升
        accept_target = trust * 0.4 + pleasure * 0.3 + arousal * 0.3
        if self.context_interpretation == "sexual" and trust > 0.6:
            accept_target += 0.1
        if self.taboo_thrill > 0.5:
            accept_target += 0.1
        self.acceptance_level += (min(1.0, accept_target) - self.acceptance_level) * 0.08 * dt

        # 主动抵抗：恐惧+低信任提升，高arousal/快感降低
        resist_target = 0.0
        if fear > 0.3:
            resist_target += fear * 0.8
        if shame > 0.8 and trust < 0.7:
            resist_target += shame * 0.3
        if trust < 0.3:
            resist_target += 0.3
        if arousal > 0.6:
            resist_target *= (1.0 - arousal * 0.5)  # 身体有反应时抵抗意志削弱
        if pleasure > 0.5:
            resist_target *= (1.0 - pleasure * 0.6)
        self.active_resistance_will += (min(1.0, resist_target) - self.active_resistance_will) * 0.1 * dt

        # 拒绝真诚度
        sincerity_target = self.active_resistance_will * 0.8
        if shame > 0.6 and trust > 0.7 and arousal > 0.5:
            sincerity_target = 0.2  # 害羞但其实愿意，所以是半推半就
        self.refusal_sincerity += (sincerity_target - self.refusal_sincerity) * 0.1 * dt

        # 半推半就程度：高shame+高arousal+中高信任
        token_target = 0.0
        if shame > 0.5 and arousal > 0.4 and 0.5 < trust < 0.9:
            token_target = min(1.0, shame * arousal * 1.5)
        self.token_resistance += (token_target - self.token_resistance) * 0.08 * dt

        # 犹豫度
        self.hesitation = max(0.0, min(1.0, conflict * 0.7 + shame * 0.3 - self.acceptance_level))

    def _update_self_perception(self, arousal: float, shame: float, pleasure: float,
                                 state, world, dt: float):
        """更新自我认知"""
        # 感觉自己淫荡：高arousal+高羞耻+有过主动反应
        slut_target = 0.0
        if arousal > 0.5 and shame > 0.5:
            slut_target = arousal * shame * 0.8
        if pleasure > 0.7 and shame > 0.6:
            slut_target += 0.2
        if state.emotion.suppressing_sounds == False and arousal > 0.7:
            slut_target += 0.2  # 意识到自己在发出声音
        self.perceived_sluttiness += (min(1.0, slut_target) - self.perceived_sluttiness) * 0.05 * dt

        # 自我意识
        selfcon_target = shame * 0.6 + self.perceived_sluttiness * 0.3
        if self._current_privacy < 0.5:
            selfcon_target += 0.3
        self.self_consciousness += (selfcon_target - self.self_consciousness) * 0.1 * dt

        # 意识到自己的声音
        if state.ans.breathing_rate > 25 and arousal > 0.5:
            self.aware_of_moans = True
        elif arousal < 0.2:
            self.aware_of_moans = False

        # 她认为你怎么看她
        if state.relationship_trust > 0.8 and pleasure > 0.5:
            self.perceived_partner_view = "loved"
        elif self.context_interpretation == "sexual" and state.relationship_trust < 0.6:
            self.perceived_partner_view = "as_sex_object"
        elif arousal > 0.4 and state.relationship_trust > 0.5:
            self.perceived_partner_view = "as_woman"
        else:
            self.perceived_partner_view = "as_sister"

    def _update_boundaries(self, action_type: str, state, dt: float):
        """更新边界意识"""
        boundary_map = {
            "hug": "personal_space",
            "kiss": "kiss_boundary",
            "deep_kiss": "romantic_boundary",
            "neck_kiss": "erogenous_zone_boundary",
            "touch_over_clothes": "body_touch_boundary",
            "touch_breast": "sexual_touch_boundary",
            "suck_nipple": "breast_sexual_boundary",
            "rub_clit_through_panties": "genital_over_clothes",
            "rub_clit": "direct_genital_boundary",
            "cunnilingus": "oral_sex_boundary",
            "finger_insert_one": "penetration_boundary",
            "penetrate": "sex_boundary",
        }
        b = boundary_map.get(action_type)
        if b and b not in self.boundaries_crossed:
            self.boundaries_crossed.append(b)
            self.surprised = True
            self.surprise_level = max(self.surprise_level, 0.5)

        # 逼近的边界
        approaching = []
        if "body_touch_boundary" in self.boundaries_crossed and "sexual_touch_boundary" not in self.boundaries_crossed:
            approaching.append("sexual_touch_boundary")
        if "sexual_touch_boundary" in self.boundaries_crossed and "direct_genital_boundary" not in self.boundaries_crossed:
            approaching.append("direct_genital_boundary")
        if "direct_genital_boundary" in self.boundaries_crossed and "penetration_boundary" not in self.boundaries_crossed:
            approaching.append("penetration_boundary")
        self.boundaries_approaching = approaching

        # 禁忌刺激感（禁果效应的主观感受）
        taboo_target = 0.0
        if len(self.boundaries_crossed) >= 2 and "siblings" in self.taboo_labels:
            taboo_target = min(1.0, 0.3 + len(self.boundaries_crossed) * 0.1)
        if state.global_arousal > 0.5 and "siblings" in self.taboo_labels:
            taboo_target += 0.2
        if state.emotion.blend.get("shame", 0) > 0.3:
            taboo_target += 0.1
        self.taboo_thrill += (min(1.0, taboo_target) - self.taboo_thrill) * 0.05 * dt

    def _update_immersion(self, arousal: float, pleasure: float, ans, state, dt: float):
        """更新沉浸/解离程度"""
        # 沉浸度随快感和arousal上升
        imm_target = pleasure * 0.5 + arousal * 0.5
        if state.orgasm.phase in ["plateau", "orgasm"]:
            imm_target = min(1.0, imm_target + 0.3)
        self.immersion += (imm_target - self.immersion) * 0.06 * dt

        # 大脑空白度
        blank_target = 0.0
        if arousal > 0.8:
            blank_target = (arousal - 0.8) * 5.0
        if state.orgasm.phase == "orgasm":
            blank_target = 0.9 + state.orgasm.intensity * 0.1
        self.mind_blank += (min(1.0, blank_target) - self.mind_blank) * 0.1 * dt

        # 解离（不真实感）
        diss_target = 0.0
        if state.orgasm.phase == "orgasm" and state.orgasm.intensity > 0.8:
            diss_target = 0.7
        if self.mind_blank > 0.8:
            diss_target = 0.5
        self.dissociation += (diss_target - self.dissociation) * 0.05 * dt

        self.is_flying = self.mind_blank > 0.6 or state.orgasm.phase == "orgasm"

        # 内心OS
        if self.mind_blank > 0.8:
            self.inner_voice = "blank"
        elif self.hesitation > 0.7:
            self.inner_voice = "conflicted"
        elif self.taboo_thrill > 0.6 and self.acceptance_level > 0.5:
            self.inner_voice = "guilty_pleasure"
        elif self.token_resistance > 0.6:
            self.inner_voice = "saying_no_wanting_yes"
        elif self.acceptance_level > 0.7:
            self.inner_voice = "surrender"
        elif self.active_resistance_will >= 0.8 or self.refusal_sincerity >= 0.75:
            self.inner_voice = "stop_please"
        elif self.hesitation > 0.5 or self.active_resistance_will >= 0.55 or self.refusal_sincerity >= 0.5:
            self.inner_voice = "needs_slow_down"
        elif self.surprised:
            self.inner_voice = "whats_happening"
        else:
            self.inner_voice = ""

    def get_dominant_state(self) -> str:
        """获取当前主导主观状态（给输出层用）"""
        if self.mind_blank > 0.8:
            return "mind_blank"
        if self.is_flying and self.dissociation > 0.5:
            return "dissociating"
        if self.active_resistance_will >= 0.8 or self.refusal_sincerity >= 0.75:
            return "resisting"
        if self.active_resistance_will >= 0.55 or self.refusal_sincerity >= 0.5:
            return "setting_boundary"
        if self.token_resistance > 0.5:
            return "token_resistance"
        if self.hesitation > 0.6:
            return "hesitant"
        if self.taboo_thrill > 0.5 and self.acceptance_level > 0.5:
            return "guilty_pleasure"
        if self.acceptance_level > 0.7:
            return "accepting"
        if self.surprised:
            return "surprised"
        if self.self_consciousness > 0.6:
            return "self_conscious"
        if self.context_interpretation == "sexual":
            return "in_the_mood"
        if self.context_interpretation == "intimate":
            return "affectionate"
        return "casual"

    def to_dict(self) -> dict:
        return {
            "context_interpretation": self.context_interpretation,
            "context_confidence": self.context_confidence,
            "perceived_intimacy": self.perceived_intimacy,
            "perceived_intent": self.perceived_intent,
            "intent_suspicion_level": self.intent_suspicion_level,
            "intent_certainty": self.intent_certainty,
            "active_resistance_will": self.active_resistance_will,
            "refusal_sincerity": self.refusal_sincerity,
            "acceptance_level": self.acceptance_level,
            "hesitation": self.hesitation,
            "token_resistance": self.token_resistance,
            "perceived_sluttiness": self.perceived_sluttiness,
            "self_perceived_skill": self.self_perceived_skill,
            "perceived_partner_view": self.perceived_partner_view,
            "self_consciousness": self.self_consciousness,
            "aware_of_moans": self.aware_of_moans,
            "boundaries_crossed": self.boundaries_crossed.copy(),
            "boundaries_approaching": self.boundaries_approaching.copy(),
            "taboo_labels": self.taboo_labels.copy(),
            "taboo_thrill": self.taboo_thrill,
            "immersion": self.immersion,
            "dissociation": self.dissociation,
            "mind_blank": self.mind_blank,
            "is_flying": self.is_flying,
            "expected_next": self.expected_next,
            "fear_next": self.fear_next,
            "secretly_want": self.secretly_want,
            "surprised": self.surprised,
            "surprise_level": self.surprise_level,
            "aware_of_arousal": self.aware_of_arousal,
            "conflicted_about": self.conflicted_about,
            "inner_voice": self.inner_voice,
            "last_update_time": self.last_update_time,
            "dominant_state": self.get_dominant_state(),
        }
