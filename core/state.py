"""
整合的妹妹状态类
包含身体、神经、ANS、情绪、技能、高潮、伙伴感知等所有Layer 0状态
"""
import json
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from .body.parts import (
    BodyRegion, build_default_body, compute_global_arousal,
    compute_readiness, compute_total_wetness, PartnerState,
)
from .body.neural import (
    build_neural_nodes, connect_nodes, stimulate_node, propagate,
    apply_cognitive_inhibition,
)
from .ans import ANSState
from .emotion import EmotionState
from .skills import default_skills, calc_effective_mastery, practice_skill, NoveltySystem, BodyAwareness
from .orgasm import OrgasmSystem
from .sensitivity import SensitivityModulators, effective_sensitivity
from .mind_state import MindState
from world.clothing import ClothingSystem
from core.physical_engine import StaminaState, configure_stamina, recover_stamina



# === 日志补接线（详细排障） ===
try:
    from app.logger import debug, info, success, warning, error, trace
except Exception:
    debug = info = success = warning = error = trace = lambda *a, **k: None

class CharacterState:
    """完整角色状态（原SisterState）"""
    def __init__(self):
        # === Layer 0a: 身体部位 ===
        self.body: Dict[str, Any] = build_default_body()

        # === 神经网络 ===
        self.nodes = build_neural_nodes(self.body)
        self.skills = default_skills()
        connect_nodes(self.nodes, skills=self.skills)

        # === Layer 0b: ANS ===
        self.ans = ANSState()

        # === Layer 0c: 情绪 ===
        self.emotion = EmotionState()

        # === Layer 0d: 技能/经验 ===
        self.novelty = NoveltySystem()
        self.body_awareness = BodyAwareness()
        self.relationship_trust: float = 0.75  # 基础信任度

        # === Layer 0e: 动态敏感度调制器 ===
        self.sensitivity_mods = SensitivityModulators()

        # === 高潮系统 ===
        self.orgasm = OrgasmSystem()

        # === 伙伴状态感知 ===
        self.partner = PartnerState()

        # === 当前刺激信息（近期各部位刺激强度记录）===
        self.current_stimulation = {
            "active": False,
            "adequate_intensity": False,
            "current_action": None,
            "current_targets": [],
            "location": "bedroom",
            "recent_stim_strength": {},  # {part_name: 近期累计刺激强度}
        }

        # === 对话历史（最近N轮）===
        self.conversation_history: List[Dict[str, str]] = []

        # === 人格参数 ===
        self.personality = {
            "extraversion": 0.4,
            "shyness_base": 0.6,
            "spoiled": 0.7,
            "tsundere": 0.5,
        }

        # === 时间 ===
        self.last_update: datetime = datetime.now()
        self.time_elapsed_since_last: float = 0.0
        self.sim_time: float = 0.0  # 游戏内累计时间（秒）

        # === 全局状态 ===
        self.global_arousal: float = 0.0
        self._world_privacy_level: float = 0.9  # 由world系统更新

        # === Layer 0.5: 主观认知层 ===
        self.mind = MindState()

        # === 衣物系统 ===
        self.clothing = ClothingSystem("summer_home")

        # === 情感记忆系统（core/memory 已移除，置空占位）===
        self.memory = None

        # === 客观体力与疲劳 ===
        self.stamina = StaminaState()

    # === 刺激注入 ===
    def apply_stimulation(self, part_name: str, intensity: float, duration: float = 1.0,
                         action_type: str = "touch", through_clothes: bool = None):
        """对某个部位施加刺激（使用动态敏感度调制+衣物阻碍计算）"""
        debug("身体引擎", f"刺激注入: part={part_name}, intensity={intensity:.3f}, duration={duration:.2f}s, action={action_type}, 隔衣={through_clothes}")
        raw_intensity = intensity  # 缩放前的动作原始强度（唤起基准用）
        # 计算衣物可接触度
        access = self.clothing.get_accessibility(part_name)
        friction = self.clothing.get_friction_mod(part_name)
        # 如果是隔着衣物的动作，通过friction获得额外质感，但直接刺激强度降低
        if through_clothes is None:
            through_clothes = access < 0.5
        if through_clothes:
            intensity *= 0.5  # 隔衣刺激强度减半
            intensity *= (1 + friction)  # 但摩擦带来额外刺激
        else:
            intensity *= access  # 直接接触按可接触度缩放

        # 获取有效敏感度
        base_sens = 1.0
        for region in self.body.values():
            if isinstance(region, BodyRegion) and part_name in region.sub_parts:
                base_sens = region.get(part_name).sensitivity
                break
        eff_sens = effective_sensitivity(
            part_name, base_sens, self, None, self.novelty, action_type
        )
        actual_intensity = intensity * eff_sens

        # 高潮后阴蒂过度敏感处理
        clit_pain = False
        gf = self.body["genital_female"]
        if part_name == "clitoris" and gf.get("clitoris").get("oversensitive_post"):
            pain = gf.get("clitoris").get("pain_overstim", 0)
            if pain > 0.5:
                clit_pain = True
                self.emotion.blend["pain"] = min(1.0, self.emotion.blend.get("pain", 0) + pain * 0.5)
                actual_intensity *= (1 - pain * 0.6)
            else:
                actual_intensity *= (1 - pain * 0.3)

        # === 后穴(anus)专门调制：润滑/适应/紧致/疼痛 ===
        if part_name == "anus":
            anus = gf.get("anus")
            lub = anus.get("lubrication")
            adapt = anus.get("adaptation")
            pain = anus.get("anal_pain")
            if lub < 0.35:
                # 润滑不足：痛感上升、快感打折、适应很慢
                pain_inc = max(0.0, (0.35 - lub)) * 0.8 * intensity
                anus.add("anal_pain", pain_inc, max_val=1.0)
                self.emotion.blend["pain"] = min(1.0, self.emotion.blend.get("pain", 0) + pain_inc * 0.5)
                actual_intensity *= 0.35
                anus.add("adaptation", intensity * 0.05, max_val=0.3)
                anus.add("lubrication", -intensity * 0.04, min_val=0.0)
            else:
                # 有润滑：适应上升、紧致下降、肛内快感累积
                anus.add("adaptation", intensity * 0.16, max_val=1.0)
                anus.set("tightness", max(0.15, anus.get("tightness") - intensity * 0.06))
                anus.add("anal_pleasure", actual_intensity * 0.12, max_val=1.0)
                anus.add("lubrication", -intensity * 0.02, min_val=0.0)
            # 已适应后疼痛渐消
            if adapt > 0.55:
                anus.add("anal_pain", -0.06, min_val=0.0)
            # 润滑不足时也会消耗，需反复用油补充

        # 通知敏感度调制器刺激变化
        self.sensitivity_mods.on_stimulation_change(part_name)

        # 认知调制
        cog_mod = self.emotion.get_cognitive_mod(part_name)
        stimulate_node(self.nodes, part_name, actual_intensity, duration, cog_mod)

        # 记录近期刺激强度（给高潮类型选择用）
        recent = self.current_stimulation["recent_stim_strength"]
        recent[part_name] = recent.get(part_name, 0) + actual_intensity * duration

        self.current_stimulation["active"] = True
        self.current_stimulation["adequate_intensity"] = self.current_stimulation.get(
            "adequate_intensity", False) or actual_intensity > 0.3
        self.current_stimulation["current_action"] = action_type
        if part_name not in self.current_stimulation["current_targets"]:
            self.current_stimulation["current_targets"].append(part_name)

        # === 性敏感部位刺激直接耦合 ANS 全局唤起 ===
        # 原路径靠"部位arousal → 局部加权聚合 → ANS追踪"层层衰减，单次非生殖器
        # 刺激（揉胸/吻脖/抚摸）几乎不抬全局唤起，模型永远看到 arousal≈0/平静而弱化输出。
        # 此处以动作本身强度（intensity）为基准直接抬升 ANS 全局信号（衣物只打折、
        # 敏感度/可接触度只影响局部触感不影响唤起基调），揉胸/抚摸一两轮即可到可感水平。
        sex_sensitive = {
            "clitoris": 0.25, "vaginal_canal": 0.25, "g_spot": 0.25,
            "vaginal_vestibule": 0.22, "cervix": 0.20, "anus": 0.25,
            "nipple_left": 0.14, "nipple_right": 0.14,
            "breast_left": 0.10, "breast_right": 0.10,
            "inner_labia": 0.22, "inner_thighs": 0.10,
            "neck": 0.09, "ear": 0.09, "back": 0.07, "waist": 0.07,
            "hip": 0.09, "shoulder": 0.06, "spine": 0.07,
        }
        if raw_intensity > 0.01 and part_name in sex_sensitive:
            _gain = sex_sensitive[part_name]
            if through_clothes:
                _gain *= 0.55  # 隔衣刺激对唤起提升打折（仍保留质感）
            _boost = raw_intensity * max(0.5, duration) * _gain
            self.ans.arousal_global = min(1.0, self.ans.arousal_global + _boost)

        debug("身体引擎", f"刺激完成: part={part_name}, 实际强度={actual_intensity:.3f}, 敏感度={eff_sens:.2f}, arousal={self.global_arousal:.2f}, 阴蒂痛={clit_pain}")
        return clit_pain

    def apply_cognitive_shock(self, shock_type: str):
        """情绪冲击（中断）"""
        self.emotion.trigger_interrupt(shock_type, self.sim_time)
        if shock_type == "mention_other_girl":
            self.ans.arousal_global *= 0.1
        # 新鲜感记录
        if hasattr(self, 'novelty'):
            self.novelty.register_interrupt(shock_type)

    def set_world_privacy(self, level: float):
        """由world系统设置当前隐私度"""
        self._world_privacy_level = max(0.0, min(1.0, level))

    # === 时间推进（神经传播+状态更新）===
    def tick(self, dt: float = 0.8, world=None):
        """推进一个时间步（神经传播+所有系统更新）"""
        debug("身体引擎", f"状态tick: dt={dt:.2f}s, sim_time={self.sim_time + dt:.1f}s, arousal={self.global_arousal:.3f}, 心率={self.ans.heart_rate}")
        if world is not None:
            self.set_world_privacy(world.get_privacy_level(self) if hasattr(world, "get_privacy_level") else world.privacy_level)
        self.sim_time += dt
        recover_stamina(self.stamina, dt, sleeping=bool(world and getattr(world, "position_detail", "") == "sleeping"))

        # 1. 认知抑制注入（恐惧/受伤/羞耻→抑制性通路）
        apply_cognitive_inhibition(self.nodes, self.emotion, dt)

        # 2. 神经传播（根据dt调整迭代次数）
        def cog_mod_fn(node):
            return self.emotion.get_cognitive_mod(node.name)
        iters = max(6, int(dt / 0.08))
        fired = propagate(self.nodes, iterations=iters, dt=0.08, cognitive_mod_fn=cog_mod_fn)

        # 3. 用发放结果更新身体部位arousal
        for region in self.body.values():
            if not isinstance(region, BodyRegion):
                continue
            for sp in region.all_parts():
                if sp.name in fired:
                    f_out = fired[sp.name]
                    sp.arousal = min(1.0, sp.arousal + f_out * 0.6)
                sp.tick_decay(dt)

        # 4. 敏感度调制器tick
        self.sensitivity_mods.tick(dt)

        # 5. 更新全局arousal（加入禁果效应）
        self.global_arousal = compute_global_arousal(self.body, self.ans)
        forbidden_bonus = self.emotion.get_forbidden_bonus()
        self.global_arousal = min(1.0, self.global_arousal * (1 + forbidden_bonus * 0.5))
        self.body["__global_arousal__"] = self.global_arousal

        # 6. 同步ANS arousal_global
        self.ans.arousal_global = max(self.ans.arousal_global, self.global_arousal * 0.8)

        # 7. 润滑分泌
        if self.ans.arousal_global > 0.3:
            gate = self.emotion.cognitive_gate
            wet_rate = 0.005 * dt * self.ans.arousal_global * gate
            vcanal = self.body["genital_female"].get("vaginal_canal")
            vcanal.extra["wetness"] = min(1.0, vcanal.get("wetness") + wet_rate)
            vest = self.body["genital_female"].get("vaginal_vestibule")
            vest.extra["wetness"] = min(1.0, vest.get("wetness") + wet_rate * 0.8)
            if self.ans.arousal_global > 0.7:
                vcanal.extra["tent_lub"] = min(1.0, vcanal.get("tent_lub") + 0.003 * dt * gate)

        # 8. 充血/勃起反应
        for pname in ["nipple_left", "nipple_right", "clitoris"]:
            part = None
            for region in self.body.values():
                if not isinstance(region, BodyRegion):
                    continue
                if pname in region.sub_parts:
                    part = region.get(pname)
                    break
            if part and part.arousal > 0.4:
                part.extra["congestion"] = min(1.0, part.get("congestion") + 0.01 * dt)
                if "erection" in part.extra:
                    part.extra["erection"] = min(1.0, part.get("erection") + 0.02 * dt)

        # 9. ANS更新（传入禁果加成）
        stim_act = self.current_stimulation["active"]
        stim_adeq = self.current_stimulation["adequate_intensity"]
        self.ans.update_from_firing(fired, dt, self.orgasm.phase,
                                    forbidden_bonus=forbidden_bonus)
        fatigue = max(0.0, min(1.0, float(getattr(self.stamina, "fatigue", 0.0))))
        emotional_load = max(
            float(self.emotion.blend.get("fear", 0.0)),
            float(self.emotion.blend.get("anxiety", 0.0)),
            float(self.emotion.blend.get("anticipation", 0.0)),
        )
        fatigue_hr_offset = -fatigue * 8.0
        emotional_hr_offset = emotional_load * 12.0
        if self.orgasm.phase == "orgasm":
            target_hr = max(155.0, self.ans.heart_rate_base + 80.0 + emotional_hr_offset)
        else:
            target_hr = self.ans.heart_rate_base + self.ans.arousal_global * 70.0 + fatigue_hr_offset + emotional_hr_offset
        self.ans.heart_rate += (target_hr - self.ans.heart_rate) * min(1.0, 0.18 * dt)
        self.ans.heart_rate = max(45.0, min(190.0, self.ans.heart_rate))
        # 平台期正反馈：性紧张持续累积，arousal不会掉
        if self.orgasm.phase == "plateau" and stim_act and stim_adeq:
            self.ans.arousal_global = min(1.0, self.ans.arousal_global + 0.008 * dt)
            self.global_arousal = min(1.0, self.global_arousal + 0.006 * dt)

        # 10. 高潮系统更新（传入active_stims）
        clit_pain = self.body["genital_female"].get("clitoris").get("pain_overstim", 0) > 0.5
        active_stims = self.current_stimulation.get("recent_stim_strength", {})
        event = self.orgasm.update(
            self.global_arousal, self.ans, self.emotion,
            stim_act, stim_adeq, dt, self.skills,
            active_stims=active_stims, on_clit_pain=clit_pain
        )
        # 高潮后arousal同步
        if self.orgasm.phase == "resolution":
            self.global_arousal = max(self.global_arousal, self.orgasm._post_arousal)
            self.ans.arousal_global = max(self.ans.arousal_global, self.orgasm._post_arousal * 0.9)
            self.body["__global_arousal__"] = self.global_arousal
        if event == "orgasm_start":
            self._on_orgasm_start()
        elif event == "orgasm_end":
            self._on_orgasm_end()

        # 11. 情绪衰减 + 重算认知闸门
        self.emotion.decay(dt)
        self.emotion.compute_cognitive_gate(self.relationship_trust, self._world_privacy_level)

        # 11b. 主观认知层更新（需要知道当前动作）
        cur_act = self.current_stimulation.get("current_action") or ""
        self.mind.update(self, world, cur_act, dt)

        # 12. 重置当前刺激（保留recent_stim_strength缓慢衰减）
        self.current_stimulation["active"] = False
        self.current_stimulation["current_targets"] = []
        self.current_stimulation["adequate_intensity"] = False
        self.current_stimulation["current_action"] = None
        # 近期刺激强度衰减
        recent = self.current_stimulation["recent_stim_strength"]
        for k in list(recent.keys()):
            recent[k] *= math.exp(-0.05 * dt)
            if recent[k] < 0.05:
                del recent[k]

    def _on_orgasm_start(self):
        """高潮开始时的状态变化"""
        success("高潮", f"高潮开始: type={self.orgasm.orgasm_type}, intensity={self.orgasm.orgasm_intensity:.2f}")
        self.emotion.blend["pleasure"] = 1.0
        self.emotion.blend["overwhelm"] = min(1.0, self.emotion.blend.get("overwhelm", 0) + 0.8)
        self.emotion.blend["loss_of_control"] = min(1.0, self.emotion.blend.get("loss_of_control", 0) + 0.9)
        self.emotion.blend["inevitability"] = 1.0
        self.emotion.suppressing_sounds = False
        self.ans.dizzy = 0.8
        self.ans.vision_blur = 0.7
        self.ans.ears_ringing = 0.6
        # 记录技能练习
        otype = self.orgasm.orgasm_type
        if otype in ["clitoral", "blended"]:
            practice_skill(self.skills, "clit_stimulation_receiving", 0.8)
        if otype in ["vaginal", "blended", "cervical"]:
            for s in ["missionary", "cowgirl", "doggy"]:
                if s in self.skills and calc_effective_mastery(self.skills[s]) > 0.1:
                    practice_skill(self.skills, s, 0.5)
        self.body_awareness.increase_from_experience(0.05)
        # 记忆标记（后面memory系统用）
        self._orgasm_just_happened = True
        self._last_orgasm_type = otype

    def _on_orgasm_end(self):
        """高潮结束"""
        success("高潮", f"高潮结束: type={self.orgasm.orgasm_type}, 累计次数={self.orgasm.orgasms_so_far}")
        self.emotion.blend["satisfaction"] = 1.0
        self.emotion.blend["overwhelm"] *= 0.5
        self.emotion.blend["loss_of_control"] *= 0.3
        self.emotion.blend["inevitability"] = 0.0
        self._orgasm_just_happened = False
        # 高潮后阴蒂过度敏感设置
        gf = self.body["genital_female"]
        gf.get("clitoris").set("oversensitive_post", True)
        if self.orgasm.orgasm_intensity > 0.8:
            gf.get("clitoris").set("pain_overstim", 0.5)

    # === 对话历史 ===
    def add_turn(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 30:
            self.conversation_history = self.conversation_history[-20:]

    def get_recent_history(self, n: int = 10) -> List[Dict[str, str]]:
        return self.conversation_history[-n:]

    # === 状态摘要（给模型用的自然语言描述）===
    def get_state_summary(self) -> str:
        """生成当前身体/情绪状态的自然语言摘要，用于prompt"""
        parts = []
        e = self.emotion
        a = self.ans
        arousal = self.global_arousal
        gf = self.body["genital_female"]

        # 中断/震惊状态优先
        if e.interrupt_triggered:
            interrupt_descs = {
                "parents_come_home": "吓得僵住了，一动不敢动",
                "door_knock": "吓得屏住呼吸",
                "mention_other_girl": "心里一沉，快感瞬间退了大半",
                "phone_ring": "吓了一跳",
                "heard_sound": "警觉地听着外面的动静",
            }
            desc = interrupt_descs.get(e.interrupt_type, "吃了一惊")
            return desc + "。"

        # 情绪
        emo_desc = []
        if e.blend.get("shame", 0) > 0.5:
            emo_desc.append("害羞得不行")
        if e.conflict_level > 0.6:
            emo_desc.append("心里又羞又乱但身体忍不住有反应")
        if e.blend.get("trust", 0) > 0.8 and arousal > 0.3:
            emo_desc.append("完全信任着你")
        if e.blend.get("pleasure", 0) > 0.6:
            emo_desc.append("身体感受到强烈快感")
        if e.blend.get("jealousy", 0) > 0.5:
            emo_desc.append("吃醋了有点小脾气")
        if e.blend.get("hurt", 0) > 0.5:
            emo_desc.append("心里有点难过委屈")
        if e.blend.get("fear", 0) > 0.5:
            emo_desc.append("紧张又害怕")
        if e.blend.get("sleepy", 0) > 0.6:
            emo_desc.append("困得睁不开眼")
        if e.blend.get("happiness", 0) > 0.6 and arousal < 0.2:
            emo_desc.append("心情很好")
        if e.blend.get("love", 0) > 0.7:
            emo_desc.append("心里满是喜欢")
        if e.primary == "calm" and not emo_desc:
            emo_desc.append("心情平静")

        # 生理状态
        phys_desc = []
        if arousal > 0.2:
            if a.heart_rate > 100:
                phys_desc.append("心跳有点快")
            if a.breathing_rate > 20:
                phys_desc.append("呼吸变急促了")
            if a.skin_flush > 0.4:
                phys_desc.append("脸颊和胸口泛着粉")
            if a.tremor > 0.3:
                phys_desc.append("身体微微发颤")
            if e.suppressing_sounds:
                phys_desc.append("咬着唇忍住声音")
            if e.forbidden_arousal_bonus > 0.08:
                phys_desc.append("因为怕被发现身体反而更敏感了")
        if arousal > 0.5:
            wet = compute_total_wetness(self.body)
            if wet > 0.5:
                phys_desc.append("下面已经湿了")
            nip_l = self.body["chest"].get("nipple_left").get("erection")
            nip_r = self.body["chest"].get("nipple_right").get("erection")
            if (nip_l + nip_r) / 2 > 0.5:
                phys_desc.append("乳尖硬了")
            if self.body["torso_core"].get("abdomen").get("butterflies") > 0.3:
                phys_desc.append("小腹发紧")
        if arousal > 0.7:
            if self.body["legs_feet"].get("thighs").get("tremor") > 0.5:
                phys_desc.append("大腿控制不住地轻颤")
            if a.muscle_tone > 0.6:
                phys_desc.append("身子发软")

        # 后穴状态
        anus = gf.get("anus")
        if anus and (anus.get("fullness") > 0.3 or anus.get("anal_pleasure") > 0.3 or anus.get("lubrication") < 0.35):
            lub = anus.get("lubrication")
            adapt = anus.get("adaptation")
            pain = anus.get("anal_pain")
            if pain > 0.5:
                phys_desc.append("后面又疼又胀")
            elif lub < 0.35:
                phys_desc.append("后面还很紧很涩")
            elif adapt < 0.5:
                phys_desc.append("后穴紧致地含着你，还有点酸胀")
            else:
                phys_desc.append("后穴已经被肏软了，又紧又热地绞着你")

        # 高潮阶段
        if self.orgasm.phase == "plateau":
            phys_desc.append("快感累积着快要到顶点了")
        elif self.orgasm.phase == "orgasm":
            type_descs = {
                "clitoral": "阴蒂高潮中",
                "vaginal": "阴道深处高潮中",
                "blended": "混合高潮中，全身都在痉挛",
                "cervical": "宫颈被顶到的深处高潮",
                "anal": "后穴深处被肏到高潮，后穴紧紧绞着",
                "multiple_chain": "一波接一波的高潮根本停不下来",
            }
            phys_desc.append(type_descs.get(self.orgasm.orgasm_type, "正在高潮中"))
            phys_desc.append("身体不受控制地痉挛收缩")
        elif self.orgasm.phase == "resolution" and self.orgasm.time_since_last < 60:
            phys_desc.append("高潮余韵还没过去，身子还在轻轻发颤")

        # 伙伴感知
        part_desc = []
        if self.partner.inside_her:
            if self.partner.pulsing_inside or self.partner.ejaculation_phase == "happening":
                part_desc.append("能感觉到他在里面射了")
            elif self.partner.thrusting:
                speed = "慢慢" if self.partner.thrust_speed < 0.5 else "快速"
                depth = "浅浅" if self.partner.thrust_depth < 0.5 else "深深"
                part_desc.append(f"他在{depth}{speed}地律动")

        summary = "。".join(emo_desc + phys_desc + part_desc)
        if not summary.endswith("。"):
            summary += "。"
        return summary

    def get_wetness(self) -> float:
        """获取当前润滑度（0-1）"""
        return compute_total_wetness(self.body)

    # === 持久化 ===
    def to_dict(self) -> Dict:
        """完整序列化所有状态"""
        def serialize_body():
            body_data = {}
            for rname, region in self.body.items():
                if isinstance(region, BodyRegion):
                    body_data[rname] = region.to_dict()
                else:
                    body_data[rname] = region
            return body_data
        return {
            "version": 2,
            "sim_time": self.sim_time,
            "relationship_trust": self.relationship_trust,
            "global_arousal": self.global_arousal,
            "stamina": self.stamina.to_dict(),
            "_world_privacy_level": self._world_privacy_level,
            "personality": self.personality,
            "ans": self.ans.to_dict(),
            "emotion": {
                "primary": self.emotion.primary,
                "blend": dict(self.emotion.blend),
                "cognitive_gate": self.emotion.cognitive_gate,
                "conflict_level": self.emotion.conflict_level,
                "forbidden_arousal_bonus": self.emotion.forbidden_arousal_bonus,
                "suppressing_sounds": self.emotion.suppressing_sounds,
                "suppressing_movements": self.emotion.suppressing_movements,
                "interrupt_triggered": self.emotion.interrupt_triggered,
                "interrupt_type": self.emotion.interrupt_type,
                "attention_focus": list(self.emotion.attention_focus),
            },
            "orgasm": {
                "phase": self.orgasm.phase,
                "orgasm_type": self.orgasm.orgasm_type,
                "orgasm_intensity": self.orgasm.orgasm_intensity,
                "orgasms_so_far": self.orgasm.orgasms_so_far,
                "plateau_timer": self.orgasm.plateau_timer,
                "orgasm_duration": self.orgasm.orgasm_duration,
                "time_since_last": self.orgasm.time_since_last,
                "orgasm_threshold": self.orgasm.orgasm_threshold,
                "_post_arousal": getattr(self.orgasm, '_post_arousal', 0.3),
            },
            "sensitivity_mods": {
                "tired": self.sensitivity_mods.tired,
                "sore_genitals": self.sensitivity_mods.sore_genitals,
                "horny_morning": self.sensitivity_mods.horny_morning,
                "adaptation": dict(self.sensitivity_mods.adaptation),
            },
            "partner": self.partner.to_dict(),
            "novelty": self.novelty.to_dict() if hasattr(self.novelty, 'to_dict') else {},
            "body": serialize_body(),
            "conversation_history": self.conversation_history[-20:],
            "saved_at": datetime.now().isoformat(),
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: Path):
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ver = data.get("version", 1)
        if ver < 2:
            # 旧版本简单加载
            for k, v in data.get("ans", {}).items():
                if hasattr(self.ans, k):
                    setattr(self.ans, k, v)
            self.emotion.primary = data.get("emotion", {}).get("primary", "calm")
            self.emotion.blend.update(data.get("emotion", {}).get("blend", {}))
            self.emotion.suppressing_sounds = data.get("emotion", {}).get("suppressing_sounds", False)
            self.relationship_trust = data.get("relationship_trust", 0.75)
            self.orgasm.phase = data.get("orgasm_phase", "excitement")
            self.orgasm.orgasms_so_far = data.get("orgasms_so_far", 0)
            self.conversation_history = data.get("conversation_history", [])
            return
        # v2完整加载
        self.sim_time = data.get("sim_time", 0.0)
        self.relationship_trust = data.get("relationship_trust", 0.75)
        self.global_arousal = data.get("global_arousal", 0.0)
        if "stamina" in data:
            self.stamina = StaminaState.from_dict(data["stamina"])
        self._world_privacy_level = data.get("_world_privacy_level", 0.9)
        self.personality.update(data.get("personality", {}))
        # ANS
        for k, v in data.get("ans", {}).items():
            if hasattr(self.ans, k):
                setattr(self.ans, k, v)
        # Emotion
        em = data.get("emotion", {})
        self.emotion.primary = em.get("primary", "calm")
        self.emotion.blend.update(em.get("blend", {}))
        self.emotion.cognitive_gate = em.get("cognitive_gate", 0.8)
        self.emotion.conflict_level = em.get("conflict_level", 0.0)
        self.emotion.forbidden_arousal_bonus = em.get("forbidden_arousal_bonus", 0.0)
        self.emotion.suppressing_sounds = em.get("suppressing_sounds", False)
        self.emotion.suppressing_movements = em.get("suppressing_movements", False)
        self.emotion.interrupt_triggered = em.get("interrupt_triggered", False)
        self.emotion.interrupt_type = em.get("interrupt_type", "")
        self.emotion.attention_focus = em.get("attention_focus", ["partner"])
        # Orgasm
        og = data.get("orgasm", {})
        self.orgasm.phase = og.get("phase", "excitement")
        self.orgasm.orgasm_type = og.get("orgasm_type", None)
        self.orgasm.orgasm_intensity = og.get("orgasm_intensity", 0.0)
        self.orgasm.orgasms_so_far = og.get("orgasms_so_far", 0)
        self.orgasm.plateau_timer = og.get("plateau_timer", 0.0)
        self.orgasm.orgasm_duration = og.get("orgasm_duration", 0.0)
        self.orgasm.time_since_last = og.get("time_since_last", 999.0)
        self.orgasm.orgasm_threshold = og.get("orgasm_threshold", 0.92)
        self.orgasm._post_arousal = og.get("_post_arousal", 0.3)
        # Sensitivity mods
        sm = data.get("sensitivity_mods", {})
        self.sensitivity_mods.tired = sm.get("tired", 0.0)
        self.sensitivity_mods.sore_genitals = sm.get("sore_genitals", 0.0)
        self.sensitivity_mods.horny_morning = sm.get("horny_morning", 0.0)
        self.sensitivity_mods.adaptation = sm.get("adaptation", {})
        # History
        self.conversation_history = data.get("conversation_history", [])

    # === 角色模板应用 ===
    def apply_character_template(self, character):
        """
        应用角色模板参数（body/mind/emotional参数）。
        character可以是app.models.CharacterBase对象或包含对应字段的dict。
        """
        self.character_id = getattr(character, 'id', 'unknown')

        # body_params
        bp = getattr(character, 'body_params', None)
        if bp is None and isinstance(character, dict):
            bp = character.get('body_params', {})
        if bp:
            def g(name, default):
                v = bp.get(name, default) if isinstance(bp, dict) else getattr(bp, name, default)
                return v
            self.orgasm.orgasm_threshold = g('orgasm_threshold', 0.82)
            self.orgasm._post_arousal = max(0.1, 0.3 / g('arousal_decay_rate', 1.0))
            hr_base = int(g('heart_rate_base', 72))
            self.ans.heart_rate = float(hr_base)
            self.ans.heart_rate_base = float(hr_base)
            self._apply_sensitivity_multipliers(
                clit=g('clitoral_sensitivity', 1.0),
                breast=g('breast_sensitivity', 1.0),
                gspot=g('g_spot_sensitivity', 0.8),
                skin=g('skin_sensitivity', 1.0),
                anal=g('anal_sensitivity', 0.3),
                base=g('sensitivity_base', 1.0),
            )
            if not g('multiple_orgasm_capable', True):
                self.orgasm.multiple_possible = False
            self.orgasm.refractory_base = g('refractory_period', 300)
            appearance = getattr(character, 'appearance', None)
            if appearance is None and isinstance(character, dict):
                appearance = character.get('appearance')
            self.stamina = configure_stamina(bp, appearance)

        # mind_params
        mp = getattr(character, 'mind_params', None)
        if mp is None and isinstance(character, dict):
            mp = character.get('mind_params', {})
        if mp:
            def g(name, default):
                v = mp.get(name, default) if isinstance(mp, dict) else getattr(mp, name, default)
                return v
            self.mind.acceptance_level = max(0.0, 1.0 - g('initial_resistance_sexual', 0.9))
            self.personality['shyness_base'] = g('shyness_base', 0.5)
            self.personality['moral_inhibition'] = g('moral_inhibition', 0.5)
            self.personality['jealousy_tendency'] = g('jealousy_tendency', 0.4)
            self.personality['initiative_base'] = g('initiative_base', 0.2)
            self.personality['teasing_tendency'] = g('teasing_tendency', 0.1)
            self.personality['cry_tendency'] = g('cry_tendency', 0.4)
            self.personality['emotional_volatility'] = g('emotional_volatility', 0.3)
            self.personality['vocal_suppression'] = g('vocal_suppression_tendency', 0.6)
            self.emotion.suppressing_sounds = g('vocal_suppression_tendency', 0.6) > 0.5
            self.relationship_trust = g('_initial_trust', getattr(character, 'initial_trust', 0.4))

        # emotional_params
        ep = getattr(character, 'emotional_params', None)
        if ep is None and isinstance(character, dict):
            ep = character.get('emotional_params', {})
        if ep:
            def g(name, default):
                v = ep.get(name, default) if isinstance(ep, dict) else getattr(ep, name, default)
                return v
        # 初始穿着
        initial_outfit = getattr(character, 'initial_outfit', None)
        if initial_outfit is None and isinstance(character, dict):
            initial_outfit = character.get('initial_outfit', 'summer_home')
        if initial_outfit:
            self.clothing.reset(initial_outfit)

    def _apply_sensitivity_multipliers(self, clit=1.0, breast=1.0, gspot=0.8,
                                        skin=1.0, anal=0.3, base=1.0):
        """调整各部位敏感度倍率"""
        multipliers = {
            "clitoris": clit * base,
            "nipple_left": breast * base,
            "nipple_right": breast * base,
            "g_spot": gspot * base,
            "anal_sphincter": anal * base,
            "earlobe": skin * base * 0.8,
            "neck": skin * base * 0.8,
            "inner_thigh_left": skin * base * 0.9,
            "inner_thigh_right": skin * base * 0.9,
            "labia": base * 0.9,
        }
        for region in self.body.values():
            if isinstance(region, BodyRegion):
                for sp_name, sp in region.sub_parts.items():
                    if sp_name in multipliers:
                        sp.sensitivity = min(2.0, sp.sensitivity * multipliers[sp_name])


# 向后兼容别名
SisterState = CharacterState
