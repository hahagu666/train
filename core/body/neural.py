"""
Layer 0a: 简化发放式神经网络（Spiking Network）
模拟阈值发放、不应期、短期facilitation、神经噪声
"""
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NeuralNode:
    """单个神经节点"""
    name: str
    region: str
    sensitivity: float = 1.0
    base_threshold: float = 0.3     # 降低阈值，更容易发放

    # 运行时状态
    potential: float = 0.0
    threshold: float = field(init=False)
    refractory_period: float = 0.0
    absolute_refractory: float = 0.15
    relative_refractory: float = 0.5
    facilitation: float = 1.0
    facilitation_decay: float = 0.98
    last_fire_time: float = -100.0
    firing_rate: float = 0.0
    current_input: float = 0.0
    efferent: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.threshold = self.base_threshold

    def stimulate(self, intensity: float, duration: float = 1.0, cognitive_mod: float = 1.0):
        """接收刺激（支持抑制性负信号）"""
        if intensity > 0:
            eff_thresh = self.threshold
            if self.refractory_period > self.absolute_refractory:
                return  # 绝对不应期只抑制兴奋性输入
            elif self.refractory_period > 0:
                eff_thresh *= (1 + self.refractory_period * 0.3)

        signal = intensity * self.facilitation * duration
        if signal > 0:
            signal *= self.sensitivity * cognitive_mod
            signal *= random.uniform(0.95, 1.05)
        else:
            # 抑制信号不受facilitation放大，直接作用于电位
            signal *= cognitive_mod
        self.current_input += signal
        self.potential += signal

    def tick(self, dt: float, current_time: float) -> Tuple[bool, float]:
        """推进一个时间步"""
        fired = False
        # 电位衰减（兴奋电位衰减，抑制性超极化恢复）
        if self.potential > 0:
            self.potential *= 0.97
        else:
            self.potential *= 0.92

        if self.potential >= self.threshold and self.refractory_period <= 0:
            fired = True
            self.last_fire_time = current_time
            self.firing_rate = min(self.firing_rate + 0.15, 1.0)
            self.potential = -0.05
            self.refractory_period = self.absolute_refractory + self.relative_refractory
            self.facilitation = min(self.facilitation * 1.08, 2.0)

        if self.refractory_period > 0:
            self.refractory_period -= dt
        # facilitation恢复到基线1.0（发放时暂时升高，不活动时慢慢恢复，不会衰减到0）
        self.facilitation += (1.0 - self.facilitation) * (1 - self.facilitation_decay)

        # 输出：兴奋时按输入比例输出，抑制时只传递抑制信号
        if fired:
            output = max(0.8, abs(self.current_input) * 1.5) if self.current_input != 0 else 1.0
            if self.current_input < 0:
                output = -output * 0.5
        else:
            if self.current_input > 0:
                output = self.current_input * 0.5 + max(0, self.potential) * 0.4
            else:
                output = self.current_input * 0.5 + min(0, self.potential) * 0.4
        self.current_input = 0.0
        self.potential = max(self.potential, -0.5)  # 防止超极化过深
        return fired, output

    def reset(self):
        self.potential = 0.0
        self.refractory_period = 0.0
        self.facilitation = 1.0
        self.firing_rate = 0.0
        self.current_input = 0.0


# === 神经连接权重 ===
BASE_CONNECTIONS: Dict[Tuple[str, str], float] = {
    # === 强连接（性反射弧，生理上的强神经通路）===
    ("nipple_left", "genital_female"): 0.70,
    ("nipple_right", "genital_female"): 0.70,
    ("clitoris", "vaginal_canal"): 0.80,
    ("clitoris", "g_spot"): 0.60,
    ("g_spot", "cervix"): 0.70,
    ("inner_thighs", "genital_female"): 0.60,
    ("neck", "chest"): 0.50,
    ("earlobes", "neck"): 0.60,
    ("buttocks", "genital_female"): 0.50,
    ("lips", "chest"): 0.40,
    ("lips", "genital_female"): 0.30,
    # === 中等连接 ===
    ("waist", "genital_female"): 0.35,
    ("back", "waist"): 0.30,
    ("navel", "genital_female"): 0.30,
    ("fingers", "arms_hands"): 0.40,
    ("throat", "chest"): 0.25,
    # === 全身联动 ===
    ("skin_global", "all"): 0.20,
    # === 抑制性连接（认知/情绪通过这些通路抑制性反应）===
    ("cognitive_fear", "genital_female"): -0.50,
    ("cognitive_fear", "skin_global"): -0.30,
    ("cognitive_hurt", "genital_female"): -0.40,
    ("cognitive_shame", "skin_global"): -0.20,
}

# 生殖器区域的核心节点（用于genital_female通配符）
GENITAL_CORE_NODES = ["clitoris", "vaginal_canal", "inner_labia", "g_spot", "vaginal_vestibule"]


def get_connection_weight(src: str, dst: str, base_weight: float, skills: dict = None) -> float:
    """
    根据技能经验调整连接权重
    经验越多，神经通路越强（身体记住了反应路径）
    抑制性连接不被经验增强
    """
    if base_weight < 0:
        return base_weight  # 抑制连接永远保留

    from ..skills import calc_effective_mastery
    w = base_weight

    if skills:
        # 胸部→生殖器通路受breast_touch经验影响
        if src in ["nipple_left", "nipple_right", "breast_left", "breast_right"]:
            mast = calc_effective_mastery(skills.get("breast_touch", type('Skill', (), {'times':0, 'mastery':0, 'last_practiced':None, 'half_life_days':14})()))
            w *= (0.5 + mast * 1.0)
        # 生殖器内部通路受性交/手指经验影响
        if src in ["clitoris", "g_spot"] or dst in ["vaginal_canal", "g_spot", "cervix"]:
            m_miss = calc_effective_mastery(skills.get("missionary", type('Skill', (), {'times':0, 'mastery':0, 'last_practiced':None, 'half_life_days':14})()))
            m_cow = calc_effective_mastery(skills.get("cowgirl", type('Skill', (), {'times':0, 'mastery':0, 'last_practiced':None, 'half_life_days':14})()))
            m_fin = calc_effective_mastery(skills.get("clit_stimulation_receiving", type('Skill', (), {'times':0, 'mastery':0, 'last_practiced':None, 'half_life_days':14})()))
            max_mast = max(m_miss, m_cow, m_fin)
            w *= (0.5 + max_mast * 1.0)
        # 口交通路
        if dst == "throat" or src == "throat":
            m_bj = calc_effective_mastery(skills.get("blowjob", type('Skill', (), {'times':0, 'mastery':0, 'last_practiced':None, 'half_life_days':10})()))
            w *= (0.3 + m_bj * 1.2)

    return w


def build_neural_nodes(body_regions) -> Dict[str, NeuralNode]:
    """为所有身体子部位创建神经节点"""
    nodes = {}
    for region_name, region in body_regions.items():
        if not hasattr(region, 'sub_parts'):
            continue
        for sp_name, sp in region.sub_parts.items():
            node = NeuralNode(
                name=sp_name,
                region=region_name,
                sensitivity=sp.sensitivity,
                base_threshold=0.2 / max(sp.sensitivity, 0.3)  # 阈值更低，更容易发放
            )
            nodes[sp_name] = node
    # 添加虚拟认知节点（用于抑制性连接的入口）
    for cog_name in ["cognitive_fear", "cognitive_hurt", "cognitive_shame"]:
        nodes[cog_name] = NeuralNode(name=cog_name, region="cognitive",
                                     sensitivity=1.0, base_threshold=0.1)
    return nodes


def connect_nodes(nodes: Dict[str, NeuralNode], connection_overrides: Dict = None, skills: dict = None):
    """建立神经连接"""
    conns = dict(BASE_CONNECTIONS)
    if connection_overrides:
        conns.update(connection_overrides)

    # 建立连接
    for (src, dst), base_weight in conns.items():
        weight = get_connection_weight(src, dst, base_weight, skills)
        if src in nodes:
            if dst == "all":
                for n in nodes.values():
                    if n.name != src and not n.name.startswith("cognitive_"):
                        nodes[src].efferent[n.name] = weight * 0.3
            elif dst in nodes:
                nodes[src].efferent[dst] = weight
            elif dst == "genital_female":
                for gn in GENITAL_CORE_NODES:
                    if gn in nodes:
                        nodes[src].efferent[gn] = weight * 0.8




def stimulate_node(nodes: Dict[str, NeuralNode], part_name: str, intensity: float,
                   duration: float = 1.0, cognitive_mod: float = 1.0):
    """刺激指定节点"""
    if part_name in nodes:
        nodes[part_name].stimulate(intensity, duration, cognitive_mod)
    # 别名映射
    alias = {
        "clit": "clitoris", "pussy": "vaginal_canal", "cock": "penis",
        "boob_left": "breast_left", "boob_right": "breast_right",
        "tit_left": "nipple_left", "tit_right": "nipple_right",
        "ass": "buttocks", "pussy_lips": "outer_labia",
        "inner_thigh": "inner_thighs", "thigh": "thighs",
    }
    if part_name in alias and alias[part_name] in nodes:
        nodes[alias[part_name]].stimulate(intensity, duration, cognitive_mod)


def propagate(nodes: Dict[str, NeuralNode], iterations: int = 8, dt: float = 0.1,
              cognitive_mod_fn=None, on_fire=None) -> Dict[str, float]:
    """
    传播神经信号
    返回: 各节点发放强度 {node_name: output}
    """
    fired_total = {}
    current_time = 0.0

    for step in range(iterations):
        current_time += dt
        fired_this_step = {}

        for node in nodes.values():
            fired, output = node.tick(dt, current_time)
            if fired:
                mod = cognitive_mod_fn(node) if cognitive_mod_fn else 1.0
                fired_this_step[node.name] = output * mod
                fired_total[node.name] = fired_total.get(node.name, 0) + output * mod

        # 传播发放信号（支持抑制连接）
        for fname, fout in fired_this_step.items():
            if fname in nodes:
                for tname, weight in nodes[fname].efferent.items():
                    if tname in nodes:
                        # 正信号用excitatory传播，负信号直接作为抑制
                        nodes[tname].stimulate(fout * weight, duration=0.3, cognitive_mod=1.0)

        if on_fire:
            on_fire(fired_this_step, step, dt)

    return fired_total


def apply_cognitive_inhibition(nodes: Dict[str, NeuralNode], emotion_state, dt: float):
    """
    根据当前情绪状态激活认知抑制节点
    恐惧/受伤/羞耻会通过抑制连接减弱性反应
    """
    fear = emotion_state.blend.get("fear", 0.0)
    hurt = emotion_state.blend.get("hurt", 0.0)
    shame = emotion_state.blend.get("shame", 0.0)

    if fear > 0.2 and "cognitive_fear" in nodes:
        nodes["cognitive_fear"].stimulate(fear * 0.6, duration=dt, cognitive_mod=1.0)
    if hurt > 0.2 and "cognitive_hurt" in nodes:
        nodes["cognitive_hurt"].stimulate(hurt * 0.5, duration=dt, cognitive_mod=1.0)
    if shame > 0.3 and "cognitive_shame" in nodes:
        nodes["cognitive_shame"].stimulate(shame * 0.3, duration=dt, cognitive_mod=1.0)
