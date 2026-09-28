"""
Layer 1b: 状态持久化 - JSON序列化/反序列化
所有组件都有to_dict()，反序列化时安全地更新属性
"""
import json
from typing import Tuple, Dict, Any, Optional

from core.state import CharacterState, SisterState
from core.body.parts import (
    build_default_body, BodyRegion, SubPart, PartnerState, compute_total_wetness
)
from core.body.neural import build_neural_nodes, connect_nodes, NeuralNode
from core.skills import default_skills, Skill, NoveltySystem, BodyAwareness
from core.emotion import EmotionState
from world.world_state import WorldState
from world.time_system import GameTime
from world.clothing import ClothingSystem
from core.mind_state import MindState
from core.memory import MemorySystem
from core.orgasm import OrgasmSystem
from core.ans import ANSState
from core.sensitivity import SensitivityModulators
from core.physical_engine import StaminaState


CURRENT_STATE_VERSION = 2
SUPPORTED_STATE_VERSIONS = {1, CURRENT_STATE_VERSION}
_STATE_DICT_FIELDS = {
    "body", "neural_potentials", "ans", "emotion", "orgasm",
    "sensitivity_mods", "novelty", "body_awareness", "partner", "mind",
    "clothing", "skills", "current_stimulation", "personality", "world",
    "memory",
}


def _migrate_state_payload(data: dict) -> dict:
    """Validate and normalize released state formats without mutating the source."""
    if not isinstance(data, dict):
        raise ValueError("状态数据必须是对象")
    payload = dict(data)
    raw_version = payload.get("version", 1)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError("状态版本无效")
    if raw_version not in SUPPORTED_STATE_VERSIONS:
        if raw_version > CURRENT_STATE_VERSION:
            raise ValueError(f"不支持未来状态版本: {raw_version}")
        raise ValueError(f"不支持状态版本: {raw_version}")
    for field in _STATE_DICT_FIELDS:
        if field in payload and not isinstance(payload[field], dict):
            raise ValueError(f"状态字段 {field} 必须是对象")
    if "conversation_history" in payload and not isinstance(payload["conversation_history"], list):
        raise ValueError("状态字段 conversation_history 必须是列表")
    payload["version"] = CURRENT_STATE_VERSION
    return payload


def _safe_update(obj, data: dict):
    """安全地将data中的键值更新到obj，跳过不存在的属性和只读property"""
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if hasattr(obj, k):
            try:
                cur = getattr(obj, k)
                # 如果是dict/list等容器类型做update而不是替换
                if isinstance(cur, dict) and isinstance(v, dict):
                    cur.update(v)
                elif isinstance(cur, list) and isinstance(v, list):
                    setattr(obj, k, v)
                else:
                    setattr(obj, k, v)
            except AttributeError:
                pass  # 只读property跳过


def _serialize_bodyregion(region: BodyRegion) -> dict:
    return region.to_dict()


def _serialize_neural_node(node: NeuralNode) -> dict:
    return {
        "name": node.name,
        "potential": node.potential,
        "threshold": node.threshold,
        "refractory_period": node.refractory_period,
        "facilitation": node.facilitation,
        "firing_rate": node.firing_rate,
    }


def serialize_state(state: CharacterState, world: WorldState = None,
                   memory: MemorySystem = None) -> dict:
    """完整序列化state到可JSON化的dict"""
    # 技能
    skills_dict = {}
    for k, v in state.skills.items():
        if isinstance(v, Skill):
            skills_dict[k] = v.to_dict()

    data = {
        "version": CURRENT_STATE_VERSION,
        # 身体
        "body": _serialize_body(state.body) if hasattr(state.body, 'items') else {},
        # 神经（保存完整状态：电位、不应期、易化）
        "neural_potentials": {
            name: _serialize_neural_node(n) for name, n in state.nodes.items()
        },
        # 子系统
        "ans": state.ans.to_dict(),
        "emotion": state.emotion.to_dict(),
        "orgasm": state.orgasm.to_dict(),
        "sensitivity_mods": state.sensitivity_mods.to_dict(),
        "novelty": state.novelty.to_dict(),
        "body_awareness": state.body_awareness.to_dict(),
        "partner": state.partner.to_dict(),
        "mind": state.mind.to_dict(),
        "clothing": state.clothing.to_dict(),
        # 技能
        "skills": skills_dict,
        # 全局状态
        "global_arousal": state.global_arousal,
        "relationship_trust": state.relationship_trust,
        "sim_time": state.sim_time,
        "stamina": state.stamina.to_dict(),
        "current_stimulation": {
            "active": state.current_stimulation.get("active", False),
            "adequate_intensity": state.current_stimulation.get("adequate_intensity", False),
            "current_action": state.current_stimulation.get("current_action"),
            "current_targets": list(state.current_stimulation.get("current_targets", [])),
            "recent_stim_strength": dict(state.current_stimulation.get("recent_stim_strength", {})),
            "last_stim_targets": list(state.current_stimulation.get("last_stim_targets", [])),
            "continuous_stim_duration": state.current_stimulation.get("continuous_stim_duration", 0.0),
        },
        "conversation_history": state.conversation_history[-30:],
        "personality": dict(state.personality),
    }

    if memory is None:
        memory = getattr(state, "memory", None)
    if memory:
        data["memory"] = memory.to_dict()

    if world:
        data["world"] = world.to_dict()

    return data


def _serialize_body(body: dict) -> dict:
    result = {}
    for rname, region in body.items():
        if isinstance(region, BodyRegion):
            result[rname] = region.to_dict()
    return result


def _deserialize_body(data: dict) -> dict:
    body = build_default_body()
    global_arousal = data.get("__global_arousal__", 0.0)
    for region_name, rdata in data.items():
        if region_name.startswith("__"):
            continue
        if region_name in body and isinstance(body[region_name], BodyRegion):
            region = body[region_name]
            for sp_name, spdata in rdata.get("sub_parts", {}).items():
                if sp_name in region.sub_parts:
                    sp = region.sub_parts[sp_name]
                    sp.arousal = spdata.get("arousal", 0.0)
                    sp.sensitivity = spdata.get("sensitivity", sp.sensitivity)
                    sp.extra.update(spdata.get("extra", {}))
    body["__global_arousal__"] = global_arousal
    return body


def _restore_neural(nodes, ndata: dict):
    """恢复神经网络完整状态"""
    for name, nd in ndata.items():
        if name in nodes:
            n = nodes[name]
            n.potential = nd.get("potential", 0.0)
            n.refractory_period = nd.get("refractory_period", 0.0)
            n.facilitation = nd.get("facilitation", 1.0)
            n.firing_rate = nd.get("firing_rate", 0.0)
            if "threshold" in nd:
                n.threshold = nd["threshold"]


def _deserialize_skill(data: dict) -> Skill:
    s = Skill()
    _safe_update(s, data)
    return s


def deserialize_state(data: dict) -> Tuple[CharacterState, Optional[WorldState], Optional[MemorySystem]]:
    """从JSON dict反序列化，返回(state, world, memory)"""
    data = _migrate_state_payload(data)
    state = CharacterState()

    # body
    if "body" in data:
        state.body = _deserialize_body(data["body"])

    # neural nodes rebuild + restore
    state.nodes = build_neural_nodes(state.body)
    state.skills = default_skills()
    if "skills" in data:
        for k, v in data["skills"].items():
            if k in state.skills:
                state.skills[k] = _deserialize_skill(v)
    connect_nodes(state.nodes, skills=state.skills)
    if "neural_potentials" in data:
        _restore_neural(state.nodes, data["neural_potentials"])

    # 子系统安全更新
    if "ans" in data:
        _safe_update(state.ans, data["ans"])
    if "emotion" in data:
        em = data["emotion"]
        # blend需要特殊处理（blend是dict，_safe_update会update而非替换）
        if "blend" in em:
            state.emotion.blend.update(em["blend"])
        _safe_update(state.emotion, {k: v for k, v in em.items() if k != "blend"})
        state.emotion.update_primary()
    if "orgasm" in data:
        _safe_update(state.orgasm, data["orgasm"])
    if "sensitivity_mods" in data:
        _safe_update(state.sensitivity_mods, data["sensitivity_mods"])
    if "novelty" in data:
        _safe_update(state.novelty, data["novelty"])
    if "body_awareness" in data:
        _safe_update(state.body_awareness, data["body_awareness"])
    if "partner" in data:
        _safe_update(state.partner, data["partner"])
    if "mind" in data:
        _safe_update(state.mind, data["mind"])

    # clothing (metadata is optional for compatibility with older saves)
    if "clothing" in data:
        cd = data["clothing"]
        state.clothing.reset(
            cd.get("base_outfit", "summer_home"),
            custom_description=cd.get("custom_description", ""),
        )
        garments_data = cd.get("garments", {})
        for gkey, gdata in garments_data.items():
            if gkey in state.clothing.garments:
                g = state.clothing.garments[gkey]
                _safe_update(g, gdata)

    # Global
    state.global_arousal = data.get("global_arousal", 0)
    state.relationship_trust = data.get("relationship_trust", 0.75)
    state.sim_time = data.get("sim_time", 0)
    if "stamina" in data:
        state.stamina = StaminaState.from_dict(data["stamina"])
    if "current_stimulation" in data:
        stimulation = data["current_stimulation"]
        if isinstance(stimulation, dict):
            state.current_stimulation.update(stimulation)
        for k in ["recent_stim_strength"]:
            if not isinstance(state.current_stimulation.get(k), dict):
                state.current_stimulation[k] = {}
        for k in ["current_targets", "last_stim_targets"]:
            if not isinstance(state.current_stimulation.get(k), list):
                state.current_stimulation[k] = []

    state.conversation_history = data.get("conversation_history", [])
    if "personality" in data:
        state.personality.update(data["personality"])

    # World
    world = None
    if "world" in data:
        wd = data["world"]
        world = WorldState()
        gt_data = wd.get("game_time", {})
        world.game_time = GameTime(
            start_hour=gt_data.get("hour", 23),
            start_day=gt_data.get("day", 1)
        )
        world.game_time.minute = gt_data.get("minute", 0)
        world.game_time.sim_seconds = gt_data.get("sim_seconds", 0.0)
        _safe_update(world, {k: v for k, v in wd.items() if k not in ("game_time", "events", "event_engine")})
        event_data = wd.get("events", wd.get("event_engine"))
        if event_data:
            world.events.load_from_dict(event_data, world=world)

    # Memory
    memory = None
    if "memory" in data:
        memory = MemorySystem.from_dict(data["memory"])
        state.memory = memory

    return state, world, memory


def save_game(state: CharacterState, world: WorldState, memory: MemorySystem, path: str):
    """保存游戏到JSON文件"""
    data = serialize_state(state, world, memory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_game(path: str) -> Tuple[CharacterState, Optional[WorldState], Optional[MemorySystem]]:
    """从JSON文件加载游戏"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return deserialize_state(data)
