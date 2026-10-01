"""
会话管理 - 支持多会话、多存档、分支时间线
"""
import os
import json
import re
import uuid
import shutil
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
from copy import deepcopy

from .models import (
    SessionMeta, CreateSessionRequest, RenameSessionRequest,
    SessionStateSnapshot, SaveSlotSummary, BranchRequest,
)
from .config import SESSIONS_DIR
from core.serialization import deserialize_state, serialize_state
from core.state import CharacterState
from world.world_state import WorldState
from world.scenario_engine import ScenarioEngine


_TRANSACTION_ADMISSION_LOCK = Lock()

CURRENT_CONTEXT_VERSION = 2
CURRENT_SNAPSHOT_VERSION = 3
CURRENT_SAVE_VERSION = 4
SUPPORTED_CONTEXT_VERSIONS = {1, CURRENT_CONTEXT_VERSION}
SUPPORTED_SNAPSHOT_VERSIONS = {1, 2, CURRENT_SNAPSHOT_VERSION}
SUPPORTED_SAVE_VERSIONS = {1, 2, 3, CURRENT_SAVE_VERSION}



# === 日志补接线（详细排障） ===
try:
    from app.logger import debug, info, success, warning, error, trace
except Exception:
    debug = info = success = warning = error = trace = lambda *a, **k: None

def _normalize_user_profile(payload: object) -> dict:
    result = deepcopy(payload) if isinstance(payload, dict) else {}
    result["name"] = str(result.get("name") or "你").strip() or "你"
    for field in (
        "gender", "personality_description", "speaking_style_hint",
        "appearance_hint", "relation", "preferred_address", "additional_context",
    ):
        value = result.get(field, "")
        result[field] = str(value).strip() if value is not None else ""
    age = result.get("age", 0)
    if isinstance(age, bool):
        age = 0
    try:
        result["age"] = min(max(int(age), 0), 120)
    except (TypeError, ValueError):
        result["age"] = 0
    traits = result.get("traits", result.get("extracted_traits", []))
    normalized_traits = []
    if isinstance(traits, list):
        for value in traits:
            item = str(value).strip()
            if item and len(item) <= 80 and item not in normalized_traits:
                normalized_traits.append(item)
            if len(normalized_traits) >= 30:
                break
    result["traits"] = normalized_traits
    result.pop("extracted_traits", None)
    return result


def _format_version(payload: dict, current: int, supported: set[int], label: str) -> int:
    if not isinstance(payload, dict):
        raise ValueError(f"{label}数据必须是对象")
    version = payload.get("format_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"{label}版本无效")
    if version not in supported:
        if version > current:
            raise ValueError(f"不支持未来{label}版本: {version}")
        raise ValueError(f"不支持{label}版本: {version}")
    return version


def _normalize_context(payload: dict, turn_count: int) -> tuple[str, int, dict]:
    _format_version(payload, CURRENT_CONTEXT_VERSION, SUPPORTED_CONTEXT_VERSIONS, "上下文")
    summary = payload.get("conversation_summary", "")
    if not isinstance(summary, str):
        summary = ""
    cursor = payload.get("compacted_through_turn", 0)
    if isinstance(cursor, bool):
        cursor = 0
    try:
        cursor = int(cursor)
    except (TypeError, ValueError):
        cursor = 0
    profile = _normalize_user_profile(payload.get("user_profile", {}))
    return summary, min(max(cursor, 0), max(int(turn_count), 0)), profile


def _normalize_snapshot(payload: dict) -> dict:
    version = _format_version(payload, CURRENT_SNAPSHOT_VERSION, SUPPORTED_SNAPSHOT_VERSIONS, "快照")
    normalized = deepcopy(payload)
    messages = normalized.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("快照消息必须是列表")
    turn = normalized.get("turn", 0)
    try:
        turn = max(int(turn), 0)
    except (TypeError, ValueError):
        turn = 0
    summary, cursor, profile = _normalize_context({
        "format_version": CURRENT_CONTEXT_VERSION,
        "conversation_summary": normalized.get("conversation_summary", ""),
        "compacted_through_turn": normalized.get("compacted_through_turn", 0),
        "user_profile": normalized.get("user_profile", {}) if version >= 3 else {},
    }, turn)
    normalized.update({
        "format_version": CURRENT_SNAPSHOT_VERSION,
        "messages": messages,
        "turn": turn,
        "conversation_summary": summary,
        "compacted_through_turn": cursor,
        "user_profile": profile,
    })
    return normalized


def _normalize_save(payload: dict) -> dict:
    version = _format_version(payload, CURRENT_SAVE_VERSION, SUPPORTED_SAVE_VERSIONS, "存档")
    normalized = deepcopy(payload)
    if "messages" in normalized and not isinstance(normalized["messages"], list):
        raise ValueError("存档消息必须是列表")
    turn = normalized.get("turn", 0)
    try:
        turn = max(int(turn), 0)
    except (TypeError, ValueError):
        turn = 0
    # Released v1/v2 saves did not persist context; v3 introduced it and v4
    # adds the immutable per-session user profile snapshot.
    has_context = version >= 3
    summary, cursor, profile = _normalize_context({
        "format_version": CURRENT_CONTEXT_VERSION,
        "conversation_summary": normalized.get("conversation_summary", "") if has_context else "",
        "compacted_through_turn": normalized.get("compacted_through_turn", 0) if has_context else 0,
        "user_profile": normalized.get("user_profile", {}) if version >= 4 else {},
    }, turn)
    normalized.update({
        "format_version": CURRENT_SAVE_VERSION,
        "turn": turn,
        "conversation_summary": summary,
        "compacted_through_turn": cursor,
        "user_profile": profile,
    })
    return normalized


def _atomic_write_json(path: str, payload: object):
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_write_messages(path: str, messages: List[Dict]):
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for message in messages:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class SessionInstance:
    """一个活跃会话实例"""
    def __init__(self):
        self.meta: Optional[SessionMeta] = None
        self.char_state: Optional[CharacterState] = None
        self.world: Optional[WorldState] = None
        self.messages: List[Dict] = []  # 对话消息列表
        self.save_slots: Dict[str, dict] = {}  # {slot_id: save_dict}
        self.turn_count: int = 0
        self.scenario_engine: Optional[ScenarioEngine] = None
        self.scenario_id: Optional[str] = None
        self.timeline_id: str = uuid.uuid4().hex[:12]
        self.current_snapshot_id: Optional[str] = None
        self.snapshot_index: List[Dict] = []
        self.conversation_summary: str = ""
        self.compacted_through_turn: int = 0
        self.user_profile: dict = _normalize_user_profile({})
        self.corrupted_sessions: List[Dict] = []

    def build_state_snapshot(self) -> SessionStateSnapshot:
        """构建给前端的状态快照"""
        s = self.char_state
        w = self.world
        raw_phase = s.orgasm.phase if s else "excitement"
        emotion_str = s.emotion.get_dominant_emotions() if s else "平静"
        hr = int(s.ans.heart_rate) if s else 72
        arousal = float(s.global_arousal) if s else 0.0
        # The simulation calls its resting pre-plateau phase "excitement". The
        # API exposes a neutral baseline while there is no measurable arousal.
        phase = "baseline" if arousal <= 0.0 and raw_phase == "excitement" else raw_phase
        trust = float(s.relationship_trust) if s else 0.5
        clothing_desc = ""
        if s and hasattr(s, 'clothing') and s.clothing is not None:
            if hasattr(s.clothing, 'clothing_description'):
                clothing_desc = s.clothing.clothing_description(verbose=False)
            elif hasattr(s.clothing, 'state_label'):
                clothing_desc = s.clothing.state_label()
        time_str = w.game_time.to_string() if w and hasattr(w.game_time, 'to_string') else ""
        location = w.location_name if w else "卧室"
        privacy_val = w.get_privacy_level(s) if w else 0.7
        danger_level = float(getattr(w, "danger_level", 0.1)) if w else 0.1
        detection_risk = float(getattr(getattr(w, "events", None), "detection_risk", 0.0)) if w else 0.0
        has_people = bool(getattr(w, "has_people", False)) if w else False
        others_present = bool(getattr(w, "others_present", False)) if w else False
        privacy = (
            "危险"
            if has_people
            else ("私密" if privacy_val > 0.6 else "半公开")
        )
        acceptance = float(getattr(getattr(s, 'mind', None), 'acceptance_level', 0.3)) if s else 0.3
        relationship_stage = (
            getattr(self.meta, "relationship_stage", self.meta.current_stage)
            if self.meta else "D"
        )
        scenario_stage = getattr(self.meta, "scenario_stage", None) if self.meta else None

        return SessionStateSnapshot(
            arousal=round(arousal, 2),
            phase=phase,
            emotion=emotion_str,
            trust=round(trust, 2),
            closeness=round(self.meta.relationship_closeness if self.meta else 0.4, 2),
            heart_rate=hr,
            clothing=clothing_desc,
            clothing_layers=s.clothing.layers_snapshot() if s and hasattr(s, 'clothing') and hasattr(s.clothing, 'layers_snapshot') else {},
            time_str=time_str,
            location=location,
            privacy=privacy,
            privacy_level=round(float(privacy_val), 2),
            danger_level=round(max(0.0, min(1.0, danger_level)), 2),
            detection_risk=round(max(0.0, min(1.0, detection_risk)), 2),
            has_people=has_people,
            others_present=others_present,
            stage=relationship_stage,
            relationship_stage=relationship_stage,
            scenario_stage=scenario_stage,
            acceptance=round(acceptance, 2),
            stamina=round(s.stamina.ratio if s and hasattr(s, "stamina") else 1.0, 2),
            fatigue=round(s.stamina.fatigue if s and hasattr(s, "stamina") else 0.0, 2),
            is_orgasm=(phase == "orgasm"),
            is_interrupted=getattr(getattr(s, 'emotion', None), 'interrupt_triggered', False) if s else False,
        )

    def get_opening_message(self) -> Optional[Dict]:
        """Return the single persisted opening message, if present."""
        return next(
            (message for message in self.messages if message.get("message_type") == "opening"),
            None,
        )


class SessionManager:
    def __init__(self, character_manager=None, scenario_engine_class=None, knowledge_service=None):
        self.sessions: Dict[str, SessionInstance] = {}
        self.active_session_id: Optional[str] = None
        self.char_mgr = character_manager
        self.knowledge_svc = knowledge_service
        self._active_transactions: set[str] = set()
        self._transaction_lock = Lock()
        self._load_all()

    def begin_transaction(self, session_id: str) -> bool:
        """独占会话写事务；事务结束前禁止其他持久化入口。"""
        with getattr(self, "_transaction_lock", _TRANSACTION_ADMISSION_LOCK):
            if session_id in self._active_transactions:
                return False
            self._active_transactions.add(session_id)
            return True

    def end_transaction(self, session_id: str):
        with getattr(self, "_transaction_lock", _TRANSACTION_ADMISSION_LOCK):
            self._active_transactions.discard(session_id)

    def transaction_active(self, session_id: str) -> bool:
        with getattr(self, "_transaction_lock", _TRANSACTION_ADMISSION_LOCK):
            return session_id in self._active_transactions

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(SESSIONS_DIR, session_id)

    def _load_all(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.corrupted_sessions = []
        for sid in os.listdir(SESSIONS_DIR):
            sdir = os.path.join(SESSIONS_DIR, sid)
            if not os.path.isdir(sdir):
                continue
            meta_path = os.path.join(sdir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            try:
                inst = self._load_session_from_disk(sid)
                if inst:
                    self.sessions[sid] = inst
            except Exception as e:
                report = {
                    "session_id": sid,
                    "detected_at": datetime.now().isoformat(),
                    "recoverable": False,
                    "components": [{"component": "session", "error": str(e)}],
                }
                self.corrupted_sessions.append(report)
                print(f"[SessionManager] 加载会话失败 {sid}: {e}")

    def list_corrupted(self) -> List[dict]:
        return deepcopy(self.corrupted_sessions)

    def _load_session_from_disk(self, session_id: str) -> Optional[SessionInstance]:
        sdir = self._session_dir(session_id)
        meta_path = os.path.join(sdir, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        # datetime反序列化
        if "created_at" in meta_dict:
            meta_dict["created_at"] = datetime.fromisoformat(meta_dict["created_at"])
        if "last_active_at" in meta_dict:
            meta_dict["last_active_at"] = datetime.fromisoformat(meta_dict["last_active_at"])
        inst = SessionInstance()
        inst.meta = SessionMeta(**meta_dict)
        inst.scenario_id = inst.meta.scenario_id

        # 加载状态
        state_path = os.path.join(sdir, "state.json")
        world_path = os.path.join(sdir, "world.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            loaded_state, embedded_world, loaded_memory = deserialize_state(state_data)
            inst.char_state = loaded_state
            if embedded_world and not os.path.exists(world_path):
                inst.world = embedded_world
            if loaded_memory:
                inst.char_state.memory = loaded_memory
        if os.path.exists(world_path):
            inst.world = WorldState()
            with open(world_path, "r", encoding="utf-8") as f:
                wd = json.load(f)
            inst.world.load_from_dict(wd)

        # 加载消息
        msg_path = os.path.join(sdir, "messages.jsonl")
        if os.path.exists(msg_path):
            with open(msg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        inst.messages.append(json.loads(line))
            inst.turn_count = max(
                (int(message.get("turn", 0)) for message in inst.messages),
                default=0,
            )

        # 加载存档
        saves_dir = os.path.join(sdir, "saves")
        if os.path.exists(saves_dir):
            for fname in os.listdir(saves_dir):
                if fname.endswith(".json"):
                    slot_id = fname[:-5]
                    with open(os.path.join(saves_dir, fname), "r", encoding="utf-8") as f:
                        inst.save_slots[slot_id] = _normalize_save(json.load(f))

        timeline_path = os.path.join(sdir, "timeline.json")
        if os.path.exists(timeline_path):
            with open(timeline_path, "r", encoding="utf-8") as f:
                timeline_data = json.load(f)
            inst.timeline_id = timeline_data.get("timeline_id", inst.timeline_id)
            inst.current_snapshot_id = timeline_data.get("current_snapshot_id")
            inst.snapshot_index = timeline_data.get("snapshot_index", [])
        context_path = os.path.join(sdir, "context.json")
        if os.path.exists(context_path):
            with open(context_path, "r", encoding="utf-8") as f:
                context_data = json.load(f)
            inst.conversation_summary, inst.compacted_through_turn, inst.user_profile = _normalize_context(
                context_data, inst.turn_count
            )

        # Scenario engine
        try:
            from world.scenario_engine import get_scenario_engine
            inst.scenario_engine = get_scenario_engine()
        except Exception:
            pass
        inst.meta.scenario_stage = self._scenario_stage(inst)

        return inst

    @staticmethod
    def _scenario_stage(inst: SessionInstance) -> Optional[str]:
        """Resolve the selected card category from its authoritative card/id."""
        if not inst.scenario_id:
            return None
        if inst.scenario_engine:
            card = inst.scenario_engine.get_card(inst.scenario_id)
            if card:
                return getattr(card, "stage", None)
        match = re.match(r"^D-([A-K])\d+$", inst.scenario_id)
        return match.group(1) if match else None

    @staticmethod
    def _sync_meta_from_state(inst: SessionInstance):
        """Keep persisted/list metadata aligned with the final runtime state."""
        if not inst.meta:
            return
        inst.meta.relationship_stage = inst.meta.current_stage
        inst.meta.scenario_stage = SessionManager._scenario_stage(inst)
        if inst.char_state:
            inst.meta.relationship_trust = float(inst.char_state.relationship_trust)
        if inst.world:
            inst.meta.current_location = inst.world.location_name
            if hasattr(inst.world, "game_time"):
                inst.meta.current_time = inst.world.game_time.to_string()
        inst.meta.total_interactions = inst.turn_count
        inst.meta.scenario_id = inst.scenario_id

    def _persist_session(self, inst: SessionInstance):
        """持久化会话到磁盘"""
        self._sync_meta_from_state(inst)
        sdir = self._session_dir(inst.meta.session_id)
        os.makedirs(sdir, exist_ok=True)
        os.makedirs(os.path.join(sdir, "saves"), exist_ok=True)
        os.makedirs(os.path.join(sdir, "memory"), exist_ok=True)

        # meta
        meta_dict = inst.meta.model_dump()
        meta_dict["created_at"] = inst.meta.created_at.isoformat()
        meta_dict["last_active_at"] = inst.meta.last_active_at.isoformat()
        _atomic_write_json(os.path.join(sdir, "meta.json"), meta_dict)

        # state
        if inst.char_state:
            state_data = serialize_state(inst.char_state, inst.world, getattr(inst.char_state, "memory", None))
            _atomic_write_json(os.path.join(sdir, "state.json"), state_data)
        if inst.world:
            wd = inst.world.to_dict() if hasattr(inst.world, 'to_dict') else {}
            _atomic_write_json(os.path.join(sdir, "world.json"), wd)

        # messages
        msg_path = os.path.join(sdir, "messages.jsonl")
        _atomic_write_messages(msg_path, inst.messages)

        # saves
        saves_dir = os.path.join(sdir, "saves")
        for slot_id, save_data in inst.save_slots.items():
            _atomic_write_json(os.path.join(saves_dir, f"{slot_id}.json"), save_data)

        _atomic_write_json(os.path.join(sdir, "timeline.json"), {
            "timeline_id": inst.timeline_id,
            "current_snapshot_id": inst.current_snapshot_id,
            "snapshot_index": inst.snapshot_index,
        })
        _atomic_write_json(os.path.join(sdir, "context.json"), {
            "format_version": CURRENT_CONTEXT_VERSION,
            "conversation_summary": inst.conversation_summary,
            "compacted_through_turn": inst.compacted_through_turn,
            "user_profile": _normalize_user_profile(inst.user_profile),
        })

    def capture_snapshot(self, inst: SessionInstance, operation: str = "turn", preview: str = "") -> dict:
        debug("存档", f"捕获快照: session={inst.meta.session_id}, operation={operation}, turn={inst.turn_count}, preview={preview[:30]}")
        snapshot_id = uuid.uuid4().hex[:16]
        snapshot = {
            "format_version": CURRENT_SNAPSHOT_VERSION,
            "snapshot_id": snapshot_id,
            "session_id": inst.meta.session_id,
            "timeline_id": inst.timeline_id,
            "parent_snapshot_id": inst.current_snapshot_id,
            "turn": inst.turn_count,
            "operation": operation,
            "created_at": datetime.now().isoformat(),
            "state_dict": serialize_state(inst.char_state, inst.world, getattr(inst.char_state, "memory", None)) if inst.char_state else {},
            "world_dict": inst.world.to_dict() if inst.world and hasattr(inst.world, "to_dict") else {},
            "messages": deepcopy(inst.messages),
            "conversation_summary": inst.conversation_summary,
            "compacted_through_turn": inst.compacted_through_turn,
            "user_profile": deepcopy(inst.user_profile),
            "meta": inst.meta.model_dump(mode="json"),
            "scenario_id": inst.scenario_id,
            "relationship_memories": self.knowledge_svc.export_relationship(inst.meta.session_id) if self.knowledge_svc else [],
            "preview": preview or (inst.messages[-1].get("content", "")[:80] if inst.messages else ""),
        }
        snapshot_dir = os.path.join(self._session_dir(inst.meta.session_id), "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        _atomic_write_json(os.path.join(snapshot_dir, f"{snapshot_id}.json"), snapshot)
        inst.current_snapshot_id = snapshot_id
        inst.snapshot_index.append({k: snapshot[k] for k in ("snapshot_id", "timeline_id", "parent_snapshot_id", "turn", "operation", "created_at", "preview")})
        self._persist_session(inst)
        return snapshot

    def list_history(self, session_id: str) -> List[dict]:
        inst = self.sessions.get(session_id)
        return deepcopy(inst.snapshot_index) if inst else []

    def get_snapshot(self, session_id: str, snapshot_id: str) -> Optional[dict]:
        inst = self.sessions.get(session_id)
        if not inst or snapshot_id not in {x.get("snapshot_id") for x in inst.snapshot_index}:
            return None
        path = os.path.join(self._session_dir(session_id), "snapshots", f"{snapshot_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return _normalize_snapshot(json.load(f))

    def restore_snapshot(self, session_id: str, snapshot_id: str,
                         new_timeline: bool = False,
                         persist: bool = True) -> Optional[SessionInstance]:
        inst = self.sessions.get(session_id)
        snapshot = self.get_snapshot(session_id, snapshot_id)
        if not inst or not snapshot:
            return None
        state, embedded_world, memory = self._restore_state_dict(snapshot.get("state_dict", {}))
        if memory:
            state.memory = memory
        world = embedded_world or WorldState()
        if snapshot.get("world_dict"):
            world.load_from_dict(snapshot["world_dict"])
        inst.char_state = state
        inst.world = world
        inst.messages = deepcopy(snapshot.get("messages", []))
        inst.turn_count = snapshot.get("turn", 0)
        inst.conversation_summary = snapshot.get("conversation_summary", "")
        inst.compacted_through_turn = min(
            int(snapshot.get("compacted_through_turn", 0) or 0),
            inst.turn_count,
        )
        inst.user_profile = _normalize_user_profile(snapshot.get("user_profile", {}))
        meta_data = deepcopy(snapshot.get("meta", {}))
        for key in ("created_at", "last_active_at"):
            if isinstance(meta_data.get(key), str):
                meta_data[key] = datetime.fromisoformat(meta_data[key])
        inst.meta = SessionMeta(**meta_data)
        inst.scenario_id = snapshot.get("scenario_id", inst.meta.scenario_id)
        inst.meta.scenario_id = inst.scenario_id
        if self.knowledge_svc:
            self.knowledge_svc.replace_relationship(session_id, snapshot.get("relationship_memories", []))
        if new_timeline:
            inst.timeline_id = uuid.uuid4().hex[:12]
            inst.current_snapshot_id = snapshot_id
        else:
            inst.timeline_id = snapshot.get("timeline_id", inst.timeline_id)
            inst.current_snapshot_id = snapshot_id
        if persist:
            self._persist_session(inst)
        return inst

    def retract_turn(self, session_id: str, turn: int) -> Optional[SessionInstance]:
        inst = self.sessions.get(session_id)
        if not inst:
            return None
        candidates = [x for x in inst.snapshot_index if x.get("turn") == max(0, turn - 1)]
        if not candidates:
            return None
        return self.restore_snapshot(session_id, candidates[-1]["snapshot_id"], new_timeline=True)

    def rollback_to_snapshot(self, session_id: str, snapshot_id: str) -> Optional[SessionInstance]:
        return self.restore_snapshot(session_id, snapshot_id, new_timeline=True)

    def list_sessions(self, char_id: str = None) -> List[SessionMeta]:
        result = []
        for inst in self.sessions.values():
            if char_id and inst.meta.character_id != char_id:
                continue
            result.append(inst.meta)
        result.sort(key=lambda m: m.last_active_at, reverse=True)
        return result

    def create_session(self, char_id: str, name: str = None,
                       scenario_id: str = None, custom_opening: str = None,
                       initial_trust: float = None,
                       initial_closeness: float = None, initial_outfit: str = None,
                       custom_outfit_description: str = None,
                       location: str = None, time_str: str = None,
                       privacy: str = None,
                       user_profile: Optional[dict] = None) -> SessionInstance:
        debug("会话", f"创建会话: char={char_id}, scenario={scenario_id}, trust={initial_trust}, closeness={initial_closeness}, location={location}")
        if self.char_mgr:
            char = self.char_mgr.get_character(char_id)
            if not char:
                raise ValueError(f"角色不存在: {char_id}")
        else:
            char = None

        sid = uuid.uuid4().hex[:12]
        inst = SessionInstance()
        inst.user_profile = _normalize_user_profile(user_profile)
        inst.char_state = CharacterState()
        inst.world = WorldState()

        # 应用角色模板
        if char:
            inst.char_state.apply_character_template(char)
            # 根据关系类型设置世界初始状态
            rel = char.relationship_type
            if rel == "step_sister":
                inst.world.set_location("bedroom")
                inst.world.door_locked = True
                inst.world.set_base_privacy(0.85)
            elif rel == "classmate":
                inst.world.set_location("school")
                inst.world.has_people = False
                inst.world.door_locked = False
                inst.world.set_base_privacy(0.15)
                # 同桌穿校服
                inst.char_state.clothing.reset("school_uniform")
            elif rel == "childhood_friend":
                inst.world.set_location("outside")
                inst.world.has_people = False
                inst.world.set_base_privacy(0.4)
                inst.char_state.clothing.reset("casual")

        # 随机开场剧情
        opening = ""
        card = None
        try:
            from world.scenario_engine import get_scenario_engine
            inst.scenario_engine = get_scenario_engine()
            # apply_initial_state期望state和world属性
            inst.state = inst.char_state  # 添加别名
            selection_trust = (
                initial_trust
                if initial_trust is not None
                else inst.char_state.relationship_trust
            )
            if scenario_id:
                card = inst.scenario_engine.get_card(scenario_id)
                if card is None:
                    raise ValueError(f"剧情卡不存在: {scenario_id}")
                from core.content_policy import evaluate_character
                from world.scenario_engine import STAGE_INFO
                stage_info = STAGE_INFO.get(card.stage, {})
                if stage_info.get("adult_only") and not evaluate_character(char).allowed:
                    raise ValueError("该剧情仅限已确认年满 18 岁的角色")
                if char and card.stage not in getattr(char, "allowed_stages", []):
                    raise ValueError("角色不允许进入该剧情阶段")
            elif not custom_opening:
                card = inst.scenario_engine.pick_intro(trust=selection_trust)
            else:
                # 自定义开场仍绑定随机场景状态，文本本身最后覆盖 opening
                card = inst.scenario_engine.pick_intro(trust=selection_trust)
            if card:
                inst.scenario_id = card.card_id
                card.apply_initial_state(inst)
            if custom_opening:
                from world.scenario_engine import normalize_character_output
                opening = normalize_character_output(custom_opening)
                # 保留随机场景状态；自定义文本只替换 opening 内容

        except ValueError:
            raise
        except Exception as e:
            print(f"[SessionManager] 剧情卡选择失败: {e}")
            import traceback
            traceback.print_exc()
            opening = ""

        # 手动初始值最后应用，覆盖角色模板和剧情卡状态。
        if initial_trust is not None:
            inst.char_state.relationship_trust = initial_trust
        if initial_outfit is not None:
            # Preset semantics: replace both simulation outfit and displayed description.
            inst.char_state.clothing.reset(initial_outfit)
        elif custom_outfit_description is not None:
            # Custom semantics: preserve the scenario/template garment simulation while
            # using the caller's prose as the initial visible outfit description.
            inst.char_state.clothing.custom_description = custom_outfit_description.strip()
        if location is not None:
            inst.world.set_location(location)
        if time_str is not None:
            time_match = re.fullmatch(r"\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[：:]([0-5]\d)\s*", time_str)
            if time_match:
                _year, _month, day, hour, minute = map(int, time_match.groups())
                if not (1 <= _month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23):
                    raise ValueError("初始时间的日期或时间超出范围")
            else:
                legacy = re.fullmatch(r"\s*(?:(?:第)?(\d+)天\s*)?(?:(?:早上|上午|中午|下午|晚上)\s*)?(\d{1,2})(?:[:：点时]\s*(\d{1,2}))?(?:分)?\s*", time_str)
                if not legacy:
                    raise ValueError("初始时间格式必须为 YYYY年M月D日HH：MM，例如 2026年5月20日21：00")
                day = int(legacy.group(1)) if legacy.group(1) else None
                hour = int(legacy.group(2))
                minute = int(legacy.group(3) or 0)
                if "下午" in time_str or "晚上" in time_str:
                    hour = hour + 12 if hour < 12 else hour
                if hour > 23 or (day is not None and day < 1):
                    raise ValueError("初始时间的日期或时间超出范围")
            inst.world.game_time.set_time(hour, minute, day)
            inst.world._update_parents_presence()
        if privacy is not None:
            privacy_levels = {"私密": 0.9, "半公开": 0.5, "危险": 0.1}
            if privacy not in privacy_levels:
                raise ValueError(f"无效的隐私级别: {privacy}")
            inst.world.set_base_privacy(privacy_levels[privacy])
            if privacy == "危险":
                if not inst.world.parents_present and not inst.world.others_present:
                    inst.world.people_present["scenario_other"] = {
                        "label": "附近有人", "location": "current", "activity": None,
                    }
                inst.world._sync_presence_flags()
            else:
                inst.world.scenario_people_override = True
                inst.world.people_present = {
                    "user": {"location": "current", "activity": None},
                    "sister": {"location": "current", "activity": None},
                }
                inst.world._sync_presence_flags()
        inst.char_state.set_world_privacy(inst.world.privacy_level)

        # 第0轮角色文本必须在剧情卡、角色模板和所有手动覆盖都落定后生成。
        if card and not custom_opening:
            opening = card.generate_turn_zero(inst, character=char)

        # If scenario generation is unavailable, keep the fallback neutral and in-role.
        if not opening and char:
            first_person = getattr(char.speech_style, "first_person", "我") or "我"
            outfit = inst.char_state.clothing.clothing_description(verbose=False)
            if rel == "step_sister":
                opening = (
                    f"（深夜，卧室里只开着床头灯。{first_person}穿着{outfit}坐在床边，"
                    f"抱着靠枕看手机。听见你进门，{first_person}抬头看向你。）"
                    "……你怎么还没睡。"
                )
            elif rel == "classmate":
                opening = (
                    f"（午后的自习课，教室里很安静。{first_person}穿着{outfit}坐在你旁边写作业，"
                    f"察觉到你的视线后，{first_person}侧过头看向你。）"
                    "……看什么，作业写完了？"
                )
            elif rel == "childhood_friend":
                opening = (
                    f"（夏日傍晚，小区楼下的长椅旁，{first_person}穿着{outfit}，手里拿着一根冰棍。"
                    f"看到你过来，{first_person}笑着朝你挥手。）喂！这里，我等你好久了。"
                )

        # 所有来源的开场在持久化前走同一个最终清洗入口。
        if opening:
            from world.scenario_engine import normalize_character_output
            opening = normalize_character_output(opening)
        session_name = name or (f"和{char.name}的对话" if char else "新对话")
        now = datetime.now()
        inst.meta = SessionMeta(
            session_id=sid,
            character_id=char_id,
            character_name=char.name if char else "",
            session_name=session_name,
            created_at=now,
            last_active_at=now,
            current_stage=char.allowed_stages[0] if char else "D",
            relationship_stage=char.allowed_stages[0] if char else "D",
            scenario_stage=getattr(card, "stage", None) if card else None,
            total_interactions=0,
            relationship_closeness=(
                initial_closeness
                if initial_closeness is not None
                else getattr(inst, "initial_closeness", char.initial_closeness if char else 0.4)
            ),
            relationship_trust=(
                initial_trust
                if initial_trust is not None
                else inst.char_state.relationship_trust
            ),
            summary="刚开始",
            is_active=True,
            current_location=inst.world.location_name,
            current_time=inst.world.game_time.to_string() if hasattr(inst.world, 'game_time') else "",
            scenario_id=inst.scenario_id,
        )
        if opening:
            inst.messages.append({
                "message_id": uuid.uuid4().hex,
                "turn_id": "opening",
                "timeline_id": inst.timeline_id,
                "message_type": "opening",
                "role": "assistant",
                "content": opening,
                "turn": 0,
                "timestamp": now.isoformat(),
            })

        self.sessions[sid] = inst
        self.active_session_id = sid
        self.capture_snapshot(inst, operation="initial", preview=opening[:80])
        return inst

    def _restore_state_dict(self, state_dict: dict):
        loaded_state, loaded_world, loaded_memory = deserialize_state(state_dict)
        return loaded_state, loaded_world, loaded_memory

    def get_session(self, session_id: str) -> Optional[SessionInstance]:
        return self.sessions.get(session_id)

    def get_active_session(self) -> Optional[SessionInstance]:
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None

    def switch_session(self, session_id: str) -> Optional[SessionInstance]:
        if session_id in self.sessions:
            self.active_session_id = session_id
            self.sessions[session_id].meta.is_active = True
            return self.sessions[session_id]
        return None

    def delete_session(self, session_id: str) -> bool:
        debug("会话", f"删除会话: {session_id}")
        if session_id not in self.sessions:
            return False
        sdir = self._session_dir(session_id)
        if os.path.exists(sdir):
            shutil.rmtree(sdir)
        if self.knowledge_svc:
            self.knowledge_svc.delete_relationship(session_id)
        del self.sessions[session_id]
        if self.active_session_id == session_id:
            self.active_session_id = None
        return True

    def rename_session(self, session_id: str, new_name: str):
        inst = self.sessions.get(session_id)
        if not inst:
            return False
        inst.meta.session_name = new_name
        self._persist_session(inst)
        return True

    def create_save(self, session_id: str, slot_name: str = None) -> str:
        inst = self.sessions.get(session_id)
        if not inst:
            return ""
        slot_id = uuid.uuid4().hex[:8]
        from pathlib import Path
        save_data = {
            "slot_id": slot_id,
            "name": slot_name or f"存档{len(inst.save_slots)+1}",
            "created_at": datetime.now().isoformat(),
            "state_dict": serialize_state(inst.char_state, inst.world, getattr(inst.char_state, "memory", None)) if inst.char_state else {},
            "world_dict": inst.world.to_dict() if inst.world and hasattr(inst.world, 'to_dict') else {},
            "messages": deepcopy(inst.messages),
            "messages_cutoff_idx": len(inst.messages),
            "turn": inst.turn_count,
            "conversation_summary": inst.conversation_summary,
            "compacted_through_turn": min(
                max(int(inst.compacted_through_turn or 0), 0), inst.turn_count
            ),
            "user_profile": deepcopy(inst.user_profile),
            "stage": inst.meta.current_stage,
            "closeness": inst.meta.relationship_closeness,
            "trust": inst.meta.relationship_trust,
            "scenario_id": inst.scenario_id,
            "relationship_memories": self.knowledge_svc.export_relationship(session_id)
            if self.knowledge_svc else [],
            "format_version": CURRENT_SAVE_VERSION,
        }
        # 预览
        preview = ""
        if inst.messages:
            last_msg = inst.messages[-1]
            preview = last_msg.get("content", "")[:40]
        save_data["preview"] = preview
        inst.save_slots[slot_id] = save_data
        self._persist_session(inst)
        return slot_id

    def load_save(self, session_id: str, slot_id: str) -> Optional[SessionInstance]:
        inst = self.sessions.get(session_id)
        if not inst or slot_id not in inst.save_slots:
            return None
        save = _normalize_save(inst.save_slots[slot_id])
        rollback = {
            "char_state": inst.char_state,
            "world": inst.world,
            "messages": inst.messages,
            "turn_count": inst.turn_count,
            "conversation_summary": inst.conversation_summary,
            "compacted_through_turn": inst.compacted_through_turn,
            "user_profile": deepcopy(inst.user_profile),
            "meta": inst.meta,
            "scenario_id": inst.scenario_id,
            "relationship_memories": self.knowledge_svc.export_relationship(session_id)
            if self.knowledge_svc else [],
        }
        try:
            loaded_state = inst.char_state
            loaded_world = inst.world
            if save.get("state_dict"):
                loaded_state, embedded_world, loaded_memory = self._restore_state_dict(save["state_dict"])
                if loaded_memory:
                    loaded_state.memory = loaded_memory
                if embedded_world:
                    loaded_world = embedded_world
            if save.get("world_dict"):
                loaded_world = loaded_world or WorldState()
                loaded_world.load_from_dict(save["world_dict"])

            cutoff = save.get("messages_cutoff_idx", len(inst.messages))
            saved_messages = save.get("messages")
            inst.char_state = loaded_state
            inst.world = loaded_world
            inst.messages = deepcopy(
                saved_messages if saved_messages is not None else inst.messages[:cutoff]
            )
            inst.turn_count = save.get("turn", 0)
            inst.conversation_summary = save.get("conversation_summary", "")
            inst.compacted_through_turn = save.get("compacted_through_turn", 0)
            inst.user_profile = _normalize_user_profile(save.get("user_profile", {}))
            inst.meta = inst.meta.model_copy(deep=True)
            inst.meta.current_stage = save.get("stage", "D")
            inst.meta.relationship_closeness = save.get("closeness", 0.4)
            inst.meta.relationship_trust = save.get("trust", 0.4)
            inst.scenario_id = save.get("scenario_id", inst.meta.scenario_id)
            inst.meta.scenario_id = inst.scenario_id
            if self.knowledge_svc:
                if "relationship_memories" in save:
                    memories = save.get("relationship_memories", [])
                else:
                    memories = self._filter_relationship_memories(
                        rollback["relationship_memories"], save
                    )
                self.knowledge_svc.replace_relationship(session_id, memories)
            self._persist_session(inst)
            return inst
        except Exception:
            inst.char_state = rollback["char_state"]
            inst.world = rollback["world"]
            inst.messages = rollback["messages"]
            inst.turn_count = rollback["turn_count"]
            inst.conversation_summary = rollback["conversation_summary"]
            inst.compacted_through_turn = rollback["compacted_through_turn"]
            inst.user_profile = rollback["user_profile"]
            inst.meta = rollback["meta"]
            inst.scenario_id = rollback["scenario_id"]
            if self.knowledge_svc:
                self.knowledge_svc.replace_relationship(
                    session_id, rollback["relationship_memories"]
                )
            raise

    @staticmethod
    def _filter_relationship_memories(items: List[dict], save: dict) -> List[dict]:
        cutoff_turn = save.get("turn", 0)
        created_at = save.get("created_at", "")
        eligible = []
        for item in items:
            turn_end = item.get("source_turn_end")
            if turn_end is not None and turn_end <= cutoff_turn:
                eligible.append(item)
            elif turn_end is None and item.get("created_at", "") <= created_at:
                eligible.append(item)
        return eligible

    def list_saves(self, session_id: str) -> List[SaveSlotSummary]:
        inst = self.sessions.get(session_id)
        if not inst:
            return []
        result = []
        for slot_id, save in inst.save_slots.items():
            result.append(SaveSlotSummary(
                slot_id=slot_id,
                name=save.get("name", "未命名"),
                created_at=datetime.fromisoformat(save["created_at"]),
                turn=save.get("turn", 0),
                preview=save.get("preview", ""),
                stage=save.get("stage", ""),
            ))
        result.sort(key=lambda s: s.created_at, reverse=True)
        return result

    def delete_save(self, session_id: str, slot_id: str) -> bool:
        inst = self.sessions.get(session_id)
        if not inst or slot_id not in inst.save_slots:
            return False
        del inst.save_slots[slot_id]
        save_file = os.path.join(self._session_dir(session_id), "saves", f"{slot_id}.json")
        if os.path.exists(save_file):
            os.remove(save_file)
        return True

    def branch_session(self, session_id: str, from_slot: str, branch_name: str) -> str:
        """从存档点创建分支（新会话）"""
        inst = self.sessions.get(session_id)
        if not inst or from_slot not in inst.save_slots:
            return ""
        save = _normalize_save(inst.save_slots[from_slot])
        new_sid = uuid.uuid4().hex[:12]
        new_inst = SessionInstance()
        new_inst.char_state = CharacterState()
        new_inst.world = WorldState()
        if save.get("state_dict"):
            loaded_state, loaded_world, loaded_memory = self._restore_state_dict(save["state_dict"])
            new_inst.char_state = loaded_state
            if loaded_memory:
                new_inst.char_state.memory = loaded_memory
            if loaded_world:
                new_inst.world = loaded_world
        if save.get("world_dict"):
            new_inst.world.load_from_dict(save["world_dict"])
        cutoff = save.get("messages_cutoff_idx", 0)
        saved_messages = save.get("messages")
        new_inst.messages = deepcopy(
            saved_messages if saved_messages is not None else inst.messages[:cutoff]
        )
        new_inst.turn_count = save.get("turn", 0)
        new_inst.conversation_summary = save.get("conversation_summary", "")
        new_inst.compacted_through_turn = save.get("compacted_through_turn", 0)
        new_inst.user_profile = _normalize_user_profile(save.get("user_profile", {}))
        new_inst.scenario_id = save.get("scenario_id", inst.scenario_id)
        new_inst.save_slots = {}
        now = datetime.now()
        new_inst.meta = SessionMeta(
            session_id=new_sid,
            character_id=inst.meta.character_id,
            character_name=inst.meta.character_name,
            session_name=branch_name,
            created_at=now,
            last_active_at=now,
            current_stage=save.get("stage", "D"),
            total_interactions=save.get("turn", 0),
            relationship_closeness=save.get("closeness", 0.4),
            relationship_trust=save.get("trust", 0.4),
            summary=f"分支自: {inst.meta.session_name}",
            scenario_id=new_inst.scenario_id,
        )
        self.sessions[new_sid] = new_inst
        if self.knowledge_svc:
            if "relationship_memories" in save:
                memories = []
                for raw_item in save.get("relationship_memories", []):
                    item = deepcopy(raw_item)
                    item["memory_id"] = uuid.uuid4().hex[:12]
                    item["associated_session_id"] = new_sid
                    item["associated_char_id"] = inst.meta.character_id
                    memories.append(item)
                self.knowledge_svc.replace_relationship(new_sid, memories)
            else:
                source_items = self.knowledge_svc.export_relationship(session_id)
                eligible = self._filter_relationship_memories(source_items, save)
                cloned = []
                for item in eligible:
                    item = deepcopy(item)
                    item["memory_id"] = uuid.uuid4().hex[:12]
                    item["associated_session_id"] = new_sid
                    item["associated_char_id"] = inst.meta.character_id
                    cloned.append(item)
                self.knowledge_svc.replace_relationship(new_sid, cloned)
        self._persist_session(new_inst)
        return new_sid

    def branch_from_snapshot(self, session_id: str, snapshot_id: str, branch_name: str) -> str:
        inst = self.sessions.get(session_id)
        snapshot = self.get_snapshot(session_id, snapshot_id)
        if not inst or not snapshot:
            return ""
        new_sid = uuid.uuid4().hex[:12]
        new_inst = SessionInstance()
        state, embedded_world, memory = self._restore_state_dict(snapshot.get("state_dict", {}))
        if memory:
            state.memory = memory
        new_inst.char_state = state
        new_inst.world = embedded_world or WorldState()
        if snapshot.get("world_dict"):
            new_inst.world.load_from_dict(snapshot["world_dict"])
        new_inst.messages = deepcopy(snapshot.get("messages", []))
        new_inst.turn_count = snapshot.get("turn", 0)
        new_inst.conversation_summary = snapshot.get("conversation_summary", "")
        new_inst.compacted_through_turn = snapshot.get("compacted_through_turn", 0)
        new_inst.user_profile = _normalize_user_profile(snapshot.get("user_profile", {}))
        new_inst.scenario_id = snapshot.get("scenario_id")
        now = datetime.now()
        source_meta = snapshot.get("meta", {})
        new_inst.meta = SessionMeta(
            session_id=new_sid,
            character_id=inst.meta.character_id,
            character_name=inst.meta.character_name,
            session_name=branch_name,
            created_at=now,
            last_active_at=now,
            current_stage=source_meta.get("current_stage", "D"),
            total_interactions=new_inst.turn_count,
            relationship_closeness=source_meta.get("relationship_closeness", 0.4),
            relationship_trust=source_meta.get("relationship_trust", 0.4),
            summary=f"分支自: {inst.meta.session_name}",
            current_location=source_meta.get("current_location", ""),
            current_time=source_meta.get("current_time", ""),
            scenario_id=new_inst.scenario_id,
        )
        for message in new_inst.messages:
            message["timeline_id"] = new_inst.timeline_id
        self.sessions[new_sid] = new_inst
        if self.knowledge_svc:
            memories = []
            for raw_item in snapshot.get("relationship_memories", []):
                item = deepcopy(raw_item)
                item["memory_id"] = uuid.uuid4().hex[:12]
                item["associated_session_id"] = new_sid
                item["associated_char_id"] = inst.meta.character_id
                memories.append(item)
            self.knowledge_svc.replace_relationship(new_sid, memories)
        self.capture_snapshot(new_inst, operation="branch", preview=snapshot.get("preview", ""))
        return new_sid

    def auto_save(self, session_id: str) -> Optional[dict]:
        """覆盖_autosave槽并返回轻量元数据；活跃事务期间不读取瞬态状态。"""
        debug("存档", f"自动存档触发: session={session_id}")
        inst = self.sessions.get(session_id)
        if not inst:
            return None
        if self.transaction_active(session_id):
            return {"status": "deferred"}
        inst.save_slots["_autosave"] = {
            "slot_id": "_autosave",
            "name": "自动存档",
            "created_at": datetime.now().isoformat(),
            "state_dict": serialize_state(inst.char_state, inst.world, getattr(inst.char_state, "memory", None)) if inst.char_state else {},
            "world_dict": inst.world.to_dict() if inst.world and hasattr(inst.world, 'to_dict') else {},
            "messages": deepcopy(inst.messages),
            "messages_cutoff_idx": len(inst.messages),
            "turn": inst.turn_count,
            "conversation_summary": inst.conversation_summary,
            "compacted_through_turn": min(
                max(int(inst.compacted_through_turn or 0), 0), inst.turn_count
            ),
            "user_profile": deepcopy(inst.user_profile),
            "stage": inst.meta.current_stage,
            "closeness": inst.meta.relationship_closeness,
            "trust": inst.meta.relationship_trust,
            "scenario_id": inst.scenario_id,
            "relationship_memories": self.knowledge_svc.export_relationship(session_id)
            if self.knowledge_svc else [],
            "format_version": CURRENT_SAVE_VERSION,
            "preview": "(自动存档)",
        }
        self._persist_session(inst)
        autosave = inst.save_slots["_autosave"]
        return {
            key: autosave[key]
            for key in ("slot_id", "name", "created_at", "turn", "stage", "preview", "format_version")
        }
