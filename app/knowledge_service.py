"""
轻量级知识库服务 - 三层记忆系统（关键词 + FAISS向量检索混合）
- shared: 全局共享知识（世界观、通用剧情参考）
- character_private: 角色私有记忆（背景、喜好、恐惧）
- relationship: 关系层记忆（共同回忆、认知标签）

FAISS索引使用小模型(Qwen2.5-1.5B)生成embedding，模型不可用时自动降级为纯关键词检索。
"""
import hashlib
import os
import json
import uuid
import re
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from app.config import KNOWLEDGE_DIR, SHARED_KNOWLEDGE_DIR, RELATIONS_DIR, EMBEDDING_DIMENSION
from app.logger import info, debug, warning

try:
    import faiss
    _faiss_available = True
except ImportError:
    _faiss_available = False
    faiss = None


class MemoryItem:
    """记忆条目"""
    def __init__(self, content: str, layer: str = "relationship", memory_type: str = "episodic",
                 importance: float = 0.5, tags: List[str] = None,
                 associated_char_id: str = "", associated_session_id: str = "",
                 emotional_valence: float = 0.0, emotional_arousal: float = 0.0,
                 source_turn_start: Optional[int] = None,
                 source_turn_end: Optional[int] = None, origin: str = "chat",
                 entities: List[str] = None, specificity: float = 0.5,
                 status: str = "active", supersedes: Optional[str] = None,
                 content_hash: str = ""):
        self.memory_id = str(uuid.uuid4())[:12]
        self.layer = layer
        self.type = memory_type
        self.content = content
        self.importance = importance
        self.tags = tags or []
        self.associated_char_id = associated_char_id
        self.associated_session_id = associated_session_id
        self.emotional_valence = emotional_valence
        self.emotional_arousal = emotional_arousal
        self.created_at = datetime.now().isoformat()
        self.last_recalled = self.created_at
        self.decay_factor = 1.0
        self.recall_count = 0
        self.source_turn_start = source_turn_start
        self.source_turn_end = source_turn_end
        self.origin = origin
        self.entities = list(dict.fromkeys(entities or []))
        self.specificity = min(max(float(specificity), 0.0), 1.0)
        self.status = status if status in {"active", "superseded"} else "active"
        self.supersedes = supersedes
        normalized = re.sub(r"\s+", "", content).casefold()
        self.content_hash = content_hash or hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()[:20]

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "layer": self.layer,
            "type": self.type,
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "associated_char_id": self.associated_char_id,
            "associated_session_id": self.associated_session_id,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "created_at": self.created_at,
            "last_recalled": self.last_recalled,
            "decay_factor": self.decay_factor,
            "recall_count": self.recall_count,
            "source_turn_start": self.source_turn_start,
            "source_turn_end": self.source_turn_end,
            "origin": self.origin,
            "entities": self.entities,
            "specificity": self.specificity,
            "status": self.status,
            "supersedes": self.supersedes,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryItem':
        m = cls(
            content=d.get("content", ""),
            layer=d.get("layer", "relationship"),
            memory_type=d.get("type", "episodic"),
            importance=d.get("importance", 0.5),
            tags=d.get("tags", []),
            associated_char_id=d.get("associated_char_id", ""),
            associated_session_id=d.get("associated_session_id", ""),
            emotional_valence=d.get("emotional_valence", 0.0),
            emotional_arousal=d.get("emotional_arousal", 0.0),
            source_turn_start=d.get("source_turn_start"),
            source_turn_end=d.get("source_turn_end"),
            origin=d.get("origin", "chat"),
            entities=d.get("entities", []),
            specificity=d.get("specificity", 0.5),
            status=d.get("status", "active"),
            supersedes=d.get("supersedes"),
            content_hash=d.get("content_hash", ""),
        )
        m.memory_id = d.get("memory_id", m.memory_id)
        m.created_at = d.get("created_at", m.created_at)
        m.last_recalled = d.get("last_recalled", m.last_recalled)
        m.decay_factor = d.get("decay_factor", 1.0)
        m.recall_count = d.get("recall_count", 0)
        return m


def _extract_keywords(text: str) -> List[str]:
    """Extract whitespace words plus Chinese bi/tri-grams for fallback recall."""
    text = re.sub(r"[，。！？、；：\"'（）【】\n\r]", " ", text)
    stopwords = {"的", "了", "是", "我", "你", "他", "她", "它", "在", "有", "和", "就", "都", "而", "及", "与", "或", "一个", "没有", "到", "会", "要", "这", "那"}
    keywords = []
    for word in text.split():
        word = word.strip().casefold()
        if len(word) >= 2 and word not in stopwords:
            keywords.append(word)
        elif len(word) == 1 and word not in stopwords and not re.match(r"[\d\s]", word):
            keywords.append(word)
        for segment in re.findall(r"[一-鿿]{2,}", word):
            for size in (2, 3):
                keywords.extend(
                    segment[index:index + size]
                    for index in range(len(segment) - size + 1)
                )
    return list(dict.fromkeys(keywords))


class _LayerIndex:
    """单层索引：关键词映射 + FAISS向量索引"""

    def __init__(self, path_prefix: str, dim: int = EMBEDDING_DIMENSION):
        self.path_prefix = path_prefix
        self.dim = dim
        self.keyword_map: Dict[str, List[str]] = {}
        self.memories: List[MemoryItem] = []
        self.memory_by_id: Dict[str, MemoryItem] = {}
        self.vector_memory_ids: List[str] = []
        self.faiss_index = None
        if _faiss_available:
            self.faiss_index = faiss.IndexFlatIP(dim)

    def _json_path(self) -> str:
        return self.path_prefix + ".json"

    def _faiss_path(self) -> str:
        return self.path_prefix + ".faiss"

    def load(self):
        """加载权威JSON；旧关键词映射会迁移为规范条目。"""
        self.keyword_map = {}
        self.memories = []
        self.memory_by_id = {}
        self.vector_memory_ids = []
        json_path = self._json_path()
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("items"), list):
                    items = [MemoryItem.from_dict(item) for item in data["items"]]
                    keyword_map = data.get("keyword_map", {})
                    self.vector_memory_ids = list(data.get("vector_memory_ids", []))
                else:
                    deduped = {}
                    keyword_map = {}
                    for keyword, raw_items in (data.items() if isinstance(data, dict) else []):
                        ids = []
                        for raw_item in raw_items if isinstance(raw_items, list) else []:
                            item = MemoryItem.from_dict(raw_item)
                            deduped.setdefault(item.memory_id, item)
                            ids.append(item.memory_id)
                        keyword_map[keyword] = list(dict.fromkeys(ids))
                    items = list(deduped.values())
                    self.vector_memory_ids = []
                self.memories = items
                self.memory_by_id = {item.memory_id: item for item in items}
                self.keyword_map = {
                    keyword: [memory_id for memory_id in ids if memory_id in self.memory_by_id]
                    for keyword, ids in keyword_map.items()
                    if isinstance(ids, list)
                }
                self._ensure_keyword_entries()
            except Exception as e:
                warning("知识库", f"加载JSON失败 {json_path}: {e}")

        self._load_faiss()

    def _ensure_keyword_entries(self):
        for item in self.memories:
            for keyword in _extract_keywords(item.content):
                ids = self.keyword_map.setdefault(keyword, [])
                if item.memory_id not in ids:
                    ids.append(item.memory_id)

    def _new_faiss_index(self):
        return faiss.IndexFlatIP(self.dim) if _faiss_available else None

    def _discard_vectors(self):
        self.vector_memory_ids = []
        self.faiss_index = self._new_faiss_index()

    def _load_faiss(self):
        if not _faiss_available:
            self.faiss_index = None
            self.vector_memory_ids = []
            return
        faiss_path = self._faiss_path()
        if not os.path.exists(faiss_path):
            self._discard_vectors()
            return
        try:
            index = faiss.read_index(faiss_path)
            valid = (
                index.d == self.dim
                and index.ntotal == len(self.vector_memory_ids)
                and all(memory_id in self.memory_by_id for memory_id in self.vector_memory_ids)
            )
            if not valid:
                warning("知识库", f"FAISS元数据不一致，将忽略向量索引: {faiss_path}")
                self._discard_vectors()
                return
            self.faiss_index = index
        except Exception as e:
            warning("知识库", f"加载FAISS失败 {faiss_path}: {e}")
            self._discard_vectors()

    def _rebuild_faiss(self):
        """重建向量加速，不删除无法生成向量的规范记忆。"""
        self._discard_vectors()
        if not self.memories or not _faiss_available:
            return
        from llm_small import embed
        embeddings = []
        vector_ids = []
        info("知识库", f"正在重建FAISS索引，共{len(self.memories)}条记忆...")
        for item in self.memories:
            try:
                embedding = embed(item.content)
            except Exception as e:
                debug("知识库", f"生成embedding失败: {e}")
                continue
            vector = self._valid_vector(embedding)
            if vector is not None:
                embeddings.append(vector)
                vector_ids.append(item.memory_id)
        if embeddings:
            self.faiss_index.add(np.stack(embeddings).astype(np.float32))
            self.vector_memory_ids = vector_ids
            info("知识库", f"FAISS索引重建完成，有效向量{len(vector_ids)}个")

    def _valid_vector(self, embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != self.dim or not np.all(np.isfinite(vector)):
            return None
        return vector

    def save(self):
        """原子保存权威JSON，FAISS失败不会破坏记忆数据。"""
        directory = os.path.dirname(self.path_prefix)
        os.makedirs(directory, exist_ok=True)
        data = {
            "version": 2,
            "dimension": self.dim,
            "items": [item.to_dict() for item in self.memories],
            "keyword_map": self.keyword_map,
            "vector_memory_ids": self.vector_memory_ids,
        }
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
            ) as temp_file:
                json.dump(data, temp_file, ensure_ascii=False, indent=2)
                temp_path = temp_file.name
            os.replace(temp_path, self._json_path())
        except Exception as e:
            warning("知识库", f"保存JSON失败: {e}")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return

        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                faiss.write_index(self.faiss_index, self._faiss_path())
            except Exception as e:
                warning("知识库", f"保存FAISS失败: {e}")

    def add(self, item: MemoryItem, embedding: Optional[np.ndarray] = None):
        """添加规范记忆，并在向量有效时建立显式行映射。"""
        if item.memory_id in self.memory_by_id:
            return
        keywords = list(dict.fromkeys(_extract_keywords(item.content)))
        item.tags = list(dict.fromkeys(item.tags + keywords))
        self.memories.append(item)
        self.memory_by_id[item.memory_id] = item
        for keyword in keywords:
            self.keyword_map.setdefault(keyword, []).append(item.memory_id)

        vector = self._valid_vector(embedding)
        if self.faiss_index is not None and vector is not None:
            self.faiss_index.add(vector.reshape(1, -1))
            self.vector_memory_ids.append(item.memory_id)

    def remove_all(self):
        self.keyword_map = {}
        self.memories = []
        self.memory_by_id = {}
        self._discard_vectors()

    def search(self, keywords: List[str], query_embedding: Optional[np.ndarray] = None,
               top_k: int = 5, layer_score_boost: float = 1.0) -> List[Tuple[MemoryItem, float]]:
        """返回候选及分数，不在分层检索阶段修改召回元数据。"""
        scored: Dict[str, float] = {}
        for keyword in keywords:
            for memory_id in self.keyword_map.get(keyword, []):
                item = self.memory_by_id.get(memory_id)
                if item is None:
                    continue
                score = (
                    item.importance
                    * item.decay_factor
                    * (0.75 + 0.5 * item.specificity)
                    * layer_score_boost
                )
                if item.status != "active":
                    continue
                scored[memory_id] = scored.get(memory_id, 0.0) + score

        vector = self._valid_vector(query_embedding)
        if self.faiss_index is not None and vector is not None and self.faiss_index.ntotal > 0:
            distances, indices = self.faiss_index.search(vector.reshape(1, -1), top_k * 2)
            for index, score in zip(indices[0], distances[0]):
                if index < 0 or index >= len(self.vector_memory_ids):
                    continue
                memory_id = self.vector_memory_ids[index]
                item = self.memory_by_id.get(memory_id)
                if item is None or item.status != "active":
                    continue
                vector_score = ((float(score) + 1.0) / 2.0) * layer_score_boost * 1.5
                scored[memory_id] = scored.get(memory_id, 0.0) + vector_score

        results = [
            (self.memory_by_id[memory_id], score)
            for memory_id, score in scored.items()
            if memory_id in self.memory_by_id
        ]
        results.sort(key=lambda pair: (-pair[1], pair[0].memory_id))
        return results[:top_k]


class KnowledgeService:
    """
    三层知识库服务
    - shared: 全局共享知识
    - character_private: 角色私有记忆
    - relationship: 关系层记忆
    """

    def __init__(self):
        self.shared = _LayerIndex(os.path.join(SHARED_KNOWLEDGE_DIR, "index"))
        self.char_layers: Dict[str, _LayerIndex] = {}
        self.session_layers: Dict[str, _LayerIndex] = {}
        self._load_all()

    def _char_path(self, char_id: str) -> str:
        return os.path.join(KNOWLEDGE_DIR, "characters", char_id, "index")

    def _session_path(self, session_id: str) -> str:
        return os.path.join(KNOWLEDGE_DIR, "relations", session_id, "index")

    def _load_all(self):
        self.shared.load()
        # 加载已有角色层
        char_dir = os.path.join(KNOWLEDGE_DIR, "characters")
        if os.path.exists(char_dir):
            for cid in os.listdir(char_dir):
                path = self._char_path(cid)
                layer = _LayerIndex(path)
                layer.load()
                self.char_layers[cid] = layer
        # 加载已有关系层
        rel_dir = os.path.join(KNOWLEDGE_DIR, "relations")
        if os.path.exists(rel_dir):
            for sid in os.listdir(rel_dir):
                path = self._session_path(sid)
                layer = _LayerIndex(path)
                layer.load()
                self.session_layers[sid] = layer

    def _get_or_create_layer(self, layer_name: str, char_id: str = "", session_id: str = "") -> _LayerIndex:
        if layer_name == "shared":
            return self.shared
        elif layer_name == "character_private":
            if char_id not in self.char_layers:
                self.char_layers[char_id] = _LayerIndex(self._char_path(char_id))
            return self.char_layers[char_id]
        else:
            if session_id not in self.session_layers:
                self.session_layers[session_id] = _LayerIndex(self._session_path(session_id))
            return self.session_layers[session_id]

    def list_memories_page(self, layer_name: str, char_id: str = "", session_id: str = "",
                           limit: int = 50, cursor: Optional[str] = None) -> Tuple[List[MemoryItem], Optional[str]]:
        if limit < 1 or limit > 100:
            raise ValueError("分页大小必须在1到100之间")
        if layer_name not in {"shared", "character_private", "relationship"}:
            raise ValueError("不支持的知识层")
        if layer_name == "character_private" and not char_id:
            raise ValueError("缺少角色ID")
        if layer_name == "relationship" and not session_id:
            raise ValueError("缺少会话ID")
        items = list(self._get_or_create_layer(layer_name, char_id, session_id).memories)
        items.sort(key=lambda item: (item.created_at, item.memory_id), reverse=True)
        if cursor:
            try:
                cursor_time, cursor_id = cursor.split("|", 1)
            except ValueError as exc:
                raise ValueError("分页游标无效") from exc
            items = [item for item in items if (item.created_at, item.memory_id) < (cursor_time, cursor_id)]
        page = items[:limit]
        next_cursor = None
        if len(items) > limit and page:
            next_cursor = f"{page[-1].created_at}|{page[-1].memory_id}"
        return page, next_cursor

    def create_api_memory(self, layer_name: str, request, char_id: str = "", session_id: str = "") -> MemoryItem:
        if layer_name not in {"shared", "character_private", "relationship"}:
            raise ValueError("不支持的知识层")
        item = MemoryItem(
            content=request.content.strip(),
            layer=layer_name,
            memory_type=request.type,
            importance=request.importance,
            tags=list(request.tags),
            associated_char_id=char_id,
            associated_session_id=session_id,
            emotional_valence=request.emotional_valence,
            emotional_arousal=request.emotional_arousal,
            origin="api",
        )
        self.add_memory(item)
        return item

    def list_memories(self, layer_name: str, char_id: str = "", session_id: str = "") -> List[MemoryItem]:
        layer = self._get_or_create_layer(layer_name, char_id, session_id)
        return list(layer.memories)

    def export_relationship(self, session_id: str) -> List[dict]:
        return [item.to_dict() for item in self.list_memories("relationship", session_id=session_id)]

    def replace_relationship(self, session_id: str, items: List[dict], save: bool = True):
        layer = self._get_or_create_layer("relationship", session_id=session_id)
        layer.remove_all()
        for raw_item in items:
            item = raw_item if isinstance(raw_item, MemoryItem) else MemoryItem.from_dict(raw_item)
            item.layer = "relationship"
            item.associated_session_id = session_id
            layer.add(item)
        if save:
            layer.save()

    def clone_relationship(self, session_id: str, new_session_id: str, char_id: str = "") -> List[dict]:
        source = self.export_relationship(session_id)
        cloned = []
        for raw_item in source:
            item = MemoryItem.from_dict(raw_item)
            item.memory_id = str(uuid.uuid4())[:12]
            item.associated_session_id = new_session_id
            if char_id:
                item.associated_char_id = char_id
            cloned.append(item.to_dict())
        self.replace_relationship(new_session_id, cloned)
        return cloned

    def delete_relationship(self, session_id: str):
        layer = self.session_layers.pop(session_id, None)
        if layer is None:
            return
        layer.remove_all()
        directory = os.path.dirname(layer.path_prefix)
        if os.path.isdir(directory):
            shutil.rmtree(directory)

    def flush(self):
        self.shared.save()
        for layer in self.char_layers.values():
            layer.save()
        for layer in self.session_layers.values():
            layer.save()

    # --- 添加记忆 ---
    def add_memory(self, item: MemoryItem, save: bool = True):
        """Add a canonical memory unless the same active fact already exists."""
        debug("记忆", f"写入记忆: layer={item.layer}, type={item.memory_type}, char={item.associated_char_id}, session={item.associated_session_id}, 内容={item.content[:30]}")

        layer = self._get_or_create_layer(
            item.layer,
            item.associated_char_id,
            item.associated_session_id,
        )
        duplicate = next(
            (
                existing
                for existing in layer.memories
                if existing.status == "active"
                and existing.content_hash == item.content_hash
                and existing.associated_char_id == item.associated_char_id
                and existing.associated_session_id == item.associated_session_id
            ),
            None,
        )
        if duplicate:
            duplicate.importance = max(duplicate.importance, item.importance)
            duplicate.specificity = max(duplicate.specificity, item.specificity)
            duplicate.tags = list(dict.fromkeys(duplicate.tags + item.tags))
            duplicate.entities = list(dict.fromkeys(duplicate.entities + item.entities))
            if item.source_turn_start is not None:
                starts = [
                    value for value in (
                        duplicate.source_turn_start,
                        item.source_turn_start,
                    ) if value is not None
                ]
                duplicate.source_turn_start = min(starts)
            if item.source_turn_end is not None:
                duplicate.source_turn_end = max(
                    duplicate.source_turn_end or 0,
                    item.source_turn_end,
                )
            if save:
                layer.save()
            return duplicate

        if item.supersedes:
            superseded = layer.memory_by_id.get(item.supersedes)
            if superseded and superseded.status == "active":
                superseded.status = "superseded"

        embedding = None
        if _faiss_available:
            try:
                from llm_small import embed
                embedding = embed(item.content)
            except Exception as e:
                debug("知识库", f"生成embedding失败: {e}")
        layer.add(item, embedding)
        if save:
            layer.save()
        return item

    # --- 检索 ---
    def query(self, session_id: str, char_id: str, query_text: str,
              top_k: int = 5, layers: List[str] = None) -> List[MemoryItem]:
        """
        三层检索：先关系层→角色层→共享层，合并去重返回top_k。
        混合关键词匹配与FAISS向量相似度。
        """
        if layers is None:
            layers = ["relationship", "character_private", "shared"]

        keywords = _extract_keywords(query_text)
        query_embedding = None
        if _faiss_available:
            try:
                from llm_small import embed
                query_embedding = embed(query_text)
            except Exception as e:
                debug("知识库", f"生成query embedding失败: {e}")

        candidates: List[Tuple[MemoryItem, float, _LayerIndex]] = []
        if "relationship" in layers and session_id in self.session_layers:
            layer = self.session_layers[session_id]
            candidates.extend((item, score, layer) for item, score in layer.search(
                keywords, query_embedding, top_k, 1.5
            ))
        if "character_private" in layers and char_id in self.char_layers:
            layer = self.char_layers[char_id]
            candidates.extend((item, score, layer) for item, score in layer.search(
                keywords, query_embedding, top_k, 1.2
            ))
        if "shared" in layers:
            candidates.extend((item, score, self.shared) for item, score in self.shared.search(
                keywords, query_embedding, top_k, 1.0
            ))

        merged: Dict[str, Tuple[MemoryItem, float, _LayerIndex]] = {}
        for item, score, layer in candidates:
            previous = merged.get(item.memory_id)
            if previous is None:
                merged[item.memory_id] = (item, score, layer)
            else:
                merged[item.memory_id] = (item, previous[1] + score, previous[2])
        ranked = sorted(merged.values(), key=lambda row: (-row[1], row[0].memory_id))[:top_k]
        recalled_at = datetime.now().isoformat()
        dirty_layers = set()
        for item, _, layer in ranked:
            item.recall_count += 1
            item.last_recalled = recalled_at
            dirty_layers.add(layer)
        for layer in dirty_layers:
            layer.save()
        return [item for item, _, _ in ranked]

    def retrieve_for_prompt(self, session_id: str, char_id: str, query_text: str,
                            current_emotion: str = "", current_location: str = "",
                            current_stage: str = "") -> str:
        """检索记忆并格式化为prompt注入文本"""
        combined_query = query_text
        if current_emotion:
            combined_query += " " + current_emotion
        if current_location:
            combined_query += " " + current_location
        if current_stage:
            combined_query += " " + current_stage

        memories = self.query(session_id, char_id, combined_query, top_k=5)
        if not memories:
            return ""

        lines = ["## 她记得的事情"]
        for m in memories:
            provenance = ""
            if m.source_turn_start is not None:
                end_turn = m.source_turn_end or m.source_turn_start
                provenance = f" [来源回合 {m.source_turn_start}-{end_turn}]"
            lines.append(f"- {m.content}{provenance}")
        return "\n".join(lines)

    # --- 记忆形成 ---
    def should_form_memory(self, user_input: str, assistant_response: str,
                          state_snapshot: dict, turn_count: int) -> Optional[MemoryItem]:
        """Form a source-grounded memory only for a durable, specific event."""
        importance = 0.0
        specificity = 0.65
        emotion = str(state_snapshot.get("emotion", ""))
        arousal = float(state_snapshot.get("arousal", 0) or 0)
        combined = f"{user_input}\n{assistant_response}"
        reasons = []

        important_keywords = [
            "第一次", "初吻", "告白", "分手", "结婚", "生日", "跨年",
            "争吵", "误会", "和解", "承诺", "约定", "边界",
        ]
        matched_events = [keyword for keyword in important_keywords if keyword in combined]
        if matched_events:
            importance = 0.8
            specificity = 0.85
            reasons.append("事件：" + "、".join(matched_events[:3]))

        # 性事标志性事件（体位/偏好/敏感点/高潮）——让性交细节也能进入长期记忆
        sexual_keywords = ["骑乘", "后入", "口交", "乳交", "69", "女上位", "侧躺",
                          "最敏感", "敏感点", "最舒服", "最喜欢这样", "高潮", "内射"]
        matched_sexual = [k for k in sexual_keywords if k in combined]
        if matched_sexual and arousal > 0.4:
            importance = max(importance, 0.55)
            specificity = max(specificity, 0.8)
            reasons.append("性事：" + "、".join(matched_sexual[:3]))


        disclosure_patterns = ["其实", "我告诉你", "秘密", "小时候", "害怕", "喜欢", "讨厌"]
        matched_disclosures = [pattern for pattern in disclosure_patterns if pattern in combined]
        if matched_disclosures:
            importance = max(importance, 0.6)
            specificity = max(specificity, 0.75)
            reasons.append("披露：" + "、".join(matched_disclosures[:3]))

        # 情绪词匹配对话原文（emotion 字段是英文键，中文词永远不会命中）
        if arousal > 0.7 or any(word in combined for word in ("哭", "怒", "生气", "难过", "伤心", "眼泪")):
            importance = max(importance, 0.65)
            reasons.append(f"状态：情绪={emotion or '未知'}，唤醒度={arousal:.2f}")

        if turn_count > 0 and turn_count % 20 == 0:
            importance = max(importance, 0.5)
            reasons.append(f"关系阶段：{state_snapshot.get('stage', 'D')}")

        if importance < 0.3:
            return None

        def excerpt(text: str, limit: int = 120) -> str:
            normalized = re.sub(r"\s+", " ", text).strip()
            return normalized if len(normalized) <= limit else normalized[:limit - 1] + "…"

        user_excerpt = excerpt(user_input)
        assistant_excerpt = excerpt(assistant_response)
        source_parts = []
        if user_excerpt:
            source_parts.append(f"用户说：“{user_excerpt}”")
        if assistant_excerpt:
            source_parts.append(f"角色回应：“{assistant_excerpt}”")
        content = f"第{turn_count}回合，{'；'.join(source_parts)}"
        if reasons:
            content += "（" + "；".join(reasons) + "）"

        entities = []
        for key in ("character_name", "location", "current_location"):
            value = str(state_snapshot.get(key, "")).strip()
            if value and value not in entities:
                entities.append(value)

        return MemoryItem(
            content=content,
            layer="relationship",
            memory_type="episodic",
            importance=importance,
            associated_char_id=state_snapshot.get("character_id", ""),
            associated_session_id=state_snapshot.get("session_id", ""),
            emotional_valence=0.5 if "喜欢" in emotion or "开心" in emotion else 0.0,
            emotional_arousal=arousal,
            source_turn_start=turn_count,
            source_turn_end=turn_count,
            entities=entities,
            specificity=specificity,
        )
