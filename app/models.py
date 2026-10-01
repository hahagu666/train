"""
Pydantic数据模型 - API层使用
所有API请求/响应都通过这些模型验证
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Literal


# ===== 通用响应 =====
class StandardResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: Optional[Any] = None


# ===== 模型状态 =====
class ModelStatusResponse(BaseModel):
    main_model_loaded: bool = False
    main_model_device: str = "cpu"
    main_model_quantization: str = "none"
    small_model_loaded: bool = False
    small_model_available: bool = False
    embedder_loaded: bool = False
    loading: bool = False
    load_progress: float = 0.0
    main_model_state: str = "unloaded"
    main_model_busy: bool = False
    main_model_error: str = ""
    main_model_device_map: str = ""
    main_model_dtype: str = ""
    main_model_footprint_bytes: int = 0
    cuda_memory: Dict[str, int] = Field(default_factory=dict)
    attention_backend: str = ""
    use_cache: bool = False
    last_generation: Dict[str, Any] = Field(default_factory=dict)
    queue_depth: int = 0
    current_job_id: Optional[str] = None
    current_session_id: Optional[str] = None


# ===== 用户人设 =====
class UserProfileBase(BaseModel):
    name: str = "你"
    gender: str = "unknown"
    age: int = 0
    personality_description: str = ""
    speaking_style_hint: str = ""
    appearance_hint: str = ""


class UserProfileResponse(UserProfileBase):
    relation_to_characters: Dict[str, str] = Field(default_factory=dict)
    custom_name_preference: Dict[str, str] = Field(default_factory=dict)
    extracted_traits: List[str] = Field(default_factory=list)


class SessionUserProfileInput(BaseModel):
    """Optional identity overrides captured when a session is created."""
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, max_length=80)
    gender: Optional[str] = Field(default=None, max_length=40)
    age: Optional[int] = Field(default=None, ge=0, le=120)
    personality_description: Optional[str] = Field(default=None, max_length=1200)
    speaking_style_hint: Optional[str] = Field(default=None, max_length=500)
    appearance_hint: Optional[str] = Field(default=None, max_length=1000)
    traits: Optional[List[str]] = Field(default=None, max_length=30)
    relation: Optional[str] = Field(default=None, max_length=120)
    preferred_address: Optional[str] = Field(default=None, max_length=80)
    additional_context: Optional[str] = Field(default=None, max_length=1200)

    @field_validator(
        "name", "gender", "personality_description", "speaking_style_hint",
        "appearance_hint", "relation", "preferred_address", "additional_context",
    )
    @classmethod
    def strip_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("traits")
    @classmethod
    def clean_traits(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return None
        result = []
        for value in values:
            item = str(value).strip()
            if item and len(item) <= 80 and item not in result:
                result.append(item)
        return result


class SetUserProfileRequest(BaseModel):
    description: str = Field(..., description="用户自然语言自我描述")


class UpdateUserNameRequest(BaseModel):
    name: str


class SetRelationRequest(BaseModel):
    character_id: str
    relation: str
    custom_address: Optional[str] = None


# ===== 角色 =====
class CharacterAppearance(BaseModel):
    hair: str = "黑色长发"
    eyes: str = "杏眼"
    height: int = 158
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bust_cm: Optional[float] = None
    cup_size: str = ""  # 罩杯（A/B/C/D/E），形象描写用
    body_type: str = "纤细"
    skin: str = "白皙"
    visual_age_hint: Optional[str] = None
    # 身体特征：性交描写时用于刻画真实感受（形象化）
    vagina_depth_cm: float = 12.0  # 阴道深度（cm）
    vagina_tightness: str = "适中"  # 穴口紧度，带形象描述，如"紧致——手指贴得很紧，只留一条窄缝，进去时能清晰感到每寸包裹"
    extra: Dict[str, Any] = Field(default_factory=dict)


class CharacterPersonality(BaseModel):
    shyness: float = Field(default=0.5, ge=0.0, le=1.0)
    tsundere: float = Field(default=0.2, ge=0.0, le=1.0)
    gentle: float = Field(default=0.5, ge=0.0, le=1.0)
    jealousy: float = Field(default=0.4, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.3, ge=0.0, le=1.0)
    maturity: float = Field(default=0.2, ge=0.0, le=1.0)
    extra: Dict[str, float] = Field(default_factory=dict)


class CharacterSpeechStyle(BaseModel):
    tone: str = "温柔"
    use_particles: bool = True
    first_person: str = "我"
    address_user: Dict[str, str] = Field(default_factory=dict)
    extra_hints: str = ""
    # 用户自定义回复风格案例：作为模型输出格式/措辞示范注入
    style_example: str = Field(default="", max_length=2000)


class BodyParams(BaseModel):
    sensitivity_base: float = 1.0
    sensitivity_erogenous: float = 1.2
    arousal_speed: float = 1.0
    arousal_decay_rate: float = 1.0
    orgasm_threshold: float = 0.82
    orgasm_intensity_base: float = 1.0
    refractory_period: float = 300.0
    wetness_base: float = 0.3
    multiple_orgasm_capable: bool = True
    skin_sensitivity: float = 1.0
    breast_sensitivity: float = 1.0
    clitoral_sensitivity: float = 1.0
    g_spot_sensitivity: float = 0.8
    anal_sensitivity: float = 0.3
    body_temperature: float = 36.5
    heart_rate_base: int = 72
    stamina: float = 1.0


class MindParams(BaseModel):
    shyness_base: float = 0.5
    moral_inhibition: float = 0.5
    resistance_threshold: float = 0.7
    initial_resistance_sexual: float = 0.9
    trust_open_rate: float = 1.0
    jealousy_tendency: float = 0.4
    attachment_style: str = "secure"
    physical_contact_comfort: float = 0.3
    eye_contact_shy: bool = True
    vocal_suppression_tendency: float = 0.6
    initiative_base: float = 0.2
    teasing_tendency: float = 0.1
    cry_tendency: float = 0.4
    emotional_volatility: float = 0.3


class EmotionalParams(BaseModel):
    trust_gain_rate: float = 1.0
    trust_loss_rate: float = 1.5
    hurt_heal_rate: float = 0.8
    pleasure_decay_rate: float = 0.7
    embarrassment_recovery_rate: float = 0.6
    fear_recovery_rate: float = 0.5
    forgiveness_rate: float = 0.6
    positive_memory_strength: float = 1.0
    negative_memory_strength: float = 1.3
    novelty_decay_rate: float = 0.3


class CharacterBase(BaseModel):
    id: str
    name: str
    avatar: str = ""
    background: str = ""
    avatar_revision: int = 0
    background_revision: int = 0
    is_preset: bool = False
    age: Optional[int] = None
    adult_verified: bool = False
    sexual_interaction_allowed: bool = False
    eligibility_reason: str = "未完成成年资格确认"
    eligibility_version: int = 1
    relationship_type: str = "classmate"
    character_description: str = ""
    appearance: CharacterAppearance = Field(default_factory=CharacterAppearance)
    personality: CharacterPersonality = Field(default_factory=CharacterPersonality)
    speech_style: CharacterSpeechStyle = Field(default_factory=CharacterSpeechStyle)
    backstory: str = ""
    body_params: BodyParams = Field(default_factory=BodyParams)
    mind_params: MindParams = Field(default_factory=MindParams)
    emotional_params: EmotionalParams = Field(default_factory=EmotionalParams)
    initial_outfit: str = "summer_home"
    initial_closeness: float = 0.4
    initial_trust: float = 0.4
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    limits: Dict[str, List[str]] = Field(default_factory=lambda: {"soft": [], "hard": []})
    allowed_stages: List[str] = Field(default_factory=lambda: ["A", "B", "C", "D", "E"])

    def refresh_eligibility(self) -> bool:
        from core.content_policy import refresh_character_eligibility

        return refresh_character_eligibility(self).allowed


class CharacterSummary(BaseModel):
    id: str
    name: str
    avatar: str
    background: str = ""
    avatar_revision: int = 0
    background_revision: int = 0
    is_preset: bool
    relationship_type: str
    current_stage: str = "D"
    description_preview: str


class CharacterLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft: List[str] = Field(default_factory=list, max_length=20)
    hard: List[str] = Field(default_factory=list, max_length=20)


class CreateCharacterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=40)
    age: int = Field(ge=0, le=120)
    adult_confirmed: bool = False
    relationship_type: str = "classmate"  # 自由字符串：妹妹/青梅竹马/同学/…或自定义，前端直接传中文
    character_description: str = Field(min_length=1, max_length=1200)
    backstory: str = Field(default="", max_length=2000)
    appearance: CharacterAppearance = Field(default_factory=CharacterAppearance)
    personality: CharacterPersonality = Field(default_factory=CharacterPersonality)
    speech_style: CharacterSpeechStyle = Field(default_factory=CharacterSpeechStyle)
    body_params: BodyParams = Field(default_factory=BodyParams)  # 体力百分比 stamina(0-1) 等身体参数
    initial_outfit: str = Field(default="casual", min_length=1, max_length=80)
    initial_closeness: float = Field(default=0.3, ge=0.0, le=1.0)
    initial_trust: float = Field(default=0.3, ge=0.0, le=1.0)
    likes: List[str] = Field(default_factory=list, max_length=20)
    dislikes: List[str] = Field(default_factory=list, max_length=20)
    fears: List[str] = Field(default_factory=list, max_length=20)
    limits: CharacterLimits = Field(default_factory=CharacterLimits)

    @field_validator("name", "character_description", "backstory", "initial_outfit")
    @classmethod
    def strip_text(cls, value, info):
        value = value.strip()
        if info.field_name in ("name", "character_description", "initial_outfit") and not value:
            raise ValueError("该字段不能为空")
        return value

    @field_validator("likes", "dislikes", "fears")
    @classmethod
    def clean_list(cls, values):
        result = []
        for value in values:
            item = str(value).strip()
            if item and len(item) <= 80 and item not in result:
                result.append(item)
        return result

    @model_validator(mode="after")
    def validate_confirmation(self):
        if self.adult_confirmed and self.age < 18:
            raise ValueError("未满 18 岁不能确认成年资格")
        return self


class UpdateCharacterRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0, le=120)
    adult_verified: Optional[bool] = None
    character_description: Optional[str] = None
    personality_description: Optional[str] = None
    backstory: Optional[str] = None
    appearance: Optional[CharacterAppearance] = None
    body_params: Optional[BodyParams] = None
    initial_outfit: Optional[str] = None
    likes: Optional[List[str]] = None
    dislikes: Optional[List[str]] = None
    fears: Optional[List[str]] = None


class AppSettings(BaseModel):
    version: int = 1
    theme: Literal["system", "light", "dark"] = "system"
    language: Literal["zh-CN"] = "zh-CN"
    auto_save: bool = True
    stream_responses: bool = True


class UpdateAppSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: Optional[Literal["system", "light", "dark"]] = None
    language: Optional[Literal["zh-CN"]] = None
    auto_save: Optional[bool] = None
    stream_responses: Optional[bool] = None


# ===== 会话 =====
class ScenarioSummary(BaseModel):
    id: str
    title: str
    stage: str
    stage_name: str
    location: str = ""
    context_preview: str = ""


class SessionMeta(BaseModel):
    session_id: str
    character_id: str
    character_name: str = ""
    session_name: str
    created_at: datetime
    last_active_at: datetime
    current_stage: str = "D"
    relationship_stage: str = "D"
    scenario_stage: Optional[str] = None
    total_interactions: int = 0
    relationship_closeness: float = 0.4
    relationship_trust: float = 0.4
    summary: str = ""
    is_active: bool = False
    current_location: str = "卧室"
    current_time: str = ""
    scenario_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_stage_fields(cls, data):
        """Migrate legacy current_stage while keeping J/K as scenario categories."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        legacy = str(values.get("current_stage") or "D").upper()
        relationship = values.get("relationship_stage")
        scenario = values.get("scenario_stage")
        if relationship is None:
            relationship = legacy if legacy in "ABCDEFGHI" else "D"
        if scenario is None and legacy in {"J", "K"}:
            scenario = legacy
        values["relationship_stage"] = relationship
        values["current_stage"] = relationship
        values["scenario_stage"] = scenario
        return values


class CreateSessionRequest(BaseModel):
    character_id: str
    name: Optional[str] = None
    scenario_id: Optional[str] = None
    custom_opening: Optional[str] = Field(default=None, max_length=2000)
    initial_trust: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    initial_closeness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    initial_outfit: Optional[str] = None
    custom_outfit_description: Optional[str] = Field(default=None, min_length=1, max_length=500)
    location: Optional[str] = None
    time_str: Optional[str] = None
    privacy: Optional[Literal["私密", "半公开", "危险"]] = None
    user_profile: Optional[SessionUserProfileInput] = None

    @field_validator("custom_outfit_description")
    @classmethod
    def validate_custom_outfit_description(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("自定义衣着描述不能为空")
        return value


class RenameSessionRequest(BaseModel):
    name: str


class SessionStateSnapshot(BaseModel):
    """精简状态快照，给前端展示用"""
    arousal: float = 0.0
    phase: str = "baseline"
    emotion: str = ""
    trust: float = 0.0
    closeness: float = 0.0
    heart_rate: int = 72
    clothing: str = ""
    clothing_layers: dict = {}
    time_str: str = ""
    location: str = "卧室"
    privacy: str = "私密"
    privacy_level: float = 0.7
    danger_level: float = 0.1
    detection_risk: float = 0.0
    has_people: bool = False
    others_present: bool = False
    stage: str = "D"
    relationship_stage: str = "D"
    scenario_stage: Optional[str] = None
    acceptance: float = 0.3
    stamina: float = 1.0
    fatigue: float = 0.0
    is_orgasm: bool = False
    is_interrupted: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_stage_fields(cls, data):
        if not isinstance(data, dict):
            return data
        values = dict(data)
        legacy = str(values.get("stage") or "D").upper()
        relationship = values.get("relationship_stage")
        scenario = values.get("scenario_stage")
        if relationship is None:
            relationship = legacy if legacy in "ABCDEFGHI" else "D"
        if scenario is None and legacy in {"J", "K"}:
            scenario = legacy
        values["relationship_stage"] = relationship
        values["stage"] = relationship
        values["scenario_stage"] = scenario
        return values


# ===== 存档 =====
class SaveSlotSummary(BaseModel):
    slot_id: str
    name: str
    created_at: datetime
    turn: int = 0
    preview: str = ""
    stage: str = ""


class CreateSaveRequest(BaseModel):
    name: Optional[str] = None


class LoadSaveRequest(BaseModel):
    slot_id: str


class BranchRequest(BaseModel):
    from_slot: Optional[str] = None
    snapshot_id: Optional[str] = None
    name: str


class RollbackRequest(BaseModel):
    snapshot_id: str


class TurnOperationRequest(BaseModel):
    suggestion: Optional[str] = None
    request_id: Optional[str] = Field(default=None, max_length=64)


# ===== 消息 =====
class ChatMessageModel(BaseModel):
    message_id: str
    timestamp: datetime
    role: str  # user/assistant/system/narrator
    content: str
    turn: int = 0
    turn_id: str = ""
    timeline_id: str = ""
    snapshot_id: Optional[str] = None
    state_snapshot: Optional[SessionStateSnapshot] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    request_id: Optional[str] = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    response: str
    state: SessionStateSnapshot
    messages: List[ChatMessageModel] = Field(default_factory=list)
    job_id: str = ""
    request_id: str = ""


class CreateKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    type: Literal["episodic", "semantic", "emotional", "procedural"] = "semantic"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    emotional_arousal: float = Field(default=0.0, ge=0.0, le=1.0)


class KnowledgePage(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    next_cursor: Optional[str] = None
    has_more: bool = False


# ===== WebSocket消息类型 =====
class WSMessage(BaseModel):
    type: str
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
