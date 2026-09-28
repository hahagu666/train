# AI Roleplay Social Backend - Architecture Design

> 版本: 1.0  
> 日期: 2026-08-05  
> 状态: 设计稿待审核

---

## 1. 项目概述

### 1.1 目标
构建一个通用的AI角色扮演社交后端框架，支持：
- 多个AI角色（预设3个：妹妹、青梅竹马、同桌）+ 用户自定义角色
- 每个角色独立状态、记忆、关系发展
- 完整的生理-情绪-认知仿真系统驱动角色反应
- 三层RAG知识库（全局/角色私有/关系层）
- 多存档/分支时间线（Galgame式存档读档）
- WebSocket流式输出
- 用户可自定义自身人设（自然语言描述，无需格式）
- RESTful API，前后端分离，前端可任意实现

### 1.2 非目标
- 不内置前端页面（由前端独立实现）
- 不做模型训练（微调小模型是未来可选优化）
- 不做多用户/鉴权系统（本地单机应用，单用户）
- 不做语音/图片生成（纯文本+未来扩展）

---

## 2. 目录结构

```
e:\sister_train\
├── app/                          # 新增：应用核心层
│   ├── __init__.py
│   ├── server.py                 # FastAPI应用入口
│   ├── config.py                 # 全局配置
│   ├── models.py                 # Pydantic数据模型（API层）
│   ├── character_manager.py      # 角色管理
│   ├── session_manager.py        # 会话/存档管理
│   ├── user_profile.py           # 用户人设管理
│   ├── chat_orchestrator.py      # 对话编排（核心调度器）
│   ├── knowledge_service.py      # RAG知识库服务
│   ├── memory_compactor.py       # 记忆压缩/摘要
│   ├── preset_characters.py      # 3个预设角色定义
│   └── deps.py                   # FastAPI依赖注入
├── core/                         # 已有：仿真核心（不改结构）
│   ├── body/                     # 神经网络+身体部位
│   ├── ans.py                    # 自主神经系统
│   ├── emotion.py                # 情绪模型
│   ├── mind_state.py             # 主观认知层
│   ├── orgasm.py                 # 高潮系统
│   ├── sensitivity.py            # 动态敏感度
│   ├── state.py                  # SisterState → 重命名为 CharacterState
│   ├── parser.py                 # 动作解析器
│   ├── beat_controller.py        # 节奏控制器
│   ├── sensation.py              # 感受生成器
│   ├── post_processor.py         # 输出后处理
│   ├── memory.py                 # 情感记忆（身体记忆）
│   ├── skills.py                 # 技能经验
│   └── serialization.py          # 序列化
├── world/                        # 已有：世界系统（不改结构）
│   ├── world_state.py            # 世界状态
│   ├── time_system.py            # 时间系统
│   ├── events.py                 # 事件引擎
│   ├── clothing.py               # 衣物系统
│   └── scenario_engine.py        # 剧情卡引擎
├── models/                       # 已有：模型文件
│   ├── Qwen--Qwen2.5-7B-Instruct/
│   └── Qwen--Qwen2.5-1.5B-Instruct/  (未来)
├── data/                         # 重构：数据持久化
│   ├── characters/               # 角色定义JSON
│   │   ├── imouto.json           # 妹妹预设
│   │   ├── childhood_friend.json # 青梅竹马预设
│   │   ├── desk_mate.json        # 同桌预设
│   │   └── custom_{id}.json      # 用户自定义角色
│   ├── sessions/                 # 会话/存档
│   │   └── {session_id}/
│   │       ├── meta.json         # 会话元数据
│   │       ├── state.json        # CharacterState快照
│   │       ├── world.json        # WorldState快照
│   │       ├── messages.jsonl    # 对话历史
│   │       ├── saves/            # 手动存档点
│   │       │   └── {slot_id}.json
│   │       └── memory/           # 该会话的长期记忆
│   │           ├── episodic.jsonl   # 情景记忆
│   │           ├── semantic.jsonl   # 语义记忆
│   │           └── embeddings/      # 向量索引
│   ├── knowledge/                # RAG知识库
│   │   ├── shared/               # 全局共享层
│   │   │   ├── worldview.md      # 世界观设定
│   │   │   └── scenarios/        # 通用剧情参考
│   │   └── relations/            # 关系层记忆（跨会话）
│   │       └── {char_id}_{rel_id}/
│   ├── user_profile.json         # 用户自身人设
│   └── embeddings_cache/         # 全局向量缓存
├── JB/                           # 已有：剧情卡原文
├── llm_qwen.py                   # 已有：7B主模型适配器（改：支持流式）
├── llm_small.py                  # 已有：1.5B小模型适配器
├── chat.py                       # 保留：命令行客户端（改成调用API）
├── train/                        # 已有：训练脚本
└── docs/
    ├── architecture_design.md
    └── specs/
        └── 2026-08-05-backend-architecture.md (本文件)
```

---

## 3. 数据模型

### 3.1 Character（角色）

```python
class CharacterBase:
    id: str                      # "imouto" / "childhood_friend" / "desk_mate" / "custom_xxx"
    name: str                    # 角色名，如"林小雨"
    avatar: str                  # 头像路径
    is_preset: bool              # 是否预设角色
    relationship_type: str       # "step_sister" / "childhood_friend" / "classmate" / "teacher" /自定义
    character_description: str   # 自然语言描述（用户输入/预设写好的）
    
    # === 角色外观 ===
    appearance: dict             # {"hair":"黑色长直发","eyes":"杏眼","height":158,...}
    
    # === 性格参数（影响情绪/行为倾向）===
    personality: dict            # {"shyness":0.7,"tsundere":0.4,"gentle":0.6,"jealousy":0.5,...}
    
    # === 说话风格 ===
    speech_style: dict           # {"tone":"软糯","use_particles":True,"first_person":"我",...}
    
    # === 背景故事 ===
    backstory: str               # 自然语言背景
    
    # === 仿真参数（决定身体反应差异）===
    body_params: dict            # 见3.1.1
    mind_params: dict            # 见3.1.2
    emotional_params: dict       # 见3.1.3
    initial_outfit: str          # 初始穿着 "pajamas"/"school_uniform"/"summer_home"
    initial_closeness: float     # 初始亲密度 0-1
    initial_trust: float         # 初始信任度 0-1
    
    # === 喜好/厌恶 ===
    likes: List[str]
    dislikes: List[str]
    fears: List[str]
    limits: dict                 # {"soft":[...],"hard":[...]}
    
    # === 可用剧情阶段 ===
    allowed_stages: List[str]    # ["A","B","C","D","E",...] 关系进展解锁
```

#### 3.1.1 body_params（身体参数）
```python
{
    "sensitivity_base": 1.0,        # 基础敏感度 0.5-1.5
    "sensitivity_erogenous": 1.2,   # 敏感带敏感度倍率
    "arousal_speed": 1.0,           # 唤起累积速度
    "arousal_decay_rate": 1.0,      # 唤起消退速度
    "orgasm_threshold": 0.82,       # 高潮阈值
    "orgasm_intensity_base": 1.0,   # 高潮基础强度
    "refractory_period": 300,       # 不应期（秒）
    "wetness_base": 0.3,            # 基础润滑度
    "multiple_orgasm_capable": True,# 是否能连续高潮
    "skin_sensitivity": 1.0,        # 皮肤敏感度
    "breast_sensitivity": 1.0,      # 胸部敏感度
    "clitoral_sensitivity": 1.0,    # 阴蒂敏感度
    "g_spot_sensitivity": 0.8,      # G点敏感度
    "anal_sensitivity": 0.3,        # 后庭敏感度
    "body_temperature": 36.5,       # 基础体温
    "heart_rate_base": 72,          # 基础心率
    "stamina": 1.0,                 # 体力/耐力
}
```

#### 3.1.2 mind_params（心理参数）
```python
{
    "shyness_base": 0.5,            # 基础害羞度
    "moral_inhibition": 0.5,        # 道德/禁忌抑制（影响禁果效应强度）
    "resistance_threshold": 0.7,    # 推开门槛
    "initial_resistance_sexual": 0.9, # 初始对性接触的抗拒
    "trust_open_rate": 1.0,         # 信任打开速度
    "jealousy_tendency": 0.4,       # 吃醋倾向
    "attachment_style": "secure",   # secure/anxious/avoidant
    "physical_contact_comfo**": 0.3, # 身体接触舒适度（初始）
    "eye_contact_shy": True,        # 对视会害羞
    "vocal_suppression_tendency": 0.6, # 抑制声音倾向
    "initiative_base": 0.2,         # 主动行动倾向
    "teasing_tendency": 0.1,        # 调戏/挑逗倾向
    "cry_tendency": 0.4,            # 爱哭程度
    "emotional_volatility": 0.3,    # 情绪波动度
}
```

#### 3.1.3 emotional_params（情绪参数）
```python
{
    "trust_gain_rate": 1.0,
    "trust_loss_rate": 1.5,         # 失去信任比获得快
    "hurt_heal_rate": 0.8,
    "pleasure_decay_rate": 0.7,
    "embarrassment_recovery_rate": 0.6,
    "fear_recovery_rate": 0.5,
    "forgiveness_rate": 0.6,        # 原谅速度
    "positive_memory_strength": 1.0,
    "negative_memory_strength": 1.3, # 负面记忆更强
    "novelty_decay_rate": 0.3,      # 新鲜感衰减
}
```

### 3.2 UserProfile（用户人设）
```python
class UserProfile:
    name: str                       # 用户名字（如"哥"/"陈老师"）
    gender: str                     # "male"/"female"/"unknown"
    age: int                        # 年龄（可选，0=未知）
    relation_to_characters: dict    # {char_id: "step_brother"/"teacher"/"classmate"}
    personality_description: str    # 用户输入的自然语言自我描述原文
    extracted_traits: dict          # 小模型提取的结构化特征
    speaking_style_hint: str        # 说话风格（如"命令式"/"温柔"/"爱逗她"）
    appearance_hint: str            # 外貌描述（用户可选填）
    custom_name_preference: dict    # 各角色对用户的称呼（如{"imouto":"哥哥"}）
    
    @classmethod
    async def from_natural_language(cls, text: str) -> 'UserProfile':
        """用小模型从自然语言提取结构化UserProfile"""
        ...
```

### 3.3 Session（会话/存档）
```python
class SessionMeta:
    session_id: str                 # UUID
    character_id: str               # 对应哪个角色
    session_name: str               # 会话名（如"和妹妹的日常"）
    created_at: datetime
    last_active_at: datetime
    current_stage: str              # 当前关系阶段 "A/B/C/D/E/F/G/H/I"
    total_interactions: int         # 总互动轮数
    relationship_closeness: float   # 亲密度（实时计算）
    relationship_trust: float       # 信任度（实时计算）
    summary: str                    # 会话一句话摘要（自动生成）
    is_active: bool                 # 是否正在进行
    
class SessionState:
    meta: SessionMeta
    char_state: CharacterState      # 完整仿真状态（=原来的SisterState，重命名）
    world_state: WorldState         # 世界状态
    messages: List[ChatMessage]     # 对话历史（最近N轮原文）
    episodic_memories: List[MemoryItem]  # 情景记忆
    save_slots: Dict[str, SaveSnapshot] # 手动存档点
```

### 3.4 ChatMessage（消息）
```python
class ChatMessage:
    message_id: str
    timestamp: datetime
    role: str                       # "user"/"assistant"/"system"/"narrator"
    content: str                    # 消息文本
    turn: int                       # 第几轮
    # 仿真快照（每轮结束后记录关键状态）
    state_snapshot: dict            # 精简版状态，用于前端展示
    action_type: str                # 如果是用户输入，解析出的动作类型
    triggered_events: List[str]     # 本轮触发的事件
    scenario_card_id: str           # 如果触发了剧情卡
    is_interrupted: bool            # 本轮是否被中断
    orgasm_this_turn: bool          # 本轮是否高潮
    memory_formation_id: str        # 如果形成了长期记忆
```

### 3.5 MemoryItem（记忆条目，三层）
```python
class MemoryItem:
    memory_id: str
    layer: str                      # "shared"/"character_private"/"relationship"
    type: str                       # "episodic"(事件)/"semantic"(事实)/"emotional"(情感)/"procedural"(技能)
    content: str                    # 文本内容
    embedding: List[float]          # 向量（可选，检索用）
    importance: float               # 重要性 0-1
    timestamp_sim: float            # 游戏内时间戳
    timestamp_real: datetime        # 真实时间
    tags: List[str]                 # 检索标签
    associated_char_id: str         # 关联角色
    associated_user_relation: str   # 关联关系
    decay_factor: float             # 记忆衰减系数
    last_recalled: datetime         # 上次被检索到的时间（回忆强化记忆）
    emotional_valence: float        # 情绪效价 -1(负)~+1(正)
    emotional_arousal: float        # 情绪唤醒度 0-1
```

### 3.6 SaveSnapshot（存档点）
```python
class SaveSnapshot:
    slot_id: str
    name: str                       # 存档名
    created_at: datetime
    char_state_dict: dict           # 序列化的CharacterState
    world_state_dict: dict          # 序列化的WorldState
    messages_cutoff_idx: int        # 到第几条消息为止
    memory_count: int               # 存档时的记忆条数
    preview: str                    # 一句话预览
```

---

## 4. 核心服务设计

### 4.1 CharacterManager（角色管理）

**职责**：角色CRUD、预设角色初始化、角色卡解析、角色列表查询

**接口**：
```python
class CharacterManager:
    def list_presets() -> List[CharacterBase]
    def list_characters() -> List[CharacterBase]
    def get_character(char_id: str) -> CharacterBase
    def create_character_from_text(text: str) -> CharacterBase  # 小模型提取
    def update_character(char_id: str, updates: dict) -> CharacterBase
    def delete_character(char_id: str) -> bool
    def create_preset_characters()  # 初始化3个预设
    def get_simulation_params(char_id: str) -> dict  # 获取仿真参数
```

**预设角色定义**（preset_characters.py）：
- **妹妹（imouto）**：林小雨，继妹，住一起3年，软糯/害羞/傲娇，初始信任0.6/亲密度0.5，校服+睡衣
- **青梅竹马（childhood_friend）**：苏晚晴，从小一起长大，住隔壁，开朗/偶尔强势/爱逗你，初始信任0.7/亲密度0.65
- **同桌（desk_mate）**：沈若溪，高中同桌，高冷学霸/实际害羞/不爱说话，初始信任0.4/亲密度0.35

### 4.2 UserProfileManager（用户人设管理）

**职责**：用户人设提取、更新、查询、对角色的称呼配置

**接口**：
```python
class UserProfileManager:
    def get_profile() -> UserProfile
    async def set_profile_from_text(text: str) -> UserProfile  # 自然语言→提取→保存
    def update_name(name: str)
    def set_relation_to_character(char_id: str, relation: str)
    def get_address_form(char_id: str) -> str  # 该角色该怎么称呼用户
    def reset()
```

**小模型提取逻辑**：
用户输入一段自然语言描述，小模型输出JSON：
```json
{
  "name": "陈默",
  "gender": "male",
  "age": 20,
  "personality_traits": ["温柔", "爱逗她", "有点坏"],
  "relationship": {
    "imouto": "step_brother",
    "childhood_friend": "childhood_friend",
    "desk_mate": "classmate"
  },
  "speaking_style": "喜欢开玩笑，偶尔霸道",
  "how_she_calls_you": {
    "imouto": "哥哥",
    "childhood_friend": "喂",
    "desk_mate": "那个..."
  }
}
```

### 4.3 SessionManager（会话管理）

**职责**：创建/切换/删除会话、存档读档、多分支时间线

**接口**：
```python
class SessionManager:
    def list_sessions(char_id: str = None) -> List[SessionMeta]
    def create_session(char_id: str, name: str = None,
                       start_stage: str = None, scenario_id: str = None) -> SessionState
    def get_session(session_id: str) -> SessionState
    def get_active_session() -> SessionState  # 当前活跃会话
    def switch_session(session_id: str) -> SessionState
    def delete_session(session_id: str) -> bool
    
    # 存档系统（Galgame式）
    def create_save(session_id: str, slot_name: str = None) -> str  # 返回slot_id
    def load_save(session_id: str, slot_id: str) -> SessionState
    def list_saves(session_id: str) -> List[SaveSnapshot]
    def delete_save(session_id: str, slot_id: str) -> bool
    def auto_save(session_id: str)  # 每轮自动存档
    
    # 分支管理
    def branch_session(session_id: str, from_save_slot: str,
                       branch_name: str) -> str  # 创建分支，返回新session_id
    def rename_session(session_id: str, new_name: str)
    
    # 状态持久化
    def persist_session(session_id: str)  # 写入磁盘
    def load_all_sessions()  # 启动时加载
```

**存档机制**：
- 每轮对话结束后自动quick save
- 用户可手动创建命名存档（"初夜前"、"暑假第二天"等）
- 从存档点可以"branch out"开新分支，不影响原来的时间线
- 存档包含完整状态快照+消息截断点，读档完全恢复

### 4.4 KnowledgeService（RAG知识库服务）

**三层知识库架构**：
```
Layer 1 - shared（全局共享）
├─ 世界观/设定（现代都市/高中背景）
├─ 常识性描写参考（接吻怎么写、拥抱怎么写）
└─ 通用剧情模板（节日/天气/校园活动）

Layer 2 - character_private（角色私有，每个角色独立）
├─ 角色的背景故事细节
├─ 角色的喜好/厌恶/恐惧
├─ 她的个人经历（如小时候被狗追所以怕狗）
└─ 她对你的刻板印象（"哥总是爱逗我"）

Layer 3 - relationship（关系层，每个session独立）
├─ 你们的共同回忆（"跨年那天我们第一次接吻"）
├─ 她对你的认知标签（"他说话算数"/"他总是很晚回家"）
├─ 你们之间发生过的特殊事件（争吵、感动、误会）
└─ 未解决的情感张力（"上次的事她还在闹别扭"）
```

**接口**：
```python
class KnowledgeService:
    def __init__(self):
        self.embedder = None  # 向量嵌入模型（用bge-small-zh或主模型pooler）
        self.shared_index = None
        self.char_indices = {}     # {char_id: FAISS index}
        self.session_indices = {}  # {session_id: FAISS index}
    
    async def query(self, session: SessionState, query: str,
                    top_k: int = 5, layers: List[str] = None) -> List[MemoryItem]:
        """三层检索：先关系层→角色层→共享层，合并去重返回top_k"""
        ...
    
    async def add_memory(self, session: SessionState, item: MemoryItem):
        """添加一条记忆到对应层"""
        ...
    
    async def add_shared_knowledge(self, text: str, tags: List[str], layer_type: str = "shared"):
        """管理员接口：添加全局知识/角色背景"""
        ...
    
    async def compact_memories(self, session: SessionState):
        """记忆压缩：把多条短记忆归纳为更少更浓缩的记忆（调用小模型）"""
        ...
    
    async def retrieve_relevant_backstory(self, session: SessionState,
                                          action_context: str) -> str:
        """根据当前动作上下文检索相关背景/记忆，返回注入prompt的文本"""
        ...
```

**向量存储选型**：FAISS（本地、轻量、无需服务）
**嵌入模型**：先不引入独立嵌入模型，1.5B小模型做embed（mean pooling）；后续可换bge-small-zh-v1.5（~100MB）

### 4.5 ChatOrchestrator（对话编排器，核心）

这是最核心的服务，把仿真引擎、模型、RAG、解析器、后处理串起来。

**对话流程（每轮）**：
```
用户消息 → 
  [1] Parser解析动作（如果是动作类输入）
  [2] 加载当前Session状态
  [3] RAG检索相关记忆/背景
  [4] Beat执行 + 仿真tick（如果是动作）
  [5] Sensation采样当前感受
  [6] 检查事件引擎触发（中断/随机事件/剧情卡）
  [7] 组装Prompt（状态+感受+记忆+对话历史+用户消息）
  [8] 调用7B主模型，流式返回token给前端
  [9] 全量响应后，Post Process解析（小模型提取信号）
  [10] 应用状态调整（情绪/接受度/信任/记忆形成）
  [11] 检查是否形成长期记忆（importance>阈值则存入RAG）
  [12] 每N轮自动压缩记忆
  [13] Auto save
  [14] 推送完整状态更新给前端（WebSocket）
```

**接口**：
```python
class ChatOrchestrator:
    def __init__(self, char_mgr, session_mgr, user_mgr, knowledge_svc,
                 llm_generate_fn, llm_stream_fn, small_llm_fn):
        ...
    
    async def process_message_stream(self, session_id: str, user_input: str) -> AsyncGenerator:
        """
        处理用户消息，通过WebSocket/StreamingResponse流式返回。
        yield的事件类型：
        - {"type":"token","content":"..."}          # 逐token输出
        - {"type":"action_start","action":"..."}    # 动作开始执行
        - {"type":"state_update","state":{...}}     # 状态变更（前端展示用）
        - {"type":"event","event":"interrupt","desc":"..."} # 事件触发
        - {"type":"orgasm","intensity":0.9}         # 高潮事件
        - {"type":"scenario","card":"D-F05",...}    # 剧情卡触发
        - {"type":"memory_formed","summary":"..."}  # 形成新记忆
        - {"type":"error","message":"..."}
        - {"type":"done","full_response":"..."}     # 完成
        """
        ...
```

### 4.6 MemoryCompactor（记忆压缩器）

**职责**：对话多了以后，把旧的详细对话历史压缩成摘要记忆，释放context窗口

**策略**：
1. 最近10轮：保留完整原文在context里
2. 10-30轮前：用小模型压缩成情景记忆（每5轮一条摘要）
3. 30轮以前：进一步压缩为高层语义记忆（"她知道你喜欢吃辣""你上次提别的女生她吃醋了"）
4. 压缩不是删除，是从"详细对话"变成"摘要记忆"，通过RAG检索时仍然能找到

---

## 5. API接口设计

### 5.1 REST API

#### 模型状态
```
GET  /api/model/status           # 模型加载状态（7B/1.5B/embedder是否就绪）
POST /api/model/preload          # 预加载模型（启动时可调用）
```

#### 用户人设
```
GET  /api/user/profile           # 获取当前用户人设
PUT  /api/user/profile           # 用自然语言设置/更新人设
                                 # body: {"description": "我是她哥哥..."}
PATCH /api/user/profile/name     # 修改名字
POST /api/user/profile/relation  # 设置与某角色的关系
```

#### 角色管理
```
GET  /api/characters             # 角色列表（含预设+自定义）
GET  /api/characters/presets     # 预设角色列表
GET  /api/characters/{id}        # 角色详情
POST /api/characters             # 创建自定义角色（自然语言描述）
                                 # body: {"description": "..."}
PATCH /api/characters/{id}       # 修改角色属性
DELETE /api/characters/{id}      # 删除自定义角色
```

#### 会话管理
```
GET  /api/sessions               # 会话列表（可按character_id筛选）
POST /api/sessions               # 创建新会话
                                 # body: {"character_id":"imouto","name":"...","scenario_id":"D-D01"}
GET  /api/sessions/{id}          # 会话详情+状态摘要
DELETE /api/sessions/{id}        # 删除会话
PATCH /api/sessions/{id}/name    # 重命名会话
POST /api/sessions/{id}/reset    # 重置会话（重新开始）
```

#### 存档系统
```
GET  /api/sessions/{id}/saves    # 存档列表
POST /api/sessions/{id}/saves    # 创建存档
                                 # body: {"name":"..."}
GET  /api/sessions/{id}/saves/{slot_id}  # 存档详情（预览）
POST /api/sessions/{id}/saves/{slot_id}/load  # 读档
DELETE /api/sessions/{id}/saves/{slot_id}
POST /api/sessions/{id}/branch   # 从存档点创建分支
                                 # body: {"from_slot":"...","name":"新分支"}
```

#### 对话
```
POST /api/chat                   # 非流式对话（用于测试）
                                 # body: {"session_id":"...","message":"..."}
                                 # returns: {"response":"...","state":{...}}
```

#### 知识库管理
```
GET  /api/knowledge/shared       # 全局知识库条目列表
POST /api/knowledge/shared       # 添加全局知识
GET  /api/knowledge/character/{id} # 角色私有记忆
POST /api/knowledge/character/{id}
GET  /api/knowledge/session/{id}   # 关系层记忆（该会话）
```

#### 系统
```
GET  /api/health                 # 健康检查
GET  /api/config                 # 获取系统配置
```

### 5.2 WebSocket接口

**连接**：`WS /ws/chat/{session_id}`

**客户端→服务器消息**：
```json
// 发送消息
{"type":"message","content":"亲她一下"}

// 停止生成
{"type":"stop"}

// 心跳
{"type":"ping"}
```

**服务器→客户端消息**：
```json
// 流式token
{"type":"token","content":"嗯……"}
{"type":"token","content":"她耳"}
{"type":"token","content":"朵一下子红了"}

// 动作开始（提示状态变化）
{"type":"action","action_type":"kiss","targets":["lips"],"through_clothes":false}

// 状态更新（每轮结束后推送完整状态，前端更新UI）
{"type":"state","data":{
  "arousal":0.32,
  "phase":"excitement",
  "emotion":"shyness/shock",
  "trust":0.61,
  "heart_rate":88,
  "clothing":"...",
  "time":"第1天 晚上11点05分",
  "location":"卧室"
}}

// 事件
{"type":"event","event_type":"scenario","card_id":"D-C05","title":"偷偷进你房间"}
{"type":"event","event_type":"interrupt","desc":"走廊传来脚步声"}
{"type":"event","event_type":"orgasm","intensity":0.95}

// 错误
{"type":"error","message":"..."}

// 完成
{"type":"done","full_response":"她耳朵一下子红了……「你、你干嘛……」"}
```

---

## 6. 仿真层适配

### 6.1 SisterState → CharacterState 重命名
现有`core/state.py`的`SisterState`重命名为`CharacterState`，增加：
- `character_id: str` 关联到哪个角色
- 初始化时接受Character的仿真参数（body/mind/emotional_params）覆盖默认值
- `apply_character_template(character: CharacterBase)` 方法

### 6.2 参数注入
每个角色的body_params/mind_params/emotional_params在初始化CharacterState时注入，替换当前的硬编码默认值。例如：
- 同桌（高冷型）：shyness_base=0.8, initiative_base=0.1, moral_inhibition=0.7
- 青梅竹马（开朗型）：shyness_base=0.3, initiative_base=0.4, teasing_tendency=0.4
- 妹妹（软糯型）：shyness_base=0.6, cry_tendency=0.5, moral_inhibition=0.6

### 6.3 关系阶段解锁
当前`scenario_engine.py`的STAGE_INFO已有min_trust/min_privacy/min_arousal门槛，新增：
- 阶段与剧情卡目录直接对应（D-A到D-I）
- 会话推进时根据closeness/trust自动解锁新阶段
- Parser在未解锁阶段时，识别到超前动作会触发抗拒行为（不是报错，是角色真实拒绝）

### 6.4 Parser的"不合时宜动作"处理
Parser识别到动作后，在执行前检查：
- 关系阶段是否允许（如陌生阶段摸胸→拒绝+信任下降）
- 当前场景是否合适（如学校里接吻→尴尬+推开）
- 角色是否在抗拒（active_resistance高→真实推开，不是半推半就）
- 用户人设的关系类型（老师/哥哥/同学的禁忌等级不同，禁果效应加成不同）

不合时宜的动作不是"不执行"，而是**真实执行但产生符合情境的反应**（她会推开、生气、尴尬、震惊），让社交后果自然发生。

---

## 7. 模型层改造

### 7.1 主模型（llm_qwen.py）改造支持流式
```python
async def generate_stream(prompt: str, system_prompt: str = "",
                          max_new_tokens: int = 400,
                          temperature: float = 0.85) -> AsyncGenerator[str, None]:
    """流式生成，逐token yield"""
    ...
```

### 7.2 小模型（llm_small.py）扩展功能
- 用户人设提取（UserProfile.from_natural_language）
- 角色创建提取（create_character_from_text）
- Post Process结构化信号提取（已有）
- 记忆摘要/压缩
- 剧情卡智能选择（已有）
- 记忆重要性评分（新记忆形成时打分）
- 会话自动命名（创建会话后自动生成名字）

### 7.3 嵌入模型
暂不引入独立模型。先用1.5B小模型做mean pooling生成embedding。后续如果需要更高质量检索，加bge-small-zh-v1.5（~100MB，CPU跑很快）。

---

## 8. 记忆形成与检索机制

### 8.1 什么触发记忆形成？
每轮对话结束后，小模型评估是否形成长期记忆：
- **重要事件**：初吻、第一次亲密接触、争吵、感动时刻、说过的重要话
- **强烈情绪**：情绪强度>0.7的时刻（她哭了、她生气了、她高潮了）
- **信息披露**：她告诉你她的秘密/喜好/恐惧
- **重复互动模式**：多次出现类似行为（"你又摸她头"→形成"他喜欢摸我头"的语义记忆）

### 8.2 记忆重要性评分
小模型对新记忆打0-1分：
- 0-0.3：很快遗忘（不存入长期，仅在短期对话里存在）
- 0.3-0.6：存入长期，但会随时间衰减
- 0.6-0.8：深刻记忆，衰减很慢
- 0.8-1.0：核心记忆（第一次、重大事件），几乎不衰减

### 8.3 记忆衰减与强化
- 时间流逝：记忆重要性随游戏内时间缓慢衰减
- 回忆强化：记忆被检索到（对话中提到相关话题）→ important升高，last_recalled更新
- 情绪共振：类似情绪的事件会激活相关记忆（如她吃醋时，会想起上次你提别的女生的事）

### 8.4 检索策略
每轮对话前，根据以下query从三层RAG检索top-k相关记忆：
1. 当前用户输入文本
2. 当前动作类型
3. 当前情绪状态
4. 当前场景/位置
5. 当前关系阶段

合并去重后，筛选最多5条记忆（避免context过载），以"她记得：……"的形式注入prompt。

---

## 9. Prompt构建策略

每轮最终发送给7B主模型的prompt结构：

```
[System: 角色设定]
你是{name}，{role}。
你的性格：{personality_summary}
你的背景：{backstory_summary}
你称呼他为：{address_form}
他的性格：{user_traits}
说话风格：{speech_style}

[System: 当前情境]
{scene_description}
{clothing_description}
当前关系阶段：{stage}
亲密度：{closeness}/信任度：{trust}
当前情绪：{emotion_state}
身体状态：{body_summary}（心率、唤起、是否湿润等）

[System: 相关记忆]
{retrieved_memories}  (0-5条)

[System: 世界观]
{shared_knowledge_snippet}  (仅在需要时)

[System: 任务]
你在和{user_name}互动，写出你的反应。包含动作、神态、声音、可能说的话。
不要做叙述者，你就是{name}。用第三人称写动作，对话用「」括起来。
2-5句话。直接写内容，不要前缀。

[对话历史]
{recent_messages}  (最近10轮)

[感官输入]
{sensation_report}
（如果是动作类输入）他{action_description}

[用户刚才说/做了]
{user_input}
```

---

## 10. 启动流程

```
1. 启动FastAPI服务
   ├─ 检查模型文件是否存在
   ├─ 初始化CharacterManager（加载/创建预设角色）
   ├─ 初始化UserProfileManager（加载user_profile.json）
   ├─ 初始化KnowledgeService（加载共享知识库）
   ├─ 初始化SessionManager（加载所有sessions）
   ├─ 异步预加载主模型到GPU（不阻塞API启动，/api/model/status显示loading）
   └─ 服务就绪，开始监听

2. 用户第一次打开前端
   ├─ 如果用户未设置人设 → 提示"先介绍一下你自己吧"
   ├─ 如果没有会话 → 显示角色列表，选角色后创建新会话（自动选开场剧情卡）
   └─ 开始聊天
```

---

## 11. 错误处理与边界情况

- **模型未加载完成**：API返回503 + 状态提示；WebSocket推送"模型加载中"
- **小模型不存在**：所有小模型功能自动降级（规则关键词/随机选择/正则提取）
- **显存不足**：加载时检测，如果7B fp16失败自动尝试int4量化
- **会话状态损坏**：加载时验证JSON，损坏的会话标记为corrupted，提供恢复选项
- **用户输入无法解析**：Parser无法识别时作为普通对话处理，不触发仿真
- **RAG检索为空**：正常，不注入记忆段
- **长时间生成无输出**：设置超时（30秒），超时后中断返回已有内容
- **WebSocket断连**：支持重连，消息不会丢失（持久化在session里）

---

## 12. 依赖项

新增需要安装的包：
```
fastapi
uvicorn[standard]
websockets
faiss-cpu          # 向量检索（CPU版足够）
pydantic>=2.0      # 数据模型
python-multipart   # 文件上传（未来扩展头像等）
sentence-transformers # （可选，未来换bge嵌入模型时需要）
```

现有依赖保留：
```
torch, transformers, peft, trl, accelerate, datasets, numpy
```

安装命令：
```bash
pip install fastapi uvicorn[standard] websockets faiss-cpu pydantic
```

---

## 13. 实施计划概览

分阶段实施，每个阶段可独立测试：

1. **Phase 1**：重构目录结构 + FastAPI骨架 + Pydantic模型 + 基础CRUD API（角色/会话/用户）
2. **Phase 2**：SisterState→CharacterState重命名 + 仿真参数注入 + 预设角色定义
3. **Phase 3**：WebSocket流式对话 + llm_qwen流式改造 + ChatOrchestrator串起来
4. **Phase 4**：命令行chat.py改成API客户端，验证端到端
5. **Phase 5**：存档/读档/分支系统
6. **Phase 6**：RAG三层知识库 + FAISS集成 + 记忆形成机制
7. **Phase 7**：小模型所有功能点（人设提取/角色创建/记忆压缩/重要性评分）
8. **Phase 8**：关系阶段解锁 + 不合时宜动作真实反应
9. **Phase 9**：用户自然语言人设提取流程完善
10. **Phase 10**：整体测试 + 文档
