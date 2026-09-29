"""
全局配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模型路径
MAIN_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "Qwen--Qwen2.5-7B-Instruct", "snapshots", "master"
)
SMALL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "Qwen--Qwen2.5-1.5B-Instruct", "snapshots", "master"
)

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHARACTERS_DIR = os.path.join(DATA_DIR, "characters")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")
SHARED_KNOWLEDGE_DIR = os.path.join(KNOWLEDGE_DIR, "shared")
RELATIONS_DIR = os.path.join(KNOWLEDGE_DIR, "relations")
EMBEDDINGS_CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")
USER_MEDIA_DIR = os.path.join(DATA_DIR, "user_media")
USER_PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")

# 剧情卡目录
JB_DIR = os.path.join(PROJECT_ROOT, "JB")

# 服务器配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
CORS_ORIGINS = ["*"]  # 本地开发允许所有来源

# 模型配置
MAIN_MODEL_DTYPE = "auto"  # auto/float16/int4/cpu
MAIN_MODEL_DEVICE = "auto"  # auto/cuda/cpu
MAIN_MODEL_FP16_MIN_FREE_GIB = 16.0
MAIN_MODEL_MAX_NEW_TOKENS = int(os.getenv("MAIN_MODEL_MAX_NEW_TOKENS", "256"))
MAIN_MODEL_TEMPERATURE = 0.95
MAIN_MODEL_TOP_P = 0.8
MAIN_MODEL_TOP_K = 20
MAIN_MODEL_REPETITION_PENALTY = 1.15
MAIN_MODEL_ATTENTION = "sdpa"
MAIN_MODEL_USE_CACHE = True
MAIN_MODEL_WARMUP = True
# 预热目标 token 数：用接近生产长度的序列做一次 prefill，把一次性冷启动预填充
# 成本从"首个用户回合"移到"模型加载阶段"，显著降低首回合延迟
MAIN_MODEL_WARMUP_TOKENS = int(os.getenv("MAIN_MODEL_WARMUP_TOKENS", "1800"))
MAIN_MODEL_GENERATION_TIMEOUT = float(os.getenv("MAIN_MODEL_GENERATION_TIMEOUT", "240"))

# ===== 推理后端切换 =====
# llama: 走 llama-server (GGUF, CUDA, 快数倍)；transformers: 走 bitsandbytes NF4
MAIN_MODEL_BACKEND = os.getenv("MAIN_MODEL_BACKEND", "llama")
# llama.cpp 二进制与 GGUF 路径（models/ 已被 .gitignore 排除，不入库）
LLAMA_BIN_PATH = os.getenv(
    "LLAMA_BIN_PATH",
    os.path.join(PROJECT_ROOT, "models", "llama-cpp", "b2", "bin", "llama-server.exe"),
)
LLAMA_GGUF_PATH = os.getenv(
    "LLAMA_GGUF_PATH",
    os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b-instruct-abliterated-v2.Q4_K_M.gguf"),
)
LLAMA_SERVER_HOST = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8081"))
LLAMA_CTX_TOKENS = int(os.getenv("LLAMA_CTX_TOKENS", "6144"))
LLAMA_N_GPU = int(os.getenv("LLAMA_N_GPU", "999"))
LLAMA_SERVER_TIMEOUT = float(os.getenv("LLAMA_SERVER_TIMEOUT", "180"))

# ===== 小模型后端（结构化提取/分类）=====
# generate/classify/extract_structured 走 GGUF (llama.cpp)；embed 仍走 transformers(CPU)，保证 FAISS 向量空间不变。
# 注意：本机 8GB 显卡不足以让两个 llama-server 并发上 GPU（会把 7B 拖垮 60 倍），
# 故小模型强制 -ngl 0 全 CPU，7B 独享 GPU，避免显存争抢与泄漏。
SMALL_LLAMA_BIN_PATH = os.getenv("SMALL_LLAMA_BIN_PATH", LLAMA_BIN_PATH)
SMALL_LLAMA_GGUF_PATH = os.getenv(
    "SMALL_LLAMA_GGUF_PATH",
    os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
)
SMALL_LLAMA_SERVER_HOST = os.getenv("SMALL_LLAMA_SERVER_HOST", "127.0.0.1")
SMALL_LLAMA_SERVER_PORT = int(os.getenv("SMALL_LLAMA_SERVER_PORT", "8082"))
SMALL_LLAMA_CTX_TOKENS = int(os.getenv("SMALL_LLAMA_CTX_TOKENS", "2048"))
SMALL_LLAMA_N_GPU = int(os.getenv("SMALL_LLAMA_N_GPU", "0"))
SMALL_LLAMA_SERVER_TIMEOUT = float(os.getenv("SMALL_LLAMA_SERVER_TIMEOUT", "120"))
# 拒绝方向消融：推理时从残差流中减去"拒绝方向"的投影，削弱基座对齐先验
# 对编排指令的对抗（话题转移式推脱）。方向文件由 train/extract_refusal_direction.py 生成。
REFUSAL_ABLATION = os.getenv("REFUSAL_ABLATION", "1") == "1"
REFUSAL_ABLATION_ALPHA = float(os.getenv("REFUSAL_ABLATION_ALPHA", "1.0"))
REFUSAL_DIRECTION_FILE = os.path.join(PROJECT_ROOT, "models", "refusal_direction_qwen.pt")
SMALL_MODEL_DEVICE = "cpu"
MAX_NEW_TOKENS = MAIN_MODEL_MAX_NEW_TOKENS
TEMPERATURE = MAIN_MODEL_TEMPERATURE
STREAM_CHUNK_SIZE = 1  # 逐token流式输出

# 对话配置
MAX_CONTEXT_MESSAGES = 15  # 兼容旧配置：保留最近N轮原文对话
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2560"))
MEMORY_COMPACT_INTERVAL = 20  # 每N轮自动压缩记忆
MEMORY_IMPORTANCE_THRESHOLD = 0.3  # 重要性>此值存入长期记忆
AUTO_SAVE_INTERVAL = 1  # 每N轮自动存档

# 向量检索配置
EMBEDDING_DIMENSION = 1536  # 后续换bge-small-zh是512，先用小模型的维度
RETRIEVAL_TOP_K = 5

# 确保目录存在
for d in [DATA_DIR, CHARACTERS_DIR, SESSIONS_DIR, KNOWLEDGE_DIR,
          SHARED_KNOWLEDGE_DIR, RELATIONS_DIR, EMBEDDINGS_CACHE_DIR, USER_MEDIA_DIR]:
    os.makedirs(d, exist_ok=True)
