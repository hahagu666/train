"""
主模型推理后端路由。

依据 config.MAIN_MODEL_BACKEND 在两种实现间切换，对外暴露与 llm_qwen.py
一致的公共接口，供 chat_orchestrator / server 通过 `import model_backend as llm_qwen`
透明调用：
  - llama:          llm_llama.py   (llama-server GGUF, CUDA, 快)
  - transformers:   llm_qwen.py    (bitsandbytes NF4)
"""
from app.config import MAIN_MODEL_BACKEND
from app.logger import info

if MAIN_MODEL_BACKEND == "llama":
    from llm_llama import (
        DEFAULT_SYSTEM_PROMPT,
        generate,
        generate_stream,
        get_status,
        is_available,
        is_loaded,
        load_model,
        reload_model,
        unload_model,
        GenerationCancelledError,
        GenerationLengthLimitError,
        GenerationOOMError,
        GenerationTimeoutError,
        ModelBusyError,
    )
    info("大模型", f"推理后端: llama.cpp (llama-server GGUF)")
else:
    from llm_qwen import (
        DEFAULT_SYSTEM_PROMPT,
        generate,
        generate_stream,
        get_status,
        is_available,
        is_loaded,
        load_model,
        reload_model,
        unload_model,
        GenerationCancelledError,
        GenerationLengthLimitError,
        GenerationOOMError,
        GenerationTimeoutError,
        ModelBusyError,
    )
    info("大模型", f"推理后端: transformers (bitsandbytes)")
