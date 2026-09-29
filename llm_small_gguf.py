"""
小模型 GGUF 后端：用 llama.cpp llama-server 跑 Qwen2.5-1.5B GGUF（GPU），
只为 generate / classify / extract_structured 提速（结构化提取是对话热路径的最大瓶颈）。

约束（保证不影响原有对话逻辑）：
- generate 返回与 llm_small.generate 相同的纯文本；JSON 解析在 llm_small.extract_structured 内完成。
- embed 仍走 transformers(CPU)，向量空间不变，FAISS 索引兼容。
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Optional

import httpx

from app.logger import info, success, warning, error, debug
from app.config import (
    SMALL_LLAMA_BIN_PATH,
    SMALL_LLAMA_GGUF_PATH,
    SMALL_LLAMA_SERVER_HOST,
    SMALL_LLAMA_SERVER_PORT,
    SMALL_LLAMA_CTX_TOKENS,
    SMALL_LLAMA_N_GPU,
    SMALL_LLAMA_SERVER_TIMEOUT,
)

_BASE_URL = f"http://{SMALL_LLAMA_SERVER_HOST}:{SMALL_LLAMA_SERVER_PORT}"
_proc: Optional[subprocess.Popen] = None
_loaded = False
_last_load_error = ""


def is_available() -> bool:
    return os.path.isfile(SMALL_LLAMA_GGUF_PATH) and os.path.isfile(SMALL_LLAMA_BIN_PATH)


def is_loaded() -> bool:
    try:
        r = httpx.get(f"{_BASE_URL}/health", timeout=3.0)
        return r.status_code == 200 and r.json().get("status") in ("ok", "ok_no_slots")
    except Exception:
        return False


def load() -> bool:
    global _proc, _loaded, _last_load_error
    if not is_available():
        _last_load_error = "小模型 GGUF 或 llama-server 不存在"
        return False
    if is_loaded():
        _loaded = True
        return True
    cmd = [
        SMALL_LLAMA_BIN_PATH,
        "-m", SMALL_LLAMA_GGUF_PATH,
        "--host", SMALL_LLAMA_SERVER_HOST,
        "--port", str(SMALL_LLAMA_SERVER_PORT),
        "-ngl", str(SMALL_LLAMA_N_GPU),
        "-c", str(SMALL_LLAMA_CTX_TOKENS),
        "--jinja",
        "-fa", "on",
        "-np", "1",
        "--no-webui",
    ]
    info("小模型", f"[gguf] 启动小模型 llama-server: {' '.join(cmd)}")
    try:
        _proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        _last_load_error = f"小模型 llama-server 启动失败: {exc}"
        error("小模型", _last_load_error)
        _proc = None
        return False
    deadline = time.monotonic() + SMALL_LLAMA_SERVER_TIMEOUT
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            _last_load_error = f"小模型 llama-server 退出 (exit={_proc.returncode})"
            error("小模型", _last_load_error)
            return False
        if is_loaded():
            _loaded = True
            success("小模型", f"✓ 小模型 GGUF 就绪 ({SMALL_LLAMA_GGUF_PATH})")
            return True
        time.sleep(1.0)
    _last_load_error = "小模型 llama-server 启动超时"
    unload()
    error("小模型", _last_load_error)
    return False


def unload() -> None:
    global _proc, _loaded
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None
    _loaded = False


def generate(prompt: str, system_prompt: str = "", max_new_tokens: int = 120,
             temperature: float = 0.3) -> str:
    """OpenAI 兼容 /v1/chat/completions；不可用时返回空字符串（调用方降级）。"""
    if not is_loaded():
        if not load():
            return ""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": "qwen2.5-1.5b-instruct",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.85,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{_BASE_URL}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"] or ""
        return content.strip()
    except Exception as exc:
        warning("小模型", f"[gguf] 生成失败: {exc}")
        return ""
