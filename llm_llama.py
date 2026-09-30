"""
llama.cpp 后端适配器：通过 llama-server 的 OpenAI 兼容 HTTP 接口提供
与 llm_qwen.py 相同的生成接口（generate / generate_stream / 生命周期 / 状态）。

要点：
- 由本模块在 load_model 时自动拉起 llama-server.exe 子进程（GPU CUDA 版），
  并通过 127.0.0.1:<port>/v1/chat/completions 调用（OpenAI 格式）。
- 相比 transformers+bitsandbytes NF4，GGUF+llama.cpp 在 CUDA 上 decode 快数倍。
- 无 Python forward hook：拒绝方向消融在本后端不生效（按"先速度后质量"顺序接受）。
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from threading import Event
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from app.logger import info, success, warning, error, debug
from app.config import (
    MAIN_MODEL_BACKEND,
    MAIN_MODEL_MAX_NEW_TOKENS,
    MAIN_MODEL_TEMPERATURE,
    LLAMA_BIN_PATH,
    LLAMA_GGUF_PATH,
    LLAMA_SERVER_HOST,
    LLAMA_SERVER_PORT,
    LLAMA_CTX_TOKENS,
    LLAMA_N_GPU,
    LLAMA_SERVER_TIMEOUT,
)


class ModelBusyError(RuntimeError):
    pass


class GenerationCancelledError(RuntimeError):
    pass


class GenerationTimeoutError(RuntimeError):
    pass


class GenerationOOMError(RuntimeError):
    pass


class GenerationLengthLimitError(RuntimeError):
    pass


DEFAULT_SYSTEM_PROMPT = """你是一个角色扮演AI。根据提供的状态和感受写出角色的反应。
台词直接写；动作、神态、声音和其他非台词内容都用全角中文括号（ ）括起来。
2-4句话，自然简短，直接写内容，不要前缀。"""


_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
_proc: Optional[subprocess.Popen] = None
_loading = False
_loaded = False
_last_load_error = ""
_start_time = 0.0

_last_generation = {
    "prompt_tokens": 0,
    "output_tokens": 0,
    "tokenize_seconds": 0.0,
    "generation_seconds": 0.0,
    "decode_seconds": 0.0,
    "tokens_per_second": 0.0,
    "stop_reason": "",
}


def _health() -> Optional[dict]:
    try:
        r = httpx.get(f"{_BASE_URL}/health", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _health_ok() -> bool:
    try:
        h = _health()
        return h is not None and h.get("status") in ("ok", "ok_no_slots")
    except Exception:
        return False


def is_loaded() -> bool:
    return _health_ok()


def is_available() -> bool:
    """模型文件与 llama-server 二进制是否存在。"""
    return os.path.isfile(LLAMA_GGUF_PATH) and os.path.isfile(LLAMA_BIN_PATH)


def _start_server() -> bool:
    """拉起 llama-server.exe 子进程，等待健康。"""
    global _proc, _loaded, _last_load_error, _start_time
    if not os.path.isfile(LLAMA_BIN_PATH):
        _last_load_error = f"llama-server 不存在: {LLAMA_BIN_PATH}"
        return False
    if not os.path.isfile(LLAMA_GGUF_PATH):
        _last_load_error = f"GGUF 模型不存在: {LLAMA_GGUF_PATH}"
        return False
    if _health_ok():
        _loaded = True
        return True
    _start_time = time.time()
    cmd = [
        LLAMA_BIN_PATH,
        "-m", LLAMA_GGUF_PATH,
        "--host", LLAMA_SERVER_HOST,
        "--port", str(LLAMA_SERVER_PORT),
        "-ngl", str(LLAMA_N_GPU),
        "-c", str(LLAMA_CTX_TOKENS),
        "--jinja",
        "-fa", "on",
        "-np", "1",
        "--no-webui",
    ]
    info("大模型", f"[llama] 启动 llama-server: {' '.join(cmd)}")
    try:
        _proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        _last_load_error = f"llama-server 启动失败: {exc}"
        error("大模型", _last_load_error)
        _proc = None
        return False

    deadline = time.monotonic() + LLAMA_SERVER_TIMEOUT
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            _last_load_error = f"llama-server 进程退出 (exit={_proc.returncode})"
            error("大模型", _last_load_error)
            return False
        if _health_ok():
            _loaded = True
            success("大模型", f"✓ llama-server 就绪 ({LLAMA_GGUF_PATH})")
            return True
        time.sleep(1.0)
    _last_load_error = f"llama-server 启动超时({LLAMA_SERVER_TIMEOUT}s)"
    _terminate_server()
    error("大模型", _last_load_error)
    return False


def _terminate_server() -> None:
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


def load_model(device: str = None) -> bool:
    global _loading
    _loading = True
    try:
        ok = _start_server()
        _loading = False
        return ok
    except Exception as exc:
        _loading = False
        _last_load_error = str(exc)
        error("大模型", f"[llama] 加载失败: {exc}")
        return False


def unload_model() -> bool:
    _terminate_server()
    return True


def reload_model(device: str = None) -> bool:
    _terminate_server()
    return load_model(device)


def _cuda_memory() -> dict:
    """尽力从 nvidia-smi 读取显存；失败时返回零值，不影响状态上报。"""
    zero = {"free_bytes": 0, "total_bytes": 0, "allocated_bytes": 0, "reserved_bytes": 0}
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        parts = [p.strip() for p in out.split(",")] if out else []
        if len(parts) >= 3:
            total, free, used = (float(x) for x in parts[:3])
            return {
                "total_bytes": int(total * 1024 * 1024),
                "free_bytes": int(free * 1024 * 1024),
                "allocated_bytes": int(used * 1024 * 1024),
                "reserved_bytes": int(used * 1024 * 1024),
            }
    except Exception:
        pass
    return zero


def get_status() -> dict:
    global _loading
    h = _health()
    return {
        "state": "loading" if _loading else ("ready" if h else "unloaded"),
        "loaded": h is not None,
        "loading": _loading,
        "busy": False,
        "device": "cuda",
        "quantization": "q4_k_m",
        "backend": MAIN_MODEL_BACKEND,
        "error": _last_load_error,
        "device_map": "cuda:0" if h else "",
        "dtype": "int4",
        "model_footprint_bytes": os.path.getsize(LLAMA_GGUF_PATH) if os.path.isfile(LLAMA_GGUF_PATH) else 0,
        "cuda_memory": _cuda_memory(),
        "attention_backend": "fa",
        "use_cache": True,
        "llama_server": f"{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}",
        "model": LLAMA_GGUF_PATH,
        "last_generation": dict(_last_generation),
    }


def _build_messages(system_prompt: str, user_prompt: str) -> list:
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_prompt},
    ]


def _payload(messages, max_new_tokens, temperature, stream: bool):
    return {
        "model": "qwen2.5-7b-instruct",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.8,
        "frequency_penalty": 0.3,
        "stream": stream,
    }


def generate(prompt: str, system_prompt: str = "",
             max_new_tokens: int = MAIN_MODEL_MAX_NEW_TOKENS,
             temperature: float = MAIN_MODEL_TEMPERATURE,
             cancel_event: Optional[Event] = None,
             timeout: Optional[float] = 120.0) -> str:
    debug("大模型", f"[llama] 完整生成调用: prompt={len(prompt)}字, max_tokens={max_new_tokens}, temperature={temperature}")
    if not is_loaded():
        if not load_model():
            raise RuntimeError(_last_load_error or "llama-server 未加载")
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("生成已取消")
    messages = _build_messages(system_prompt, prompt)
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout or 240.0) as client:
            r = client.post(
                f"{_BASE_URL}/v1/chat/completions",
                json=_payload(messages, max_new_tokens, temperature, stream=False),
            )
            r.raise_for_status()
            data = r.json()
    except httpx.TimeoutException:
        _last_generation.update({"stop_reason": "timeout"})
        raise GenerationTimeoutError("llama-server 生成超时")
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 500:
            _last_generation.update({"stop_reason": "error"})
            raise GenerationLengthLimitError("llama-server 返回500，可能上下文超限") from exc
        raise
    except Exception as exc:
        raise GenerationOOMError(f"llama-server 请求失败: {exc}") from exc

    content = ""
    try:
        content = data["choices"][0]["message"]["content"] or ""
        finish = data.get("choices", [{}])[0].get("finish_reason", "")
    except Exception:
        content = ""
        finish = ""

    elapsed = time.monotonic() - t0
    n_tokens = data.get("usage", {}).get("completion_tokens", 0) or max(
        1, len(content) // 2
    )
    _last_generation.update({
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
        "output_tokens": n_tokens,
        "generation_seconds": round(elapsed, 3),
        "decode_seconds": 0.0,
        "tokens_per_second": round(n_tokens / elapsed, 2) if elapsed else 0.0,
        "stop_reason": finish or ("length" if n_tokens >= max_new_tokens else "eos"),
    })
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("生成已取消")
    if _last_generation["stop_reason"] == "length":
        raise GenerationLengthLimitError(
            f"模型回复达到 {max_new_tokens} token 上限，未检测到自然结束"
        )
    debug("大模型", f"[llama] 完整生成完成: {n_tokens} token, {elapsed:.1f}s, {n_tokens / elapsed:.1f} tok/s, stop={finish}")
    return content


async def generate_stream(prompt: str, system_prompt: str = "",
                          max_new_tokens: int = MAIN_MODEL_MAX_NEW_TOKENS,
                          temperature: float = MAIN_MODEL_TEMPERATURE,
                          cancel_event: Optional[Event] = None,
                          timeout: float = 120.0) -> AsyncGenerator[str, None]:
    debug("大模型", f"[llama] 流式生成调用: prompt={len(prompt)}字, max_tokens={max_new_tokens}, temperature={temperature}")
    if not is_loaded():
        if not load_model():
            yield "[模型未加载，请先下载模型]"
            return
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("生成已取消")
    messages = _build_messages(system_prompt, prompt)
    generation_started = time.monotonic()
    token_count = 0
    first_token_received = False
    try:
        async with httpx.AsyncClient(timeout=timeout or 240.0) as client:
            async with client.stream(
                "POST",
                f"{_BASE_URL}/v1/chat/completions",
                json=_payload(messages, max_new_tokens, temperature, stream=True),
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "ignore")
                    raise GenerationOOMError(f"llama-server HTTP {resp.status_code}: {body[:200]}")
                async for line in resp.aiter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        raise GenerationCancelledError("生成已取消")
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload or payload == "[DONE]":
                        break
                    try:
                        import json
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        first_token_received = True
                        token_count += len(content)
                        yield content
    except httpx.TimeoutException:
        _last_generation.update({"stop_reason": "timeout"})
        raise GenerationTimeoutError("llama-server 流式生成超时")
    finally:
        pass

    elapsed = time.monotonic() - generation_started
    _last_generation.update({
        "output_tokens": token_count,
        "generation_seconds": round(elapsed, 3),
        "tokens_per_second": round(token_count / elapsed, 2) if elapsed else 0.0,
        "stop_reason": "ok",
    })
    debug("大模型", f"[llama] 流式生成完成: {token_count} token, {elapsed:.1f}s, {token_count / elapsed:.1f} tok/s")


if __name__ == "__main__":
    async def _test():
        print("测试 llama 后端流式生成...")
        async for tok in generate_stream("说你好"):
            print(tok, end="", flush=True)
        print("\n完成")
    asyncio.run(_test())
