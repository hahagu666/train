"""
Qwen2.5-7B-Instruct LLM适配器
支持非流式和流式生成
"""
import multiprocessing as mp
import os
import logging
import asyncio
import queue
import time
import re
import traceback
from typing import Optional, AsyncGenerator, Callable
from threading import Thread, Event, Lock, RLock, Semaphore

_STREAM_POLL_INTERVAL = 0.05

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from app.logger import info, success, warning, error, debug
from app.config import (
    MAIN_MODEL_ATTENTION,
    MAIN_MODEL_DEVICE,
    MAIN_MODEL_DTYPE,
    MAIN_MODEL_FP16_MIN_FREE_GIB,
    MAIN_MODEL_MAX_NEW_TOKENS,
    MAIN_MODEL_REPETITION_PENALTY,
    MAIN_MODEL_TEMPERATURE,
    MAIN_MODEL_TOP_K,
    MAIN_MODEL_TOP_P,
    MAIN_MODEL_USE_CACHE,
    MAIN_MODEL_WARMUP,
    REFUSAL_ABLATION,
    REFUSAL_ABLATION_ALPHA,
    REFUSAL_DIRECTION_FILE,
)

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
    StoppingCriteria, StoppingCriteriaList,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models",
                          "Qwen--Qwen2.5-7B-Instruct", "snapshots", "master")

_model = None
_tokenizer = None
_model_loaded = False
_device = "cpu"
_load_start_time = 0
_loaded_in_4bit = False
_ablation_handles = []
_lifecycle_lock = RLock()
_generation_slot = Semaphore(1)
_last_load_error = ""
_loading = False
_attention_backend = MAIN_MODEL_ATTENTION
_device_map_summary = ""
_model_footprint_bytes = 0
_last_generation = {
    "prompt_tokens": 0,
    "output_tokens": 0,
    "tokenize_seconds": 0.0,
    "generation_seconds": 0.0,
    "decode_seconds": 0.0,
    "tokens_per_second": 0.0,
    "stop_reason": "",
}


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


class ModelWorkerUnterminableError(RuntimeError):
    """A timed-out worker is still alive, so generation ownership stays held."""


class ModelWorkerRecoveryError(RuntimeError):
    """The old worker is dead, but a replacement could not become ready."""


class _GenerationStopCriteria(StoppingCriteria):
    def __init__(self, cancel_event: Optional[Event], deadline: Optional[float]):
        self.cancel_event = cancel_event
        self.deadline = deadline
        self.reason = ""

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.reason = "cancelled"
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.reason = "timeout"
            return True
        return False


def _cuda_memory() -> dict:
    if not torch.cuda.is_available():
        return {"free_bytes": 0, "total_bytes": 0, "allocated_bytes": 0, "reserved_bytes": 0}
    try:
        free, total = torch.cuda.mem_get_info(0)
        return {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        }
    except Exception:
        return {"free_bytes": 0, "total_bytes": 0, "allocated_bytes": 0, "reserved_bytes": 0}


def _select_load_mode(device: str = None) -> tuple[str, str]:
    requested_device = device or MAIN_MODEL_DEVICE
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device != "cuda":
        return requested_device, "fp32"
    if not torch.cuda.is_available():
        raise RuntimeError("配置要求使用CUDA，但当前CUDA不可用")
    requested_dtype = MAIN_MODEL_DTYPE.lower()
    if requested_dtype in {"int4", "4bit", "nf4"}:
        return "cuda", "4bit"
    if requested_dtype in {"float16", "fp16"}:
        return "cuda", "fp16"
    free_gib = _get_available_vram_gb()
    return "cuda", "fp16" if free_gib >= MAIN_MODEL_FP16_MIN_FREE_GIB else "4bit"


def _summarize_device_map(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        values = sorted({str(value) for value in device_map.values()})
        return ",".join(values)
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return str(getattr(model, "device", "unknown"))


def _validate_cuda_residency(model):
    summary = _summarize_device_map(model).lower()
    if "cpu" in summary or "disk" in summary:
        raise RuntimeError(f"检测到模型权重卸载到CPU/磁盘: {summary}")
    if "cuda" not in summary and summary not in {"0", "cuda:0"}:
        raise RuntimeError(f"无法确认模型权重驻留CUDA: {summary}")


# ===== 拒绝方向消融 =====
# 参考 Arditi et al. 2024（Refusal in LLMs is mediated by a single direction）：
# Instruct模型的拒绝/回避倾向集中在残差流的一个方向上。推理时把每层残差流
# 在该方向上的投影减掉（alpha=1即完全投影消除），可以削弱对齐先验对角色
# 扮演指令的对抗（如"喝杯茶/坐下来聊聊"式话题转移推脱）。
# 方向文件由 train/extract_refusal_direction.py 离线生成。

def _make_ablation_hook(direction: torch.Tensor, alpha: float):
    """返回 decoder layer 的 forward hook：output[0] 减去方向投影 * alpha。"""
    def hook(module, args, output):
        if isinstance(output, tuple) and torch.is_tensor(output[0]):
            h = output[0]
            if h.dim() == 3 and h.shape[-1] == direction.shape[0]:
                d = direction.to(device=h.device, dtype=h.dtype)
                proj = (h @ d).unsqueeze(-1) * d
                return (h - alpha * proj,) + tuple(output[1:])
        return output
    return hook


def _install_refusal_ablation() -> bool:
    """模型加载完成后挂载消融hook；无文件或关闭开关时静默跳过。

    支持两种方向文件格式：
      {"direction": tensor[hidden]}     单一方向，用于所有层（Arditi et al. 2024）
      {"directions": {idx: tensor}}     逐层方向（旧格式兼容）
    """
    global _ablation_handles
    _remove_refusal_ablation()
    if not REFUSAL_ABLATION or _model is None:
        return False
    if not os.path.isfile(REFUSAL_DIRECTION_FILE):
        debug("大模型", f"消融未启用：缺少方向文件 {REFUSAL_DIRECTION_FILE}")
        return False
    try:
        payload = torch.load(REFUSAL_DIRECTION_FILE, map_location="cpu")
        layers = _model.model.layers
        if isinstance(payload, dict) and "direction" in payload:
            d = torch.as_tensor(payload["direction"]).float()
            d = d / (d.norm() + 1e-8)
            for layer in layers:
                handle = layer.register_forward_hook(
                    _make_ablation_hook(d, REFUSAL_ABLATION_ALPHA))
                _ablation_handles.append(handle)
            installed = len(layers)
        else:
            directions = payload.get("directions", payload) if isinstance(payload, dict) else payload
            installed = 0
            for idx, layer in enumerate(layers):
                d = directions.get(idx)
                if d is None:
                    continue
                d = torch.as_tensor(d).float()
                d = d / (d.norm() + 1e-8)
                handle = layer.register_forward_hook(
                    _make_ablation_hook(d, REFUSAL_ABLATION_ALPHA))
                _ablation_handles.append(handle)
                installed += 1
        if installed:
            success("大模型", f"✓ 拒绝方向消融已挂载: {installed}/{len(layers)}层, "
                             f"alpha={REFUSAL_ABLATION_ALPHA}")
        return installed > 0
    except Exception as e:
        _remove_refusal_ablation()
        warning("大模型", f"拒绝方向消融挂载失败（继续用原模型生成）: {e}")
        return False


def _remove_refusal_ablation():
    global _ablation_handles
    for handle in _ablation_handles:
        try:
            handle.remove()
        except Exception:
            pass
    _ablation_handles = []


def _try_load_4bit(device: str) -> bool:
    """低显存策略：用NF4量化加载，且拒绝CPU/磁盘offload。"""
    global _model, _tokenizer, _model_loaded, _device, _loaded_in_4bit
    global _attention_backend, _device_map_summary, _model_footprint_bytes
    if device != "cuda":
        error("大模型", "4-bit量化仅支持CUDA加载")
        return False
    try:
        from transformers import BitsAndBytesConfig
        info("大模型", "使用NF4 4-bit量化加载...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=MAIN_MODEL_ATTENTION,
        )
        _validate_cuda_residency(_model)
        _model.eval()
        _model_loaded = True
        _device = "cuda"
        _loaded_in_4bit = True
        _attention_backend = MAIN_MODEL_ATTENTION
        _device_map_summary = _summarize_device_map(_model)
        footprint = getattr(_model, "get_memory_footprint", None)
        _model_footprint_bytes = int(footprint()) if callable(footprint) else 0
        success("大模型", f"✓ NF4 4-bit模型加载完成，device_map={_device_map_summary}")
        _install_refusal_ablation()
        return True
    except Exception as e:
        _model = None
        _tokenizer = None
        _model_loaded = False
        _loaded_in_4bit = False
        error("大模型", f"4-bit量化加载失败: {e}", exc_info=True)
        return False


def is_model_available() -> bool:
    """检查模型文件是否存在"""
    exists = os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH)
    if not exists:
        debug("大模型", f"模型路径不存在: {MODEL_PATH}")
    return exists


def _get_available_vram_gb() -> float:
    """获取可用显存（GB），CPU返回0"""
    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(0)[0]
        return free / (1024**3)
    except Exception:
        return 0.0


def _load_model_unlocked(device: str = None) -> bool:
    """按显存选择加载模式；低显存CUDA直接NF4，避免WDDM分页。"""
    global _model, _tokenizer, _model_loaded, _device, _load_start_time
    global _loaded_in_4bit, _attention_backend, _device_map_summary, _model_footprint_bytes
    if _model_loaded:
        return True

    if not is_model_available():
        error("大模型", f"模型路径不存在: {MODEL_PATH}")
        return False

    device, load_mode = _select_load_mode(device)
    _device = device
    _load_start_time = time.time()

    info("大模型", f"正在加载模型 from {MODEL_PATH} ...")
    debug("大模型", f"  目标设备: {device}")
    debug("大模型", f"  加载模式: {load_mode}")
    if load_mode == "4bit":
        return _try_load_4bit(device)

    try:
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        debug("大模型", f"  tokenizer加载完成 ({time.time()-t0:.1f}s)")

        t1 = time.time()
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": 0} if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=MAIN_MODEL_ATTENTION,
        )
        if device != "cuda":
            _model = _model.to(device)
        else:
            _validate_cuda_residency(_model)
        _model.eval()
        _model_loaded = True
        _loaded_in_4bit = False
        _attention_backend = MAIN_MODEL_ATTENTION
        _device_map_summary = _summarize_device_map(_model)
        footprint = getattr(_model, "get_memory_footprint", None)
        _model_footprint_bytes = int(footprint()) if callable(footprint) else 0
        success("大模型", f"✓ 模型加载完成！设备: {device}，总耗时: {time.time()-_load_start_time:.1f}s")
        _install_refusal_ablation()
        return True
    except torch.cuda.OutOfMemoryError as oom:
        error("大模型", f"显存不足，FP16加载失败: {oom}")
        _model = None
        _tokenizer = None
        _model_loaded = False
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if device == "cuda" and MAIN_MODEL_DTYPE.lower() == "auto":
            return _try_load_4bit(device)
        return False
    except Exception as e:
        _model = None
        _tokenizer = None
        _model_loaded = False
        error("大模型", f"模型加载失败: {e}", exc_info=True)
        return False


def _build_messages(system_prompt: str, user_prompt: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _output_token_count(generated_ids) -> int:
    if callable(getattr(generated_ids, "numel", None)):
        return int(generated_ids.numel())
    return len(generated_ids.values)


def _stop_reason(generated_ids, max_new_tokens: int) -> str:
    output_tokens = _output_token_count(generated_ids)
    eos_ids = getattr(_tokenizer, "eos_token_id", None)
    if eos_ids is not None and output_tokens:
        if not isinstance(eos_ids, (list, tuple, set)):
            eos_ids = {eos_ids}
        try:
            last_token = generated_ids[-1]
            if callable(getattr(last_token, "item", None)):
                last_token = last_token.item()
            if int(last_token) in eos_ids:
                return "eos"
        except (IndexError, TypeError, ValueError):
            pass
    if output_tokens >= max_new_tokens:
        return "length"
    return "eos" if eos_ids is None or output_tokens == 0 else "stopped"


def _generate_unlocked(prompt: str, system_prompt: str = "",
             max_new_tokens: int = MAIN_MODEL_MAX_NEW_TOKENS,
             temperature: float = MAIN_MODEL_TEMPERATURE,
             cancel_event: Optional[Event] = None, timeout: Optional[float] = 120.0) -> str:
    """非流式生成，返回完整文本"""
    if not _model_loaded:
        warning("大模型", "模型未加载，尝试加载...")
        if not load_model():
            raise RuntimeError(_last_load_error or "模型未加载，请先下载模型")

    stop_criteria = _GenerationStopCriteria(cancel_event, time.monotonic() + timeout if timeout else None)
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("生成已取消")
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = _build_messages(sys_p, prompt)
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenize_started = time.monotonic()
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    tokenize_seconds = time.monotonic() - tokenize_started
    prompt_tokens = int(inputs["input_ids"].shape[1])

    t0 = time.monotonic()
    debug("大模型", f"[完整生成] 开始，device={_device}, 4bit={_loaded_in_4bit}, max_tokens={max_new_tokens}")
    try:
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=MAIN_MODEL_TOP_P,
                top_k=MAIN_MODEL_TOP_K,
                repetition_penalty=MAIN_MODEL_REPETITION_PENALTY,
                use_cache=MAIN_MODEL_USE_CACHE,
                do_sample=temperature > 0.01,
                pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
            )

        if stop_criteria.reason == "cancelled":
            raise GenerationCancelledError("生成已取消")
        if stop_criteria.reason == "timeout":
            raise GenerationTimeoutError("模型完整生成超时")
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generation_seconds = time.monotonic() - t0
        decode_started = time.monotonic()
        response = _tokenizer.decode(generated_ids, skip_special_tokens=True)
        decode_seconds = time.monotonic() - decode_started
        output_tokens = _output_token_count(generated_ids)
        stop_reason = _stop_reason(generated_ids, max_new_tokens)
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "decode_seconds": round(decode_seconds, 4),
            "tokens_per_second": round(output_tokens / generation_seconds, 3) if generation_seconds else 0.0,
            "stop_reason": stop_reason,
        })
        if stop_reason == "length":
            raise GenerationLengthLimitError(
                f"模型回复达到 {max_new_tokens} token 上限，未检测到自然结束"
            )
        elapsed = time.monotonic() - t0
        debug("大模型", f"[完整生成] 完成，{len(response)}字，耗时{elapsed:.1f}s")
        return _clean_response(response)
    except GenerationLengthLimitError:
        raise
    except (GenerationCancelledError, GenerationTimeoutError) as exc:
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - t0, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "timeout" if isinstance(exc, GenerationTimeoutError) else "cancelled",
        })
        raise
    except torch.cuda.OutOfMemoryError as oom:
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - t0, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "oom",
        })
        error("大模型", f"完整生成OOM: {oom}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise GenerationOOMError("显存不足，请缩短输入或降低生成长度") from oom
    except Exception:
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - t0, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "error",
        })
        error("大模型", "完整生成失败", exc_info=True)
        raise


async def _generate_stream_unlocked(prompt: str, system_prompt: str = "",
                          max_new_tokens: int = MAIN_MODEL_MAX_NEW_TOKENS,
                          temperature: float = MAIN_MODEL_TEMPERATURE,
                          cancel_event: Optional[Event] = None,
                          timeout: float = 120.0) -> AsyncGenerator[str, None]:
    """异步流式生成，逐token yield文本（不阻塞事件循环）"""
    started_at = time.monotonic()
    t0 = time.time()
    overall_deadline = started_at + timeout if timeout else None
    first_token_deadline = overall_deadline

    if not _model_loaded:
        warning("大模型", "模型未加载，尝试加载...")
        if not load_model():
            yield "[模型未加载，请先下载模型]"
            return

    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = _build_messages(sys_p, prompt)
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenize_started = time.monotonic()
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    tokenize_seconds = time.monotonic() - tokenize_started
    prompt_tokens = int(inputs["input_ids"].shape[1])

    generation_started = time.monotonic()

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True,
        timeout=_STREAM_POLL_INTERVAL,
    )

    # Stream cleanup must not mutate the coordinator-owned cancellation event.
    # The local event only stops this model invocation when the stream closes
    # or reaches its deadline.
    local_stop_event = Event()

    class _CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return local_stop_event.is_set() or (
                cancel_event is not None and cancel_event.is_set()
            )

    stopping = StoppingCriteriaList([_CancelCriteria()])
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=MAIN_MODEL_TOP_P,
        top_k=MAIN_MODEL_TOP_K,
        repetition_penalty=MAIN_MODEL_REPETITION_PENALTY,
        use_cache=MAIN_MODEL_USE_CACHE,
        do_sample=temperature > 0.01,
        pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        streamer=streamer,
    )
    if stopping is not None:
        gen_kwargs["stopping_criteria"] = stopping

    generation_error = []
    generation_outputs = []
    token_queue = queue.Queue()
    stream_done = object()

    def _run_generation():
        try:
            generation_outputs.append(_model.generate(**gen_kwargs))
        except Exception as exc:
            generation_error.append(exc)
            try:
                streamer.on_finalized_text("", stream_end=True)
            except Exception:
                pass

    def _pump_stream():
        try:
            while True:
                try:
                    token_queue.put(next(streamer))
                except queue.Empty:
                    if local_stop_event.is_set() and not generation_thread.is_alive():
                        break
                    continue
                except StopIteration:
                    break
        except Exception as exc:
            if not local_stop_event.is_set() and not (
                cancel_event is not None and cancel_event.is_set()
            ):
                generation_error.append(exc)
        finally:
            token_queue.put(stream_done)

    generation_thread = Thread(target=_run_generation, daemon=True)
    pump_thread = Thread(target=_pump_stream, daemon=True)
    generation_thread.start()
    pump_thread.start()

    poll_timeout = object()

    def _get_next_token(wait_timeout: float):
        try:
            return token_queue.get(timeout=wait_timeout)
        except queue.Empty:
            return poll_timeout

    token_count = 0
    first_token_received = False
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                local_stop_event.set()
                break
            now = time.monotonic()
            active_deadline = overall_deadline if first_token_received else first_token_deadline
            if active_deadline is not None and now >= active_deadline:
                local_stop_event.set()
                if first_token_received:
                    raise GenerationTimeoutError("模型流式生成超时")
                raise GenerationTimeoutError("模型首个token生成超时")
            wait_timeout = _STREAM_POLL_INTERVAL
            if active_deadline is not None:
                wait_timeout = min(wait_timeout, max(0.001, active_deadline - now))
            new_text = await asyncio.to_thread(_get_next_token, wait_timeout)
            if new_text is poll_timeout:
                continue
            if new_text is stream_done:
                if generation_error:
                    raise generation_error[0]
                break
            if new_text:
                first_token_received = True
                token_count += len(new_text)
                yield new_text
    except (GenerationCancelledError, GenerationTimeoutError) as exc:
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - generation_started, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "timeout" if isinstance(exc, GenerationTimeoutError) else "cancelled",
        })
        raise
    except torch.cuda.OutOfMemoryError as oom:
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - generation_started, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "oom",
        })
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise GenerationOOMError("显存不足，请缩短输入或降低生成长度") from oom
    except Exception:
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - generation_started, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "error",
        })
        raise
    finally:
        local_stop_event.set()
        # The model thread owns the actual CUDA generation. Do not let the
        # stream finish while it is still running, otherwise the caller could
        # start another generation on the same model.
        await asyncio.to_thread(generation_thread.join)
        await asyncio.to_thread(pump_thread.join)

    if cancel_event is not None and cancel_event.is_set():
        _last_generation.update({
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "tokenize_seconds": round(tokenize_seconds, 4),
            "generation_seconds": round(time.monotonic() - generation_started, 4),
            "decode_seconds": 0.0,
            "tokens_per_second": 0.0,
            "stop_reason": "cancelled",
        })
        raise GenerationCancelledError("生成已取消")

    if not generation_outputs:
        raise RuntimeError("模型流式生成未返回token序列")
    generated_ids = generation_outputs[0][0][prompt_tokens:]
    generation_seconds = time.monotonic() - generation_started
    output_tokens = _output_token_count(generated_ids)
    stop_reason = _stop_reason(generated_ids, max_new_tokens)
    _last_generation.update({
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokenize_seconds": round(tokenize_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "decode_seconds": 0.0,
        "tokens_per_second": round(output_tokens / generation_seconds, 3) if generation_seconds else 0.0,
        "stop_reason": stop_reason,
    })
    if stop_reason == "length":
        raise GenerationLengthLimitError(
            f"模型回复达到 {max_new_tokens} token 上限，未检测到自然结束"
        )

    debug("大模型", f"[流式] 生成完成，{token_count}字，耗时{generation_seconds:.1f}s")


def _clean_response(response: str) -> str:
    response = response.strip()
    response = response.replace("\r\n", "\n")
    lines = response.splitlines()
    while lines:
        line = lines[0].strip()
        if not line:
            lines.pop(0)
            continue
        if re.match(r"^(?:assistant|Assistant|我|角色|妹妹|[^：:]{1,30})[：:]\s*", line):
            lines[0] = re.sub(r"^(?:assistant|Assistant|我|角色|妹妹|[^：:]{1,30})[：:]\s*", "", line, count=1).strip()
            if not lines[0]:
                lines.pop(0)
                continue
        break
    return "\n".join(lines).strip()


def get_model_device() -> str:
    if not _IN_MODEL_WORKER:
        return _supervisor.status().get("device", "none")
    return _device if _model_loaded else "none"


def is_loaded() -> bool:
    if not _IN_MODEL_WORKER:
        return _supervisor.status()["loaded"]
    return _model_loaded


def is_4bit() -> bool:
    if not _IN_MODEL_WORKER:
        return _supervisor.status().get("quantization") == "nf4-4bit"
    return _loaded_in_4bit


_IN_MODEL_WORKER = False
_WORKER_POLL_INTERVAL = 0.02
_WORKER_START_TIMEOUT = 600.0


def _worker_snapshot() -> dict:
    """Return model metadata from inside the process that owns the model."""
    return {
        "device": _device,
        "quantization": "nf4-4bit" if _loaded_in_4bit else ("fp16" if _device == "cuda" else "fp32"),
        "device_map": _device_map_summary,
        "dtype": "float16" if _device == "cuda" else "float32",
        "model_footprint_bytes": _model_footprint_bytes,
        "cuda_memory": _cuda_memory(),
        "attention_backend": _attention_backend,
    }


def _model_worker_main(command_queue, result_queue, cancel_signal, device=None):
    """Spawn-safe model process entrypoint. It alone imports weights onto the GPU."""
    global _IN_MODEL_WORKER
    _IN_MODEL_WORKER = True
    try:
        loaded = load_model(device)
        result_queue.put(("ready", loaded, _last_load_error, _worker_snapshot() if loaded else {}))
        if not loaded:
            return
        while True:
            command = command_queue.get()
            if command[0] == "shutdown":
                return
            if command[0] not in {"generate", "generate_stream"}:
                continue
            command_kind, request_id, args = command
            try:
                if command_kind == "generate_stream":
                    async def relay_stream():
                        async for chunk in _generate_stream_unlocked(
                            *args, cancel_event=cancel_signal, timeout=None,
                        ):
                            if chunk:
                                result_queue.put(("stream_chunk", request_id, chunk))
                        return ""

                    text = asyncio.run(relay_stream())
                else:
                    text = _generate_unlocked(*args, cancel_event=cancel_signal, timeout=None)
                result_queue.put(("result", request_id, "ok", text, dict(_last_generation)))
            except GenerationCancelledError as exc:
                result_queue.put(("result", request_id, "cancelled", str(exc), dict(_last_generation)))
            except GenerationLengthLimitError as exc:
                result_queue.put(("result", request_id, "length", str(exc), dict(_last_generation)))
            except GenerationOOMError as exc:
                result_queue.put(("result", request_id, "oom", str(exc), dict(_last_generation)))
            except BaseException as exc:
                result_queue.put((
                    "result", request_id, "error", f"{type(exc).__name__}: {exc}",
                    dict(_last_generation), traceback.format_exc(),
                ))
    except BaseException as exc:
        try:
            result_queue.put(("ready", False, f"{type(exc).__name__}: {exc}", {}))
        except Exception:
            pass


class _ModelWorkerSupervisor:
    """Own one long-lived spawned worker and serialize its lifecycle/generation."""

    def __init__(self, context=None, worker_target: Callable = _model_worker_main,
                 start_timeout: float = _WORKER_START_TIMEOUT):
        self._context = context or mp.get_context("spawn")
        self._worker_target = worker_target
        self._start_timeout = start_timeout
        self._lock = Lock()
        self._process = None
        self._commands = None
        self._results = None
        self._cancel_signal = None
        self._request_id = 0
        self._state = "unloaded"
        self._error = ""
        self._metadata = {}
        self._device_request = None

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _start_locked(self, device=None, timeout=None) -> bool:
        if self._process_alive() and self._state in {"ready", "generating"}:
            return True
        if self._process is not None:
            self._process.join()
            self._process = None
        self._state = "loading"
        self._error = ""
        self._device_request = device
        self._commands = self._context.Queue()
        self._results = self._context.Queue()
        self._cancel_signal = self._context.Event()
        process = self._context.Process(
            target=self._worker_target,
            args=(self._commands, self._results, self._cancel_signal, device),
            name="qwen-model-worker",
        )
        process.start()
        self._process = process
        start_timeout = self._start_timeout if timeout is None else min(
            self._start_timeout, max(0.0, timeout)
        )
        deadline = time.monotonic() + start_timeout
        while time.monotonic() < deadline:
            if not process.is_alive():
                self._error = f"模型工作进程启动失败 (exitcode={process.exitcode})"
                self._state = "error"
                process.join()
                return False
            try:
                message = self._results.get(timeout=min(_WORKER_POLL_INTERVAL, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if message[0] == "ready":
                loaded, load_error, metadata = message[1:]
                self._error = load_error
                self._metadata = metadata
                self._state = "ready" if loaded else "error"
                if not loaded:
                    process.join(timeout=1.0)
                return bool(loaded)
        self._error = "模型工作进程加载超时"
        self._terminate_locked()
        self._state = "error"
        return False

    def load(self, device=None) -> bool:
        with self._lock:
            return self._start_locked(device)

    def _terminate_locked(self):
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            process.join(timeout=10.0)
        if process.is_alive():
            self._state = "fatal_worker_unterminable"
            self._error = "无法终止失去响应的模型工作进程"
            raise ModelWorkerUnterminableError(self._error)
        self._process = None
        # Windows may briefly retain mapped CUDA DLL/image handles after the
        # process handle is signalled. Give process teardown a bounded settling
        # interval before spawning a replacement that maps the same libraries.
        time.sleep(0.1)

    def _replace_after_timeout_locked(self):
        # Never load the replacement until the old process has been reaped. This
        # ordering prevents two copies of the model from occupying the GPU.
        self._state = "timeout_terminating"
        self._terminate_locked()
        self._metadata = {}
        self._state = "recovering"
        if not self._start_locked(self._device_request):
            self._state = "recovery_failed"
            raise ModelWorkerRecoveryError(
                self._error or "模型替换工作进程加载失败"
            )

    def _schedule_recovery_after_timeout_locked(self):
        """Reap the timed-out worker now and reload after request ownership ends."""
        self._state = "timeout_terminating"
        self._terminate_locked()
        self._metadata = {}
        self._state = "recovering"

        def recover():
            with self._lock:
                if self._process_alive() and self._state in {"ready", "generating"}:
                    return
                try:
                    if not self._start_locked(self._device_request):
                        self._state = "recovery_failed"
                except Exception as exc:
                    self._error = f"超时后模型恢复失败: {exc}"
                    self._state = "recovery_failed"

        Thread(target=recover, name="qwen-model-recovery", daemon=True).start()

    def generate(self, prompt, system_prompt, max_new_tokens, temperature,
                 cancel_event, timeout):
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("模型正在生成")
        release_ownership = True
        try:
            if not self._start_locked(self._device_request):
                raise RuntimeError(self._error or "模型未加载")
            self._request_id += 1
            request_id = self._request_id
            self._cancel_signal.clear()
            self._commands.put(("generate", request_id, (
                prompt, system_prompt, max_new_tokens, temperature,
            )))
            self._state = "generating"
            deadline = time.monotonic() + timeout if timeout else None
            cancellation_sent = False
            while True:
                if cancel_event is not None and cancel_event.is_set() and not cancellation_sent:
                    self._cancel_signal.set()
                    cancellation_sent = True
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    _last_generation.update({"stop_reason": "timeout"})
                    try:
                        self._replace_after_timeout_locked()
                    except ModelWorkerUnterminableError:
                        # The generation may still be executing on CUDA. Keep the
                        # lease permanently owned and reject every later request.
                        release_ownership = False
                        raise
                    except ModelWorkerRecoveryError as exc:
                        self._error = f"超时后模型恢复失败: {exc}"
                        raise
                    raise GenerationTimeoutError("模型完整生成超时")
                wait = _WORKER_POLL_INTERVAL
                if deadline is not None:
                    wait = min(wait, max(0.001, deadline - now))
                try:
                    message = self._results.get(timeout=wait)
                except queue.Empty:
                    if not self._process_alive():
                        self._state = "error"
                        self._error = f"模型工作进程意外退出 (exitcode={self._process.exitcode})"
                        raise RuntimeError(self._error)
                    continue
                if message[0] != "result" or message[1] != request_id:
                    continue
                _, _, outcome, payload, telemetry, *extra = message
                _last_generation.update(telemetry)
                self._state = "ready"
                if outcome == "ok":
                    return payload
                if outcome == "cancelled":
                    raise GenerationCancelledError(payload)
                if outcome == "length":
                    raise GenerationLengthLimitError(payload)
                if outcome == "oom":
                    raise GenerationOOMError(payload)
                detail = extra[0] if extra else payload
                raise RuntimeError(detail)
        finally:
            if release_ownership:
                self._lock.release()

    async def generate_stream(self, prompt, system_prompt, max_new_tokens,
                              temperature, cancel_event, timeout):
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("模型正在生成")
        release_ownership = True
        terminal_received = False
        request_id = None
        deadline = time.monotonic() + timeout if timeout else None
        cancellation_sent = False

        async def next_message():
            wait = _WORKER_POLL_INTERVAL
            if deadline is not None:
                wait = min(wait, max(0.001, deadline - time.monotonic()))
            try:
                return await asyncio.to_thread(self._results.get, True, wait)
            except queue.Empty:
                return None

        async def recover_timeout():
            nonlocal release_ownership, terminal_received
            _last_generation.update({"stop_reason": "timeout"})
            try:
                await asyncio.to_thread(self._schedule_recovery_after_timeout_locked)
                terminal_received = True
            except ModelWorkerUnterminableError:
                release_ownership = False
                raise
            except ModelWorkerRecoveryError as exc:
                self._error = f"超时后模型恢复失败: {exc}"
                raise

        async def drain_after_abandonment():
            nonlocal terminal_received
            self._cancel_signal.set()
            while not terminal_received:
                if deadline is not None and time.monotonic() >= deadline:
                    await recover_timeout()
                    return
                message = await next_message()
                if message is None:
                    if not self._process_alive():
                        self._state = "error"
                        self._error = (
                            "模型工作进程意外退出 "
                            f"(exitcode={self._process.exitcode})"
                        )
                        return
                    continue
                if message[0] == "result" and message[1] == request_id:
                    telemetry = message[4]
                    _last_generation.update(telemetry)
                    self._state = "ready"
                    terminal_received = True

        try:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise GenerationTimeoutError("模型流式生成超时")
            startup = asyncio.create_task(asyncio.to_thread(
                self._start_locked, self._device_request, remaining,
            ))
            try:
                started = await asyncio.shield(startup)
            except asyncio.CancelledError:
                # asyncio.to_thread cannot cancel an in-progress worker startup.
                # Keep lifecycle ownership until that thread has stopped mutating it.
                await startup
                raise
            if not started:
                if deadline is not None and time.monotonic() >= deadline:
                    terminal_received = True
                    raise GenerationTimeoutError("模型工作进程启动超时")
                raise RuntimeError(self._error or "模型未加载")
            if deadline is not None and time.monotonic() >= deadline:
                terminal_received = True
                raise GenerationTimeoutError("模型工作进程启动超时")
            self._request_id += 1
            request_id = self._request_id
            self._cancel_signal.clear()
            self._commands.put(("generate_stream", request_id, (
                prompt, system_prompt, max_new_tokens, temperature,
            )))
            self._state = "generating"

            while True:
                if (
                    cancel_event is not None
                    and cancel_event.is_set()
                    and not cancellation_sent
                ):
                    self._cancel_signal.set()
                    cancellation_sent = True
                if deadline is not None and time.monotonic() >= deadline:
                    await recover_timeout()
                    raise GenerationTimeoutError("模型流式生成超时")
                message = await next_message()
                if message is None:
                    if not self._process_alive():
                        self._state = "error"
                        self._error = (
                            "模型工作进程意外退出 "
                            f"(exitcode={self._process.exitcode})"
                        )
                        raise RuntimeError(self._error)
                    continue
                if len(message) < 2 or message[1] != request_id:
                    continue
                if message[0] == "stream_chunk":
                    if message[2]:
                        yield message[2]
                    continue
                if message[0] != "result":
                    continue
                _, _, outcome, payload, telemetry, *extra = message
                _last_generation.update(telemetry)
                self._state = "ready"
                terminal_received = True
                if outcome == "ok":
                    return
                if outcome == "cancelled":
                    raise GenerationCancelledError(payload)
                if outcome == "length":
                    raise GenerationLengthLimitError(payload)
                if outcome == "oom":
                    raise GenerationOOMError(payload)
                detail = extra[0] if extra else payload
                raise RuntimeError(detail)
        finally:
            if request_id is not None and not terminal_received and release_ownership:
                cleanup = asyncio.create_task(drain_after_abandonment())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
            if release_ownership:
                self._lock.release()

    def unload(self) -> bool:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("模型正在生成")
        try:
            if self._process_alive():
                self._commands.put(("shutdown",))
                self._process.join(timeout=10.0)
                if self._process.is_alive():
                    self._terminate_locked()
            elif self._process is not None:
                self._process.join()
            self._process = None
            self._metadata = {}
            self._error = ""
            self._state = "unloaded"
            return True
        finally:
            self._lock.release()

    def status(self) -> dict:
        # Do not take the ownership lock: status must remain observable while a
        # generation or a replacement load holds it.
        alive = self._process_alive()
        state = self._state
        if (
            self._process is not None
            and not alive
            and state in {"loading", "ready", "generating", "timeout_terminating", "recovering"}
        ):
            state = "error"
        metadata = dict(self._metadata)
        return {
            "state": state,
            "loaded": alive and state in {"ready", "generating"},
            "loading": state in {"loading", "recovering"},
            "busy": state in {
                "loading", "generating", "timeout_terminating", "recovering",
                "fatal_worker_unterminable",
            },
            "error": self._error,
            **metadata,
        }


_supervisor = _ModelWorkerSupervisor()


def _warmup_model() -> None:
    """初始化量化 CUDA 内核，避免首个真实请求承担一次性编译成本。"""
    if not MAIN_MODEL_WARMUP or not _model_loaded or _device != "cuda":
        return
    started = time.monotonic()
    inputs = _tokenizer("你好", return_tensors="pt").to(_model.device)
    with torch.no_grad():
        _model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=MAIN_MODEL_USE_CACHE,
            pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        )
    debug("大模型", f"CUDA预热完成，耗时{time.monotonic()-started:.1f}s")


def load_model(device: str = None) -> bool:
    global _last_load_error, _loading
    if not _IN_MODEL_WORKER:
        result = _supervisor.load(device)
        _last_load_error = _supervisor.status().get("error", "")
        return result
    with _lifecycle_lock:
        if _model_loaded:
            return True
        _loading = True
        try:
            result = _load_model_unlocked(device)
            if result:
                try:
                    _warmup_model()
                except Exception as exc:
                    warning("大模型", f"CUDA预热失败，将在首次生成时重试: {exc}")
            _last_load_error = "" if result else "模型加载失败"
            return result
        except Exception as exc:
            _last_load_error = str(exc)
            raise
        finally:
            _loading = False


def _acquire_generation():
    if not _generation_slot.acquire(blocking=False):
        raise ModelBusyError("模型正在生成")
    if not _model_loaded and not load_model():
        _generation_slot.release()
        raise RuntimeError("模型未加载")


def generate(prompt: str, system_prompt: str = "",
             max_new_tokens: int = MAIN_MODEL_MAX_NEW_TOKENS,
             temperature: float = MAIN_MODEL_TEMPERATURE,
             cancel_event: Optional[Event] = None, timeout: Optional[float] = 120.0) -> str:
    if not _IN_MODEL_WORKER:
        return _supervisor.generate(
            prompt, system_prompt, max_new_tokens, temperature, cancel_event, timeout,
        )
    _acquire_generation()
    try:
        return _generate_unlocked(prompt, system_prompt, max_new_tokens, temperature, cancel_event, timeout)
    finally:
        _generation_slot.release()


async def generate_stream(prompt: str, system_prompt: str = "",
                          max_new_tokens: int = MAIN_MODEL_MAX_NEW_TOKENS,
                          temperature: float = MAIN_MODEL_TEMPERATURE,
                          cancel_event: Optional[Event] = None,
                          timeout: float = 120.0) -> AsyncGenerator[str, None]:
    if not _IN_MODEL_WORKER:
        async for token in _supervisor.generate_stream(
            prompt, system_prompt, max_new_tokens, temperature,
            cancel_event, timeout,
        ):
            yield token
        return
    _acquire_generation()
    try:
        async for token in _generate_stream_unlocked(prompt, system_prompt, max_new_tokens, temperature, cancel_event, timeout):
            yield token
    finally:
        _generation_slot.release()


def unload_model() -> bool:
    global _model, _tokenizer, _model_loaded, _loaded_in_4bit
    if not _IN_MODEL_WORKER:
        return _supervisor.unload()
    if not _generation_slot.acquire(blocking=False):
        raise ModelBusyError("模型正在生成")
    try:
        with _lifecycle_lock:
            _remove_refusal_ablation()
            _model = None
            _tokenizer = None
            _model_loaded = False
            _loaded_in_4bit = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    finally:
        _generation_slot.release()


def reload_model(device: str = None) -> bool:
    unload_model()
    return load_model(device)


def get_status() -> dict:
    if not _IN_MODEL_WORKER:
        worker = _supervisor.status()
        loaded = worker["loaded"]
        device = worker.get("device", "none") if loaded else "none"
        return {
            "state": worker["state"],
            "loaded": loaded,
            "loading": worker["loading"],
            "busy": worker["busy"],
            "device": device,
            "quantization": worker.get("quantization", "fp32"),
            "error": worker["error"],
            "device_map": worker.get("device_map", ""),
            "dtype": worker.get("dtype", "float32"),
            "model_footprint_bytes": worker.get("model_footprint_bytes", 0),
            "cuda_memory": worker.get("cuda_memory", {"free_bytes": 0, "total_bytes": 0, "allocated_bytes": 0, "reserved_bytes": 0}),
            "attention_backend": worker.get("attention_backend", MAIN_MODEL_ATTENTION),
            "use_cache": MAIN_MODEL_USE_CACHE,
            "last_generation": dict(_last_generation),
        }
    available = _generation_slot.acquire(blocking=False)
    if available:
        _generation_slot.release()
    cuda_memory = _cuda_memory()
    state = "loading" if _loading else ("generating" if not available else ("ready" if _model_loaded else "unloaded"))
    return {
        "state": state,
        "loaded": _model_loaded,
        "loading": _loading,
        "busy": not available,
        "device": get_model_device(),
        "quantization": "nf4-4bit" if _loaded_in_4bit else ("fp16" if _model_loaded and _device == "cuda" else "fp32"),
        "error": _last_load_error,
        "device_map": _device_map_summary,
        "dtype": "float16" if _model_loaded and _device == "cuda" else "float32",
        "model_footprint_bytes": _model_footprint_bytes,
        "cuda_memory": cuda_memory,
        "attention_backend": _attention_backend,
        "use_cache": MAIN_MODEL_USE_CACHE,
        "last_generation": dict(_last_generation),
    }

DEFAULT_SYSTEM_PROMPT = """你是一个角色扮演AI。根据提供的状态和感受写出角色的反应。
台词直接写；动作、神态、声音和其他非台词内容都用全角中文括号（ ）括起来。
2-4句话，自然简短，直接写内容，不要前缀。"""


if __name__ == "__main__":
    import asyncio

    async def test():
        print("测试流式生成...")
        full = ""
        async for tok in generate_stream("说你好"):
            print(tok, end="", flush=True)
            full += tok
        print()

    asyncio.run(test())
