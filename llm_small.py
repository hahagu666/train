"""
小模型适配器 - Qwen2.5-1.5B-Instruct（可选，按需加载到CPU）
用于：
  1. Post Processor: 从7B输出中提取结构化情绪/动作信号（替代正则）
  2. Scenario Engine: 智能选择下一个剧情卡/事件（分类任务）

特点：
  - 默认不加载，模型路径不存在时所有函数安全降级为None（调用方自动用回正则/随机）
  - 加载到CPU而非GPU，用完不主动卸载（CPU内存3GB，不影响GPU显存）
  - 短文本任务（输入<1000字，输出<100字），CPU推理约200-500ms
  - 模型下载好后放到 SMALL_MODEL_PATH 即可自动启用
"""
import os
import logging
import time

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

from app.logger import info, success, warning, error, debug

SMALL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "Qwen--Qwen2.5-1.5B-Instruct", "snapshots", "master"
)

_model = None
_tokenizer = None
_available: bool = None  # None=未检测, True/False=是否可用


def is_available() -> bool:
    """检测小模型是否存在（不加载权重，只检查路径和config）"""
    global _available
    if _available is not None:
        return _available
    config_path = os.path.join(SMALL_MODEL_PATH, "config.json")
    _available = os.path.exists(config_path)
    if _available:
        debug("小模型", f"检测到模型文件: {SMALL_MODEL_PATH}")
    else:
        debug("小模型", "模型未找到，将使用降级模式")
    return _available


def _load():
    """延迟加载模型到CPU（只加载一次）"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if not is_available():
        return None, None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info("小模型", "加载Qwen2.5-1.5B到CPU...")
    t0 = time.time()
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_PATH, trust_remote_code=True)
        debug("小模型", f"  tokenizer加载完成 ({time.time()-t0:.1f}s)")
        
        t1 = time.time()
        _model = AutoModelForCausalLM.from_pretrained(
            SMALL_MODEL_PATH,
            dtype=torch.float32,  # CPU用fp32兼容性最好
            device_map={"": "cpu"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _model.eval()
        elapsed = time.time() - t0
        success("小模型", f"✓ 加载完成，总耗时{elapsed:.1f}s (CPU模式)")
        return _model, _tokenizer
    except Exception as e:
        error("小模型", f"加载失败: {e}", exc_info=True)
        return None, None


def generate(prompt: str, system_prompt: str = "", max_new_tokens: int = 120,
             temperature: float = 0.3) -> str:
    """
    小模型生成（短文本任务）。
    模型不可用时返回空字符串，调用方负责降级。
    """
    model, tokenizer = _load()
    if model is None:
        return ""

    import torch
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    t0 = time.time()
    debug("小模型", f"[生成] 输入{inputs['input_ids'].shape[1]}tokens, max_new={max_new_tokens}")
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.85,
                do_sample=temperature > 0.01,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        elapsed = time.time() - t0
        debug("小模型", f"[生成] 完成，{len(response)}字，耗时{elapsed*1000:.0f}ms")
        return response
    except Exception as e:
        error("小模型", f"生成失败: {e}", exc_info=True)
        return ""


def extract_structured(prompt: str, system_prompt: str = "") -> dict:
    """
    结构化提取：要求模型输出JSON，解析后返回dict。
    模型不可用或解析失败返回空dict。
    """
    import json
    sys = system_prompt or (
        "你是一个文本分析器。根据用户提供的文本，提取其中的信号，"
        "严格输出JSON，不要输出其他内容。字段缺失时填0。"
    )
    debug("小模型", "[结构化提取] 开始解析...")
    raw = generate(prompt, system_prompt=sys, temperature=0.05, max_new_tokens=200)
    if not raw:
        debug("小模型", "[结构化提取] 无输出，返回空dict")
        return {}
    # 找到JSON部分
    try:
        # 去掉可能的 ```json ... ``` 包裹
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        # 找第一个 { 到最后一个 }
        l, r = raw.find("{"), raw.rfind("}")
        if l >= 0 and r > l:
            result = json.loads(raw[l:r+1])
            debug("小模型", f"[结构化提取] 成功，字段: {list(result.keys())}")
            return result
    except Exception as e:
        warning("小模型", f"[结构化提取] JSON解析失败: {e}")
        debug("小模型", f"  原始输出: {raw[:100]}...")
    return {}


def classify(prompt: str, labels: list) -> str:
    """
    分类任务：从labels中选一个最合适的返回。
    模型不可用返回labels[0]。
    """
    sys = f"你是一个分类器。根据用户描述，从以下选项中选择最合适的一个，只输出选项文本，不要解释：{', '.join(labels)}"
    debug("小模型", f"[分类] 选项数: {len(labels)}")
    raw = generate(prompt, system_prompt=sys, temperature=0.05, max_new_tokens=20)
    if not raw:
        debug("小模型", "[分类] 无输出，返回默认选项")
        return labels[0] if labels else ""
    # 模糊匹配：哪个label出现在输出里就返回哪个
    for label in labels:
        if label in raw:
            debug("小模型", f"[分类] 结果: {label}")
            return label
    debug("小模型", f"[分类] 模糊匹配失败，返回默认选项，原始输出: {raw[:50]}")
    return labels[0] if labels else ""


def embed(text: str):
    """
    生成文本embedding，返回归一化后的numpy数组 (dim,)。
    模型不可用时返回None。
    """
    model, tokenizer = _load()
    if model is None:
        return None

    import torch
    import torch.nn.functional as F
    import numpy as np

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        sum_embeddings = torch.sum(hidden * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        emb = sum_embeddings / sum_mask  # (1, hidden_dim)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def unload():
    """主动卸载模型释放CPU内存（可选调用）"""
    global _model, _tokenizer, _available
    import gc
    if _model is not None:
        info("小模型", "卸载模型释放内存...")
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        success("小模型", "✓ 模型已卸载")
