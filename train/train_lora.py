"""
LoRA微调训练脚本
使用PEFT + TRL进行SFT训练
数据集格式：JSONL，每行{"conversations": [{"role":"user", "content":"..."}, {"role":"assistant", "content":"..."}]}
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import config


def load_dataset(data_path: Path, max_samples: int = None) -> Dataset:
    """加载JSONL格式对话数据集"""
    conversations = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "conversations" in item:
                    conversations.append(item["conversations"])
            except json.JSONDecodeError:
                continue
    if max_samples:
        conversations = conversations[:max_samples]
    print(f"[Train] 加载了 {len(conversations)} 条对话")
    return Dataset.from_dict({"conversations": conversations})


def format_conversation(conv: List[Dict], tokenizer) -> str:
    """将对话格式化为ChatML格式"""
    messages = []
    for msg in conv:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append({"role": role, "content": content})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def main():
    p = argparse.ArgumentParser(description="LoRA微调训练")
    p.add_argument("--data", type=str, required=True, help="训练数据JSONL路径")
    p.add_argument("--output", type=str, default=str(config.TRAIN_OUTPUT_DIR), help="输出目录")
    p.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="批大小")
    p.add_argument("--lr", type=float, default=config.TRAIN_LR, help="学习率")
    p.add_argument("--max-length", type=int, default=config.TRAIN_MAX_LENGTH, help="最大序列长度")
    p.add_argument("--max-samples", type=int, default=None, help="最大样本数（调试用）")
    p.add_argument("--no-4bit", action="store_true", help="不使用4bit量化")
    p.add_argument("--lora-r", type=int, default=config.TRAIN_LORA_R, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=config.TRAIN_LORA_ALPHA, help="LoRA alpha")
    p.add_argument("--resume", type=str, default=None, help="从checkpoint恢复")
    args = p.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LoRA 微调训练")
    print(f"  基座模型: {config.MODEL_PATH}")
    print(f"  训练数据: {data_path}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.MODEL_PATH), trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 量化配置
    if not args.no_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = torch.bfloat16
    else:
        bnb_config = None
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 加载模型
    print("[Train] 加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        str(config.MODEL_PATH),
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=config.TRAIN_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.TRAIN_LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    dataset = load_dataset(data_path, args.max_samples)

    # 只对assistant部分计算loss
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # 手动格式化
    def preprocess(example):
        text = format_conversation(example["conversations"], tokenizer)
        tokenized = tokenizer(
            text, truncation=True, max_length=args.max_length,
            padding=False, return_tensors=None,
        )
        tokenized["text"] = text
        return tokenized

    print("[Train] 预处理数据集...")
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=args.max_length,
        dataset_text_field="text",
        packing=False,
    )

    print("[Train] 开始训练...")
    trainer.train(resume_from_checkpoint=args.resume)

    # 保存LoRA权重
    final_save = output_dir / "final_lora"
    print(f"[Train] 保存LoRA权重到 {final_save}")
    model.save_pretrained(str(final_save))
    tokenizer.save_pretrained(str(final_save))
    print("[Train] 训练完成！")


if __name__ == "__main__":
    main()
