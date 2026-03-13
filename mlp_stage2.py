#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 multitask training (non-T2TT subset) for Qwen-style causal LM, with LoRA.

Tasks included:
- PR   : speech_tokens -> phoneme_token_text
- ASR  : speech_tokens -> transcription
- G2P  : transcription -> phoneme_token_text
- P2G  : phoneme_token_text -> transcription

Design choices:
- Load from stage1 checkpoint (model + tokenizer)
- Use tokenizer.apply_chat_template() for Qwen-style formatting
- No language sampling control
- No packing
- Multitask joint training via plain concatenation of all train files
- Labels are masked on the prompt/user part; only assistant response contributes to loss
- Use LoRA for parameter-efficient stage2 training

Expected data layout:
stage2_data/
  de/
    de_train_PR.jsonl
    de_train_ASR.jsonl
    de_train_G2P.jsonl
    de_train_P2G.jsonl
    de_validation_PR.jsonl
    ...
  es/
  sl/
  sv-SE/
"""

from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, TaskType, get_peft_model


# -------------------------
# Task templates
# -------------------------
def build_messages(task: str, lang: str, inp: str, tgt: Optional[str] = None) -> List[Dict[str, str]]:
    task = task.upper().strip()
    lang = (lang or "unknown").strip()

    if task == "PR":
        user_text = (
            f"Task: Phoneme Recognition (PR)\n"
            f"Language: {lang}\n"
            f"Input speech tokens:\n{inp}\n\n"
            f"Output the phoneme token sequence."
        )
    elif task == "ASR":
        user_text = (
            f"Task: Automatic Speech Recognition (ASR)\n"
            f"Language: {lang}\n"
            f"Input speech tokens:\n{inp}\n\n"
            f"Transcribe the speech into graphemes/text in {lang}."
        )
    elif task == "G2P":
        user_text = (
            f"Task: Grapheme-to-Phoneme (G2P)\n"
            f"Language: {lang}\n"
            f"Input text:\n{inp}\n\n"
            f"Convert the text into phoneme tokens."
        )
    elif task == "P2G":
        user_text = (
            f"Task: Phoneme-to-Grapheme (P2G)\n"
            f"Language: {lang}\n"
            f"Input phoneme tokens:\n{inp}\n\n"
            f"Convert the phoneme tokens into graphemes/text in {lang}."
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    messages = [{"role": "user", "content": user_text}]
    if tgt is not None:
        messages.append({"role": "assistant", "content": tgt})
    return messages


# -------------------------
# File discovery
# -------------------------
LANG_DIR_TO_CODE = {
    "de": "de",
    "es": "es",
    "sl": "sl",
    "sv-SE": "sv",
}


def arg_to_dir_name(lang_arg: str) -> str:
    if lang_arg in LANG_DIR_TO_CODE:
        return lang_arg
    if lang_arg == "sv":
        return "sv-SE"
    if lang_arg in {"de", "es", "sl"}:
        return lang_arg
    raise ValueError(f"Unsupported language argument: {lang_arg}")


def find_task_files(data_root: Path, languages: List[str], split: str, tasks: List[str]) -> List[Path]:
    files: List[Path] = []
    for lang_arg in languages:
        lang_dir_name = arg_to_dir_name(lang_arg)
        lang_dir = data_root / lang_dir_name
        if not lang_dir.exists():
            print(f"[WARN] missing language dir: {lang_dir}")
            continue

        lang_code = LANG_DIR_TO_CODE[lang_dir_name]
        for task in tasks:
            candidates = [
                lang_dir / f"{lang_code}_{split}_{task}.jsonl",
                lang_dir / f"{lang_dir_name}_{split}_{task}.jsonl",
            ]
            found = None
            for p in candidates:
                if p.exists():
                    found = p
                    break
            if found is None:
                print(f"[WARN] missing file for lang={lang_dir_name}, split={split}, task={task}")
                continue
            files.append(found)
    return files


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} line {line_num}: {e}") from e


# -------------------------
# Dataset
# -------------------------
class Stage2Dataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        tokenizer,
        max_length: int,
        limit_per_file: Optional[int] = None,
        verbose: bool = True,
    ):
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "Tokenizer does not provide apply_chat_template(). "
                "Please load a chat-capable Qwen tokenizer/checkpoint."
            )

        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.samples: List[Dict[str, List[int]]] = []

        raw_count = 0
        kept_count = 0
        truncated_count = 0
        lengths: List[int] = []

        for fp in files:
            local_count = 0
            for rec in read_jsonl(fp):
                task = str(rec.get("task", "")).strip().upper()
                lang = str(rec.get("language", "unknown")).strip()
                inp = str(rec.get("input", "")).strip()
                tgt = str(rec.get("target", "")).strip()

                if not task or not inp or not tgt:
                    continue

                raw_count += 1
                local_count += 1

                prompt_messages = build_messages(task=task, lang=lang, inp=inp, tgt=None)
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_ids = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                )["input_ids"]

                full_messages = build_messages(task=task, lang=lang, inp=inp, tgt=tgt)
                full_text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                full_ids = tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                )["input_ids"]

                if len(prompt_ids) >= len(full_ids):
                    continue

                if len(full_ids) > self.max_length:
                    full_ids = full_ids[: self.max_length]
                    truncated_count += 1

                labels = full_ids.copy()
                prompt_len = min(len(prompt_ids), len(full_ids))
                for i in range(prompt_len):
                    labels[i] = -100

                attn = [1] * len(full_ids)

                self.samples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": attn,
                        "labels": labels,
                    }
                )
                kept_count += 1
                lengths.append(len(full_ids))

                if limit_per_file is not None and local_count >= limit_per_file:
                    break

        if not self.samples:
            raise RuntimeError("No training samples were built.")

        if verbose:
            print(f"[Stage2Dataset] files={len(files)} raw={raw_count} kept={kept_count} truncated={truncated_count}")
            if lengths:
                print(
                    f"[Stage2Dataset] seq len stats: "
                    f"count={len(lengths)} "
                    f"p50={percentile(lengths, 50)} "
                    f"p90={percentile(lengths, 90)} "
                    f"p95={percentile(lengths, 95)} "
                    f"max={max(lengths)}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def percentile(xs: List[int], q: int) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    k = int(math.ceil((q / 100.0) * len(xs))) - 1
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


# -------------------------
# Debug / stats
# -------------------------
def print_trainable_params(model):
    total, trainable = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"[Stage2] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


def debug_model_and_tokenizer(tok, model):
    print("[Stage2] tokenizer class:", tok.__class__.__name__)
    print("[Stage2] pad_token:", tok.pad_token, tok.pad_token_id)
    print("[Stage2] eos_token:", tok.eos_token, tok.eos_token_id)
    print("[Stage2] has chat template:", hasattr(tok, "apply_chat_template"))

    print("[Stage2] model class:", model.__class__.__name__)

    input_emb = model.get_input_embeddings()
    print("[Stage2] input embedding shape:", tuple(input_emb.weight.shape))

    output_emb = model.get_output_embeddings()
    print("[Stage2] has output head:", output_emb is not None)
    if output_emb is not None and hasattr(output_emb, "weight"):
        print("[Stage2] output head shape:", tuple(output_emb.weight.shape))
        print("[Stage2] tied embeddings:", output_emb.weight.data_ptr() == input_emb.weight.data_ptr())


# -------------------------
# LoRA helpers
# -------------------------
def parse_lora_target_modules(text: str) -> List[str]:
    mods = [x.strip() for x in text.split(",") if x.strip()]
    if not mods:
        raise ValueError("LoRA target modules cannot be empty.")
    return mods


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--model_path", required=True, help="Stage1 checkpoint path")
    ap.add_argument("--data_root", required=True, help="Root of stage2_data")
    ap.add_argument("--out_dir", required=True)

    # data
    ap.add_argument("--languages", nargs="+", default=["de", "es", "sl", "sv-SE"])
    ap.add_argument("--tasks", nargs="+", default=["PR", "ASR", "G2P", "P2G"])
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--limit_train_per_file", type=int, default=None)
    ap.add_argument("--limit_eval_per_file", type=int, default=None)

    # train
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_total_limit", type=int, default=2)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules for LoRA",
    )

    # precision / runtime
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--deepspeed_config", type=str, default=None)

    # misc
    ap.add_argument("--do_eval", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = ap.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[Stage2] Loading tokenizer from:", args.model_path)
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print("[Stage2] Loading base model from:", args.model_path)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.gradient_checkpointing:
        print("[Stage2] enable gradient checkpointing")
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    debug_model_and_tokenizer(tok, model)

    lora_targets = parse_lora_target_modules(args.lora_target_modules)
    print("[Stage2] Applying LoRA")
    print(f"[Stage2] LoRA targets: {lora_targets}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print_trainable_params(model)

    train_files = find_task_files(
        data_root=Path(args.data_root),
        languages=args.languages,
        split="train",
        tasks=args.tasks,
    )
    print("[Stage2] train files:")
    for p in train_files:
        print("  ", p)

    eval_files = []
    if args.do_eval:
        eval_files = find_task_files(
            data_root=Path(args.data_root),
            languages=args.languages,
            split="validation",
            tasks=args.tasks,
        )
        print("[Stage2] eval files:")
        for p in eval_files:
            print("  ", p)

    train_dataset = Stage2Dataset(
        files=train_files,
        tokenizer=tok,
        max_length=args.max_length,
        limit_per_file=args.limit_train_per_file,
        verbose=True,
    )

    eval_dataset = None
    if args.do_eval and eval_files:
        eval_dataset = Stage2Dataset(
            files=eval_files,
            tokenizer=tok,
            max_length=args.max_length,
            limit_per_file=args.limit_eval_per_file,
            verbose=True,
        )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=False,

        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=1.0,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        bf16=args.bf16,
        fp16=args.fp16,

        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,

        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,

        deepspeed=args.deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tok,
        data_collator=default_data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    print(f"[Stage2] Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
