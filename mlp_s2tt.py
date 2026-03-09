#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

SPEECH_TOKEN_RE = re.compile(r"<[^>]+>")


# -------------------------
# Token scan (add to vocab)
# -------------------------
def scan_new_tokens(jsonl_path: str, existing_vocab: set) -> Tuple[List[str], List[str], List[str]]:
    speech_tokens, phoneme_tokens = set(), set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            st = item.get("speech_tokens", "")
            if isinstance(st, str) and st:
                speech_tokens.update(SPEECH_TOKEN_RE.findall(st))

            ph = item.get("phonemes", "")
            if isinstance(ph, str) and ph.strip():
                phoneme_tokens.update(ph.split())

    new_speech = sorted(t for t in speech_tokens if t not in existing_vocab)
    new_ph = sorted(t for t in phoneme_tokens if t not in existing_vocab)
    all_new = new_speech + new_ph
    return all_new, new_speech, new_ph


# -------------------------
# Helpers
# -------------------------
def split_speech_tokens(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return SPEECH_TOKEN_RE.findall(text)


def split_phoneme_tokens(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def chunk_list(xs: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def print_trainable_params(model):
    total, trainable = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"[Stage1] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


def debug_model_and_tokenizer(tok, model):
    print("[Stage1] tokenizer class:", tok.__class__.__name__)
    print("[Stage1] pad_token:", tok.pad_token, tok.pad_token_id)
    print("[Stage1] eos_token:", tok.eos_token, tok.eos_token_id)
    print("[Stage1] has chat template:", hasattr(tok, "apply_chat_template"))

    print("[Stage1] model class:", model.__class__.__name__)

    input_emb = model.get_input_embeddings()
    if input_emb is None:
        raise RuntimeError("Model has no input embeddings")
    print("[Stage1] input embedding shape:", tuple(input_emb.weight.shape))

    output_emb = model.get_output_embeddings()
    print("[Stage1] has output head:", output_emb is not None)

    if output_emb is not None and hasattr(output_emb, "weight"):
        print("[Stage1] output head shape:", tuple(output_emb.weight.shape))
        print(
            "[Stage1] tied embeddings:",
            output_emb.weight.data_ptr() == input_emb.weight.data_ptr()
        )


# -------------------------
# Qwen-style message builders
# -------------------------
def build_messages_speech(lang: str, speech_chunk: str):
    user_text = (
        f"Language: {lang}\n"
        f"Speech tokens:\n{speech_chunk}\n"
        "Repeat the speech tokens exactly."
    )
    assistant_text = speech_chunk
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def build_messages_phoneme(lang: str, ph_chunk: str):
    user_text = (
        f"Language: {lang}\n"
        f"Phoneme tokens:\n{ph_chunk}\n"
        "Repeat the phoneme tokens exactly."
    )
    assistant_text = ph_chunk
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


# -------------------------
# Dataset
# -------------------------
class PackedChatDataset(Dataset):
    """
    先对 speech/phoneme token 本体切块，再构造成完整对话样本，最后 packing。
    使用 tokenizer.apply_chat_template 来适配 Qwen 风格模板。
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int,
        mode: str,
        chunk_tokens_per_sample: int,
        add_eos_between_packed_samples: bool = True,
    ):
        assert mode in ("speech", "phoneme")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.mode = mode
        self.chunk_tokens_per_sample = int(chunk_tokens_per_sample)
        self.add_eos_between_packed_samples = bool(add_eos_between_packed_samples)

        if self.chunk_tokens_per_sample <= 0:
            raise ValueError("chunk_tokens_per_sample must be > 0")

        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "This tokenizer does not provide apply_chat_template(). "
                "Please use a chat-capable Qwen tokenizer."
            )

        seqs: List[List[int]] = []
        raw_lengths: List[int] = []
        kept_lengths: List[int] = []
        skipped_too_long = 0
        built_samples = 0

        eos_sep: List[int] = []
        if self.add_eos_between_packed_samples and tokenizer.eos_token_id is not None:
            eos_sep = [tokenizer.eos_token_id]

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)
                lang = item.get("language", "unknown")

                if mode == "speech":
                    tokens = split_speech_tokens(item.get("speech_tokens", ""))
                else:
                    tokens = split_phoneme_tokens(item.get("phonemes", ""))

                if not tokens:
                    continue

                chunks = chunk_list(tokens, self.chunk_tokens_per_sample)

                for chunk in chunks:
                    chunk_text = " ".join(chunk)

                    if mode == "speech":
                        messages = build_messages_speech(lang, chunk_text)
                    else:
                        messages = build_messages_phoneme(lang, chunk_text)

                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    enc = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False,
                    )
                    ids = enc["input_ids"]
                    raw_lengths.append(len(ids))

                    if len(ids) > self.max_length:
                        skipped_too_long += 1
                        continue

                    if len(ids) > 0:
                        seqs.append(ids + eos_sep)
                        kept_lengths.append(len(ids))
                        built_samples += 1

        if not seqs:
            raise RuntimeError(
                f"No sequences built for mode={mode} from {jsonl_path}. "
                f"Try decreasing --chunk_tokens_per_sample."
            )

        self._print_length_stats(
            mode=mode,
            raw_lengths=raw_lengths,
            kept_lengths=kept_lengths,
            skipped_too_long=skipped_too_long,
            built_samples=built_samples,
        )

        self.packed: List[List[int]] = []
        cur: List[int] = []

        for ids in seqs:
            if len(ids) > self.max_length:
                continue

            if len(cur) + len(ids) > self.max_length:
                if cur:
                    self.packed.append(cur[:self.max_length])
                cur = list(ids)
            else:
                cur.extend(ids)

        if cur:
            self.packed.append(cur[:self.max_length])

        print(f"[Stage1][{mode}] packed chunks: {len(self.packed)} (max_length={self.max_length})")

    @staticmethod
    def _percentile(values: List[int], q: float) -> Optional[int]:
        if not values:
            return None
        vs = sorted(values)
        idx = int(math.ceil(q * len(vs))) - 1
        idx = max(0, min(idx, len(vs) - 1))
        return vs[idx]

    def _print_length_stats(
        self,
        mode: str,
        raw_lengths: List[int],
        kept_lengths: List[int],
        skipped_too_long: int,
        built_samples: int,
    ):
        if raw_lengths:
            p50 = self._percentile(raw_lengths, 0.50)
            p90 = self._percentile(raw_lengths, 0.90)
            p95 = self._percentile(raw_lengths, 0.95)
            mx = max(raw_lengths)
            print(
                f"[Stage1][{mode}] raw tokenized lengths: "
                f"count={len(raw_lengths)} p50={p50} p90={p90} p95={p95} max={mx}"
            )

        if kept_lengths:
            kp50 = self._percentile(kept_lengths, 0.50)
            kp90 = self._percentile(kept_lengths, 0.90)
            kp95 = self._percentile(kept_lengths, 0.95)
            kmx = max(kept_lengths)
            print(
                f"[Stage1][{mode}] kept tokenized lengths: "
                f"count={len(kept_lengths)} p50={kp50} p90={kp90} p95={kp95} max={kmx}"
            )

        print(
            f"[Stage1][{mode}] built_samples={built_samples} "
            f"skipped_too_long={skipped_too_long}"
        )

    def __len__(self):
        return len(self.packed)

    def __getitem__(self, idx):
        ids = self.packed[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


# -------------------------
# Freeze: train embeddings + output head only
# -------------------------
def freeze_backbone_train_emb_and_output(model):
    for p in model.parameters():
        p.requires_grad = False

    input_emb = model.get_input_embeddings()
    if input_emb is None:
        raise RuntimeError("Model has no input embeddings")
    for p in input_emb.parameters():
        p.requires_grad = True

    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        for p in output_emb.parameters():
            p.requires_grad = True


# -------------------------
# Train phase
# -------------------------
def run_phase(
    model,
    tokenizer,
    dataset,
    out_dir,
    phase_name,
    lr,
    per_device_bs,
    grad_accum,
    warmup_ratio,
    bf16,
    fp16,
    gradient_checkpointing,
    save_steps,
    logging_steps,
    deepspeed_config,
):
    phase_out_dir = os.path.join(out_dir, phase_name)

    args = TrainingArguments(
        output_dir=phase_out_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        num_train_epochs=1,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_pin_memory=True,
        deepspeed=deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(phase_out_dir)
    tokenizer.save_pretrained(phase_out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--chunk_tokens_per_sample", type=int, default=64)

    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=20)

    ap.add_argument("--deepspeed_config", type=str, default=None)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")

    print("[Stage1] Load tokenizer")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    existing = set(tok.get_vocab().keys())
    all_new, new_speech, new_ph = scan_new_tokens(args.train_jsonl, existing)
    print(
        f"[Stage1] new tokens total={len(all_new)} "
        f"speech={len(new_speech)} phoneme={len(new_ph)}"
    )
    if all_new:
        tok.add_tokens(all_new)

    print("[Stage1] Load model")
    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    # fp16 时不要把模型直接加载成 float16；交给 Trainer/AMP 处理

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print("[Stage1] resize embeddings")
    model.resize_token_embeddings(len(tok))

    if args.gradient_checkpointing:
        print("[Stage1] enable gradient checkpointing")
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "config"):
        model.config.use_cache = False

    debug_model_and_tokenizer(tok, model)

    print("[Stage1] freeze backbone; train emb + output head")
    freeze_backbone_train_emb_and_output(model)
    print_trainable_params(model)

    print("[Stage1] Phase A: speech tokens")
    ds_speech = PackedChatDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tok,
        max_length=args.max_length,
        mode="speech",
        chunk_tokens_per_sample=args.chunk_tokens_per_sample,
        add_eos_between_packed_samples=True,
    )
    run_phase(
        model=model,
        tokenizer=tok,
        dataset=ds_speech,
        out_dir=args.out_dir,
        phase_name="stage1_speech_chatml",
        lr=args.lr,
        per_device_bs=args.per_device_bs,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        deepspeed_config=args.deepspeed_config,
    )

    print("[Stage1] Phase B: phoneme tokens")
    ds_ph = PackedChatDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tok,
        max_length=args.max_length,
        mode="phoneme",
        chunk_tokens_per_sample=args.chunk_tokens_per_sample,
        add_eos_between_packed_samples=True,
    )
    run_phase(
        model=model,
        tokenizer=tok,
        dataset=ds_ph,
        out_dir=args.out_dir,
        phase_name="stage1_phoneme_chatml",
        lr=args.lr,
        per_device_bs=args.per_device_bs,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        deepspeed_config=args.deepspeed_config,
    )

    print(f"[Stage1] Done. Saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
