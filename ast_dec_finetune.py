#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AST + Decoder fine-tuning on LibriSpeech (cross-attention seq2seq)

Goal:
- Train a decoder with cross-attention to AST encoder states
- Probe ordering effects (time_first vs freq_first) for patch-token sequences
- Show cross-attention maps (decoder->encoder) as evidence that cross-attn can "fix" ordering issues

What is kept the same as AST+CTC code:
- LibriSpeech dataset, normalization, tokenizer/vocab loading
- AST official FeatureExtractor pipeline
- Chunking + overlap + dedup_overlap logic (now applied to encoder memory)
- AST positional embedding resize when max_frames changes
- Optimizer + per-group TriStage LR scheduler style
- Head warm-up: freeze AST for first N epochs

Key differences:
- Replace CTC head with an autoregressive Transformer decoder
- Training uses teacher forcing with CE loss (ignore pad)
- Decoding uses greedy autoregressive generation (no EOS token; uses max_len or early-stop on all-pad)
- Optional saving of cross-attention heatmap for the first evaluated utterance
"""

from __future__ import annotations
import os, time, random, argparse, re, math, json
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    ASTModel,
    ASTConfig,
)
from jiwer import wer, cer


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text_for_model(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    return s.replace(" ", "|")


def normalize_text_for_wer(s: str) -> str:
    s = s.replace("|", " ")
    s = s.upper()
    s = re.sub(r"[^A-Z' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _infer_hf_cache_dir(arg_cache_dir: Optional[str] = None) -> Optional[str]:
    if arg_cache_dir:
        return arg_cache_dir
    return os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None


def _infer_offline_flag(arg_offline: bool) -> bool:
    if arg_offline:
        return True
    env_offline = (os.environ.get("TRANSFORMERS_OFFLINE") == "1") or (os.environ.get("HF_DATASETS_OFFLINE") == "1")
    return bool(env_offline)


def _check_librispeech_present(root: str, split: str):
    expected = os.path.join(root, "LibriSpeech", split)
    if not os.path.isdir(expected):
        raise FileNotFoundError(
            f"[AST-DEC] LibriSpeech split not found locally:\n"
            f"  expected directory: {expected}\n"
            f"Fix:\n"
            f"  1) In sbatch: rsync *.tar.gz to node scratch and extract into <data_root>/LibriSpeech/...\n"
            f"  2) Run with --data_root pointing to that scratch directory.\n"
        )


# -------------------------
# Processor (tokenizer) - reuse vocab.json
# -------------------------
def build_processor_from_vocab_dir(vocab_dir: str) -> Wav2Vec2Processor:
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.json not found in {vocab_dir}")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        bos_token="<bos>",
        eos_token="<eos>",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# -------------------------
# Per-group Tri-stage LR scheduler
# -------------------------
class TriStageLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_update: int,
        warmup_steps: int = 8000,
        hold_steps: int = 0,
        decay_steps: int = 72000,
        init_lr_scale: float = 0.1,
        final_lr_scale: float = 0.05,
    ):
        self.optimizer = optimizer
        self.max_update = int(max_update)
        self.warmup_steps = int(warmup_steps)
        self.hold_steps = int(hold_steps)
        self.decay_steps = int(decay_steps)
        self.init_lr_scale = float(init_lr_scale)
        self.final_lr_scale = float(final_lr_scale)

        self.num_updates = 0

        self.peak_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
        self.init_lrs = [lr * self.init_lr_scale for lr in self.peak_lrs]
        self.final_lrs = [lr * self.final_lr_scale for lr in self.peak_lrs]

        for g, lr0 in zip(self.optimizer.param_groups, self.init_lrs):
            g["lr"] = lr0

    def _stage_and_offset(self, step: int) -> Tuple[int, int]:
        s = step
        if s <= self.warmup_steps:
            return 0, s
        s -= self.warmup_steps
        if s <= self.hold_steps:
            return 1, s
        s -= self.hold_steps
        if s <= self.decay_steps:
            return 2, s
        s -= self.decay_steps
        return 3, s

    def _lr_at(self, peak: float, init: float, final: float, stage: int, steps_in_stage: int) -> float:
        if stage == 0:
            frac = steps_in_stage / max(self.warmup_steps, 1)
            return init + (peak - init) * frac
        if stage == 1:
            return peak
        if stage == 2:
            frac = steps_in_stage / max(self.decay_steps, 1)
            frac = min(max(frac, 0.0), 1.0)
            return peak + (final - peak) * frac
        return final

    def step(self):
        self.num_updates += 1
        stage, off = self._stage_and_offset(self.num_updates)
        for i, g in enumerate(self.optimizer.param_groups):
            lr = self._lr_at(self.peak_lrs[i], self.init_lrs[i], self.final_lrs[i], stage, off)
            g["lr"] = float(lr)


# -------------------------
# Dataset: waveform + normalized text
# -------------------------
class LibriSpeechWaveText(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        max_utts: Optional[int] = None,
        shuffle: bool = True,
    ):
        _check_librispeech_present(root, split)
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=False)
        self.indices = list(range(len(self.ds)))
        if shuffle:
            random.shuffle(self.indices)
        if max_utts is not None:
            self.indices = self.indices[:max_utts]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        item = self.ds[self.indices[i]]
        wav = item[0]
        sr = item[1]

        transcript = ""
        for x in item[2:]:
            if isinstance(x, str):
                transcript = x
                break

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        txt = normalize_text_for_model(transcript)
        return wav, txt


# -------------------------
# Feature extraction: AST official FeatureExtractor -> chunk/pad/mask
# -------------------------
def wav_to_ast_features(
    wav: torch.Tensor,
    ast_feature_extractor,
    sample_rate: int = 16000,
) -> torch.Tensor:
    if wav.dim() != 1:
        wav = wav.view(-1)
    wav_np = wav.detach().cpu().numpy().astype(np.float32)
    feats = ast_feature_extractor(
        wav_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    )
    x = feats["input_values"][0].to(torch.float32)
    return x


def chunk_and_pad_feat(
    feat: torch.Tensor,
    max_frames: int = 1024,
    overlap: int = 0,
) -> Tuple[torch.Tensor, List[int]]:
    T = feat.size(0)

    if overlap < 0 or overlap >= max_frames:
        raise ValueError(f"overlap must be in [0, max_frames-1], got {overlap} with max_frames={max_frames}")

    stride = max_frames - overlap

    if T <= max_frames:
        chunk = torch.zeros((max_frames, feat.size(1)), dtype=feat.dtype)
        chunk[:T] = feat
        return chunk.unsqueeze(0), [T]

    chunks: List[torch.Tensor] = []
    lens: List[int] = []
    start = 0
    while start < T:
        end = min(start + max_frames, T)
        cur = feat[start:end]
        L = end - start

        buf = torch.zeros((max_frames, feat.size(1)), dtype=feat.dtype)
        buf[:L] = cur
        chunks.append(buf)
        lens.append(L)

        if end == T:
            break
        start += stride

    return torch.stack(chunks, dim=0), lens


def collate_ast_dec(batch, max_frames: int, ast_feature_extractor, chunk_overlap: int = 0):
    waves, texts = zip(*batch)
    texts = list(texts)

    all_chunks = []
    all_lens = []
    utt_slices = []
    cursor = 0

    for wav in waves:
        feat = wav_to_ast_features(wav, ast_feature_extractor, 16000)
        chunks, lens = chunk_and_pad_feat(feat, max_frames=max_frames, overlap=chunk_overlap)
        n = chunks.size(0)
        all_chunks.append(chunks)
        all_lens.extend(lens)
        utt_slices.append((cursor, n))
        cursor += n

    chunk_fbanks = torch.cat(all_chunks, dim=0)
    chunk_frame_lens = torch.tensor(all_lens, dtype=torch.long)

    t = torch.arange(max_frames).unsqueeze(0)
    chunk_attn_mask = (t < chunk_frame_lens.unsqueeze(1)).long()

    return chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts


# -------------------------
# Resize AST pos embeddings when max_frames changes
# -------------------------
def resize_ast_positional_embeddings(ast: ASTModel, freq_bins: int, max_frames: int):
    pos = ast.embeddings.position_embeddings
    old_len = pos.shape[1]
    H = pos.shape[2]

    proj = ast.embeddings.patch_embeddings.projection
    k_f, k_t = map(int, proj.kernel_size)
    s_f, s_t = map(int, proj.stride)

    Fp_new = (freq_bins - k_f) // s_f + 1
    Tp_new = (max_frames - k_t) // s_t + 1
    num_patches_new = Fp_new * Tp_new

    Fp_old = (freq_bins - k_f) // s_f + 1
    rem = old_len % Fp_old
    special = rem if rem != 0 else 0
    num_patches_old = old_len - special
    Tp_old = num_patches_old // Fp_old

    if Fp_old * Tp_old != num_patches_old:
        raise RuntimeError(
            f"Cannot infer old grid: old_len={old_len}, Fp_old={Fp_old}, Tp_old={Tp_old}, special={special}"
        )

    if num_patches_new == num_patches_old:
        return

    pos_special = pos[:, :special, :]
    pos_patches = pos[:, special:, :]
    pos_patches = pos_patches.reshape(1, Fp_old, Tp_old, H).permute(0, 3, 1, 2)

    pos_patches_new = F.interpolate(
        pos_patches,
        size=(Fp_new, Tp_new),
        mode="bicubic",
        align_corners=False,
    )

    pos_patches_new = pos_patches_new.permute(0, 2, 3, 1).reshape(1, Fp_new * Tp_new, H)
    pos_new = torch.cat([pos_special, pos_patches_new], dim=1)

    ast.embeddings.position_embeddings = nn.Parameter(pos_new)
    print(
        f"[AST-DEC] Resized position embeddings: {old_len} -> {pos_new.shape[1]} "
        f"(special={special}, grid {Fp_old}x{Tp_old} -> {Fp_new}x{Tp_new})"
    )


# -------------------------
# Decoder building blocks (with cross-attn weights)
# -------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        mem_key_padding_mask: Optional[torch.Tensor],
        need_cross_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop(y))

        y, w = self.cross_attn(
            x, memory, memory,
            attn_mask=None,
            key_padding_mask=mem_key_padding_mask,
            need_weights=need_cross_attn,
            average_attn_weights=False,
        )
        x = self.norm2(x + self.drop(y))

        y = self.lin2(self.drop(F.gelu(self.lin1(x))))
        x = self.norm3(x + self.drop(y))
        return x, w


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.pos_emb = nn.Embedding(self.max_len, d_model)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
        mem_key_padding_mask: Optional[torch.Tensor],
        return_cross_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H = x.shape
        if L > self.max_len:
            raise ValueError(f"Decoder sequence length {L} exceeds max_len={self.max_len} (increase --dec_max_len)")

        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)

        tgt_mask = self._causal_mask(L, x.device)

        last_w = None
        for i, layer in enumerate(self.layers):
            need = return_cross_attn and (i == len(self.layers) - 1)
            x, w = layer(
                x, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                mem_key_padding_mask=mem_key_padding_mask,
                need_cross_attn=need,
            )
            if need:
                last_w = w
        return x, last_w


# -------------------------
# AST + DEC model
# -------------------------
class ASTDecModel(nn.Module):
    def __init__(
        self,
        ast: ASTModel,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        freq_bins: int = 128,
        max_frames: int = 1024,
        upsample_factor: int = 1,
        freq_pool: str = "mean",
        enc_axis: str = "time",
        token_order: str = "time_first",
        dec_layers: int = 4,
        dec_heads: int = 8,
        dec_ff: int = 2048,
        dec_dropout: float = 0.1,
        dec_max_len: int = 256,
        tie_output: bool = False,
    ):
        super().__init__()
        self.ast = ast

        self.freq_bins = int(freq_bins)
        self.max_frames = int(max_frames)
        self.upsample_factor = int(upsample_factor)
        if self.upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")

        self.enc_axis = str(enc_axis)
        if self.enc_axis not in ("time", "token"):
            raise ValueError("enc_axis must be one of: time, token")

        self.token_order = str(token_order)
        if self.token_order not in ("time_first", "freq_first"):
            raise ValueError("token_order must be one of: time_first, freq_first")

        self.freq_pool = str(freq_pool)
        if self.freq_pool not in ("mean", "max", "attn"):
            raise ValueError("freq_pool must be one of: mean, max, attn")
        if self.freq_pool == "attn":
            self.freq_attn = nn.Linear(ast.config.hidden_size, 1)

        proj = self.ast.embeddings.patch_embeddings.projection
        k_f, k_t = proj.kernel_size
        s_f, s_t = proj.stride
        self.k_f, self.k_t = int(k_f), int(k_t)
        self.s_f, self.s_t = int(s_f), int(s_t)

        if self.freq_bins < self.k_f or self.max_frames < self.k_t:
            raise ValueError(
                f"Input too small: freq_bins={self.freq_bins}, max_frames={self.max_frames}, "
                f"kernel=({self.k_f},{self.k_t})"
            )

        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.bos_id = int(bos_token_id)
        self.eos_id = int(eos_token_id)

        d_model = int(ast.config.hidden_size)
        self.tok_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_token_id)
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=int(dec_heads),
            num_layers=int(dec_layers),
            dim_ff=int(dec_ff),
            dropout=float(dec_dropout),
            max_len=int(dec_max_len),
        )
        self.out_proj = nn.Linear(d_model, self.vocab_size, bias=False)

        if tie_output:
            if self.tok_emb.weight.shape[0] != self.vocab_size:
                raise RuntimeError("unexpected embedding shape")
            self.out_proj.weight = nn.Parameter(self.tok_emb.weight.data)

    def _grid(self, F_bins: int, T_frames: int) -> Tuple[int, int, int]:
        Fp = (F_bins - self.k_f) // self.s_f + 1
        Tp = (T_frames - self.k_t) // self.s_t + 1
        return int(Fp), int(Tp), int(Fp * Tp)

    def forward_chunk_encoder(
        self,
        fbank_chunk: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.ast(input_values=fbank_chunk, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state

        Bp = hs.size(0)
        T_frames = fbank_chunk.size(1)
        F_bins = fbank_chunk.size(2)

        Fp, Tp, num_patches = self._grid(F_bins, T_frames)
        special = hs.size(1) - num_patches

        if special <= 0:
            raise RuntimeError(
                f"[AST-DEC] token length mismatch: hs_len={hs.size(1)} num_patches={num_patches} "
                f"(Fp={Fp},Tp={Tp}, kernel=({self.k_f},{self.k_t}), stride=({self.s_f},{self.s_t}))"
            )

        patch_hs = hs[:, special:, :].view(Bp, Fp, Tp, -1)

        if self.enc_axis == "token":
            if self.token_order == "time_first":
                mem = patch_hs.permute(0, 2, 1, 3).reshape(Bp, Tp * Fp, -1)
            else:
                mem = patch_hs.reshape(Bp, Fp * Tp, -1)
            return mem

        if self.freq_pool == "mean":
            time_hs = patch_hs.mean(dim=1)
        elif self.freq_pool == "max":
            time_hs = patch_hs.max(dim=1).values
        else:
            x = patch_hs.permute(0, 2, 1, 3)
            scores = self.freq_attn(x).squeeze(-1)
            alpha = torch.softmax(scores, dim=-1)
            time_hs = (alpha.unsqueeze(-1) * x).sum(dim=2)

        r = self.upsample_factor
        if r > 1:
            time_hs = time_hs.repeat_interleave(r, dim=1)
        return time_hs

    def forward_decoder(
        self,
        memory: torch.Tensor,
        mem_key_padding_mask: torch.Tensor,
        dec_in_ids: torch.Tensor,
        dec_key_padding_mask: torch.Tensor,
        return_cross_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(dec_in_ids)
        x, cross_w = self.decoder(
            x=x,
            memory=memory,
            tgt_key_padding_mask=dec_key_padding_mask,
            mem_key_padding_mask=mem_key_padding_mask,
            return_cross_attn=return_cross_attn,
        )
        logits = self.out_proj(x)
        return logits, cross_w


# -------------------------
# Length computation + assembly for encoder memory
# -------------------------
def compute_time_patch_lens(frame_lens: torch.Tensor, k_t: int, s_t: int) -> torch.Tensor:
    tp = (frame_lens - k_t) // s_t + 1
    tp = torch.clamp(tp, min=0)
    return tp.to(torch.long)


def compute_chunk_memory_lens(
    enc_axis: str,
    frame_lens: torch.Tensor,
    k_t: int,
    s_t: int,
    upsample_factor: int,
    Fp: int,
) -> torch.Tensor:
    tp_valid = compute_time_patch_lens(frame_lens, k_t, s_t)
    if enc_axis == "token":
        return (tp_valid * int(Fp)).to(torch.long)
    return (tp_valid * int(upsample_factor)).to(torch.long)


def assemble_utt_memory(
    chunk_mem: torch.Tensor,
    chunk_mem_lens: torch.Tensor,
    utt_slices: List[Tuple[int, int]],
    drop_left: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = chunk_mem.device
    B = len(utt_slices)
    H = chunk_mem.size(-1)

    utt_lens = []
    utt_seqs = []
    for (s, n) in utt_slices:
        parts = []
        total = 0
        for k in range(n):
            idx = s + k
            Lk = int(chunk_mem_lens[idx].item())
            seg = chunk_mem[idx, :Lk, :]

            if drop_left > 0 and k > 0:
                if drop_left < seg.size(0):
                    seg = seg[drop_left:, :]
                else:
                    seg = seg[:0, :]

            parts.append(seg)
            total += seg.size(0)

        utt_lens.append(total)
        utt_seqs.append(torch.cat(parts, dim=0) if len(parts) > 0 else torch.zeros((0, H), device=device))

    max_S = max(utt_lens) if utt_lens else 0
    padded = torch.zeros((B, max_S, H), device=device, dtype=chunk_mem.dtype)
    for i, seq in enumerate(utt_seqs):
        if seq.numel() > 0:
            padded[i, : seq.size(0)] = seq

    return padded, torch.tensor(utt_lens, device=device, dtype=torch.long)


def make_len_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    B = int(lengths.size(0))
    L = int(max_len) if max_len is not None else int(lengths.max().item()) if B > 0 else 0
    t = torch.arange(L, device=lengths.device).unsqueeze(0).expand(B, L)
    return t >= lengths.unsqueeze(1)


def shift_right_with_bos(label_ids: torch.Tensor, pad_id: int, bos_id: int) -> torch.Tensor:
    B, L = label_ids.shape
    dec_in = torch.full((B, L), pad_id, dtype=label_ids.dtype, device=label_ids.device)
    dec_in[:, 0] = bos_id
    if L > 1:
        dec_in[:, 1:] = label_ids[:, :-1]
    return dec_in

def append_eos_inplace(label_ids: torch.Tensor, pad_id: int, eos_id: int) -> torch.Tensor:
    # label_ids: [B, L] padded
    B, L = label_ids.shape
    out = label_ids.clone()
    for b in range(B):
        row = out[b]
        pad_pos = (row == pad_id).nonzero(as_tuple=False)
        if pad_pos.numel() > 0:
            j = int(pad_pos[0].item())
            row[j] = eos_id
        else:
            row[L - 1] = eos_id
    return out

# -------------------------
# Greedy decoding + text conversion
# -------------------------
def ids_to_text(tokenizer, ids: List[int]) -> str:
    toks = tokenizer.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in ("<pad>", "<unk>", "<bos>", "<eos>")]
    s = "".join(toks).replace("|", " ")
    return s

@torch.no_grad()
def greedy_decode(
    model: ASTDecModel,
    memory: torch.Tensor,
    mem_lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    device = memory.device
    B = memory.size(0)
    mem_kpm = make_len_mask(mem_lengths, max_len=memory.size(1))

    ys = torch.full((B, 1), model.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros((B,), dtype=torch.bool, device=device)

    for _ in range(max_len):
        dec_kpm = (ys == model.pad_token_id)
        logits, _ = model.forward_decoder(
            memory=memory,
            mem_key_padding_mask=mem_kpm,
            dec_in_ids=ys,
            dec_key_padding_mask=dec_kpm,
            return_cross_attn=False,
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1)

        # finished 的句子后续全部 pad
        next_id = torch.where(finished, torch.full_like(next_id, model.pad_token_id), next_id)

        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)


        finished = finished | (next_id == model.eos_id)
        if bool(finished.all()):
            break

    return ys[:, 1:]  # 去掉起始 bos

# -------------------------
# Attention map saving
# -------------------------
def save_cross_attention_heatmap(attn: torch.Tensor, out_png: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if attn.dim() != 4:
        raise ValueError(f"attn must be [B, heads, T, S], got {tuple(attn.shape)}")

    x = attn[0].mean(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(x, aspect="auto", origin="lower")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.xlabel("Encoder position (memory)")
    plt.ylabel("Decoder position (target)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model: ASTDecModel,
    processor: Wav2Vec2Processor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[TriStageLRScheduler],
    pad_token_id: int,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
    global_step_start: int = 0,
    debug_first_batch: bool = True,
    chunk_overlap: int = 0,
    dedup_overlap: bool = False,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    global_step = global_step_start

    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts in loader:
        global_step += 1
        optimizer.zero_grad(set_to_none=True)

        enc = processor(text=texts, return_tensors=None, padding=False)
        seqs = [ids + [model.eos_id] for ids in enc["input_ids"]]
        maxL = max(len(s) for s in seqs)
        label_ids = torch.full((len(seqs), maxL), pad_token_id, dtype=torch.long)
        for i,s in enumerate(seqs):
            label_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        label_ids = label_ids.to(device)

        chunk_fbanks = chunk_fbanks.to(device)
        chunk_frame_lens = chunk_frame_lens.to(device)
        chunk_attn_mask = chunk_attn_mask.to(device)

        Fp, Tp, _ = model._grid(chunk_fbanks.size(2), chunk_fbanks.size(1))

        if use_amp:
            with torch.amp.autocast("cuda"):
                chunk_mem = model.forward_chunk_encoder(chunk_fbanks, attention_mask=chunk_attn_mask)
        else:
            chunk_mem = model.forward_chunk_encoder(chunk_fbanks, attention_mask=chunk_attn_mask)

        chunk_mem_lens = compute_chunk_memory_lens(
            enc_axis=model.enc_axis,
            frame_lens=chunk_frame_lens,
            k_t=model.k_t,
            s_t=model.s_t,
            upsample_factor=model.upsample_factor,
            Fp=Fp,
        )

        drop_left = 0
        if chunk_overlap > 0 and dedup_overlap:
            overlap_tp = max(0, int(chunk_overlap // max(model.s_t, 1)))
            if model.enc_axis == "token":
                drop_left = overlap_tp * int(Fp)
            else:
                drop_left = overlap_tp * int(model.upsample_factor)

        memory, mem_lengths = assemble_utt_memory(chunk_mem, chunk_mem_lens, utt_slices, drop_left=drop_left)

        mem_kpm = make_len_mask(mem_lengths, max_len=memory.size(1))

        dec_in = shift_right_with_bos(label_ids, pad_id=pad_token_id, bos_id=model.bos_id)
        dec_kpm = (dec_in == pad_token_id)

        if debug_first_batch and global_step == global_step_start + 1:
            print("[DEBUG] chunk_fbanks:", tuple(chunk_fbanks.shape))
            print("[DEBUG] chunk_mem:", tuple(chunk_mem.shape))
            print("[DEBUG] chunk_mem_lens min/max:", int(chunk_mem_lens.min()), int(chunk_mem_lens.max()))
            print("[DEBUG] drop_left(mem_steps):", int(drop_left))
            print("[DEBUG] memory:", tuple(memory.shape))
            print("[DEBUG] mem_lengths min/max:", int(mem_lengths.min()), int(mem_lengths.max()))
            print("[DEBUG] label_ids:", tuple(label_ids.shape))
            print("[DEBUG] dec_in:", tuple(dec_in.shape))

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits, _ = model.forward_decoder(
                    memory=memory,
                    mem_key_padding_mask=mem_kpm,
                    dec_in_ids=dec_in,
                    dec_key_padding_mask=dec_kpm,
                    return_cross_attn=False,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label_ids.reshape(-1),
                    ignore_index=pad_token_id,
                )
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model.forward_decoder(
                memory=memory,
                mem_key_padding_mask=mem_kpm,
                dec_in_ids=dec_in,
                dec_key_padding_mask=dec_kpm,
                return_cross_attn=False,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                label_ids.reshape(-1),
                ignore_index=pad_token_id,
            )
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: ASTDecModel,
    processor: Wav2Vec2Processor,
    loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    gen_max_len: int = 256,
    save_attn: bool = False,
    out_dir: Optional[str] = None,
    chunk_overlap: int = 0,
    dedup_overlap: bool = False,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_refs: List[str] = []
    all_hyps: List[str] = []

    saved_any_attn = False

    for chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts in loader:
        enc = processor(text=texts, return_tensors=None, padding=False)
        seqs = [ids + [model.eos_id] for ids in enc["input_ids"]]
        maxL = max(len(s) for s in seqs)
        label_ids = torch.full((len(seqs), maxL), pad_token_id, dtype=torch.long)
        for i,s in enumerate(seqs):
            label_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        label_ids = label_ids.to(device)

        chunk_fbanks = chunk_fbanks.to(device)
        chunk_frame_lens = chunk_frame_lens.to(device)
        chunk_attn_mask = chunk_attn_mask.to(device)

        Fp, Tp, _ = model._grid(chunk_fbanks.size(2), chunk_fbanks.size(1))
        chunk_mem = model.forward_chunk_encoder(chunk_fbanks, attention_mask=chunk_attn_mask)

        chunk_mem_lens = compute_chunk_memory_lens(
            enc_axis=model.enc_axis,
            frame_lens=chunk_frame_lens,
            k_t=model.k_t,
            s_t=model.s_t,
            upsample_factor=model.upsample_factor,
            Fp=Fp,
        )

        drop_left = 0
        if chunk_overlap > 0 and dedup_overlap:
            overlap_tp = max(0, int(chunk_overlap // max(model.s_t, 1)))
            if model.enc_axis == "token":
                drop_left = overlap_tp * int(Fp)
            else:
                drop_left = overlap_tp * int(model.upsample_factor)

        memory, mem_lengths = assemble_utt_memory(chunk_mem, chunk_mem_lens, utt_slices, drop_left=drop_left)
        mem_kpm = make_len_mask(mem_lengths, max_len=memory.size(1))

        dec_in = shift_right_with_bos(label_ids, pad_id=pad_token_id, bos_id=model.bos_id)
        dec_kpm = (dec_in == pad_token_id)

        logits, cross_w = model.forward_decoder(
            memory=memory,
            mem_key_padding_mask=mem_kpm,
            dec_in_ids=dec_in,
            dec_key_padding_mask=dec_kpm,
            return_cross_attn=(save_attn and (not saved_any_attn)),
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            label_ids.reshape(-1),
            ignore_index=pad_token_id,
        )

        total_loss += float(loss.item())
        n_batches += 1

        if save_attn and (not saved_any_attn) and (cross_w is not None) and (out_dir is not None):
            os.makedirs(out_dir, exist_ok=True)
            attn_path = os.path.join(out_dir, "cross_attention.npy")
            png_path = os.path.join(out_dir, "cross_attention.png")
            np.save(attn_path, cross_w.detach().cpu().numpy())
            save_cross_attention_heatmap(
                cross_w,
                out_png=png_path,
                title=f"Cross-attn (enc_axis={model.enc_axis}, order={model.token_order})",
            )
            print(f"[AST-DEC] Saved cross-attention -> {attn_path} and {png_path}")
            saved_any_attn = True

        gen_ids = greedy_decode(model, memory=memory, mem_lengths=mem_lengths, max_len=int(gen_max_len))

        pred_str: List[str] = []
        for i in range(gen_ids.size(0)):
            ids = gen_ids[i].detach().cpu().tolist()
            pred_str.append(ids_to_text(processor.tokenizer, ids))

        refs_norm = [normalize_text_for_wer(s) for s in texts]
        hyps_norm = [normalize_text_for_wer(s) for s in pred_str]

        all_refs.extend(refs_norm)
        all_hyps.extend(hyps_norm)

    avg_loss = total_loss / max(1, n_batches)
    wer_score = wer(all_refs, all_hyps)
    cer_score = cer(all_refs, all_hyps)
    return avg_loss, float(wer_score), float(cer_score), all_refs[:3], all_hyps[:3]


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_split", type=str, default="train-clean-100")
    ap.add_argument("--val_split", type=str, default="dev-clean")
    ap.add_argument("--test_split", type=str, default="test-clean")
    ap.add_argument("--max_train_utts", type=int, default=2000)
    ap.add_argument("--max_val_utts", type=int, default=200)
    ap.add_argument("--max_test_utts", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--no_shuffle", action="store_true", help="Do not shuffle dataset indices before truncation.")

    ap.add_argument("--chunk_overlap", type=int, default=0)
    ap.add_argument("--dedup_overlap", action="store_true")

    ap.add_argument("--ast_ckpt", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--ast_from_scratch", action="store_true")
    ap.add_argument("--max_frames", type=int, default=2048)
    ap.add_argument("--freq_bins", type=int, default=128)

    ap.add_argument("--enc_axis", type=str, default="time", choices=["time", "token"])
    ap.add_argument("--token_order", type=str, default="time_first", choices=["time_first", "freq_first"])
    ap.add_argument("--freq_pool", type=str, default="mean", choices=["mean", "max", "attn"])
    ap.add_argument("--upsample_factor", type=int, default=1)

    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--dec_heads", type=int, default=8)
    ap.add_argument("--dec_ff", type=int, default=2048)
    ap.add_argument("--dec_dropout", type=float, default=0.1)
    ap.add_argument("--dec_max_len", type=int, default=256)
    ap.add_argument("--tie_output", action="store_true")

    ap.add_argument("--gen_max_len", type=int, default=256)

    ap.add_argument("--save_attn", action="store_true")
    ap.add_argument("--vocab_dir", type=str, required=True)

    ap.add_argument("--hf_cache_dir", type=str, default=None)
    ap.add_argument("--offline", action="store_true")

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--warmup_head_epochs", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_update", type=int, default=25000)
    ap.add_argument("--warmup_steps", type=int, default=8000)
    ap.add_argument("--hold_steps", type=int, default=0)
    ap.add_argument("--decay_steps", type=int, default=72000)
    ap.add_argument("--init_lr_scale", type=float, default=0.1)
    ap.add_argument("--final_lr_scale", type=float, default=0.05)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[AST-DEC] device: {device} | torch {torch.__version__}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"[AST-DEC] cuda device: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.out_dir, exist_ok=True)

    processor = build_processor_from_vocab_dir(args.vocab_dir)
    bos_id = processor.tokenizer.bos_token_id
    eos_id = processor.tokenizer.eos_token_id
    
    pad_token_id = processor.tokenizer.pad_token_id
    vocab_size = len(processor.tokenizer)
    print(f"[AST-DEC] vocab_size={vocab_size} pad_token_id={pad_token_id} bos_id={bos_id} eos_id={eos_id}")
    print("[TOK DEBUG]", [processor.tokenizer.convert_ids_to_tokens(i) for i in range(40)])

    cache_dir = _infer_hf_cache_dir(args.hf_cache_dir)
    offline = _infer_offline_flag(args.offline)
    print(f"[AST-DEC] HF cache_dir: {cache_dir} | offline={offline}")

    print(f"[AST-DEC] enc_axis={args.enc_axis} token_order={args.token_order} freq_pool={args.freq_pool}")
    print(f"[AST-DEC] dec_layers={args.dec_layers} dec_heads={args.dec_heads} dec_ff={args.dec_ff} "
          f"dec_dropout={args.dec_dropout} dec_max_len={args.dec_max_len}")
    print(f"[AST-DEC] gen_max_len={args.gen_max_len} save_attn={args.save_attn}")
    print(f"[AST-DEC] chunk_overlap={args.chunk_overlap} dedup_overlap={args.dedup_overlap}")

    print(f"[AST-DEC] Loading AST FeatureExtractor for {args.ast_ckpt}")
    ast_feat_extractor = AutoFeatureExtractor.from_pretrained(
        args.ast_ckpt,
        cache_dir=cache_dir,
        local_files_only=offline,
    )

    if args.ast_from_scratch:
        print("[AST-DEC] Initializing AST from scratch (ASTConfig default).")
        ast = ASTModel(ASTConfig())
    else:
        print(f"[AST-DEC] Loading pretrained AST encoder: {args.ast_ckpt}")
        ast = ASTModel.from_pretrained(
            args.ast_ckpt,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        resize_ast_positional_embeddings(ast, freq_bins=args.freq_bins, max_frames=args.max_frames)

    model = ASTDecModel(
        ast=ast,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        freq_bins=args.freq_bins,
        max_frames=args.max_frames,
        upsample_factor=args.upsample_factor,
        freq_pool=args.freq_pool,
        enc_axis=args.enc_axis,
        token_order=args.token_order,
        dec_layers=args.dec_layers,
        dec_heads=args.dec_heads,
        dec_ff=args.dec_ff,
        dec_dropout=args.dec_dropout,
        dec_max_len=args.dec_max_len,
        tie_output=args.tie_output,
    ).to(device)

    print(f"[AST-DEC] patch kernel/stride: kernel=({model.k_f},{model.k_t}) stride=({model.s_f},{model.s_t})")
    print(f"[AST-DEC] lr_head={args.lr_head} lr_enc={args.lr_enc} warmup_head_epochs={args.warmup_head_epochs}")

    for p in model.ast.parameters():
        p.requires_grad = False
    for p in model.tok_emb.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.out_proj.parameters():
        p.requires_grad = True
    if hasattr(model, "freq_attn"):
        for p in model.freq_attn.parameters():
            p.requires_grad = True

    head_params = []
    head_params += list(model.tok_emb.parameters())
    head_params += list(model.decoder.parameters())
    head_params += list(model.out_proj.parameters())
    if hasattr(model, "freq_attn"):
        head_params += list(model.freq_attn.parameters())

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": model.ast.parameters(), "lr": args.lr_enc},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    scheduler = TriStageLRScheduler(
        optimizer,
        max_update=args.max_update,
        warmup_steps=args.warmup_steps,
        hold_steps=args.hold_steps,
        decay_steps=args.decay_steps,
        init_lr_scale=args.init_lr_scale,
        final_lr_scale=args.final_lr_scale,
    )

    max_train = None if args.max_train_utts < 0 else args.max_train_utts
    max_val = None if args.max_val_utts < 0 else args.max_val_utts
    max_test = None if args.max_test_utts < 0 else args.max_test_utts

    shuffle = not args.no_shuffle
    train_ds = LibriSpeechWaveText(args.data_root, args.train_split, max_train, shuffle=shuffle)
    val_ds   = LibriSpeechWaveText(args.data_root, args.val_split,   max_val,   shuffle=shuffle)
    test_ds  = LibriSpeechWaveText(args.data_root, args.test_split,  max_test,  shuffle=shuffle)
    print(f"[AST-DEC] dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    collate_fn = lambda b: collate_ast_dec(
        b,
        max_frames=args.max_frames,
        ast_feature_extractor=ast_feat_extractor,
        chunk_overlap=args.chunk_overlap,
    )

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_fn)

    best_val_wer = 1e9
    best_path = os.path.join(args.out_dir, "ast_dec_finetuned.pt")
    global_step = 0
    history = []

    for ep in range(1, args.epochs + 1):
        if ep == args.warmup_head_epochs + 1:
            print(f"[AST-DEC] Unfreezing AST encoder at epoch {ep}.")
            for p in model.ast.parameters():
                p.requires_grad = True

        t0 = time.time()
        tr_loss, global_step = train_one_epoch(
            model=model,
            processor=processor,
            loader=train_ld,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            pad_token_id=pad_token_id,
            max_grad_norm=args.max_grad_norm,
            fp16=args.fp16,
            global_step_start=global_step,
            debug_first_batch=(ep == 1),
            chunk_overlap=args.chunk_overlap,
            dedup_overlap=args.dedup_overlap,
        )

        val_loss, val_wer, val_cer, vrefs, vhyps = evaluate(
            model, processor, val_ld, device, pad_token_id,
            gen_max_len=args.gen_max_len,
            save_attn=args.save_attn,
            out_dir=args.out_dir,
            chunk_overlap=args.chunk_overlap,
            dedup_overlap=args.dedup_overlap,
        )

        dt = time.time() - t0
        print(f"[AST-DEC] Epoch {ep:02d} | train loss {tr_loss:.4f} | "
              f"val loss {val_loss:.4f} | val WER {val_wer*100:.2f}% | CER {val_cer*100:.2f}% | "
              f"{len(train_ds)/max(dt,1e-9):.1f} utts/s | global_step={global_step}")

        for i in range(min(2, len(vrefs))):
            print(f"  [val-{i}] REF: {vrefs[i]}")
            print(f"         HYP: {vhyps[i]}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)
            print(f"[AST-DEC] Saved best ckpt -> {best_path}")

        history.append({
            "epoch": int(ep),
            "global_step": int(global_step),
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_wer": float(val_wer),
            "val_cer": float(val_cer),
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "lr_enc": float(optimizer.param_groups[1]["lr"]),
            "enc_axis": str(args.enc_axis),
            "token_order": str(args.token_order),
            "freq_pool": str(args.freq_pool),
            "upsample_factor": int(args.upsample_factor),
            "gen_max_len": int(args.gen_max_len),
            "chunk_overlap": int(args.chunk_overlap),
            "dedup_overlap": bool(args.dedup_overlap),
            "save_attn": bool(args.save_attn),
        })

    test_loss, test_wer, test_cer, trefs, thyps = evaluate(
        model, processor, test_ld, device, pad_token_id,
        gen_max_len=args.gen_max_len,
        save_attn=False,
        out_dir=args.out_dir,
        chunk_overlap=args.chunk_overlap,
        dedup_overlap=args.dedup_overlap,
    )
    print(f"[AST-DEC] Test: loss {test_loss:.4f} | WER {test_wer*100:.2f}% | CER {test_cer*100:.2f}%")
    for i in range(min(3, len(trefs))):
        print(f"  [test-{i}] REF: {trefs[i]}")
        print(f"          HYP: {thyps[i]}")

    hist_path = os.path.join(args.out_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[AST-DEC] Saved training history -> {hist_path}")

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"AST ckpt: {args.ast_ckpt}\n")
        f.write(f"HF cache_dir: {cache_dir}\n")
        f.write(f"offline: {offline}\n")
        f.write(f"kernel=({model.k_f},{model.k_t}) stride=({model.s_f},{model.s_t})\n")
        f.write(f"max_frames={args.max_frames} freq_bins={args.freq_bins} upsample_factor={args.upsample_factor}\n")
        f.write(f"enc_axis: {args.enc_axis}\n")
        f.write(f"token_order: {args.token_order}\n")
        f.write(f"freq_pool: {args.freq_pool}\n")
        f.write(f"dec_layers={args.dec_layers} dec_heads={args.dec_heads} dec_ff={args.dec_ff} "
                f"dec_dropout={args.dec_dropout} dec_max_len={args.dec_max_len}\n")
        f.write(f"gen_max_len={args.gen_max_len}\n")
        f.write(f"chunk_overlap(frames): {args.chunk_overlap}\n")
        f.write(f"dedup_overlap: {args.dedup_overlap}\n")
        f.write(f"save_attn: {args.save_attn}\n")
        f.write(f"lr_head={args.lr_head} lr_enc={args.lr_enc} warmup_head_epochs={args.warmup_head_epochs}\n")
        f.write(f"Train split: {args.train_split}\nVal split: {args.val_split}\nTest split: {args.test_split}\n")
        f.write(f"Max train/val/test utts: {args.max_train_utts}/{args.max_val_utts}/{args.max_test_utts}\n")
        f.write(f"Best val WER={best_val_wer*100:.2f}%\n")
        f.write(f"Test WER={test_wer*100:.2f}% | Test CER={test_cer*100:.2f}%\n")


if __name__ == "__main__":
    main()
