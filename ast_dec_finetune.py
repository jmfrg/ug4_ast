#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AST + Decoder fine-tuning on LibriSpeech (cross-attention seq2seq)

Fixes applied:
1) tie_output: proper weight tying (share same Parameter) + avoid duplicate params in optimizer
2) val/test deterministic: do NOT shuffle indices; max_val_utts/max_test_utts are now controllable (first N)
3) greedy decode now uses the same autocast policy as eval/train (fp16/bf16)
4) reproducibility: add numpy seed + dataloader worker seeding (worker_init_fn + generator)

New training stabilizers (requested):
5) Label smoothing (default 0.1) + eos_weight in CE loss
6) Reduce decoding eval frequency/scale: always compute val loss on full val set, but decode WER/CER only
   every N epochs and on a small subset.
7) Moderate decoder dropout: default dec_dropout=0.2
"""

from __future__ import annotations
import os, time, random, argparse, re, json, math
from typing import List, Tuple, Optional

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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    # Make dataloader workers deterministic across runs
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def text_to_ids_strict(tokenizer: Wav2Vec2CTCTokenizer, s: str) -> List[int]:
    """
    STRICT char-level mapping:
    - s is expected to already be normalize_text_for_model output: only [A-Z'|]
    - map each character to vocab id; unknown -> unk_token_id
    """
    if not s:
        return []
    vocab = tokenizer.get_vocab()
    unk = int(tokenizer.unk_token_id)
    return [int(vocab.get(ch, unk)) for ch in s]


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


class LibriSpeechWaveText(Dataset):
    def __init__(self, root: str, split: str, max_utts: Optional[int] = None, shuffle: bool = False):
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
        # torchaudio LIBRISPEECH returns (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        for x in item[2:]:
            if isinstance(x, str):
                transcript = x
                break

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        txt = normalize_text_for_model(transcript)
        return wav, txt


def wav_to_ast_features(wav: torch.Tensor, ast_feature_extractor, sample_rate: int = 16000) -> torch.Tensor:
    if wav.dim() != 1:
        wav = wav.view(-1)
    wav_np = wav.detach().cpu().numpy().astype(np.float32)
    feats = ast_feature_extractor(wav_np, sampling_rate=sample_rate, return_tensors="pt", padding=False)
    x = feats["input_values"][0].to(torch.float32)  # [T, F]
    return x


def chunk_and_pad_feat(feat: torch.Tensor, max_frames: int = 1024, overlap: int = 0) -> Tuple[torch.Tensor, List[int]]:
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

    chunk_fbanks = torch.cat(all_chunks, dim=0)  # [sum_chunks, T, F]
    chunk_frame_lens = torch.tensor(all_lens, dtype=torch.long)

    t = torch.arange(max_frames).unsqueeze(0)
    chunk_attn_mask = (t < chunk_frame_lens.unsqueeze(1)).long()

    return chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts


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

    pos_patches_new = F.interpolate(pos_patches, size=(Fp_new, Tp_new), mode="bicubic", align_corners=False)
    pos_patches_new = pos_patches_new.permute(0, 2, 3, 1).reshape(1, Fp_new * Tp_new, H)
    pos_new = torch.cat([pos_special, pos_patches_new], dim=1)

    ast.embeddings.position_embeddings = nn.Parameter(pos_new)
    print(
        f"[AST-DEC] Resized position embeddings: {old_len} -> {pos_new.shape[1]} "
        f"(special={special}, grid {Fp_old}x{Tp_old} -> {Fp_new}x{Tp_new})"
    )


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
        dec_dropout: float = 0.2,
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

        self.tie_output = bool(tie_output)
        if self.tie_output:
            self.out_proj.weight = self.tok_emb.weight

    def _grid(self, F_bins: int, T_frames: int) -> Tuple[int, int, int]:
        Fp = (F_bins - self.k_f) // self.s_f + 1
        Tp = (T_frames - self.k_t) // self.s_t + 1
        return int(Fp), int(Tp), int(Fp * Tp)

    def forward_chunk_encoder(self, fbank_chunk: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def compute_time_patch_lens(frame_lens: torch.Tensor, k_t: int, s_t: int) -> torch.Tensor:
    tp = (frame_lens - k_t) // s_t + 1
    tp = torch.clamp(tp, min=0)
    return tp.to(torch.long)


def compute_chunk_memory_lens(enc_axis: str, frame_lens: torch.Tensor, k_t: int, s_t: int, upsample_factor: int, Fp: int) -> torch.Tensor:
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


def ids_to_text(tokenizer: Wav2Vec2CTCTokenizer, ids: List[int], show_unk: bool = True) -> str:
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None
    bos_id = int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None
    unk_tok = tokenizer.unk_token

    out = []
    for tid in ids:
        if eos_id is not None and tid == eos_id:
            break
        if pad_id is not None and tid == pad_id:
            continue
        if bos_id is not None and tid == bos_id:
            continue

        tok = tokenizer.convert_ids_to_tokens(int(tid))
        if tok == unk_tok:
            out.append("?" if show_unk else "")
        elif tok in ("<pad>", "<bos>", "<eos>"):
            continue
        else:
            out.append(tok)

    s = "".join(out).replace("|", " ")
    return s


def eos_weighted_ce_loss_ls(
    logits: torch.Tensor,      # [B,L,V]
    targets: torch.Tensor,     # [B,L]
    pad_id: int,
    eos_id: int,
    eos_weight: float = 1.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Cross-entropy with:
    - ignore pad
    - label smoothing
    - extra weight on EOS positions (optional)
    """
    B, L, V = logits.shape
    logits_f = logits.reshape(B * L, V)
    targets_f = targets.reshape(B * L)

    mask = targets_f != pad_id
    if mask.sum().item() == 0:
        # nothing to train on
        return logits_f.sum() * 0.0

    logits_f = logits_f[mask]
    targets_f = targets_f[mask]

    per_tok = F.cross_entropy(
        logits_f,
        targets_f,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )

    if float(eos_weight) != 1.0:
        w = torch.ones_like(per_tok)
        w = torch.where(targets_f == eos_id, w * float(eos_weight), w)
        return (per_tok * w).sum() / w.sum().clamp_min(1.0)

    return per_tok.mean()


def _autocast_ctx(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    raise ValueError(f"Unknown amp_dtype={amp_dtype}")


@torch.no_grad()
def greedy_decode(
    model: ASTDecModel,
    memory: torch.Tensor,
    mem_lengths: torch.Tensor,
    max_len: int,
    amp_dtype: str = "bf16",
) -> torch.Tensor:
    device = memory.device
    B = memory.size(0)
    mem_kpm = make_len_mask(mem_lengths, max_len=memory.size(1))

    ys = torch.full((B, 1), model.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros((B,), dtype=torch.bool, device=device)

    use_amp = (device.type == "cuda") and (amp_dtype in ("fp16", "bf16"))

    for _ in range(max_len):
        dec_kpm = (ys == model.pad_token_id)

        with _autocast_ctx(device, amp_dtype) if use_amp else torch.autocast(device_type="cpu", enabled=False):
            logits, _ = model.forward_decoder(
                memory=memory,
                mem_key_padding_mask=mem_kpm,
                dec_in_ids=ys,
                dec_key_padding_mask=dec_kpm,
                return_cross_attn=False,
            )

        logits_last = logits[:, -1, :]
        next_id = torch.argmax(logits_last, dim=-1)

        next_id = torch.where(finished, torch.full_like(next_id, model.pad_token_id), next_id)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

        finished = finished | (next_id == model.eos_id)
        if bool(finished.all()):
            break

    return ys[:, 1:]


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


def train_one_epoch(
    model: ASTDecModel,
    processor: Wav2Vec2Processor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[TriStageLRScheduler],
    pad_token_id: int,
    max_grad_norm: float = 1.0,
    amp_dtype: str = "bf16",
    global_step_start: int = 0,
    debug_first_batch: bool = True,
    chunk_overlap: int = 0,
    dedup_overlap: bool = False,
    eos_weight: float = 1.0,
    label_smoothing: float = 0.1,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    global_step = global_step_start

    use_amp = (device.type == "cuda") and (amp_dtype in ("fp16", "bf16"))
    use_scaler = (device.type == "cuda") and (amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    unk_id = int(processor.tokenizer.unk_token_id)

    for chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts in loader:
        global_step += 1
        optimizer.zero_grad(set_to_none=True)

        seqs = [text_to_ids_strict(processor.tokenizer, t) + [model.eos_id] for t in texts]
        maxL = max((len(s) for s in seqs), default=1)
        label_ids = torch.full((len(seqs), maxL), pad_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            if len(s) > 0:
                label_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        label_ids = label_ids.to(device)

        chunk_fbanks = chunk_fbanks.to(device)
        chunk_frame_lens = chunk_frame_lens.to(device)
        chunk_attn_mask = chunk_attn_mask.to(device)

        Fp, _, _ = model._grid(chunk_fbanks.size(2), chunk_fbanks.size(1))

        with _autocast_ctx(device, amp_dtype) if use_amp else torch.autocast(device_type="cpu", enabled=False):
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
            unk_ratio = float((label_ids == unk_id).sum().item()) / max(1, float((label_ids != pad_token_id).sum().item()))
            ex_lbl = ids_to_text(processor.tokenizer, label_ids[0].detach().cpu().tolist(), show_unk=True)
            print("[DEBUG] chunk_fbanks:", tuple(chunk_fbanks.shape))
            print("[DEBUG] chunk_mem:", tuple(chunk_mem.shape))
            print("[DEBUG] chunk_mem_lens min/max:", int(chunk_mem_lens.min()), int(chunk_mem_lens.max()))
            print("[DEBUG] drop_left(mem_steps):", int(drop_left))
            print("[DEBUG] memory:", tuple(memory.shape))
            print("[DEBUG] mem_lengths min/max:", int(mem_lengths.min()), int(mem_lengths.max()))
            print("[DEBUG] label_ids:", tuple(label_ids.shape))
            print("[DEBUG] dec_in:", tuple(dec_in.shape))
            print(f"[DEBUG] label <unk> ratio (non-pad): {unk_ratio:.4f}")
            print(f"[DEBUG] decoded label[0] (stop@eos, show_unk): {ex_lbl}")

        if use_scaler:
            with _autocast_ctx(device, amp_dtype):
                logits, _ = model.forward_decoder(
                    memory=memory,
                    mem_key_padding_mask=mem_kpm,
                    dec_in_ids=dec_in,
                    dec_key_padding_mask=dec_kpm,
                    return_cross_attn=False,
                )
                loss = eos_weighted_ce_loss_ls(
                    logits=logits,
                    targets=label_ids,
                    pad_id=pad_token_id,
                    eos_id=model.eos_id,
                    eos_weight=eos_weight,
                    label_smoothing=label_smoothing,
                )
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            with _autocast_ctx(device, amp_dtype) if use_amp else torch.autocast(device_type="cpu", enabled=False):
                logits, _ = model.forward_decoder(
                    memory=memory,
                    mem_key_padding_mask=mem_kpm,
                    dec_in_ids=dec_in,
                    dec_key_padding_mask=dec_kpm,
                    return_cross_attn=False,
                )
                loss = eos_weighted_ce_loss_ls(
                    logits=logits,
                    targets=label_ids,
                    pad_id=pad_token_id,
                    eos_id=model.eos_id,
                    eos_weight=eos_weight,
                    label_smoothing=label_smoothing,
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
    eos_weight: float = 1.0,
    label_smoothing: float = 0.1,
    amp_dtype: str = "bf16",
    do_decode: bool = True,
    decode_max_utts: int = 50,   # <=0 => no decode; <0 => decode all
    shuffle_memory_eval: bool = False,
    shuffle_memory_eval_seed: int = 12345,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_refs: List[str] = []
    all_hyps: List[str] = []

    saved_any_attn = False
    decoded = 0

    for chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts in loader:
        seqs = [text_to_ids_strict(processor.tokenizer, t) + [model.eos_id] for t in texts]
        maxL = max((len(s) for s in seqs), default=1)
        label_ids = torch.full((len(seqs), maxL), pad_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            if len(s) > 0:
                label_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        label_ids = label_ids.to(device)

        chunk_fbanks = chunk_fbanks.to(device)
        chunk_frame_lens = chunk_frame_lens.to(device)
        chunk_attn_mask = chunk_attn_mask.to(device)

        Fp, _, _ = model._grid(chunk_fbanks.size(2), chunk_fbanks.size(1))

        use_amp = (device.type == "cuda") and (amp_dtype in ("fp16", "bf16"))
        with _autocast_ctx(device, amp_dtype) if use_amp else torch.autocast(device_type="cpu", enabled=False):
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
        # ---- NEW: memory shuffle test (batch-level) ----
        if shuffle_memory_eval and memory.size(0) > 1:
            gen = torch.Generator(device=memory.device)
            # make deterministic across runs
            gen.manual_seed(int(shuffle_memory_eval_seed) + int(n_batches))
            perm = torch.randperm(memory.size(0), generator=gen, device=memory.device)
            memory = memory[perm]
            mem_lengths = mem_lengths[perm]
        mem_kpm = make_len_mask(mem_lengths, max_len=memory.size(1))

        dec_in = shift_right_with_bos(label_ids, pad_id=pad_token_id, bos_id=model.bos_id)
        dec_kpm = (dec_in == pad_token_id)

        with _autocast_ctx(device, amp_dtype) if use_amp else torch.autocast(device_type="cpu", enabled=False):
            logits, cross_w = model.forward_decoder(
                memory=memory,
                mem_key_padding_mask=mem_kpm,
                dec_in_ids=dec_in,
                dec_key_padding_mask=dec_kpm,
                return_cross_attn=(save_attn and (not saved_any_attn)),
            )
            loss = eos_weighted_ce_loss_ls(
                logits=logits,
                targets=label_ids,
                pad_id=pad_token_id,
                eos_id=model.eos_id,
                eos_weight=eos_weight,
                label_smoothing=label_smoothing,
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

        # -------- optional greedy decode on subset --------
        if not do_decode or decode_max_utts == 0:
            continue

        if decode_max_utts > 0 and decoded >= decode_max_utts:
            continue

        B = memory.size(0)
        take = B
        if decode_max_utts > 0:
            take = min(B, decode_max_utts - decoded)
        if take <= 0:
            continue

        gen_ids = greedy_decode(
            model,
            memory=memory[:take],
            mem_lengths=mem_lengths[:take],
            max_len=int(gen_max_len),
            amp_dtype=amp_dtype,
        )

        # ---- DEBUG: confirm EOS-first vs delimiter collapse ----
        if not hasattr(evaluate, "_printed_first_gen"):
            evaluate._printed_first_gen = True
            eos_id = int(model.eos_id)
            bos_id = int(model.bos_id)
            pad_id = int(model.pad_token_id)

            if gen_ids.size(1) == 0:
                print("[DBG] gen_ids has zero length!")
            else:
                first_ids = gen_ids[:, 0].detach().cpu().tolist()
                first_toks = [processor.tokenizer.convert_ids_to_tokens(int(x)) for x in first_ids]
                print("[DBG] first_gen_ids:", first_ids,
                    "| first_gen_toks:", first_toks,
                    "| eos/bos/pad:", eos_id, bos_id, pad_id)

        pred_str: List[str] = []
        for i in range(gen_ids.size(0)):
            ids = gen_ids[i].detach().cpu().tolist()
            pred_str.append(ids_to_text(processor.tokenizer, ids, show_unk=True))

        refs_norm = [normalize_text_for_wer(s) for s in texts[:take]]
        hyps_norm = [normalize_text_for_wer(s) for s in pred_str]

        all_refs.extend(refs_norm)
        all_hyps.extend(hyps_norm)
        decoded += take

    avg_loss = total_loss / max(1, n_batches)

    if len(all_refs) == 0:
        # decoding skipped
        return avg_loss, float("nan"), float("nan"), [], []

    wer_score = wer(all_refs, all_hyps)
    cer_score = cer(all_refs, all_hyps)
    return avg_loss, float(wer_score), float(cer_score), all_refs[:3], all_hyps[:3]


def _unique_params(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen = set()
    out = []
    for p in params:
        if p is None:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


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
    ap.add_argument("--no_shuffle", action="store_true")

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
    ap.add_argument("--dec_dropout", type=float, default=0.2)  # <-- increased default
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

    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--eos_weight", type=float, default=1.0)

    # ---- new: label smoothing + decode frequency/scale ----
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--decode_every", type=int, default=5,
                    help="Run greedy decoding/WER every N epochs (and always on epoch 1). 0 disables decoding during training.")
    ap.add_argument("--decode_max_utts", type=int, default=50,
                    help="Max #utterances to decode for WER/CER when decoding is enabled. -1 means decode all.")

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--shuffle_memory_eval", action="store_true",
                    help="In eval, shuffle encoder memory across batch to test whether decoder uses audio.")
    ap.add_argument("--shuffle_memory_eval_seed", type=int, default=12345)

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
          f"dec_dropout={args.dec_dropout} dec_max_len={args.dec_max_len} tie_output={args.tie_output}")
    print(f"[AST-DEC] gen_max_len={args.gen_max_len} save_attn={args.save_attn}")
    print(f"[AST-DEC] chunk_overlap={args.chunk_overlap} dedup_overlap={args.dedup_overlap}")
    print(f"[AST-DEC] eos_weight={args.eos_weight} label_smoothing={args.label_smoothing}")
    print(f"[AST-DEC] decode_every={args.decode_every} decode_max_utts={args.decode_max_utts}")
    print(f"[AST-DEC] amp_dtype={args.amp_dtype}")

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

    # freeze encoder initially
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

    head_params: List[torch.nn.Parameter] = []
    head_params += list(model.tok_emb.parameters())
    head_params += list(model.decoder.parameters())
    if not model.tie_output:
        head_params += list(model.out_proj.parameters())
    if hasattr(model, "freq_attn"):
        head_params += list(model.freq_attn.parameters())
    head_params = _unique_params(head_params)

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": list(model.ast.parameters()), "lr": args.lr_enc},
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

    train_shuffle_indices = (not args.no_shuffle)
    train_ds = LibriSpeechWaveText(args.data_root, args.train_split, max_train, shuffle=train_shuffle_indices)
    val_ds   = LibriSpeechWaveText(args.data_root, args.val_split,   max_val,   shuffle=False)
    test_ds  = LibriSpeechWaveText(args.data_root, args.test_split,  max_test,  shuffle=False)
    print(f"[AST-DEC] dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    collate_fn = lambda b: collate_ast_dec(
        b,
        max_frames=args.max_frames,
        ast_feature_extractor=ast_feat_extractor,
        chunk_overlap=args.chunk_overlap,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    best_val_loss = 1e9
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
            amp_dtype=args.amp_dtype,
            global_step_start=global_step,
            debug_first_batch=(ep == 1),
            chunk_overlap=args.chunk_overlap,
            dedup_overlap=args.dedup_overlap,
            eos_weight=args.eos_weight,
            label_smoothing=args.label_smoothing,
        )

        do_decode = False
        if args.decode_every > 0:
            do_decode = (ep == 1) or (ep % args.decode_every == 0)

        val_loss, val_wer, val_cer, vrefs, vhyps = evaluate(
            model, processor, val_ld, device, pad_token_id,
            gen_max_len=args.gen_max_len,
            save_attn=args.save_attn,
            out_dir=args.out_dir,
            chunk_overlap=args.chunk_overlap,
            dedup_overlap=args.dedup_overlap,
            eos_weight=args.eos_weight,
            label_smoothing=args.label_smoothing,
            amp_dtype=args.amp_dtype,
            do_decode=do_decode,
            decode_max_utts=args.decode_max_utts,
        )
        if args.shuffle_memory_eval:
            shuf_loss, shuf_wer, shuf_cer, _, _ = evaluate(
                model, processor, val_ld, device, pad_token_id,
                gen_max_len=args.gen_max_len,
                save_attn=False,          # shuffle 测试不保存 attn（避免混淆）
                out_dir=args.out_dir,
                chunk_overlap=args.chunk_overlap,
                dedup_overlap=args.dedup_overlap,
                eos_weight=args.eos_weight,
                label_smoothing=args.label_smoothing,
                amp_dtype=args.amp_dtype,
                do_decode=do_decode,      # 如果本轮你有 decode，就也对 shuffle 做 decode
                decode_max_utts=args.decode_max_utts,
                shuffle_memory_eval=True,
                shuffle_memory_eval_seed=args.shuffle_memory_eval_seed + ep * 1000,
            )
            if not math.isnan(shuf_wer):
                print(f"[AST-DEC] (shuffle_memory) val loss {shuf_loss:.4f} | "
                      f"val WER {shuf_wer*100:.2f}% | CER {shuf_cer*100:.2f}%")
            else:
                print(f"[AST-DEC] (shuffle_memory) val loss {shuf_loss:.4f}")

        dt = time.time() - t0

        if not math.isnan(val_wer):
            print(f"[AST-DEC] Epoch {ep:02d} | train loss {tr_loss:.4f} | "
                  f"val loss {val_loss:.4f} | val WER {val_wer*100:.2f}% | CER {val_cer*100:.2f}% | "
                  f"{len(train_ds)/max(dt,1e-9):.1f} utts/s | global_step={global_step}")
            for i in range(min(2, len(vrefs))):
                print(f"  [val-{i}] REF: {vrefs[i]}")
                print(f"         HYP: {vhyps[i]}")
            best_val_wer = min(best_val_wer, val_wer)
        else:
            print(f"[AST-DEC] Epoch {ep:02d} | train loss {tr_loss:.4f} | "
                  f"val loss {val_loss:.4f} | val WER/CER: (skipped) | "
                  f"{len(train_ds)/max(dt,1e-9):.1f} utts/s | global_step={global_step}")

        # save best by val loss (stable + always available)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)
            print(f"[AST-DEC] Saved best (by val loss) ckpt -> {best_path}")

        history.append({
            "epoch": int(ep),
            "global_step": int(global_step),
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_wer": None if math.isnan(val_wer) else float(val_wer),
            "val_cer": None if math.isnan(val_cer) else float(val_cer),
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
            "eos_weight": float(args.eos_weight),
            "label_smoothing": float(args.label_smoothing),
            "decode_every": int(args.decode_every),
            "decode_max_utts": int(args.decode_max_utts),
            "amp_dtype": str(args.amp_dtype),
            "tie_output": bool(args.tie_output),
            "seed": int(args.seed),
        })

    # final test: decode on all test items (max_test_utts controls size)
    test_loss, test_wer, test_cer, trefs, thyps = evaluate(
        model, processor, test_ld, device, pad_token_id,
        gen_max_len=args.gen_max_len,
        save_attn=False,
        out_dir=args.out_dir,
        chunk_overlap=args.chunk_overlap,
        dedup_overlap=args.dedup_overlap,
        eos_weight=args.eos_weight,
        label_smoothing=args.label_smoothing,
        amp_dtype=args.amp_dtype,
        do_decode=True,
        decode_max_utts=-1,
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
        f.write(f"tie_output={args.tie_output}\n")
        f.write(f"gen_max_len={args.gen_max_len}\n")
        f.write(f"chunk_overlap(frames): {args.chunk_overlap}\n")
        f.write(f"dedup_overlap: {args.dedup_overlap}\n")
        f.write(f"save_attn: {args.save_attn}\n")
        f.write(f"lr_head={args.lr_head} lr_enc={args.lr_enc} warmup_head_epochs={args.warmup_head_epochs}\n")
        f.write(f"eos_weight={args.eos_weight}\n")
        f.write(f"label_smoothing={args.label_smoothing}\n")
        f.write(f"decode_every={args.decode_every}\n")
        f.write(f"decode_max_utts={args.decode_max_utts}\n")
        f.write(f"amp_dtype={args.amp_dtype}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"Train split: {args.train_split}\nVal split: {args.val_split}\nTest split: {args.test_split}\n")
        f.write(f"Max train/val/test utts: {args.max_train_utts}/{args.max_val_utts}/{args.max_test_utts}\n")
        f.write(f"Best val loss={best_val_loss:.6f}\n")
        if best_val_wer < 1e9:
            f.write(f"Best val WER(seen)={best_val_wer*100:.2f}%\n")
        f.write(f"Test WER={test_wer*100:.2f}% | Test CER={test_cer*100:.2f}%\n")


if __name__ == "__main__":
    main()
