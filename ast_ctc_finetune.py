#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AST + CTC fine-tuning on LibriSpeech (explicit CTC, shared vocab with HuBERT baseline)

Key features / fixes:
- Dynamic patch grid: compute F_patches/T_patches from AST patch embedding Conv2d kernel/stride
- Auto special tokens: skip CLS / DIST tokens by (seq_len - num_patches)
- Correct CTC input_lengths:
    * ctc_axis=time  : input_lengths = Tp_valid * upsample_factor
    * ctc_axis=token : input_lengths = Tp_valid * Fp  (flattened patch tokens; order configurable)
- Position embedding resize (2D interpolation) when max_frames changes (e.g., 1024 -> 2048)
- FP16-friendly CTC: always feed float32 log_probs to CTCLoss on CUDA
- Debug: print lengths for first batch of epoch 1 only
- Drop invalid samples (input_len < target_len) to avoid zero_infinity masking issues
- Temporal upsampling (repeat_interleave) for ctc_axis=time
- Head warm-up: freeze AST for first N epochs, train head; then unfreeze encoder and continue
- Offline/HPC safe: validates local LibriSpeech; HF cache_dir + offline/local_files_only support
- Save training_history.json (per-epoch metrics + per-group LR)

Input preprocessing:
- Uses AST's official FeatureExtractor (AutoFeatureExtractor.from_pretrained(ast_ckpt)) to produce
  log-mel features consistent with the checkpoint.

Minimal updates retained:
- collate returns frame-level attention_mask; forward passes attention_mask into AST
- eval decode masks padded positions (beyond input_lengths) as blank before batch_decode

Adapters / ablations:
- --ctc_axis {time,token} (default=time)
    * time: pool freq patches -> 1D time sequence for CTC (uses --freq_pool)
    * token: CTC axis is the full patch-token sequence, flatten order controlled by --token_order
- --token_order {time_first,freq_first} (default=time_first)
    * only used when ctc_axis=token
    * time_first: flatten (t,f): [Tp, Fp]
    * freq_first: flatten (f,t): [Fp, Tp]
- --freq_pool {mean,max,attn} (default=mean), only used when ctc_axis=time

Decoding (evaluation only):
- --decode {greedy,beam} (default=greedy)
- --beam_size INT (default=20)
- --beam_topk  INT (default=40): per-frame token expansion top-k (speed control)
  (beam decoding is pure CTC prefix beam search, no LM)
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
            f"[AST-CTC] LibriSpeech split not found locally:\n"
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
    )

    # Kept for API consistency; AST features are produced by AutoFeatureExtractor.
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
    """
    Tri-stage schedule applied per param-group:
      warmup:   init_lr -> peak_lr (linear)
      hold:     peak_lr
      decay:    peak_lr -> final_lr (linear)
      final:    final_lr

    peak_lr is taken from optimizer.param_groups[i]["lr"] at construction time.
    """
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
# CTC targets
# -------------------------
def build_ctc_targets(label_ids: torch.Tensor, pad_token_id: int):
    with torch.no_grad():
        nonpad_mask = (label_ids != pad_token_id)
        target_lengths = nonpad_mask.sum(dim=1, dtype=torch.long)
        pieces = []
        for b in range(label_ids.size(0)):
            pieces.append(label_ids[b, nonpad_mask[b]])
        targets = torch.cat(pieces, dim=0) if len(pieces) > 0 else torch.empty(
            0, dtype=label_ids.dtype, device=label_ids.device
        )
    return targets, target_lengths


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
        wav = item[0]  # [1, T]
        sr = item[1]

        transcript = ""
        for x in item[2:]:
            if isinstance(x, str):
                transcript = x
                break

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)  # [T]
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
    """
    Uses HF ASTFeatureExtractor/AutoFeatureExtractor to convert waveform -> log-mel features.
    Returns: [frames, 128] float32 torch tensor on CPU.
    """
    if wav.dim() != 1:
        wav = wav.view(-1)
    wav_np = wav.detach().cpu().numpy().astype(np.float32)
    feats = ast_feature_extractor(
        wav_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    )
    x = feats["input_values"][0].to(torch.float32)  # [frames, 128]
    return x


def chunk_and_pad_feat(feat: torch.Tensor, max_frames: int = 1024) -> Tuple[torch.Tensor, List[int]]:
    """
    feat: [T, 128]
    returns: chunks [n_chunks, max_frames, 128], lens list[int]
    """
    T = feat.size(0)
    if T <= max_frames:
        chunk = torch.zeros((max_frames, feat.size(1)), dtype=feat.dtype)
        chunk[:T] = feat
        return chunk.unsqueeze(0), [T]

    n_chunks = math.ceil(T / max_frames)
    chunks = torch.zeros((n_chunks, max_frames, feat.size(1)), dtype=feat.dtype)
    lens = []
    for k in range(n_chunks):
        s = k * max_frames
        e = min((k + 1) * max_frames, T)
        cur = feat[s:e]
        chunks[k, : (e - s)] = cur
        lens.append(e - s)
    return chunks, lens


def collate_ast_ctc(batch, max_frames: int, ast_feature_extractor):
    """
    Returns:
      chunk_fbanks:      [B_total_chunks, max_frames, 128]
      chunk_frame_lens:  [B_total_chunks]
      chunk_attn_mask:   [B_total_chunks, max_frames]  (1=valid frame, 0=pad)
      utt_slices:        List[(start_idx, n_chunks)] per utterance
      texts:             List[str] length=batch_size (utterance-level)
    """
    waves, texts = zip(*batch)
    texts = list(texts)

    all_chunks = []
    all_lens = []
    utt_slices = []
    cursor = 0

    for wav in waves:
        feat = wav_to_ast_features(wav, ast_feature_extractor, 16000)  # [frames, 128]
        chunks, lens = chunk_and_pad_feat(feat, max_frames=max_frames)
        n = chunks.size(0)
        all_chunks.append(chunks)
        all_lens.extend(lens)
        utt_slices.append((cursor, n))
        cursor += n

    chunk_fbanks = torch.cat(all_chunks, dim=0)  # [B_total_chunks, max_frames, 128]
    chunk_frame_lens = torch.tensor(all_lens, dtype=torch.long)

    t = torch.arange(max_frames).unsqueeze(0)  # [1, max_frames]
    chunk_attn_mask = (t < chunk_frame_lens.unsqueeze(1)).long()  # [B_total_chunks, max_frames]

    return chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts


# -------------------------
# Resize AST pos embeddings when max_frames changes
# -------------------------
def resize_ast_positional_embeddings(ast: ASTModel, freq_bins: int, max_frames: int):
    pos = ast.embeddings.position_embeddings  # [1, old_len, H]
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

    pos_special = pos[:, :special, :]                      # [1, special, H]
    pos_patches = pos[:, special:, :]                      # [1, Fp_old*Tp_old, H]
    pos_patches = pos_patches.reshape(1, Fp_old, Tp_old, H).permute(0, 3, 1, 2)  # [1,H,Fp_old,Tp_old]

    pos_patches_new = F.interpolate(
        pos_patches,
        size=(Fp_new, Tp_new),
        mode="bicubic",
        align_corners=False,
    )  # [1,H,Fp_new,Tp_new]

    pos_patches_new = pos_patches_new.permute(0, 2, 3, 1).reshape(1, Fp_new * Tp_new, H)
    pos_new = torch.cat([pos_special, pos_patches_new], dim=1)

    ast.embeddings.position_embeddings = nn.Parameter(pos_new)
    print(
        f"[AST-CTC] Resized position embeddings: {old_len} -> {pos_new.shape[1]} "
        f"(special={special}, grid {Fp_old}x{Tp_old} -> {Fp_new}x{Tp_new})"
    )


# -------------------------
# CTC decoding: greedy / prefix beam search (no LM)
# -------------------------
def _logsumexp(a: float, b: float) -> float:
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,   # [T, V] log-probabilities (float32 on CPU recommended)
    beam_size: int,
    blank_id: int,
    topk: int = 40,
) -> List[int]:
    """
    Prefix beam search for CTC (no LM).
    Returns best label sequence as a list of token ids (NO blanks).
    Complexity: O(T * beam_size * topk)
    """
    assert log_probs.dim() == 2, f"log_probs must be [T,V], got {tuple(log_probs.shape)}"
    T, V = log_probs.shape

    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -float("inf"))}  # (p_blank, p_nonblank)

    topk = int(topk)
    if topk <= 0:
        topk = V
    topk = min(topk, V)

    for t in range(T):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        lp_t = log_probs[t]  # [V]
        if topk < V:
            topv, topi = torch.topk(lp_t, k=topk)
            topi = topi.tolist()
            topv = topv.tolist()
        else:
            topi = list(range(V))
            topv = lp_t.tolist()

        for prefix, (pb, pnb) in beams.items():
            p_total = _logsumexp(pb, pnb)

            for c, lp in zip(topi, topv):
                if c == blank_id:
                    nb_pb, nb_pnb = next_beams.get(prefix, (-float("inf"), -float("inf")))
                    nb_pb = _logsumexp(nb_pb, p_total + lp)
                    next_beams[prefix] = (nb_pb, nb_pnb)
                    continue

                last = prefix[-1] if len(prefix) > 0 else None
                new_prefix = prefix + (c,)

                if last == c:
                    # stay at prefix from nonblank
                    nb_pb, nb_pnb = next_beams.get(prefix, (-float("inf"), -float("inf")))
                    nb_pnb = _logsumexp(nb_pnb, pnb + lp)
                    next_beams[prefix] = (nb_pb, nb_pnb)

                    # extend from blank only
                    nb_pb2, nb_pnb2 = next_beams.get(new_prefix, (-float("inf"), -float("inf")))
                    nb_pnb2 = _logsumexp(nb_pnb2, pb + lp)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)
                else:
                    nb_pb2, nb_pnb2 = next_beams.get(new_prefix, (-float("inf"), -float("inf")))
                    nb_pnb2 = _logsumexp(nb_pnb2, p_total + lp)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)

        # prune
        scored = []
        for pfx, (pb, pnb) in next_beams.items():
            scored.append((pfx, _logsumexp(pb, pnb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {pfx: next_beams[pfx] for pfx, _ in scored[:beam_size]}

    best_prefix = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)


def ids_to_text(tokenizer, ids: List[int]) -> str:
    toks = tokenizer.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in ("<pad>", "<unk>")]
    s = "".join(toks).replace("|", " ")
    return s


# -------------------------
# AST + CTC model
# -------------------------
class ASTCTCModel(nn.Module):
    def __init__(
        self,
        ast: ASTModel,
        vocab_size: int,
        freq_bins: int = 128,
        max_frames: int = 1024,
        upsample_factor: int = 1,
        freq_pool: str = "mean",
        ctc_axis: str = "time",  # time | token
        token_order: str = "time_first",  # time_first | freq_first (only for ctc_axis=token)
    ):
        super().__init__()
        self.ast = ast
        self.lm_head = nn.Linear(ast.config.hidden_size, vocab_size)

        self.freq_bins = int(freq_bins)
        self.max_frames = int(max_frames)
        self.upsample_factor = int(upsample_factor)
        if self.upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")

        self.ctc_axis = str(ctc_axis)
        if self.ctc_axis not in ("time", "token"):
            raise ValueError("ctc_axis must be one of: time, token")

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

    def _grid(self, F_bins: int, T_frames: int) -> Tuple[int, int, int]:
        Fp = (F_bins - self.k_f) // self.s_f + 1
        Tp = (T_frames - self.k_t) // self.s_t + 1
        return int(Fp), int(Tp), int(Fp * Tp)

    def forward_chunk(
        self,
        fbank_chunk: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input:
          fbank_chunk: [B', max_frames, 128]
          attention_mask: [B', max_frames] (frame-level; helps in your setup)
        output:
          if ctc_axis=time  : logits [B', Tp*upsample, V]
          if ctc_axis=token : logits [B', Tp*Fp, V]  (flatten order controlled by token_order)
        """
        out = self.ast(input_values=fbank_chunk, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state  # [B', seq_len, H]

        Bp = hs.size(0)
        T_frames = fbank_chunk.size(1)
        F_bins = fbank_chunk.size(2)

        Fp, Tp, num_patches = self._grid(F_bins, T_frames)
        special = hs.size(1) - num_patches  # CLS / (CLS + DIST)

        if special <= 0:
            raise RuntimeError(
                f"[AST-CTC] token length mismatch: hs_len={hs.size(1)} num_patches={num_patches} "
                f"(Fp={Fp},Tp={Tp}, kernel=({self.k_f},{self.k_t}), stride=({self.s_f},{self.s_t}))"
            )

        # patches: [B', Fp, Tp, H]
        patch_hs = hs[:, special:, :].view(Bp, Fp, Tp, -1)

        if self.ctc_axis == "token":
            if self.token_order == "time_first":
                # time-first flatten: [B, Fp, Tp, H] -> [B, Tp, Fp, H] -> [B, Tp*Fp, H]
                tok_hs = patch_hs.permute(0, 2, 1, 3).reshape(Bp, Tp * Fp, -1)
            else:
                # freq-first flatten: [B, Fp, Tp, H] -> [B, Fp*Tp, H]
                tok_hs = patch_hs.reshape(Bp, Fp * Tp, -1)

            logits = self.lm_head(tok_hs)
            return logits

        # ctc_axis == "time": frequency pooling -> [B', Tp, H]
        if self.freq_pool == "mean":
            time_hs = patch_hs.mean(dim=1)
        elif self.freq_pool == "max":
            time_hs = patch_hs.max(dim=1).values
        else:  # attn
            x = patch_hs.permute(0, 2, 1, 3)                 # [B', Tp, Fp, H]
            scores = self.freq_attn(x).squeeze(-1)           # [B', Tp, Fp]
            alpha = torch.softmax(scores, dim=-1)            # [B', Tp, Fp]
            time_hs = (alpha.unsqueeze(-1) * x).sum(dim=2)   # [B', Tp, H]

        r = self.upsample_factor
        if r > 1:
            time_hs = time_hs.repeat_interleave(r, dim=1)  # [B', Tp*r, H]

        logits = self.lm_head(time_hs)
        return logits


# -------------------------
# Length computation
# -------------------------
def compute_time_patch_lens(frame_lens: torch.Tensor, k_t: int, s_t: int) -> torch.Tensor:
    tp = (frame_lens - k_t) // s_t + 1
    tp = torch.clamp(tp, min=0)
    return tp.to(torch.long)


def compute_chunk_output_lens(
    ctc_axis: str,
    frame_lens: torch.Tensor,
    k_t: int,
    s_t: int,
    upsample_factor: int,
    Fp: int,
) -> torch.Tensor:
    """
    Returns per-chunk valid output lengths in the logits time dimension.
      - time axis : Tp_valid * upsample_factor
      - token axis: Tp_valid * Fp  (valid time patches across all frequencies; order independent)
    """
    tp_valid = compute_time_patch_lens(frame_lens, k_t, s_t)  # [B_chunks]
    if ctc_axis == "token":
        return (tp_valid * int(Fp)).to(torch.long)
    return (tp_valid * int(upsample_factor)).to(torch.long)


def assemble_utt_logits(
    chunk_logits: torch.Tensor,           # [B_total_chunks, T_max, V]
    chunk_out_lens: torch.Tensor,         # [B_total_chunks] valid T for each chunk
    utt_slices: List[Tuple[int, int]],    # per utterance: (start_idx, n_chunks)
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = chunk_logits.device
    B = len(utt_slices)
    V = chunk_logits.size(-1)

    utt_lens = []
    utt_seqs = []
    for (s, n) in utt_slices:
        parts = []
        total = 0
        for k in range(n):
            idx = s + k
            Lk = int(chunk_out_lens[idx].item())
            parts.append(chunk_logits[idx, :Lk, :])
            total += Lk
        utt_lens.append(total)
        utt_seqs.append(torch.cat(parts, dim=0))  # [total, V]

    max_T = max(utt_lens) if utt_lens else 0
    padded = torch.zeros((B, max_T, V), device=device, dtype=chunk_logits.dtype)
    for i, seq in enumerate(utt_seqs):
        padded[i, : seq.size(0)] = seq

    return padded, torch.tensor(utt_lens, device=device, dtype=torch.long)


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model: ASTCTCModel,
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
    drop_invalid: bool = True,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    global_step = global_step_start

    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ctc_loss = nn.CTCLoss(blank=pad_token_id, zero_infinity=True)

    for chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts in loader:
        global_step += 1
        optimizer.zero_grad(set_to_none=True)

        label_ids = processor(text=texts, return_tensors="pt", padding=True).input_ids.to(device)
        targets, target_lengths = build_ctc_targets(label_ids, pad_token_id)

        chunk_fbanks = chunk_fbanks.to(device)
        chunk_frame_lens = chunk_frame_lens.to(device)
        chunk_attn_mask = chunk_attn_mask.to(device)

        # grid for this batch (fixed by max_frames/freq_bins)
        Fp, Tp, _ = model._grid(chunk_fbanks.size(2), chunk_fbanks.size(1))

        if use_amp:
            with torch.amp.autocast("cuda"):
                chunk_logits = model.forward_chunk(chunk_fbanks, attention_mask=chunk_attn_mask)
        else:
            chunk_logits = model.forward_chunk(chunk_fbanks, attention_mask=chunk_attn_mask)

        chunk_out_lens = compute_chunk_output_lens(
            ctc_axis=model.ctc_axis,
            frame_lens=chunk_frame_lens,
            k_t=model.k_t,
            s_t=model.s_t,
            upsample_factor=model.upsample_factor,
            Fp=Fp,
        )

        utt_logits, input_lengths = assemble_utt_logits(chunk_logits, chunk_out_lens, utt_slices)

        if debug_first_batch and global_step == global_step_start + 1:
            print("[DEBUG] chunk_fbanks:", tuple(chunk_fbanks.shape))
            print("[DEBUG] chunk_frame_lens min/max:", int(chunk_frame_lens.min()), int(chunk_frame_lens.max()))
            print("[DEBUG] chunk_out_lens   min/max:", int(chunk_out_lens.min()), int(chunk_out_lens.max()))
            print("[DEBUG] utt_logits:", tuple(utt_logits.shape))
            print("[DEBUG] input_lengths  min/max:", int(input_lengths.min()), int(input_lengths.max()))
            print("[DEBUG] target_lengths min/max:", int(target_lengths.min()), int(target_lengths.max()))
            invalid = (input_lengths < target_lengths) | (input_lengths <= 0) | (target_lengths <= 0)
            print(f"[DEBUG] invalid_utts = {int(invalid.sum())}/{invalid.numel()} (drop_invalid={drop_invalid})")

        if drop_invalid:
            valid_mask = (input_lengths >= target_lengths) & (input_lengths > 0) & (target_lengths > 0)
            if valid_mask.sum().item() == 0:
                if debug_first_batch and global_step == global_step_start + 1:
                    print("[DEBUG] All utterances invalid for CTC in this batch. Skipping update.")
                continue
            utt_logits = utt_logits[valid_mask]
            input_lengths = input_lengths[valid_mask]
            label_ids = label_ids[valid_mask]
            targets, target_lengths = build_ctc_targets(label_ids, pad_token_id)

        log_probs = utt_logits.log_softmax(dim=-1).transpose(0, 1)  # [T,B,V]

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss = ctc_loss(log_probs.float(), targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = ctc_loss(log_probs.float(), targets, input_lengths, target_lengths)
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
    model: ASTCTCModel,
    processor: Wav2Vec2Processor,
    loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    decode: str = "greedy",
    beam_size: int = 20,
    beam_topk: int = 40,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_refs: List[str] = []
    all_hyps: List[str] = []

    ctc_loss = nn.CTCLoss(blank=pad_token_id, zero_infinity=True)

    for chunk_fbanks, chunk_frame_lens, chunk_attn_mask, utt_slices, texts in loader:
        label_ids = processor(text=texts, return_tensors="pt", padding=True).input_ids.to(device)
        targets, target_lengths = build_ctc_targets(label_ids, pad_token_id)

        chunk_fbanks = chunk_fbanks.to(device)
        chunk_frame_lens = chunk_frame_lens.to(device)
        chunk_attn_mask = chunk_attn_mask.to(device)

        Fp, Tp, _ = model._grid(chunk_fbanks.size(2), chunk_fbanks.size(1))

        chunk_logits = model.forward_chunk(chunk_fbanks, attention_mask=chunk_attn_mask)

        chunk_out_lens = compute_chunk_output_lens(
            ctc_axis=model.ctc_axis,
            frame_lens=chunk_frame_lens,
            k_t=model.k_t,
            s_t=model.s_t,
            upsample_factor=model.upsample_factor,
            Fp=Fp,
        )

        utt_logits, input_lengths = assemble_utt_logits(chunk_logits, chunk_out_lens, utt_slices)

        log_probs_TBV = utt_logits.log_softmax(dim=-1).transpose(0, 1)  # [T,B,V]
        loss = ctc_loss(log_probs_TBV.float(), targets, input_lengths, target_lengths)

        total_loss += float(loss.item())
        n_batches += 1

        # ---- decode ----
        decode = str(decode)
        if decode == "beam":
            # per-utt beam search on CPU float32
            lp_btv = utt_logits.log_softmax(dim=-1)  # [B, T, V]
            B = lp_btv.size(0)
            pred_str: List[str] = []
            for i in range(B):
                L = int(input_lengths[i].item())
                lp_tv = lp_btv[i, :L, :].detach().to("cpu", dtype=torch.float32)  # [T,V]
                best_ids = ctc_prefix_beam_search(
                    lp_tv, beam_size=int(beam_size), blank_id=pad_token_id, topk=int(beam_topk)
                )
                pred_str.append(ids_to_text(processor.tokenizer, best_ids))
        else:
            # greedy
            pred_ids = torch.argmax(utt_logits, dim=-1)  # [B, max_T]
            for i, L in enumerate(input_lengths.tolist()):
                if L < pred_ids.size(1):
                    pred_ids[i, L:] = pad_token_id  # blank
            pred_str = processor.batch_decode(pred_ids)

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

    # data
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

    # AST / features
    ap.add_argument("--ast_ckpt", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--ast_from_scratch", action="store_true")
    ap.add_argument("--max_frames", type=int, default=2048)
    ap.add_argument("--freq_bins", type=int, default=128)

    ap.add_argument("--ctc_axis", type=str, default="time", choices=["time", "token"],
                    help="CTC time axis: time=pooled time tokens; token=full patch-token sequence.")

    ap.add_argument("--token_order", type=str, default="time_first", choices=["time_first", "freq_first"],
                    help="Only used when ctc_axis=token. time_first flattens (t,f), freq_first flattens (f,t).")

    ap.add_argument("--freq_pool", type=str, default="mean", choices=["mean", "max", "attn"],
                    help="Pool frequency patches into a 1D time sequence for CTC (only used if ctc_axis=time).")

    # temporal upsampling (only meaningful for ctc_axis=time)
    ap.add_argument("--upsample_factor", type=int, default=4,
                    help="repeat_interleave factor on time tokens for CTC (ctc_axis=time)")

    # shared tokenizer/vocab
    ap.add_argument("--vocab_dir", type=str, required=True)

    # HF cache + offline
    ap.add_argument("--hf_cache_dir", type=str, default=None)
    ap.add_argument("--offline", action="store_true", help="Force local_files_only=True for HF models.")

    # optimization
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

    # decoding (eval only)
    ap.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"],
                    help="Decoding for evaluation: greedy or CTC prefix beam search (no LM).")
    ap.add_argument("--beam_size", type=int, default=20,
                    help="Beam size for --decode beam.")
    ap.add_argument("--beam_topk", type=int, default=40,
                    help="Per-frame token expansion top-k for beam search (speed control).")

    # output
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[AST-CTC] device: {device} | torch {torch.__version__}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"[AST-CTC] cuda device: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.out_dir, exist_ok=True)

    processor = build_processor_from_vocab_dir(args.vocab_dir)
    pad_token_id = processor.tokenizer.pad_token_id
    vocab_size = len(processor.tokenizer)
    print(f"[AST-CTC] vocab_size={vocab_size} pad_token_id={pad_token_id}")

    cache_dir = _infer_hf_cache_dir(args.hf_cache_dir)
    offline = _infer_offline_flag(args.offline)
    print(f"[AST-CTC] HF cache_dir: {cache_dir} | offline={offline}")

    print(f"[AST-CTC] decode={args.decode} beam_size={args.beam_size} beam_topk={args.beam_topk}")

    print(f"[AST-CTC] Loading AST FeatureExtractor for {args.ast_ckpt}")
    ast_feat_extractor = AutoFeatureExtractor.from_pretrained(
        args.ast_ckpt,
        cache_dir=cache_dir,
        local_files_only=offline,
    )

    if args.ast_from_scratch:
        print("[AST-CTC] Initializing AST from scratch (ASTConfig default).")
        ast = ASTModel(ASTConfig())
    else:
        print(f"[AST-CTC] Loading pretrained AST encoder: {args.ast_ckpt}")
        ast = ASTModel.from_pretrained(
            args.ast_ckpt,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        resize_ast_positional_embeddings(ast, freq_bins=args.freq_bins, max_frames=args.max_frames)

    model = ASTCTCModel(
        ast=ast,
        vocab_size=vocab_size,
        freq_bins=args.freq_bins,
        max_frames=args.max_frames,
        upsample_factor=args.upsample_factor,
        freq_pool=args.freq_pool,
        ctc_axis=args.ctc_axis,
        token_order=args.token_order,
    ).to(device)

    print(f"[AST-CTC] patch kernel/stride: kernel=({model.k_f},{model.k_t}) stride=({model.s_f},{model.s_t})")
    print(f"[AST-CTC] max_frames={args.max_frames} upsample_factor={model.upsample_factor} "
          f"ctc_axis={args.ctc_axis} token_order={args.token_order} freq_pool={args.freq_pool}")
    print(f"[AST-CTC] lr_head={args.lr_head} lr_enc={args.lr_enc} warmup_head_epochs={args.warmup_head_epochs}")

    # freeze encoder for warm-up
    for p in model.ast.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.lm_head.parameters(), "lr": args.lr_head},
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
    print(f"[AST-CTC] dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    collate_fn = lambda b: collate_ast_ctc(b, max_frames=args.max_frames, ast_feature_extractor=ast_feat_extractor)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_fn)

    best_val_wer = 1e9
    best_path = os.path.join(args.out_dir, "ast_ctc_finetuned.pt")
    global_step = 0
    history = []

    for ep in range(1, args.epochs + 1):
        if ep == args.warmup_head_epochs + 1:
            print(f"[AST-CTC] Unfreezing AST encoder at epoch {ep}.")
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
            drop_invalid=True,
        )
        val_loss, val_wer, val_cer, vrefs, vhyps = evaluate(
            model, processor, val_ld, device, pad_token_id,
            decode=args.decode, beam_size=args.beam_size, beam_topk=args.beam_topk
        )
        dt = time.time() - t0
        print(f"[AST-CTC] Epoch {ep:02d} | train loss {tr_loss:.4f} | "
              f"val loss {val_loss:.4f} | val WER {val_wer*100:.2f}% | CER {val_cer*100:.2f}% | "
              f"{len(train_ds)/max(dt,1e-9):.1f} utts/s | global_step={global_step}")

        for i in range(min(2, len(vrefs))):
            print(f"  [val-{i}] REF: {vrefs[i]}")
            print(f"         HYP: {vhyps[i]}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)
            print(f"[AST-CTC] Saved best ckpt -> {best_path}")

        history.append({
            "epoch": int(ep),
            "global_step": int(global_step),
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_wer": float(val_wer),
            "val_cer": float(val_cer),
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "lr_enc": float(optimizer.param_groups[1]["lr"]),
            "decode": str(args.decode),
            "beam_size": int(args.beam_size),
            "beam_topk": int(args.beam_topk),
        })

    test_loss, test_wer, test_cer, trefs, thyps = evaluate(
        model, processor, test_ld, device, pad_token_id,
        decode=args.decode, beam_size=args.beam_size, beam_topk=args.beam_topk
    )
    print(f"[AST-CTC] Test: loss {test_loss:.4f} | WER {test_wer*100:.2f}% | CER {test_cer*100:.2f}%")
    for i in range(min(3, len(trefs))):
        print(f"  [test-{i}] REF: {trefs[i]}")
        print(f"          HYP: {thyps[i]}")

    hist_path = os.path.join(args.out_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[AST-CTC] Saved training history -> {hist_path}")

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"AST ckpt: {args.ast_ckpt}\n")
        f.write(f"HF cache_dir: {cache_dir}\n")
        f.write(f"offline: {offline}\n")
        f.write(f"kernel=({model.k_f},{model.k_t}) stride=({model.s_f},{model.s_t})\n")
        f.write(f"max_frames={args.max_frames} freq_bins={args.freq_bins} upsample_factor={model.upsample_factor}\n")
        f.write(f"ctc_axis: {args.ctc_axis}\n")
        f.write(f"token_order: {args.token_order}\n")
        f.write(f"freq_pool: {args.freq_pool}\n")
        f.write(f"decode: {args.decode}\n")
        f.write(f"beam_size: {args.beam_size}\n")
        f.write(f"beam_topk: {args.beam_topk}\n")
        f.write(f"lr_head={args.lr_head} lr_enc={args.lr_enc} warmup_head_epochs={args.warmup_head_epochs}\n")
        f.write(f"Train split: {args.train_split}\nVal split: {args.val_split}\nTest split: {args.test_split}\n")
        f.write(f"Max train/val/test utts: {args.max_train_utts}/{args.max_val_utts}/{args.max_test_utts}\n")
        f.write(f"Best val WER={best_val_wer*100:.2f}%\n")
        f.write(f"Test WER={test_wer*100:.2f}% | Test CER={test_cer*100:.2f}%\n")


if __name__ == "__main__":
    main()
