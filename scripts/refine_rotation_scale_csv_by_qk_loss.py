#!/usr/bin/env python3
"""
Refine per-pair rotation angle and residual scale using direct QK-preserving losses.

This script treats the current rotation CSV as the baseline heuristic and searches
small residual updates around it:
  - phi_p  = phi_init_p + dphi_p
  - s_k,p  = gamma_p * s_base_p
  - s_q,p  = 1 / s_k,p

where s_base_p comes from alpha/stat style RPN calibration on pre-RoPE K values.

For each candidate (layer, kv_head, pair) from the input CSV:
  1. Load original final-Q / final-K ACTV dumps
  2. Load pre-RoPE K ROPV to recover the baseline RPN scale for the pair
  3. Apply the pair transform to sampled Q and dense K for that head only
  4. Fake-quantize K with q4_0_head
  5. Evaluate normalized loss terms against original logits/attention
  6. Keep the (gamma, dphi) candidate with the lowest weighted score

This is an offline calibration step; it does not modify the model directly.
Use the companion fuse script to apply the refined per-pair scale+rotation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

MAGIC_ACTV = 0x41435456  # "ACTV"
MAGIC_ROPV = 0x524F5056  # "ROPV"
QK4_0_HEAD = 128


@dataclass
class LayerActivations:
    positions: np.ndarray  # [n_tokens]
    values: np.ndarray     # [n_tokens, n_heads, n_dims]


class ActvReader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")

        magic, version, n_layers, n_heads, n_dims = struct.unpack("IIIII", self.f.read(20))
        if magic != MAGIC_ACTV:
            raise ValueError(f"{path}: invalid ACTV magic {hex(magic)}")

        self.version = version
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_dims = n_dims
        self._layer_idx = 0

    def read_next_layer(self) -> LayerActivations:
        if self._layer_idx >= self.n_layers:
            raise EOFError(f"{self.path}: no more layers")

        sample_count = struct.unpack("I", self.f.read(4))[0]
        positions = np.frombuffer(self.f.read(sample_count * 4), dtype=np.uint32).copy()
        value_count = struct.unpack("I", self.f.read(4))[0]
        values = np.frombuffer(self.f.read(value_count * 4), dtype=np.float32).copy()

        if sample_count > 0:
            expected = sample_count * self.n_heads * self.n_dims
            if value_count != expected:
                raise ValueError(
                    f"{self.path}: layer {self._layer_idx} value_count={value_count}, expected={expected}"
                )
            values = values.reshape(sample_count, self.n_heads, self.n_dims)
        else:
            values = values.reshape(0, self.n_heads, self.n_dims)

        self._layer_idx += 1
        return LayerActivations(positions=positions, values=values)

    def close(self) -> None:
        self.f.close()


@dataclass
class QuerySample:
    q_local: np.ndarray      # [q_per_k, n_dims]
    key_end: int
    sink_mask: np.ndarray    # [key_end]
    orig_logits: np.ndarray  # [q_per_k, key_end]
    orig_attn: np.ndarray    # [q_per_k, key_end]
    orig_margin: np.ndarray  # [q_per_k]


@dataclass
class Metrics:
    logit_mse: float
    sink_logit_mse: float
    attention_kl: float
    margin_mse: float
    top1_match_rate: float
    mean_attention_l1: float


@dataclass
class EvalResult:
    gamma: float
    dphi_deg: float
    alpha_rad: float
    k_scale: float
    q_scale: float
    metrics: Metrics
    score: float


@dataclass
class BaselineContext:
    query_samples: List[QuerySample]
    k_head_orig: np.ndarray
    q_per_k: int
    q_head_indices: List[int]
    pair: int
    phi_init: float
    base_k_scale: float
    base_q_scale: float
    baseline_metrics: Metrics


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("Grid must not be empty")
    return sorted(set(vals))


def load_rotation_csv(path: str):
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            row["_layer"] = int(row["layer"])
            row["_head"] = int(row["head"])
            row["_pair"] = int(row["pair"])
            row["_alpha_rad"] = float(row["suggested_alpha_rad"])
            rows.append(row)
    return fieldnames, rows


def select_evenly_spaced_indices(n: int, max_items: int) -> np.ndarray:
    if n <= max_items:
        return np.arange(n, dtype=np.int64)
    if max_items <= 1:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, n - 1, max_items, dtype=np.int64)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    ez = np.exp(z)
    return ez / np.sum(ez)


def fake_quant_dequant_q4_0_head(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n_tokens, n_dims = x.shape
    if n_dims % QK4_0_HEAD != 0:
        raise ValueError(f"Expected n_dims multiple of {QK4_0_HEAD}, got {n_dims}")

    out = np.empty_like(x, dtype=np.float32)
    for lo in range(0, n_dims, QK4_0_HEAD):
        hi = lo + QK4_0_HEAD
        xb = x[:, lo:hi].astype(np.float32, copy=False)

        arg_absmax = np.abs(xb).argmax(axis=1)
        maxv = xb[np.arange(n_tokens), arg_absmax].astype(np.float32, copy=False)

        d = maxv / -8.0
        inv_d = np.zeros_like(d, dtype=np.float32)
        nz = d != 0
        inv_d[nz] = 1.0 / d[nz]

        scaled = xb * inv_d[:, None]
        q = np.trunc(scaled + 8.5).astype(np.int16)
        q = np.minimum(q, 15)
        q = np.maximum(q, 0)

        out[:, lo:hi] = (q.astype(np.float32) - 8.0) * d[:, None]

    return out


def read_chunk_f32(f, n_floats: int) -> np.ndarray:
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def summarize_pair_l2(l2_per_token: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    l2_max = l2_per_token.max(axis=0)
    if stat_mode == "max":
        return l2_max.astype(np.float32, copy=False)

    q = np.percentile(l2_per_token, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == "percentile":
        return q
    if stat_mode == "blend":
        return (q + tail_lambda * (l2_max - q)).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported l2 stat mode: {stat_mode}")


def compute_pair_stats_from_ropv(path: str, max_tokens: int, stat_mode: str,
                                 percentile: float, tail_lambda: float):
    pair_stats: Dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        stride = n_heads * n_dims
        n_pairs = n_dims // 2

        print(
            f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, "
            f"tokens={n_tokens}, l2_stat={stat_mode}"
        )

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre_tokens = pre_count // stride
                pre_loaded = min(pre_tokens, max_tokens)
                read_floats = pre_loaded * stride
                skip_floats = pre_count - read_floats
                pre = read_chunk_f32(f, read_floats).reshape(pre_loaded, n_heads, n_pairs, 2)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)
                l2_per_token = np.sqrt(np.sum(pre * pre, axis=3, dtype=np.float32), dtype=np.float32)
                pair_stats[layer] = summarize_pair_l2(l2_per_token, stat_mode, percentile, tail_lambda)
            else:
                pair_stats[layer] = np.ones((n_heads, n_pairs), dtype=np.float32)

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                f.seek(post_count * 4, 1)

            if (layer + 1) % 8 == 0:
                print(f"  processed {layer + 1}/{n_layers} layers")

    return pair_stats


def rotate_pair_inplace(x: np.ndarray, d0: int, d1: int, angle_rad: float, scale: float) -> None:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x0 = x[:, d0].copy()
    x1 = x[:, d1].copy()
    x[:, d0] = scale * (c * x0 - s * x1)
    x[:, d1] = scale * (s * x0 + c * x1)


def apply_pair_transform_to_q(q_local: np.ndarray, d0: int, d1: int, angle_rad: float, scale: float) -> np.ndarray:
    out = q_local.copy()
    rotate_pair_inplace(out.reshape(-1, out.shape[-1]), d0, d1, angle_rad, scale)
    return out


def build_query_samples(q_positions: np.ndarray, q_values: np.ndarray, k_positions: np.ndarray, k_values: np.ndarray,
                        max_queries: int, kv_head: int, q_head_indices: List[int]) -> List[QuerySample]:
    q_sel_idx = select_evenly_spaced_indices(q_values.shape[0], max_queries)
    q_sel_pos = q_positions[q_sel_idx]
    q_sel_vals = q_values[q_sel_idx][:, q_head_indices, :].astype(np.float32, copy=False)

    samples: List[QuerySample] = []
    for local_idx, token_pos in enumerate(q_sel_pos):
        key_end = int(np.searchsorted(k_positions, token_pos, side="right"))
        if key_end <= 0:
            continue

        key_pos = k_positions[:key_end].astype(np.int64)
        sink_mask = key_pos < 4
        key_vals = k_values[:key_end, kv_head, :].astype(np.float32, copy=False)

        q_local = q_sel_vals[local_idx]
        orig_logits = np.empty((len(q_head_indices), key_end), dtype=np.float32)
        orig_attn = np.empty((len(q_head_indices), key_end), dtype=np.float32)
        orig_margin = np.empty((len(q_head_indices),), dtype=np.float32)

        for q_local_idx in range(len(q_head_indices)):
            logits = key_vals @ q_local[q_local_idx]
            orig_logits[q_local_idx] = logits
            orig_attn[q_local_idx] = stable_softmax(logits)
            if key_end >= 2:
                top2 = np.partition(logits, -2)[-2:]
                orig_margin[q_local_idx] = float(top2[-1] - top2[-2])
            else:
                orig_margin[q_local_idx] = 0.0

        samples.append(
            QuerySample(
                q_local=q_local,
                key_end=key_end,
                sink_mask=sink_mask,
                orig_logits=orig_logits,
                orig_attn=orig_attn,
                orig_margin=orig_margin,
            )
        )

    return samples


def evaluate_metrics(query_samples: List[QuerySample], q_trial: np.ndarray, k_trial_quant: np.ndarray,
                     kv_head: int) -> Metrics:
    total_sum = 0.0
    total_count = 0
    sink_sum = 0.0
    sink_count = 0
    attn_kl_sum = 0.0
    attn_l1_sum = 0.0
    margin_err_sum = 0.0
    top1_match = 0
    query_head_count = 0

    for sample_idx, sample in enumerate(query_samples):
        k_slice = k_trial_quant[:sample.key_end]
        for q_local_idx in range(q_trial.shape[1]):
            test_logits = k_slice @ q_trial[sample_idx, q_local_idx]
            orig_logits = sample.orig_logits[q_local_idx]
            diff = orig_logits - test_logits
            sq_err = diff * diff
            total_sum += float(sq_err.sum())
            total_count += int(sq_err.size)

            if sample.sink_mask.any():
                sink_sum += float(sq_err[sample.sink_mask].sum())
                sink_count += int(sample.sink_mask.sum())

            orig_top1 = int(np.argmax(orig_logits))
            test_top1 = int(np.argmax(test_logits))
            top1_match += int(orig_top1 == test_top1)

            test_attn = stable_softmax(test_logits)
            orig_attn = sample.orig_attn[q_local_idx]
            attn_kl_sum += float(np.sum(orig_attn * (np.log(orig_attn + 1e-12) - np.log(test_attn + 1e-12))))
            attn_l1_sum += float(np.sum(np.abs(orig_attn - test_attn)))

            if sample.key_end >= 2:
                top2 = np.partition(test_logits, -2)[-2:]
                test_margin = float(top2[-1] - top2[-2])
            else:
                test_margin = 0.0
            margin_err = float(sample.orig_margin[q_local_idx] - test_margin)
            margin_err_sum += margin_err * margin_err

            query_head_count += 1

    if total_count == 0 or query_head_count == 0:
        raise ValueError("No logits were evaluated")

    return Metrics(
        logit_mse=total_sum / total_count,
        sink_logit_mse=sink_sum / sink_count if sink_count > 0 else total_sum / total_count,
        attention_kl=attn_kl_sum / query_head_count,
        margin_mse=margin_err_sum / query_head_count,
        top1_match_rate=top1_match / query_head_count,
        mean_attention_l1=attn_l1_sum / query_head_count,
    )


def normalized_score(metrics: Metrics, base: Metrics, gamma: float, dphi_deg: float,
                     loss_weights: Dict[str, float], lambda_reg: float, gamma_radius: float,
                     phi_radius_deg: float, eps: float) -> float:
    score = 0.0
    score += loss_weights["logit"] * (metrics.logit_mse / max(base.logit_mse, eps))
    score += loss_weights["sink"] * (metrics.sink_logit_mse / max(base.sink_logit_mse, eps))
    score += loss_weights["attn"] * (metrics.attention_kl / max(base.attention_kl, eps))
    score += loss_weights["margin"] * (metrics.margin_mse / max(base.margin_mse, eps))

    if lambda_reg > 0.0:
        reg_gamma = ((gamma - 1.0) / max(gamma_radius, eps)) ** 2
        reg_phi = (dphi_deg / max(phi_radius_deg, eps)) ** 2
        score += lambda_reg * (reg_gamma + reg_phi)

    return float(score)


def evaluate_candidate(context: BaselineContext, gamma: float, dphi_deg: float, kv_head: int) -> EvalResult:
    d0 = 2 * context.pair
    d1 = d0 + 1
    if d1 >= context.k_head_orig.shape[1]:
        raise ValueError(f"Pair {context.pair} out of range for head_dim={context.k_head_orig.shape[1]}")

    alpha = context.phi_init + math.radians(dphi_deg)
    k_scale = gamma * context.base_k_scale
    q_scale = 1.0 / max(k_scale, 1e-12)

    k_trial = context.k_head_orig.copy()
    rotate_pair_inplace(k_trial, d0, d1, alpha, k_scale)
    k_trial_quant = fake_quant_dequant_q4_0_head(k_trial)

    q_orig = np.stack([sample.q_local for sample in context.query_samples], axis=0)
    q_trial = apply_pair_transform_to_q(q_orig, d0, d1, alpha, q_scale)
    metrics = evaluate_metrics(context.query_samples, q_trial, k_trial_quant, kv_head)

    return EvalResult(
        gamma=gamma,
        dphi_deg=dphi_deg,
        alpha_rad=alpha,
        k_scale=k_scale,
        q_scale=q_scale,
        metrics=metrics,
        score=0.0,
    )


def enrich_row(row: Dict[str, str], chosen: EvalResult, base_metrics: Metrics,
               base_k_scale: float, base_q_scale: float) -> Dict[str, str]:
    out = dict(row)
    out["orig_suggested_alpha_rad"] = row["suggested_alpha_rad"]
    out["orig_suggested_alpha_deg"] = row["suggested_alpha_deg"]
    out["suggested_alpha_rad"] = f"{chosen.alpha_rad:.10f}"
    out["suggested_alpha_deg"] = f"{math.degrees(chosen.alpha_rad):.10f}"
    out["residual_gamma"] = f"{chosen.gamma:.10f}"
    out["delta_phi_deg"] = f"{chosen.dphi_deg:.10f}"
    out["delta_phi_rad"] = f"{math.radians(chosen.dphi_deg):.10f}"
    out["base_k_scale"] = f"{base_k_scale:.10f}"
    out["base_q_scale"] = f"{base_q_scale:.10f}"
    out["final_k_scale"] = f"{chosen.k_scale:.10f}"
    out["final_q_scale"] = f"{chosen.q_scale:.10f}"
    out["baseline_logit_mse"] = f"{base_metrics.logit_mse:.10f}"
    out["baseline_sink_logit_mse"] = f"{base_metrics.sink_logit_mse:.10f}"
    out["baseline_attention_kl"] = f"{base_metrics.attention_kl:.10f}"
    out["baseline_margin_mse"] = f"{base_metrics.margin_mse:.10f}"
    out["baseline_top1_match_rate"] = f"{base_metrics.top1_match_rate:.10f}"
    out["baseline_attention_l1"] = f"{base_metrics.mean_attention_l1:.10f}"
    out["best_logit_mse"] = f"{chosen.metrics.logit_mse:.10f}"
    out["best_sink_logit_mse"] = f"{chosen.metrics.sink_logit_mse:.10f}"
    out["best_attention_kl"] = f"{chosen.metrics.attention_kl:.10f}"
    out["best_margin_mse"] = f"{chosen.metrics.margin_mse:.10f}"
    out["best_top1_match_rate"] = f"{chosen.metrics.top1_match_rate:.10f}"
    out["best_attention_l1"] = f"{chosen.metrics.mean_attention_l1:.10f}"
    out["qk_loss_score"] = f"{chosen.score:.10f}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine pair-wise rotation and residual scale using normalized QK-preserving losses"
    )
    parser.add_argument("input_csv", help="Input candidate rotation CSV")
    parser.add_argument("orig_q_actv", help="Original sampled-Q ACTV file")
    parser.add_argument("orig_k_actv", help="Original dense-K ACTV file")
    parser.add_argument("orig_ropv", help="Original ROPV file for base RPN scales")
    parser.add_argument("output_csv", help="Output CSV with refined alpha and scales")
    parser.add_argument("--analysis-csv", help="Optional CSV with all candidate rows and metrics")
    parser.add_argument("--rpn-alpha", type=float, default=8.0, help="Baseline RPN alpha (default: 8.0)")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max ROPV tokens to use for base scales")
    parser.add_argument("--max-queries-per-layer", type=int, default=8,
                        help="Max sampled queries per layer from ACTV to evaluate")
    parser.add_argument("--max-layers", type=int, default=None, help="Optional layer cap")
    parser.add_argument("--gamma-grid", default="0.9,1.0,1.1",
                        help="Comma-separated residual gamma candidates (default: 0.9,1.0,1.1)")
    parser.add_argument("--delta-phi-deg-grid", default="-5,0,5",
                        help="Comma-separated residual delta-phi degrees (default: -5,0,5)")
    parser.add_argument("--l2-stat", choices=["max", "percentile", "blend"], default="max",
                        help="Statistic for base RPN scale from pre-RoPE K (default: max)")
    parser.add_argument("--l2-percentile", type=float, default=99.9,
                        help="Percentile used when --l2-stat is percentile/blend")
    parser.add_argument("--tail-lambda", type=float, default=0.1,
                        help="Blend factor when --l2-stat=blend")
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--lambda-sink", type=float, default=2.0)
    parser.add_argument("--lambda-attn", type=float, default=2.0)
    parser.add_argument("--lambda-margin", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    gamma_grid = parse_float_grid(args.gamma_grid)
    dphi_grid = parse_float_grid(args.delta_phi_deg_grid)
    if 1.0 not in gamma_grid:
        gamma_grid.append(1.0)
        gamma_grid = sorted(set(gamma_grid))
    if 0.0 not in dphi_grid:
        dphi_grid.append(0.0)
        dphi_grid = sorted(set(dphi_grid))

    gamma_radius = max(abs(min(gamma_grid) - 1.0), abs(max(gamma_grid) - 1.0), args.eps)
    phi_radius_deg = max(abs(min(dphi_grid)), abs(max(dphi_grid)), args.eps)

    print(f"gamma_grid={gamma_grid}")
    print(f"delta_phi_deg_grid={dphi_grid}")
    print(
        "loss_weights="
        f"logit:{args.lambda_logit} sink:{args.lambda_sink} attn:{args.lambda_attn} margin:{args.lambda_margin} reg:{args.lambda_reg}"
    )

    fieldnames, rows = load_rotation_csv(args.input_csv)
    print(f"Input CSV: {args.input_csv}")
    print(f"  rows={len(rows)}")

    rows_by_layer_head = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row["_layer"], row["_head"])].append(row)

    pair_stats = compute_pair_stats_from_ropv(
        args.orig_ropv,
        max_tokens=args.max_tokens,
        stat_mode=args.l2_stat,
        percentile=args.l2_percentile,
        tail_lambda=args.tail_lambda,
    )

    q_reader = ActvReader(args.orig_q_actv)
    k_reader = ActvReader(args.orig_k_actv)
    if q_reader.n_layers != k_reader.n_layers:
        raise ValueError("Q/K layer mismatch")
    if q_reader.n_dims != k_reader.n_dims:
        raise ValueError("Q/K dim mismatch")
    if q_reader.n_heads % k_reader.n_heads != 0:
        raise ValueError(
            f"Q/K head mismatch not divisible: q_heads={q_reader.n_heads}, k_heads={k_reader.n_heads}"
        )

    n_layers = q_reader.n_layers
    if args.max_layers is not None:
        n_layers = min(n_layers, args.max_layers)
    q_per_k = q_reader.n_heads // k_reader.n_heads

    loss_weights = {
        "logit": args.lambda_logit,
        "sink": args.lambda_sink,
        "attn": args.lambda_attn,
        "margin": args.lambda_margin,
    }

    all_enriched: List[Dict[str, str]] = []
    cache_by_layer_head: Dict[Tuple[int, int], BaselineContext] = {}

    for layer in range(n_layers):
        q_layer = q_reader.read_next_layer()
        k_layer = k_reader.read_next_layer()

        layer_keys = [key for key in rows_by_layer_head.keys() if key[0] == layer]
        if not layer_keys:
            continue

        print(f"Layer {layer}: evaluating {len(layer_keys)} kv-head groups")
        for _, kv_head in sorted(layer_keys):
            head_rows = rows_by_layer_head[(layer, kv_head)]
            q_head_indices = [kv_head * q_per_k + g for g in range(q_per_k)]
            query_samples = build_query_samples(
                q_positions=q_layer.positions,
                q_values=q_layer.values,
                k_positions=k_layer.positions,
                k_values=k_layer.values,
                max_queries=args.max_queries_per_layer,
                kv_head=kv_head,
                q_head_indices=q_head_indices,
            )
            if not query_samples:
                print(f"  L{layer} H{kv_head}: no usable query samples, skipping")
                continue

            k_head_orig = k_layer.values[:, kv_head, :].astype(np.float32, copy=False)
            for row in head_rows:
                pair = row["_pair"]
                phi_init = row["_alpha_rad"]
                stat = float(pair_stats[layer][kv_head, pair])
                stat = max(stat, args.eps)
                base_k_scale = args.rpn_alpha / stat
                base_q_scale = 1.0 / base_k_scale

                context = BaselineContext(
                    query_samples=query_samples,
                    k_head_orig=k_head_orig,
                    q_per_k=q_per_k,
                    q_head_indices=q_head_indices,
                    pair=pair,
                    phi_init=phi_init,
                    base_k_scale=base_k_scale,
                    base_q_scale=base_q_scale,
                    baseline_metrics=Metrics(0, 0, 0, 0, 0, 0),
                )

                baseline = evaluate_candidate(context, gamma=1.0, dphi_deg=0.0, kv_head=kv_head)
                baseline.metrics  # keep for clarity
                context.baseline_metrics = baseline.metrics

                best = EvalResult(
                    gamma=baseline.gamma,
                    dphi_deg=baseline.dphi_deg,
                    alpha_rad=baseline.alpha_rad,
                    k_scale=baseline.k_scale,
                    q_scale=baseline.q_scale,
                    metrics=baseline.metrics,
                    score=normalized_score(
                        baseline.metrics,
                        baseline.metrics,
                        baseline.gamma,
                        baseline.dphi_deg,
                        loss_weights,
                        args.lambda_reg,
                        gamma_radius,
                        phi_radius_deg,
                        args.eps,
                    ),
                )

                for gamma in gamma_grid:
                    for dphi_deg in dphi_grid:
                        if gamma == 1.0 and dphi_deg == 0.0:
                            continue
                        cand = evaluate_candidate(context, gamma=gamma, dphi_deg=dphi_deg, kv_head=kv_head)
                        cand.score = normalized_score(
                            cand.metrics,
                            baseline.metrics,
                            cand.gamma,
                            cand.dphi_deg,
                            loss_weights,
                            args.lambda_reg,
                            gamma_radius,
                            phi_radius_deg,
                            args.eps,
                        )
                        if cand.score < best.score - 1e-12:
                            best = cand

                all_enriched.append(enrich_row(row, best, baseline.metrics, base_k_scale, base_q_scale))

    q_reader.close()
    k_reader.close()

    output_fieldnames = list(fieldnames) + [
        "orig_suggested_alpha_rad",
        "orig_suggested_alpha_deg",
        "residual_gamma",
        "delta_phi_deg",
        "delta_phi_rad",
        "base_k_scale",
        "base_q_scale",
        "final_k_scale",
        "final_q_scale",
        "baseline_logit_mse",
        "baseline_sink_logit_mse",
        "baseline_attention_kl",
        "baseline_margin_mse",
        "baseline_top1_match_rate",
        "baseline_attention_l1",
        "best_logit_mse",
        "best_sink_logit_mse",
        "best_attention_kl",
        "best_margin_mse",
        "best_top1_match_rate",
        "best_attention_l1",
        "qk_loss_score",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in all_enriched:
            writer.writerow({k: row.get(k, "") for k in output_fieldnames})
    print(f"Saved refined CSV: {args.output_csv}")

    if args.analysis_csv:
        with open(args.analysis_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in all_enriched:
                writer.writerow({k: row.get(k, "") for k in output_fieldnames})
        print(f"Saved analysis CSV: {args.analysis_csv}")

    if all_enriched:
        gammas = np.array([float(r["residual_gamma"]) for r in all_enriched], dtype=np.float64)
        dphis = np.array([float(r["delta_phi_deg"]) for r in all_enriched], dtype=np.float64)
        scores = np.array([float(r["qk_loss_score"]) for r in all_enriched], dtype=np.float64)
        improved_top1 = sum(
            float(r["best_top1_match_rate"]) > float(r["baseline_top1_match_rate"]) + 1e-12
            for r in all_enriched
        )
        print(
            f"Residual gamma stats: min={gammas.min():.6f} mean={gammas.mean():.6f} "
            f"median={np.median(gammas):.6f} max={gammas.max():.6f}"
        )
        print(
            f"Delta-phi stats (deg): min={dphis.min():.6f} mean={dphis.mean():.6f} "
            f"median={np.median(dphis):.6f} max={dphis.max():.6f}"
        )
        print(
            f"Score stats: min={scores.min():.6f} mean={scores.mean():.6f} "
            f"median={np.median(scores):.6f} max={scores.max():.6f}"
        )
        print(f"Rows with improved top1 match vs baseline: {improved_top1} / {len(all_enriched)}")


if __name__ == "__main__":
    main()
