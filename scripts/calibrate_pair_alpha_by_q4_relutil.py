#!/usr/bin/env python3
"""
Calibrate per-pair alpha_p on top of a fixed rotation CSV by minimizing a
relative quantization error + level-utilization objective under q4_0_head.

For each (layer, head, pair) row in the input rotation CSV:
  1. Build the full head after applying all fixed pair rotations from the CSV
  2. Apply a base pre-RoPE RPN scale s_base,p = base_alpha / m_p to all pairs
  3. Sweep candidate alpha values only for the target pair
  4. Fake-quant/dequant the full 128-d head with q4_0_head
  5. Score each candidate using:
       rel_mse + center_weight * center_frac - entropy_weight * entropy
  6. Emit residual_gamma = alpha_p / base_alpha so fuse_rpn_alpha.py can use it

This targets the objective the user actually wants:
  - not absolute SSE minimization by shrinking values
  - but better token-wise q4 level utilization with relative error control.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"
QK4_0_HEAD = 128
LOG16 = math.log(16.0)


def parse_csv_grid(text: str) -> np.ndarray:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Alpha grid is empty")
    return np.array(vals, dtype=np.float32)


def load_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            row["_layer"] = int(row["layer"])
            row["_head"] = int(row["head"])
            row["_pair"] = int(row["pair"])
            row["_alpha_rad"] = float(row["suggested_alpha_rad"])
            rows.append(row)
    return fieldnames, rows


def summarize_pair_l2(l2_per_token: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    l2_max = l2_per_token.max(axis=0)
    if stat_mode == "max":
        return l2_max.astype(np.float32, copy=False)

    q = np.percentile(l2_per_token, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == "percentile":
        return q
    if stat_mode == "blend":
        return (q + tail_lambda * (l2_max - q)).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported stat mode: {stat_mode}")


def read_chunk_f32(f, n_floats: int) -> np.ndarray:
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def rotate_pair(post_pair: np.ndarray, alpha_rad: float) -> np.ndarray:
    c = math.cos(alpha_rad)
    s = math.sin(alpha_rad)
    x = post_pair[:, 0]
    y = post_pair[:, 1]
    out = np.empty_like(post_pair, dtype=np.float32)
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def fake_quant_dequant_q4_0_head_with_codes_batch(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    batch, n_tokens, n_dims = x.shape
    if n_dims % QK4_0_HEAD != 0:
        raise ValueError(f"Expected n_dims multiple of {QK4_0_HEAD}, got {n_dims}")

    n_blocks = n_dims // QK4_0_HEAD
    out = np.empty_like(x, dtype=np.float32)
    codes = np.empty_like(x, dtype=np.uint8)

    for bi in range(n_blocks):
        lo = bi * QK4_0_HEAD
        hi = lo + QK4_0_HEAD
        xb = x[:, :, lo:hi].astype(np.float32, copy=False)

        arg_absmax = np.abs(xb).argmax(axis=2)
        maxv = np.take_along_axis(xb, arg_absmax[..., None], axis=2)[..., 0].astype(np.float32, copy=False)

        d = maxv / -8.0
        inv_d = np.zeros_like(d, dtype=np.float32)
        nz = d != 0
        inv_d[nz] = 1.0 / d[nz]

        scaled = xb * inv_d[:, :, None]
        q = np.trunc(scaled + 8.5).astype(np.int16)
        q = np.minimum(q, 15)
        q = np.maximum(q, 0)

        out[:, :, lo:hi] = (q.astype(np.float32) - 8.0) * d[:, :, None]
        codes[:, :, lo:hi] = q.astype(np.uint8)

    return out, codes


def rel_mse_per_token_batch(x: np.ndarray, x_hat: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    xh32 = x_hat.astype(np.float32, copy=False)
    diff = x32 - xh32
    num = np.sum(diff * diff, axis=2, dtype=np.float64)
    den = np.sum(x32 * x32, axis=2, dtype=np.float64) + eps
    return num / den


def center_fraction_per_token_batch(codes: np.ndarray) -> np.ndarray:
    return ((codes == 7) | (codes == 8)).mean(axis=2, dtype=np.float64)


def entropy_per_token_batch(codes: np.ndarray) -> np.ndarray:
    batch, n_tokens, n_dims = codes.shape
    hist = np.zeros((batch, n_tokens, 16), dtype=np.float64)
    for code in range(16):
        hist[:, :, code] = np.sum(codes == code, axis=2, dtype=np.int32)
    probs = hist / float(n_dims)
    ent_terms = np.zeros_like(probs)
    mask = probs > 0
    ent_terms[mask] = probs[mask] * np.log(probs[mask])
    ent = -np.sum(ent_terms, axis=2)
    return ent / LOG16


def score_candidates(
    x: np.ndarray,
    center_weight: float,
    entropy_weight: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_hat, codes = fake_quant_dequant_q4_0_head_with_codes_batch(x)
    rel_mse = rel_mse_per_token_batch(x, x_hat, eps=eps)
    center_frac = center_fraction_per_token_batch(codes)
    entropy = entropy_per_token_batch(codes)
    score = rel_mse + center_weight * center_frac - entropy_weight * entropy
    return score.mean(axis=1), rel_mse.mean(axis=1), center_frac.mean(axis=1), entropy.mean(axis=1)


def build_output_fieldnames(base_fieldnames: Sequence[str]) -> list[str]:
    extra = [
        "base_alpha",
        "pair_alpha",
        "base_pair_stat",
        "base_k_scale",
        "base_q_scale",
        "residual_gamma",
        "final_k_scale",
        "final_q_scale",
        "score_before",
        "score_after",
        "score_delta",
        "relative_mse_before",
        "relative_mse_after",
        "center_frac_before",
        "center_frac_after",
        "entropy_before",
        "entropy_after",
        "alpha_grid",
        "alpha_search_mode",
    ]
    out = list(base_fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def row_allowed(row: dict, layers: set[int] | None, heads: set[int] | None, pairs: set[int] | None) -> bool:
    if layers is not None and row["_layer"] not in layers:
        return False
    if heads is not None and row["_head"] not in heads:
        return False
    if pairs is not None and row["_pair"] not in pairs:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-pair alpha by relative q4 error + code utilization objective"
    )
    parser.add_argument("rotation_csv", help="Rotation CSV with fixed suggested alpha per pair")
    parser.add_argument("ropv_path", help="Input ROPV dump with pre/post activations")
    parser.add_argument("output_csv", help="Output CSV with residual_gamma chosen from pair alpha")
    parser.add_argument("--analysis-csv", help="Optional analysis CSV")
    parser.add_argument("--base-alpha", type=float, default=8.0, help="Base RPN target alpha (default: 8.0)")
    parser.add_argument("--alpha-grid", default="5.6,6.4,7.2,8.0,8.8,9.6",
                        help="Comma-separated candidate pair alpha values")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max tokens per layer to load")
    parser.add_argument("--pair-stat", choices=["max", "percentile", "blend"], default="max",
                        help="Statistic for pre-RoPE pair L2 used in base RPN scale")
    parser.add_argument("--pair-percentile", type=float, default=99.9,
                        help="Percentile used when --pair-stat is percentile/blend")
    parser.add_argument("--pair-tail-lambda", type=float, default=0.1,
                        help="Blend factor when --pair-stat is blend")
    parser.add_argument("--center-weight", type=float, default=0.10,
                        help="Penalty weight for codes collapsing to center bins 7/8")
    parser.add_argument("--entropy-weight", type=float, default=0.05,
                        help="Reward weight for higher normalized code entropy")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Numerical epsilon for relative MSE")
    parser.add_argument("--layers", type=int, nargs="*", help="Optional layer filter")
    parser.add_argument("--heads", type=int, nargs="*", help="Optional head filter")
    parser.add_argument("--pairs", type=int, nargs="*", help="Optional pair filter")
    args = parser.parse_args()

    alpha_grid = parse_csv_grid(args.alpha_grid)
    base_fieldnames, rows = load_rows(args.rotation_csv)

    layers_filter = set(args.layers) if args.layers else None
    heads_filter = set(args.heads) if args.heads else None
    pairs_filter = set(args.pairs) if args.pairs else None

    rows = [r for r in rows if row_allowed(r, layers_filter, heads_filter, pairs_filter)]
    if not rows:
        raise ValueError("No rows left after filters")

    rows_by_layer_head: Dict[tuple[int, int], List[dict]] = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row["_layer"], row["_head"])] .append(row)

    output_fieldnames = build_output_fieldnames(base_fieldnames)
    out_rows: list[dict] = []
    chosen_alpha_counter: Counter = Counter()

    with open(args.ropv_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        n_pairs = n_dims // 2
        stride = n_heads * n_dims

        print(f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")
        print(
            f"  base_alpha={args.base_alpha}, alpha_grid={alpha_grid.tolist()}, "
            f"center_weight={args.center_weight}, entropy_weight={args.entropy_weight}"
        )
        print(f"  filtered rows={len(rows)}")

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre_tokens = pre_count // stride
                pre_loaded = min(pre_tokens, args.max_tokens)
                pre_read = pre_loaded * stride
                pre_skip = pre_count - pre_read
                pre = read_chunk_f32(f, pre_read).reshape(pre_loaded, n_heads, n_pairs, 2)
                if pre_skip > 0:
                    f.seek(pre_skip * 4, 1)
            else:
                pre_loaded = 0
                pre = np.zeros((0, n_heads, n_pairs, 2), dtype=np.float32)

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                post_tokens = post_count // stride
                post_loaded = min(post_tokens, args.max_tokens)
                post_read = post_loaded * stride
                post_skip = post_count - post_read
                post = read_chunk_f32(f, post_read).reshape(post_loaded, n_heads, n_pairs, 2)
                if post_skip > 0:
                    f.seek(post_skip * 4, 1)
            else:
                post_loaded = 0
                post = np.zeros((0, n_heads, n_pairs, 2), dtype=np.float32)

            layer_heads = sorted({head for lyr, head in rows_by_layer_head if lyr == layer})
            if not layer_heads:
                continue
            if pre_loaded == 0 or post_loaded == 0:
                raise ValueError(f"Layer {layer} has no loaded tokens")

            l2_per_token = np.sqrt(np.sum(pre * pre, axis=3, dtype=np.float32), dtype=np.float32)
            pair_stats = summarize_pair_l2(
                l2_per_token,
                stat_mode=args.pair_stat,
                percentile=args.pair_percentile,
                tail_lambda=args.pair_tail_lambda,
            )

            print(f"  Layer {layer}: evaluating heads {layer_heads} with {post_loaded} tokens")

            for head in layer_heads:
                rows_for_head = sorted(rows_by_layer_head[(layer, head)], key=lambda r: r["_pair"])
                post_pairs = post[:, head, :, :].astype(np.float32, copy=True)

                for row in rows_for_head:
                    pair = row["_pair"]
                    post_pairs[:, pair, :] = rotate_pair(post_pairs[:, pair, :], row["_alpha_rad"])

                rotated_head = post_pairs.reshape(post_loaded, n_dims)
                base_pair_stats = np.maximum(pair_stats[head], 1e-8).astype(np.float32, copy=False)
                base_pair_scales = (args.base_alpha / base_pair_stats).astype(np.float32, copy=False)
                base_scale_vec = np.repeat(base_pair_scales, 2, axis=0)
                base_scaled_head = rotated_head * base_scale_vec[None, :]

                base_batch = base_scaled_head[None, :, :]
                base_score, base_rel_mse, base_center, base_entropy = score_candidates(
                    base_batch,
                    center_weight=args.center_weight,
                    entropy_weight=args.entropy_weight,
                    eps=args.eps,
                )
                base_score = float(base_score[0])
                base_rel_mse = float(base_rel_mse[0])
                base_center = float(base_center[0])
                base_entropy = float(base_entropy[0])

                for row in rows_for_head:
                    pair = row["_pair"]
                    d0 = 2 * pair
                    d1 = d0 + 1

                    cand = np.broadcast_to(base_scaled_head, (len(alpha_grid),) + base_scaled_head.shape).copy()
                    alpha_ratios = alpha_grid / args.base_alpha
                    cand[:, :, d0:d1 + 1] *= alpha_ratios[:, None, None]

                    cand_score, cand_rel_mse, cand_center, cand_entropy = score_candidates(
                        cand,
                        center_weight=args.center_weight,
                        entropy_weight=args.entropy_weight,
                        eps=args.eps,
                    )

                    tie_break = cand_score + 1e-12 * np.abs(alpha_grid - args.base_alpha)
                    best_idx = int(np.argmin(tie_break))
                    best_alpha = float(alpha_grid[best_idx])
                    best_gamma = float(alpha_ratios[best_idx])
                    best_score = float(cand_score[best_idx])
                    best_rel_mse = float(cand_rel_mse[best_idx])
                    best_center = float(cand_center[best_idx])
                    best_entropy = float(cand_entropy[best_idx])

                    pair_stat = float(base_pair_stats[pair])
                    base_k_scale = float(base_pair_scales[pair])
                    final_k_scale = float(best_alpha / pair_stat)
                    base_q_scale = float(1.0 / base_k_scale)
                    final_q_scale = float(1.0 / final_k_scale)
                    delta = float(base_score - best_score)

                    out = dict(row)
                    for k in list(out.keys()):
                        if k.startswith("_"):
                            del out[k]
                    out["base_alpha"] = f"{args.base_alpha:.10f}"
                    out["pair_alpha"] = f"{best_alpha:.10f}"
                    out["base_pair_stat"] = f"{pair_stat:.10f}"
                    out["base_k_scale"] = f"{base_k_scale:.10f}"
                    out["base_q_scale"] = f"{base_q_scale:.10f}"
                    out["residual_gamma"] = f"{best_gamma:.10f}"
                    out["final_k_scale"] = f"{final_k_scale:.10f}"
                    out["final_q_scale"] = f"{final_q_scale:.10f}"
                    out["score_before"] = f"{base_score:.10f}"
                    out["score_after"] = f"{best_score:.10f}"
                    out["score_delta"] = f"{delta:.10f}"
                    out["relative_mse_before"] = f"{base_rel_mse:.10f}"
                    out["relative_mse_after"] = f"{best_rel_mse:.10f}"
                    out["center_frac_before"] = f"{base_center:.10f}"
                    out["center_frac_after"] = f"{best_center:.10f}"
                    out["entropy_before"] = f"{base_entropy:.10f}"
                    out["entropy_after"] = f"{best_entropy:.10f}"
                    out["alpha_grid"] = ",".join(f"{float(a):.6f}" for a in alpha_grid)
                    out["alpha_search_mode"] = "pair_rel_mse_plus_level_util"
                    out_rows.append(out)
                    chosen_alpha_counter[round(best_alpha, 4)] += 1

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in output_fieldnames})
    print(f"Saved alpha/gamma CSV: {args.output_csv}")

    if args.analysis_csv:
        with open(args.analysis_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow({k: row.get(k, "") for k in output_fieldnames})
        print(f"Saved analysis CSV: {args.analysis_csv}")

    if chosen_alpha_counter:
        print("Chosen alpha counts:", dict(sorted(chosen_alpha_counter.items())))


if __name__ == "__main__":
    main()
