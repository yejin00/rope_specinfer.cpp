#!/usr/bin/env python3
"""
fuse_rpn_alpha.py - Fuse RPN (pre-RoPE Per-pair Normalization) into GGUF Wq/Wk weights.

RPN normalizes the L2 norm of each RoPE pair to a target alpha, ensuring uniform
per-pair activation magnitude. This is equivalent to the pre-RoPE RPN in fireq.

For each layer l, KV head h, pair p (llama.cpp non-interleaved: dims 2p, 2p+1):
  stat[l][h][p] = representative_over_tokens( sqrt(K[t,h,2p]^2 + K[t,h,2p+1]^2) )
  Wk[h*hd + 2p, :]   *= alpha / stat    (suppress large pairs)
  Wk[h*hd + 2p+1, :] *= alpha / stat
  Wq (for all Q heads in GQA group):
  Wq[qh*hd + 2p, :]   *= stat / alpha   (compensate)
  Wq[qh*hd + 2p+1, :] *= stat / alpha

Q·K product is preserved:  (Q * stat/alpha) · (K * alpha/stat) = Q·K

The representative statistic can be:
  - max         : original fireq-style max L2 over tokens
  - percentile  : robust tail statistic such as q99.9
  - blend       : q + lambda * (max - q)

NOTE: Rotation is orthogonal, so it preserves L2 norms per pair.
      L2 norms from the original (non-rotated) model can be used even after rotation.

Usage:
    # Original max-based RPN from ROPV full dump:
    python3 fuse_rpn_alpha.py <gguf_in> <ropv_dump.bin> --alpha 8.0 -o <gguf_out>

    # Use different alphas for early/late dimensions (e.g. dims 0-63 vs 64-127):
    python3 fuse_rpn_alpha.py <gguf_in> <ropv_dump.bin> \
        --alpha-early 4.0 --alpha-late 2.0 --alpha-split-dim 64 -o <gguf_out>

    # Percentile-RPN from ROPV full dump:
    python3 fuse_rpn_alpha.py <gguf_in> <ropv_dump.bin> --alpha 8.0 \
        --l2-stat percentile --l2-percentile 99.9 -o <gguf_out>

    # Tail-aware blended RPN from ROPV full dump:
    python3 fuse_rpn_alpha.py <gguf_in> <ropv_dump.bin> --alpha 8.0 \
        --l2-stat blend --l2-percentile 99.9 --tail-lambda 0.1 -o <gguf_out>

    # From APOR absmax file (approximate, upper bound; max only):
    python3 fuse_rpn_alpha.py <gguf_in> <apor_absmax.bin> --alpha 8.0 -o <gguf_out> --use-absmax
"""

import argparse
import csv
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    print("Error: gguf library not found. pip install gguf")
    sys.exit(1)


MAGIC_ROPV = 0x524F5056  # "ROPV"
MAGIC_APOR = 0x524F5041  # "APOR"


# ── L2/stat computation from ROPV ────────────────────────────────────

def summarize_pair_l2(l2_per_token, stat_mode, percentile, tail_lambda):
    l2_max = l2_per_token.max(axis=0)

    if stat_mode == "max":
        return l2_max.astype(np.float32, copy=False)

    q = np.percentile(l2_per_token, percentile, axis=0).astype(np.float32, copy=False)

    if stat_mode == "percentile":
        return q

    if stat_mode == "blend":
        return (q + tail_lambda * (l2_max - q)).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported l2 stat mode: {stat_mode}")


def compute_l2_from_ropv(ropv_path, stat_mode="max", percentile=99.9, tail_lambda=0.1):
    """
    Read ROPV full dump and compute per-pair per-head representative statistics.

    Returns:
        l2_norms: dict {layer_idx: np.ndarray shape (n_heads, n_pairs)}
        n_heads, n_dims from header
    """
    l2_norms = {}
    with open(ropv_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected ROPV (0x{MAGIC_ROPV:08X})")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]

        print(f"  ROPV header: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")

        n_pairs = n_dims // 2

        for layer in range(n_layers):
            # Read pre-RoPE data
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre_data = np.frombuffer(f.read(pre_count * 4), dtype=np.float32).copy()
                n_tok = pre_count // (n_heads * n_dims)
                pre_data = pre_data.reshape(n_tok, n_heads, n_pairs, 2)

                # L2 norm per pair per token: sqrt(real^2 + imag^2)
                l2_per_token = np.sqrt(pre_data[:, :, :, 0] ** 2 + pre_data[:, :, :, 1] ** 2)
                l2_norms[layer] = summarize_pair_l2(
                    l2_per_token=l2_per_token,
                    stat_mode=stat_mode,
                    percentile=percentile,
                    tail_lambda=tail_lambda,
                )
            else:
                l2_norms[layer] = np.ones((n_heads, n_pairs), dtype=np.float32)

            # Read (and skip) post-RoPE data
            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                f.seek(post_count * 4, 1)

            if (layer + 1) % 8 == 0:
                print(f"  Processed {layer + 1}/{n_layers} layers...")

    return l2_norms, n_layers, n_heads, n_dims


def compute_l2_from_apor(apor_path):
    """
    Approximate L2 norms from APOR absmax file.
    L2_approx[h][p] = sqrt(absmax[h][2p]^2 + absmax[h][2p+1]^2)
    This is an UPPER BOUND on the true max L2 norm.

    Returns:
        l2_norms: dict {layer_idx: np.ndarray shape (n_heads, n_pairs)}
        n_heads, n_dims from header
    """
    l2_norms = {}
    with open(apor_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected APOR (0x{MAGIC_APOR:08X})")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]

        print(f"  APOR header: layers={n_layers}, heads={n_heads}, dims={n_dims}")

        n_pairs = n_dims // 2

        for layer in range(n_layers):
            layer_l2 = np.zeros((n_heads, n_pairs), dtype=np.float32)
            for head in range(n_heads):
                pre_absmax = np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy()
                post_absmax = np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy()

                # Use pre-RoPE absmax for L2 approximation
                # Reshape to pairs: (n_pairs, 2)
                pairs = pre_absmax.reshape(n_pairs, 2)
                layer_l2[head] = np.sqrt(pairs[:, 0] ** 2 + pairs[:, 1] ** 2)

            l2_norms[layer] = layer_l2

    return l2_norms, n_layers, n_heads, n_dims


# ── Scale computation ─────────────────────────────────────────────────

def resolve_alpha_per_pair(alpha, alpha_early, alpha_late, alpha_split_dim, n_dims):
    """
    Build a per-pair alpha target vector.

    Modes:
      - global: all pairs use `alpha`
      - split : dims [0, alpha_split_dim) use alpha_early/default,
                dims [alpha_split_dim, n_dims) use alpha_late/default
    """
    n_pairs = n_dims // 2
    use_split = alpha_early is not None or alpha_late is not None or alpha_split_dim is not None

    if not use_split:
        if alpha is None:
            raise ValueError("--alpha is required unless split-band alphas are provided")
        alpha_per_pair = np.full(n_pairs, alpha, dtype=np.float32)
        return alpha_per_pair, {
            "mode": "global",
            "global_alpha": float(alpha),
        }

    if alpha_split_dim is None:
        alpha_split_dim = n_dims // 2

    if alpha_split_dim <= 0 or alpha_split_dim >= n_dims:
        raise ValueError(f"--alpha-split-dim must be in (0, {n_dims}), got {alpha_split_dim}")
    if alpha_split_dim % 2 != 0:
        raise ValueError(f"--alpha-split-dim must be even so it aligns with RoPE pairs, got {alpha_split_dim}")

    early_alpha = alpha_early if alpha_early is not None else alpha
    late_alpha = alpha_late if alpha_late is not None else alpha

    if early_alpha is None or late_alpha is None:
        raise ValueError(
            "Need alpha values for both bands: provide --alpha, or provide both "
            "--alpha-early and --alpha-late"
        )

    split_pair = alpha_split_dim // 2
    alpha_per_pair = np.empty(n_pairs, dtype=np.float32)
    alpha_per_pair[:split_pair] = early_alpha
    alpha_per_pair[split_pair:] = late_alpha

    return alpha_per_pair, {
        "mode": "split",
        "split_dim": int(alpha_split_dim),
        "split_pair": int(split_pair),
        "early_alpha": float(early_alpha),
        "late_alpha": float(late_alpha),
    }


def compute_rpn_scales(l2_norms, alpha_per_pair, n_heads, n_dims, eps=1e-6):
    """
    Compute per-layer RPN scale vectors.

    Returns:
        scales: dict {layer_idx: {
            'k': np.ndarray shape (n_heads * n_dims,),  -- multiply K rows
            'q_per_kv': np.ndarray shape (n_heads, n_dims)  -- per KV head scale for Q
        }}
    """
    n_pairs = n_dims // 2
    alpha_per_pair = np.asarray(alpha_per_pair, dtype=np.float32)
    if alpha_per_pair.shape != (n_pairs,):
        raise ValueError(f"alpha_per_pair must have shape ({n_pairs},), got {alpha_per_pair.shape}")

    alpha_row = alpha_per_pair[np.newaxis, :]
    scales = {}

    for layer, l2 in l2_norms.items():
        # Clamp to avoid division by zero
        l2_clamped = np.maximum(l2, eps)  # (n_heads, n_pairs)

        # K scale: alpha / L2 (suppress large pairs)
        k_scale_per_pair = alpha_row / l2_clamped  # (n_heads, n_pairs)
        # Expand pairs to full dims: each pair's scale applies to both dims
        k_scale_full = np.repeat(k_scale_per_pair, 2, axis=1)  # (n_heads, n_dims)

        # Q scale: L2 / alpha (compensate)
        q_scale_per_pair = l2_clamped / alpha_row  # (n_heads, n_pairs)
        q_scale_full = np.repeat(q_scale_per_pair, 2, axis=1)  # (n_heads, n_dims)

        scales[layer] = {
            "k": k_scale_full.reshape(-1),          # (n_kv_heads * n_dims,)
            "q_per_kv": q_scale_full,               # (n_kv_heads, n_dims) — will be expanded for GQA
        }

    return scales


def load_residual_gamma_csv(csv_path):
    gamma_map = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])
            pair = int(row["pair"])
            gamma = float(row.get("residual_gamma", "1.0") or 1.0)
            gamma_map[(layer, head, pair)] = gamma
    return gamma_map


def apply_residual_gamma_to_scales(scales, gamma_map, n_dims):
    if not gamma_map:
        return 0

    applied = 0
    for (layer, head, pair), gamma in gamma_map.items():
        if layer not in scales:
            continue
        q_per_kv = scales[layer]["q_per_kv"]
        k_full = scales[layer]["k"].reshape(q_per_kv.shape[0], n_dims)
        if not (0 <= head < q_per_kv.shape[0]):
            continue
        d0 = 2 * pair
        d1 = d0 + 1
        if d1 >= n_dims:
            continue

        k_full[head, d0:d1 + 1] *= gamma
        q_per_kv[head, d0:d1 + 1] /= gamma
        applied += 1

    return applied


# ── Weight fusion ─────────────────────────────────────────────────────

def apply_rpn_to_weight(weight_f32, layer_idx, scales, head_dim, n_kv_heads, wtype):
    """
    Apply RPN scaling to weight matrix rows.

    weight_f32: shape [out_features, in_features]
    """
    if layer_idx not in scales:
        return False

    layer_scales = scales[layer_idx]

    if wtype == "k":
        # K weight shape: (n_kv_heads * head_dim, hidden_size)
        k_vec = layer_scales["k"]  # (n_kv_heads * head_dim,)
        assert k_vec.shape[0] == weight_f32.shape[0], \
            f"K scale size {k_vec.shape[0]} != weight rows {weight_f32.shape[0]}"
        weight_f32 *= k_vec[:, np.newaxis]

    elif wtype == "q":
        # Q weight shape: (n_q_heads * head_dim, hidden_size)
        n_q_heads = weight_f32.shape[0] // head_dim
        n_group = n_q_heads // n_kv_heads
        q_per_kv = layer_scales["q_per_kv"]  # (n_kv_heads, head_dim)

        # Expand KV head scales to Q heads via GQA group repetition
        # KV head 0 → Q heads 0..n_group-1, KV head 1 → Q heads n_group..2*n_group-1, ...
        q_scale_full = np.repeat(q_per_kv, n_group, axis=0)  # (n_q_heads, head_dim)
        q_vec = q_scale_full.reshape(-1)  # (n_q_heads * head_dim,)

        assert q_vec.shape[0] == weight_f32.shape[0], \
            f"Q scale size {q_vec.shape[0]} != weight rows {weight_f32.shape[0]}"
        weight_f32 *= q_vec[:, np.newaxis]

    return True


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuse RPN (per-pair L2 normalization) into GGUF Wq/Wk weights"
    )
    parser.add_argument("input_gguf", help="Input F16 GGUF file")
    parser.add_argument("activation_dump", help="ROPV full dump or APOR absmax file")
    parser.add_argument("--alpha", type=float,
                        help="Default target L2 norm per pair (used globally unless band-specific alphas override it)")
    parser.add_argument("--alpha-early", type=float,
                        help="Target alpha for dims [0, alpha_split_dim)")
    parser.add_argument("--alpha-late", type=float,
                        help="Target alpha for dims [alpha_split_dim, head_dim)")
    parser.add_argument("--alpha-split-dim", type=int,
                        help="Dimension boundary for early/late alpha split (default: n_dims/2)")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF file")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--use-absmax", action="store_true",
                        help="Use APOR absmax file instead of ROPV full dump (approximate)")
    parser.add_argument("--l2-stat", choices=["max", "percentile", "blend"], default="max",
                        help="Statistic to summarize per-token pair L2 from ROPV (default: max)")
    parser.add_argument("--l2-percentile", type=float, default=99.9,
                        help="Percentile used when --l2-stat is percentile/blend (default: 99.9)")
    parser.add_argument("--tail-lambda", type=float, default=0.1,
                        help="Blend factor for --l2-stat blend: q + lambda*(max-q) (default: 0.1)")
    parser.add_argument("--q-only", action="store_true", help="Only fuse into Wq (skip Wk)")
    parser.add_argument("--k-only", action="store_true", help="Only fuse into Wk (skip Wq)")
    parser.add_argument("--gamma-csv", help="Optional CSV with per-pair residual_gamma to multiply on top of base RPN scale")
    args = parser.parse_args()

    print("=" * 60)
    print("GGUF RPN (Per-Pair L2 Normalization) Fusion")
    print("=" * 60)
    if args.alpha is not None:
        print(f"  alpha = {args.alpha}")
    if args.alpha_early is not None:
        print(f"  alpha_early = {args.alpha_early}")
    if args.alpha_late is not None:
        print(f"  alpha_late = {args.alpha_late}")
    if args.alpha_split_dim is not None:
        print(f"  alpha_split_dim = {args.alpha_split_dim}")
    print(f"  head_dim = {args.head_dim}")
    print(f"  l2_stat = {args.l2_stat}")
    if args.l2_stat != "max":
        print(f"  l2_percentile = {args.l2_percentile}")
    if args.l2_stat == "blend":
        print(f"  tail_lambda = {args.tail_lambda}")

    # 1. Load L2 norms
    print(f"\n[1/4] Loading activation data: {args.activation_dump}")
    if args.use_absmax:
        if args.l2_stat != "max":
            raise ValueError("--use-absmax supports only --l2-stat max")
        print("  Mode: APOR absmax (approximate L2 — upper bound)")
        l2_norms, n_layers, n_heads, n_dims = compute_l2_from_apor(args.activation_dump)
    else:
        print("  Mode: ROPV full dump")
        l2_norms, n_layers, n_heads, n_dims = compute_l2_from_ropv(
            args.activation_dump,
            stat_mode=args.l2_stat,
            percentile=args.l2_percentile,
            tail_lambda=args.tail_lambda,
        )

    # Print statistics
    all_l2 = np.concatenate([v.flatten() for v in l2_norms.values()])
    alpha_per_pair, alpha_cfg = resolve_alpha_per_pair(
        alpha=args.alpha,
        alpha_early=args.alpha_early,
        alpha_late=args.alpha_late,
        alpha_split_dim=args.alpha_split_dim,
        n_dims=n_dims,
    )
    print(f"\n  L2 statistics across all layers/heads/pairs:")
    print(f"    min={all_l2.min():.4f}  mean={all_l2.mean():.4f}  "
          f"median={np.median(all_l2):.4f}  max={all_l2.max():.4f}")
    if alpha_cfg["mode"] == "global":
        print(f"    Target alpha={alpha_cfg['global_alpha']} → scale range: "
              f"[{alpha_per_pair.min() / all_l2.max():.4f}, {alpha_per_pair.max() / max(all_l2.min(), 1e-6):.4f}]")
    else:
        print(
            f"    Alpha bands: dims [0,{alpha_cfg['split_dim']}) → {alpha_cfg['early_alpha']}, "
            f"dims [{alpha_cfg['split_dim']},{n_dims}) → {alpha_cfg['late_alpha']}"
        )
        print(f"    Combined scale range: "
              f"[{alpha_per_pair.min() / all_l2.max():.4f}, {alpha_per_pair.max() / max(all_l2.min(), 1e-6):.4f}]")

    # Sample layer 0
    if 0 in l2_norms:
        l0 = l2_norms[0]
        print(f"\n  Layer 0 sample (head 0, first 8 pairs):")
        print(f"    Stat:  {l0[0, :8]}")
        print(f"    Alpha: {alpha_per_pair[:8]}")
        print(f"    K_scl: {alpha_per_pair[:8] / l0[0, :8]}")

    # 2. Compute scales
    if alpha_cfg["mode"] == "global":
        print(f"\n[2/4] Computing RPN scales (alpha={alpha_cfg['global_alpha']}) ...")
    else:
        print(
            f"\n[2/4] Computing RPN scales "
            f"(dims [0,{alpha_cfg['split_dim']})={alpha_cfg['early_alpha']}, "
            f"dims [{alpha_cfg['split_dim']},{n_dims})={alpha_cfg['late_alpha']}) ..."
        )
    scales = compute_rpn_scales(l2_norms, alpha_per_pair, n_heads, n_dims)
    print(f"  Computed scales for {len(scales)} layers")

    gamma_applied = 0
    if args.gamma_csv:
        print(f"\n  Loading residual gamma CSV: {args.gamma_csv}")
        gamma_map = load_residual_gamma_csv(args.gamma_csv)
        gamma_applied = apply_residual_gamma_to_scales(scales, gamma_map, n_dims)
        print(f"  Applied residual gamma to {gamma_applied} (layer, head, pair) entries")

    # 3. Copy GGUF
    print(f"\n[3/4] Copying {args.input_gguf} → {args.output} ...")
    shutil.copy2(args.input_gguf, args.output)
    sz_gb = Path(args.input_gguf).stat().st_size / (1024 ** 3)
    print(f"  Copied {sz_gb:.2f} GB")

    # 4. Read tensor info and apply
    print(f"\n[4/4] Applying RPN to weights ...")
    reader = GGUFReader(args.input_gguf)

    tensors_to_modify = []
    for tensor in reader.tensors:
        is_q = ".attn_q.weight" in tensor.name
        is_k = ".attn_k.weight" in tensor.name

        if args.q_only and is_k:
            continue
        if args.k_only and is_q:
            continue

        if is_q or is_k:
            parts = tensor.name.split(".")
            if len(parts) > 1 and parts[0] == "blk":
                layer_idx = int(parts[1])
                wtype = "q" if is_q else "k"
                tensors_to_modify.append({
                    "name": tensor.name,
                    "layer": layer_idx,
                    "type": wtype,
                    "offset": tensor.data_offset,
                    "shape": tensor.shape,
                    "original_data": tensor.data,
                })

    print(f"  Found {len(tensors_to_modify)} tensors to modify")

    # Infer n_kv_heads from K tensor
    k_tensors = [t for t in tensors_to_modify if t["type"] == "k"]
    if k_tensors:
        n_kv_heads = k_tensors[0]["original_data"].shape[0] // args.head_dim
    else:
        n_kv_heads = n_heads  # fallback to dump header
    print(f"  n_kv_heads={n_kv_heads}, head_dim={args.head_dim}")

    # Verify consistency
    if n_kv_heads != n_heads:
        print(f"  WARNING: GGUF n_kv_heads={n_kv_heads} != dump n_heads={n_heads}")
        print(f"  This may indicate a mismatch between model and activation dump.")

    # Apply
    total_modified = 0
    with open(args.output, "r+b") as f:
        for i, info in enumerate(tensors_to_modify):
            weight = info["original_data"].astype(np.float32)

            ok = apply_rpn_to_weight(
                weight, info["layer"], scales, args.head_dim, n_kv_heads, info["type"]
            )

            if ok:
                weight_f16 = weight.astype(np.float16)
                f.seek(info["offset"])
                f.write(weight_f16.tobytes())
                total_modified += 1

            if (i + 1) % 10 == 0 or i == len(tensors_to_modify) - 1:
                print(f"  {i + 1}/{len(tensors_to_modify)} tensors processed")

    print(f"\n{'=' * 60}")
    print(f"  RPN fusion complete: {total_modified} tensors modified")
    if alpha_cfg["mode"] == "global":
        print(f"  alpha = {alpha_cfg['global_alpha']}")
    else:
        print(
            f"  alpha bands = dims [0,{alpha_cfg['split_dim']}) -> {alpha_cfg['early_alpha']}, "
            f"dims [{alpha_cfg['split_dim']},{n_dims}) -> {alpha_cfg['late_alpha']}"
        )
    if args.gamma_csv:
        print(f"  residual_gamma entries applied = {gamma_applied}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")

    print(f"\nNext steps:")
    out_q4 = args.output.replace(".gguf", "_q4_0.gguf")
    print(f"  # Quantize to Q4_0:")
    print(f"  ./build/bin/llama-quantize {args.output} {out_q4} q4_0")
    print(f"")
    print(f"  # Or run perplexity directly on F16:")
    print(f"  ./build/bin/llama-perplexity -m {args.output} -f wikitext-2-raw/wiki.test.raw")


if __name__ == "__main__":
    main()
