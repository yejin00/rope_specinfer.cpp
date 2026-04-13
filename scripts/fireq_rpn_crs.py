#!/usr/bin/env python3
"""
fireq_rpn_crs.py - Full fireq-style pipeline:
  1. Post-RoPE absmax → identify top-k outlier PAIRS per head per layer
  2. Offline RPN:  fuse alpha/L2 into Wq/Wk for NON-outlier pairs
  3. Online CRS:   generate SCRS binary for outlier dims in those pairs
                   (runtime post-RoPE beta scaling)

Pipeline:
  python3 scripts/fireq_rpn_crs.py \\
      <gguf_in> \\
      --ropv <ropv_dump.bin> \\
      --absmax <apor_absmax.bin> \\
      --alpha 8.0 --beta 8.0 --top-k 8 \\
      -o <gguf_out> --crs-out <crs_scales.bin>

Then run:
  LLAMA_CRS_LATE_SCALES=<crs_scales.bin> ./build/bin/llama-perplexity -m <gguf_out> ...
"""

import argparse
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
MAGIC_SCRS = 0x53435253  # "SCRS"


# ── 1. Read post-RoPE absmax → find outlier pairs ───────────────────

def read_apor_absmax(apor_path):
    """Read APOR file and return both pre/post absmax per layer/head/dim."""
    with open(apor_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]

        print(f"  APOR: layers={n_layers}, heads={n_heads}, dims={n_dims}")

        # post_absmax[layer][head] = np.ndarray(n_dims,)
        post_absmax = []
        pre_absmax = []
        for layer in range(n_layers):
            layer_pre = []
            layer_post = []
            for head in range(n_heads):
                pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy()
                post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy()
                layer_pre.append(pre)
                layer_post.append(post)
            pre_absmax.append(layer_pre)
            post_absmax.append(layer_post)

    return pre_absmax, post_absmax, n_layers, n_heads, n_dims


def choose_calibration_absmax(pre_absmax, post_absmax):
    """
    Prefer the post-RoPE slot from APOR, but fall back to the pre slot when the
    post slot is empty. Some locally generated APOR files store only one side.
    """
    calib_absmax = []
    n_fallback = 0

    for layer in range(len(post_absmax)):
        layer_abs = []
        for head in range(len(post_absmax[layer])):
            post = post_absmax[layer][head]
            pre = pre_absmax[layer][head]

            if np.any(post != 0):
                layer_abs.append(post)
            else:
                layer_abs.append(pre)
                if np.any(pre != 0):
                    n_fallback += 1
        calib_absmax.append(layer_abs)

    if n_fallback > 0:
        print(f"  Note: {n_fallback} layer/head entries had empty post-RoPE slots; "
              f"used pre slot as fallback")

    return calib_absmax


def find_outlier_pairs_and_dims(calib_absmax, n_layers, n_heads, n_dims, top_k, post_dim_start):
    """
    Find late post-RoPE outliers per head per layer.

    Method:
      1. Restrict to late dimensions starting at post_dim_start
      2. Select top-k individual dims by post-RoPE absmax
      3. Expand each selected dim to its adjacent RoPE pair (2p, 2p+1)
      4. Deduplicate with a set

    This mirrors the FireQ Python behavior: top-k is a dim budget, not a
    guaranteed unique-pair budget. If two selected dims belong to the same
    pair, the final number of unique pairs can be less than top-k.

    This repo uses llama.cpp's adjacent non-interleaved pairing:
      pair p -> dims (2p, 2p+1)

    Returns:
        outlier_pairs: dict {(layer, head): set of pair indices}
        outlier_dims:  dict {(layer, head): list of dim indices}
    """
    n_pairs = n_dims // 2
    outlier_pairs = {}
    outlier_dims = {}

    post_pair_start = max(0, post_dim_start // 2)
    late_start_dim = min(n_dims, 2 * post_pair_start)
    late_dims = list(range(late_start_dim, n_dims))

    if not late_dims:
        raise ValueError(f"No late dims available: post_dim_start={post_dim_start}, n_dims={n_dims}")

    for layer in range(n_layers):
        for head in range(n_heads):
            absmax = calib_absmax[layer][head]
            late_absmax = absmax[late_dims]
            actual_top_k = min(top_k, len(late_dims))

            if actual_top_k <= 0:
                outlier_pairs[(layer, head)] = set()
                outlier_dims[(layer, head)] = []
                continue

            top_local = np.argsort(late_absmax)[-actual_top_k:][::-1]
            top_dim_indices = [late_dims[i] for i in top_local]

            pair_set = set()
            dim_set = set()
            for dim_idx in top_dim_indices:
                pair_idx = dim_idx // 2
                pair_set.add(pair_idx)
                dim_set.add(2 * pair_idx)
                dim_set.add(2 * pair_idx + 1)

            outlier_pairs[(layer, head)] = pair_set
            outlier_dims[(layer, head)] = sorted(dim_set)

    return outlier_pairs, outlier_dims


# ── 2. Compute L2 norms from ROPV ───────────────────────────────────

def compute_l2_from_ropv(ropv_path):
    """Read ROPV full dump → max L2 norm per pair per head per layer."""
    l2_norms = {}
    with open(ropv_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]

        print(f"  ROPV: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")
        n_pairs = n_dims // 2

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre_data = np.frombuffer(f.read(pre_count * 4), dtype=np.float32).copy()
                n_tok = pre_count // (n_heads * n_dims)
                pre_data = pre_data.reshape(n_tok, n_heads, n_pairs, 2)
                l2_per_token = np.sqrt(pre_data[:, :, :, 0] ** 2 + pre_data[:, :, :, 1] ** 2)
                l2_norms[layer] = l2_per_token.max(axis=0)  # (n_heads, n_pairs)
            else:
                l2_norms[layer] = np.ones((n_heads, n_pairs), dtype=np.float32)

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                f.seek(post_count * 4, 1)

            if (layer + 1) % 8 == 0:
                print(f"    Processed {layer + 1}/{n_layers} layers...")

    return l2_norms, n_layers, n_heads, n_dims


# ── 3. Offline RPN (excluding outlier pairs) ─────────────────────────

def apply_rpn_to_gguf(output_path, input_path, l2_norms, outlier_pairs,
                       alpha, head_dim, n_kv_heads_ropv):
    """
    Apply offline RPN to Wq/Wk, EXCLUDING outlier pairs.
    Outlier pairs will be handled online by CRS.
    """
    reader = GGUFReader(input_path)

    tensors_to_modify = []
    for tensor in reader.tensors:
        is_q = ".attn_q.weight" in tensor.name
        is_k = ".attn_k.weight" in tensor.name
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
                    "original_data": tensor.data,
                })

    # Infer n_kv_heads from K tensor
    k_tensors = [t for t in tensors_to_modify if t["type"] == "k"]
    n_kv_heads = k_tensors[0]["original_data"].shape[0] // head_dim if k_tensors else n_kv_heads_ropv
    n_pairs = head_dim // 2

    print(f"  n_kv_heads={n_kv_heads}, head_dim={head_dim}, n_pairs={n_pairs}")
    print(f"  Tensors to modify: {len(tensors_to_modify)}")

    total_rpn = 0
    total_skip = 0

    with open(output_path, "r+b") as f:
        for i, info in enumerate(tensors_to_modify):
            weight = info["original_data"].astype(np.float32)
            layer = info["layer"]

            if layer not in l2_norms:
                continue

            l2 = l2_norms[layer]  # (n_kv_heads, n_pairs)
            eps = 1e-6

            if info["type"] == "k":
                for h in range(n_kv_heads):
                    for p in range(n_pairs):
                        if p in outlier_pairs.get((layer, h), set()):
                            total_skip += 1
                            continue  # Skip outlier pairs
                        l2_val = max(l2[h, p], eps)
                        s = alpha / l2_val
                        d0 = h * head_dim + 2 * p
                        d1 = h * head_dim + 2 * p + 1
                        weight[d0, :] *= s
                        weight[d1, :] *= s
                        total_rpn += 1

            elif info["type"] == "q":
                n_q_heads = weight.shape[0] // head_dim
                n_group = n_q_heads // n_kv_heads
                for kv_h in range(n_kv_heads):
                    for p in range(n_pairs):
                        if p in outlier_pairs.get((layer, kv_h), set()):
                            total_skip += 1
                            continue
                        l2_val = max(l2[kv_h, p], eps)
                        s = l2_val / alpha  # inverse for Q
                        for g in range(n_group):
                            q_h = kv_h * n_group + g
                            d0 = q_h * head_dim + 2 * p
                            d1 = q_h * head_dim + 2 * p + 1
                            weight[d0, :] *= s
                            weight[d1, :] *= s
                        total_rpn += 1

            weight_f16 = weight.astype(np.float16)
            f.seek(info["offset"])
            f.write(weight_f16.tobytes())

            if (i + 1) % 10 == 0 or i == len(tensors_to_modify) - 1:
                print(f"    {i + 1}/{len(tensors_to_modify)} tensors processed")

    print(f"  RPN applied: {total_rpn} pairs, skipped (outlier): {total_skip} pairs")


# ── 4. Generate online CRS scales for outlier pairs ──────────────────

def generate_crs_scales(outlier_dims, calib_absmax, beta, n_layers, n_heads, n_dims, output_path):
    """
    Generate SCRS binary for late CRS.

    For each selected post-RoPE outlier dim d:
      K_scale[d] = beta / absmax_post[d]
      Q_scale[d] = absmax_post[d] / beta

    The runtime stores Q scales in SCRS and applies:
      Q *= Q_scale
      K *= 1 / Q_scale

    This matches the basic FireQ late-CRS idea, using per-dim post-RoPE absmax
    while the corresponding pair is excluded from pre-RoPE RPN.
    """
    top_k_dims = max((len(dims) for dims in outlier_dims.values()), default=0)
    eps = 1e-6

    print(f"  Generating SCRS: {n_layers} layers, {n_heads} heads, "
          f"top_k={top_k_dims} dims")

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("I", MAGIC_SCRS))
        f.write(struct.pack("I", 1))  # version
        f.write(struct.pack("I", n_layers))
        f.write(struct.pack("I", n_heads))
        f.write(struct.pack("I", n_dims))
        f.write(struct.pack("I", top_k_dims))

        for layer in range(n_layers):
            for head in range(n_heads):
                dims = list(outlier_dims.get((layer, head), []))

                indices = []
                scales = []
                absmax = calib_absmax[layer][head]

                for dim_idx in dims:
                    absmax_val = max(float(absmax[dim_idx]), eps)
                    q_scale = absmax_val / beta
                    indices.append(dim_idx)
                    scales.append(q_scale)

                # Pad to top_k_dims
                while len(indices) < top_k_dims:
                    indices.append(-1)
                    scales.append(1.0)

                f.write(np.array(indices[:top_k_dims], dtype=np.int32).tobytes())
                f.write(np.array(scales[:top_k_dims], dtype=np.float32).tobytes())

    sz = Path(output_path).stat().st_size
    print(f"  Saved: {output_path} ({sz} bytes)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="fireq-style pipeline: offline RPN (non-outlier pairs) + online CRS (selected outlier dims)"
    )
    parser.add_argument("input_gguf", help="Input F16 GGUF (original or with rotation fused)")
    parser.add_argument("--ropv", required=True, help="ROPV full dump for L2 norms (pre-RoPE, from ORIGINAL model)")
    parser.add_argument("--absmax", required=True, help="APOR absmax file (post-RoPE for outlier selection, from ORIGINAL model)")
    parser.add_argument("--alpha", type=float, default=8.0, help="Target L2 norm for RPN (default: 8.0)")
    parser.add_argument("--beta", type=float, default=8.0, help="Target absmax for post-RoPE CRS (default: 8.0)")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top dims to select (→ up to k pairs as outliers, default: 8)")
    parser.add_argument("--post-dim-start", type=int, default=64,
                        help="Only consider late dims starting from this dimension (default: 64)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF file (offline RPN fused)")
    parser.add_argument("--crs-out", required=True, help="Output SCRS binary for online CRS")

    args = parser.parse_args()

    print("=" * 60)
    print("fireq-style Pipeline: Offline RPN + Online CRS")
    print("=" * 60)
    print(f"  alpha={args.alpha}, beta={args.beta}, top_k_dims={args.top_k}")
    print(f"  post_dim_start={args.post_dim_start}")
    print(f"  head_dim={args.head_dim}")

    # Step 1: Read post-RoPE absmax → find outlier pairs
    print(f"\n[1/5] Reading post-RoPE absmax: {args.absmax}")
    pre_absmax, post_absmax, n_layers_abs, n_heads_abs, n_dims_abs = \
        read_apor_absmax(args.absmax)
    calib_absmax = choose_calibration_absmax(pre_absmax, post_absmax)

    print(f"\n[2/5] Finding late outlier pairs (top-{args.top_k} dims → adjacent pairs) per head...")
    outlier_pairs, outlier_dims = find_outlier_pairs_and_dims(
        calib_absmax, n_layers_abs, n_heads_abs, n_dims_abs, args.top_k, args.post_dim_start
    )

    # Print sample
    sample_key = (0, 0)
    if sample_key in outlier_pairs:
        sample = sorted(outlier_pairs[sample_key])
        sample_dims = outlier_dims.get(sample_key, [])
        print(f"  Layer 0, Head 0: {len(sample)} outlier pairs from top-{args.top_k} late dims")
        print(f"  Pairs: {sample}")
        print(f"  → dims: {sample_dims}")

    # Step 2: Read ROPV → compute L2 norms
    print(f"\n[3/5] Reading ROPV dump for L2 norms: {args.ropv}")
    l2_norms, n_layers_ropv, n_heads_ropv, n_dims_ropv = compute_l2_from_ropv(args.ropv)

    # Verify consistency
    if n_layers_abs != n_layers_ropv:
        print(f"  WARNING: absmax layers={n_layers_abs} != ROPV layers={n_layers_ropv}")
    if n_heads_abs != n_heads_ropv:
        print(f"  WARNING: absmax heads={n_heads_abs} != ROPV heads={n_heads_ropv}")

    # Step 4: Copy GGUF
    print(f"\n[4/6] Copying {args.input_gguf} → {args.output} ...")
    shutil.copy2(args.input_gguf, args.output)
    sz_gb = Path(args.input_gguf).stat().st_size / (1024 ** 3)
    print(f"  Copied {sz_gb:.2f} GB")

    # Step 5: Apply offline RPN (excluding outlier pairs)
    print(f"\n[5/6] Applying offline RPN (alpha={args.alpha}, excluding selected outlier pairs)...")
    apply_rpn_to_gguf(
        args.output, args.input_gguf, l2_norms, outlier_pairs,
        args.alpha, args.head_dim, n_heads_ropv
    )

    # Step 6: Generate online CRS scales for outlier dims
    print(f"\n[6/6] Generating online CRS scales for outlier dims (beta={args.beta})...")
    generate_crs_scales(
        outlier_dims, calib_absmax, args.beta,
        n_layers_ropv, n_heads_ropv, n_dims_ropv,
        args.crs_out
    )

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete!")
    print(f"  Offline RPN GGUF:  {args.output}")
    print(f"  Online CRS scales: {args.crs_out}")
    print(f"{'=' * 60}")

    out_q4 = args.output.replace(".gguf", "_q4_0.gguf")
    print(f"\nNext steps:")
    print(f"  # 1. Quantize:")
    print(f"  ./build/bin/llama-quantize {args.output} {out_q4} q4_0")
    print(f"")
    print(f"  # 2. Run perplexity with online CRS:")
    print(f"  LLAMA_CRS_LATE_SCALES={args.crs_out} \\")
    print(f"    ./build/bin/llama-perplexity -m {out_q4} \\")
    print(f"    -f wikitext-2-raw/wiki.test.raw -ngl 99 -ctk q4_0 -ctv q4_0")


if __name__ == "__main__":
    main()
