#!/usr/bin/env python3
"""
Apply post-RoPE pair-norm clipping to selected pairs and write a new ROPV file.

Selection rule:
  choose (layer, head, pair) where max/q99.9 >= ratio_thresh

Clipping rule for selected pairs:
  r_t = sqrt(x_t^2 + y_t^2)
  tau = q99.9
  if r_t > tau:
      (x_t, y_t) <- (tau / r_t) * (x_t, y_t)

This preserves the pair angle and only shrinks excessive magnitudes.
The input file is never modified in-place.
"""

import argparse
import struct

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_chunk(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} floats, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def load_ropv_full(path):
    with open(path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected ROPV")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        stride = n_heads * n_dims

        print(f"Input: {path}")
        print(f"  version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")

        layers = []
        for layer_idx in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre = read_chunk(f, pre_count)
                pre = pre.reshape(pre_count // stride, n_heads, n_dims)
            else:
                pre = None

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                post = read_chunk(f, post_count)
                post = post.reshape(post_count // stride, n_heads, n_dims)
            else:
                post = None

            layers.append(
                {
                    "pre": pre,
                    "post": post,
                    "pre_count": pre_count,
                    "post_count": post_count,
                }
            )

            if layer_idx == 0:
                pre_tok = pre.shape[0] if pre is not None else 0
                post_tok = post.shape[0] if post is not None else 0
                print(f"  Layer 0: pre_count={pre_count} -> {pre_tok} tokens, post_count={post_count} -> {post_tok} tokens")

        return {
            "version": version,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_dims": n_dims,
            "n_tokens": n_tokens,
            "layers": layers,
        }


def analyze_post_pairs(data, pair_start, pair_end, ratio_thresh):
    n_pairs_total = data["n_dims"] // 2
    if pair_end is None:
        pair_end = n_pairs_total
    pair_end = min(pair_end, n_pairs_total)

    selected = {}
    summary_rows = []
    eps = 1e-12

    for layer_idx in range(data["n_layers"]):
        post = data["layers"][layer_idx]["post"]
        if post is None:
            continue

        for head_idx in range(data["n_heads"]):
            for pair_idx in range(pair_start, pair_end):
                d0 = pair_idx * 2
                d1 = d0 + 1
                x = post[:, head_idx, d0].astype(np.float64)
                y = post[:, head_idx, d1].astype(np.float64)
                r = np.sqrt(x * x + y * y)
                if r.size == 0:
                    continue

                tau = float(np.percentile(r, 99.9))
                rmax = float(np.max(r))
                ratio = rmax / max(tau, eps)
                if ratio >= ratio_thresh:
                    key = (layer_idx, head_idx, pair_idx)
                    selected[key] = tau
                    summary_rows.append(
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "pair": pair_idx,
                            "tau": tau,
                            "r_max": rmax,
                            "ratio_max_q99_9": ratio,
                        }
                    )

    summary_rows.sort(key=lambda row: (-row["ratio_max_q99_9"], row["layer"], row["head"], row["pair"]))
    return selected, summary_rows


def apply_post_pair_clip(data, selected_pairs):
    touched_pairs = 0
    touched_values = 0

    for (layer_idx, head_idx, pair_idx), tau in selected_pairs.items():
        post = data["layers"][layer_idx]["post"]
        if post is None:
            continue

        d0 = pair_idx * 2
        d1 = d0 + 1

        x = post[:, head_idx, d0].astype(np.float64)
        y = post[:, head_idx, d1].astype(np.float64)
        r = np.sqrt(x * x + y * y)
        mask = r > tau
        if not np.any(mask):
            continue

        scale = np.ones_like(r, dtype=np.float64)
        scale[mask] = tau / r[mask]

        post[mask, head_idx, d0] = (x[mask] * scale[mask]).astype(np.float32)
        post[mask, head_idx, d1] = (y[mask] * scale[mask]).astype(np.float32)

        touched_pairs += 1
        touched_values += int(np.count_nonzero(mask))

    return touched_pairs, touched_values


def write_ropv(path, data):
    with open(path, "wb") as f:
        f.write(struct.pack("I", MAGIC_ROPV))
        f.write(struct.pack("I", data["version"]))
        f.write(struct.pack("I", data["n_layers"]))
        f.write(struct.pack("I", data["n_heads"]))
        f.write(struct.pack("I", data["n_dims"]))
        f.write(struct.pack("I", data["n_tokens"]))

        for layer in data["layers"]:
            pre = layer["pre"]
            if pre is not None:
                pre_count = pre.size
                f.write(struct.pack("I", pre_count))
                f.write(pre.astype(np.float32).tobytes())
            else:
                f.write(struct.pack("I", 0))

            post = layer["post"]
            if post is not None:
                post_count = post.size
                f.write(struct.pack("I", post_count))
                f.write(post.astype(np.float32).tobytes())
            else:
                f.write(struct.pack("I", 0))


def main():
    parser = argparse.ArgumentParser(description="Apply post-RoPE pair-norm clipping to selected pairs")
    parser.add_argument("input_ropv", help="Input ROPV file")
    parser.add_argument("output_ropv", help="Output ROPV file")
    parser.add_argument("--pair-start", type=int, default=32, help="First pair index to analyze")
    parser.add_argument("--pair-end", type=int, default=None, help="Exclusive end pair index")
    parser.add_argument("--ratio-thresh", type=float, default=1.5, help="Select pairs with max/q99.9 >= this value")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report, but do not write output")
    args = parser.parse_args()

    data = load_ropv_full(args.input_ropv)
    selected_pairs, summary_rows = analyze_post_pairs(
        data=data,
        pair_start=args.pair_start,
        pair_end=args.pair_end,
        ratio_thresh=args.ratio_thresh,
    )

    print(f"\nSelected pairs: {len(selected_pairs)}")
    if summary_rows:
        print("Top selected pairs:")
        for row in summary_rows[:15]:
            print(
                f"  L{row['layer']:02d} H{row['head']:02d} P{row['pair']:02d} | "
                f"ratio={row['ratio_max_q99_9']:.3f} | tau={row['tau']:.5f} | max={row['r_max']:.5f}"
            )

    if args.dry_run:
        print("\nDry run requested. No output file written.")
        return

    touched_pairs, touched_values = apply_post_pair_clip(data, selected_pairs)
    print(f"\nClipped pairs actually modified: {touched_pairs}")
    print(f"Clipped token instances: {touched_values}")

    write_ropv(args.output_ropv, data)
    print(f"\nSaved clipped ROPV: {args.output_ropv}")


if __name__ == "__main__":
    main()
