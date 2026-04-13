#!/usr/bin/env python3
"""
Compare absmax between a reference model and a FireQ-style model,
highlighting only the CRS-selected pairs stored in a late SCRS file.

This is similar in spirit to compare_absmax_rotation_nolabel.py, but the
highlight source is a .scrs file instead of a rotation CSV.
"""

import argparse
import math
import struct
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_ROPV = 0x524F5056  # "ROPV"
MAGIC_SCRS = 0x53435253  # "SCRS"


def load_absmax(file_path):
    """
    Load absmax values from:
    - APOR file: already contains per-dim pre/post absmax
    - ROPV file: compute absmax from full dumps
    """
    with open(file_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]

        if magic == MAGIC_APOR:
            version = struct.unpack("I", f.read(4))[0]
            n_layers = struct.unpack("I", f.read(4))[0]
            n_heads = struct.unpack("I", f.read(4))[0]
            n_dims = struct.unpack("I", f.read(4))[0]

            print(f"  APOR header: layers={n_layers}, heads={n_heads}, dims={n_dims}")

            pre_absmax = []
            post_absmax = []
            for _layer in range(n_layers):
                layer_pre = []
                layer_post = []
                for _head in range(n_heads):
                    layer_pre.append(np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy())
                    layer_post.append(np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy())
                pre_absmax.append(np.array(layer_pre))
                post_absmax.append(np.array(layer_post))

            return {
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_dims": n_dims,
                "pre_absmax": pre_absmax,
                "post_absmax": post_absmax,
            }

        if magic == MAGIC_ROPV:
            version = struct.unpack("I", f.read(4))[0]
            n_layers = struct.unpack("I", f.read(4))[0]
            n_heads = struct.unpack("I", f.read(4))[0]
            n_dims = struct.unpack("I", f.read(4))[0]
            n_tokens = struct.unpack("I", f.read(4))[0]
            stride = n_heads * n_dims

            print(f"  ROPV header: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")

            pre_absmax = []
            post_absmax = []
            for _layer in range(n_layers):
                pre_count = struct.unpack("I", f.read(4))[0]
                if pre_count > 0:
                    pre = np.frombuffer(f.read(pre_count * 4), dtype=np.float32).copy()
                    pre = pre.reshape(pre_count // stride, n_heads, n_dims)
                    layer_pre = np.max(np.abs(pre), axis=0)
                else:
                    layer_pre = np.zeros((n_heads, n_dims), dtype=np.float32)

                post_count = struct.unpack("I", f.read(4))[0]
                if post_count > 0:
                    post = np.frombuffer(f.read(post_count * 4), dtype=np.float32).copy()
                    post = post.reshape(post_count // stride, n_heads, n_dims)
                    layer_post = np.max(np.abs(post), axis=0)
                else:
                    layer_post = np.zeros((n_heads, n_dims), dtype=np.float32)

                pre_absmax.append(layer_pre)
                post_absmax.append(layer_post)

            return {
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_dims": n_dims,
                "pre_absmax": pre_absmax,
                "post_absmax": post_absmax,
            }

        raise ValueError(f"Invalid magic: {hex(magic)}, expected APOR or ROPV")


def load_scrs_pairs(scrs_path):
    """
    Return {(layer, head): [(pair_idx, [(dim_idx, scale), ...]), ...]}
    grouping selected CRS dims by adjacent RoPE pair.
    """
    with open(scrs_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid SCRS magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        top_k = struct.unpack("I", f.read(4))[0]

        print(f"  SCRS header: layers={n_layers}, heads={n_heads}, dims={n_dims}, top_k={top_k}")

        grouped = {}
        for layer in range(n_layers):
            for head in range(n_heads):
                indices = np.frombuffer(f.read(top_k * 4), dtype=np.int32).copy()
                scales = np.frombuffer(f.read(top_k * 4), dtype=np.float32).copy()

                pair_map = defaultdict(list)
                for idx, scale in zip(indices, scales):
                    if idx < 0:
                        continue
                    pair_map[idx // 2].append((int(idx), float(scale)))

                grouped[(layer, head)] = sorted(pair_map.items(), key=lambda x: x[0])

        return grouped


def select_absmax_slot(data, layer_idx, head_idx, use_post_rope, label):
    if use_post_rope:
        vals = data["post_absmax"][layer_idx][head_idx]
        if float(np.max(vals)) == 0.0:
            print(f"  ⚠ {label} post slot is empty, falling back to pre slot")
            vals = data["pre_absmax"][layer_idx][head_idx]
        return vals
    return data["pre_absmax"][layer_idx][head_idx]


def visualize_comparison(orig_absmax, fireq_absmax, crs_pairs, layer_idx, head_idx,
                         use_post_rope, output_path=None):
    n_dims = len(orig_absmax)
    dims = np.arange(n_dims)
    rope_stage = "Post-RoPE" if use_post_rope else "Pre-RoPE"

    cmap = plt.cm.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    base_width = 0.65
    highlight_width = 0.85

    pair_entries = []
    for ci, (pair_idx, dim_entries) in enumerate(crs_pairs):
        color = cmap(ci % 10)
        dim_entries = sorted(dim_entries, key=lambda x: x[0])
        pair_entries.append((pair_idx, dim_entries, color))

    for ax, absmax, title_prefix, base_color in [
        (axes[0], orig_absmax, "Reference Model", "steelblue"),
        (axes[1], fireq_absmax, "FireQ Model", "forestgreen"),
    ]:
        ax.bar(dims, absmax, color=base_color, alpha=0.6, width=base_width, edgecolor="none")

        for pair_idx, dim_entries, color in pair_entries:
            valid_dims = [d for d, _s in dim_entries if d < n_dims]
            if not valid_dims:
                continue

            ax.bar(valid_dims, absmax[valid_dims], color=color, alpha=0.85,
                   width=highlight_width, edgecolor="black", linewidth=0.8)

            if len(valid_dims) >= 2:
                d0 = valid_dims[0]
                d1 = valid_dims[-1]
                y0 = absmax[d0]
                y1 = absmax[d1]
                ax.annotate(
                    "",
                    xy=(d1, y1),
                    xytext=(d0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
                )

        ax.axvspan(0, 32, alpha=0.08, color="red", label="Block 0 (0-31)")
        ax.axvspan(32, 64, alpha=0.08, color="orange", label="Block 1 (32-63)")
        ax.axvspan(64, 96, alpha=0.08, color="green", label="Block 2 (64-95)")
        ax.axvspan(96, 128, alpha=0.08, color="blue", label="Block 3 (96-127)")

        ax.set_title(f"{title_prefix}\nLayer {layer_idx}, Head {head_idx} ({rope_stage})",
                     fontsize=12, weight="bold")
        ax.set_xlim([-2, n_dims + 2])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    y_max = max(float(np.max(orig_absmax)), float(np.max(fireq_absmax))) * 1.2
    axes[0].set_ylim([0, y_max])
    axes[1].set_ylim([0, y_max])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\n✓ Saved: {output_path}")
    else:
        plt.show()

    print(f"\n{'=' * 80}")
    print(f"FireQ CRS Comparison: Layer {layer_idx}, Head {head_idx} ({rope_stage})")
    print(f"{'=' * 80}")
    print(f"{'Pair':<6} {'Dims':<14} {'Scales':<22} {'Ref vals':<18} {'FireQ vals':<18}")
    print("-" * 80)

    for pair_idx, dim_entries, _color in pair_entries:
        dims_str = ",".join(str(d) for d, _s in dim_entries)
        scales_str = ",".join(f"{s:.3f}" for _d, s in dim_entries)
        ref_vals = ",".join(f"{orig_absmax[d]:.3f}" for d, _s in dim_entries if d < n_dims)
        fireq_vals = ",".join(f"{fireq_absmax[d]:.3f}" for d, _s in dim_entries if d < n_dims)
        print(f"p{pair_idx:<5} {dims_str:<14} {scales_str:<22} {ref_vals:<18} {fireq_vals:<18}")

    print(f"\nOverall: ref   max={np.max(orig_absmax):.4f} mean={np.mean(orig_absmax):.4f}")
    print(f"         fireq max={np.max(fireq_absmax):.4f} mean={np.mean(fireq_absmax):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare absmax with FireQ SCRS pair highlights")
    parser.add_argument("orig_absmax", help="Reference absmax/APOR/ROPV file")
    parser.add_argument("fireq_absmax", help="FireQ absmax/APOR/ROPV file")
    parser.add_argument("scrs_file", help="FireQ late CRS .scrs file")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument("--post-rope", action="store_true",
                        help="Compare using post-rope absmax slot (fallback to pre if empty)")
    parser.add_argument("-o", "--output", help="Output image path")
    args = parser.parse_args()

    print("=" * 80)
    print("AbsMax Comparison: Reference vs FireQ (SCRS-highlighted)")
    print("=" * 80)

    print(f"\nLoading {args.orig_absmax} ...")
    orig_data = load_absmax(args.orig_absmax)

    print(f"Loading {args.fireq_absmax} ...")
    fireq_data = load_absmax(args.fireq_absmax)

    print(f"Loading {args.scrs_file} ...")
    scrs_pairs = load_scrs_pairs(args.scrs_file)

    layer_idx = args.layer
    head_idx = args.head

    orig_vals = select_absmax_slot(orig_data, layer_idx, head_idx, args.post_rope, "Reference")
    fireq_vals = select_absmax_slot(fireq_data, layer_idx, head_idx, args.post_rope, "FireQ")

    pairs = scrs_pairs.get((layer_idx, head_idx), [])
    if not pairs:
        print(f"\n⚠ No SCRS pairs for Layer {layer_idx}, Head {head_idx}")
        print(f"  Available keys sample: {sorted(scrs_pairs.keys())[:20]} ...")

    visualize_comparison(
        orig_vals,
        fireq_vals,
        pairs,
        layer_idx,
        head_idx,
        args.post_rope,
        args.output,
    )


if __name__ == "__main__":
    main()
