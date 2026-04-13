#!/usr/bin/env python3
"""
Plot position-bucketed channel statistics for selected RoPE pairs.

For each selected (layer, head, pair), this script loads pre/post values from ROPV,
splits tokens by position bucket, and computes qXX(|dim0|), qXX(|dim1|) per bucket.
This is useful to check whether RoPE mixing is actually visible position-wise even
when global dim-wise absmax hides it.
"""

import argparse
import math
import os
import struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_chunk_f32(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} floats, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def read_layer(filepath, target_layer, max_tokens):
    with open(filepath, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        stride = n_heads * n_dims

        for layer in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            pre = None
            if pre_count > 0:
                pre_tokens = pre_count // stride
                pre_loaded = min(pre_tokens, max_tokens)
                pre_read = pre_loaded * stride
                pre_skip = pre_count - pre_read
                pre = read_chunk_f32(f, pre_read).reshape(pre_loaded, n_heads, n_dims)
                if pre_skip > 0:
                    f.seek(pre_skip * 4, 1)
            post_count = struct.unpack('I', f.read(4))[0]
            post = None
            if post_count > 0:
                post_tokens = post_count // stride
                post_loaded = min(post_tokens, max_tokens)
                post_read = post_loaded * stride
                post_skip = post_count - post_read
                post = read_chunk_f32(f, post_read).reshape(post_loaded, n_heads, n_dims)
                if post_skip > 0:
                    f.seek(post_skip * 4, 1)
            if layer == target_layer:
                return {
                    'version': version,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'n_dims': n_dims,
                    'n_tokens': n_tokens,
                    'pre': pre,
                    'post': post,
                }
    raise ValueError(f"Layer {target_layer} not found")


def bucket_stats(arr, bucket_size, percentile):
    n = arr.shape[0]
    xs = []
    q0 = []
    q1 = []
    for start in range(0, n, bucket_size):
        end = min(start + bucket_size, n)
        chunk = arr[start:end]
        if chunk.shape[0] == 0:
            continue
        xs.append((start + end - 1) / 2.0)
        q0.append(float(np.percentile(np.abs(chunk[:, 0]), percentile)))
        q1.append(float(np.percentile(np.abs(chunk[:, 1]), percentile)))
    return np.array(xs), np.array(q0), np.array(q1)


def main():
    ap = argparse.ArgumentParser(description='Analyze position-bucketed pair channel stats from ROPV')
    ap.add_argument('ropv_path')
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--head', type=int, required=True)
    ap.add_argument('--pairs', type=int, nargs='+', required=True)
    ap.add_argument('--max-tokens', type=int, default=20480)
    ap.add_argument('--bucket-size', type=int, default=256)
    ap.add_argument('--percentile', type=float, default=99.0)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = read_layer(args.ropv_path, args.layer, args.max_tokens)
    pre = data['pre']
    post = data['post']
    if pre is None or post is None:
        raise ValueError('Selected layer does not have both pre and post values loaded')

    n_tokens = min(pre.shape[0], post.shape[0])
    pre = pre[:n_tokens]
    post = post[:n_tokens]

    fig, axes = plt.subplots(len(args.pairs), 2, figsize=(12, 3.6 * len(args.pairs)), squeeze=False)

    rows = []
    for i, pair in enumerate(args.pairs):
        d0 = 2 * pair
        d1 = d0 + 1
        pre_pair = pre[:, args.head, [d0, d1]]
        post_pair = post[:, args.head, [d0, d1]]

        x_pre, q0_pre, q1_pre = bucket_stats(pre_pair, args.bucket_size, args.percentile)
        x_post, q0_post, q1_post = bucket_stats(post_pair, args.bucket_size, args.percentile)

        ax = axes[i, 0]
        ax.plot(x_pre, q0_pre, marker='o', label=f'dim {d0}')
        ax.plot(x_pre, q1_pre, marker='s', label=f'dim {d1}')
        ax.set_title(f'Pre-RoPE P{pair} (q{args.percentile:g})')
        ax.set_xlabel('token position bucket center')
        ax.set_ylabel(f'q{args.percentile:g}(|value|)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.plot(x_post, q0_post, marker='o', label=f'dim {d0}')
        ax.plot(x_post, q1_post, marker='s', label=f'dim {d1}')
        ax.set_title(f'Post-RoPE P{pair} (q{args.percentile:g})')
        ax.set_xlabel('token position bucket center')
        ax.set_ylabel(f'q{args.percentile:g}(|value|)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        for j in range(len(x_post)):
            rows.append({
                'pair': pair,
                'bucket_center': float(x_post[j]),
                'post_dim0_q': float(q0_post[j]),
                'post_dim1_q': float(q1_post[j]),
                'post_ratio_minmax': float(min(q0_post[j], q1_post[j]) / max(q0_post[j], q1_post[j])) if max(q0_post[j], q1_post[j]) > 0 else 1.0,
            })

    fig.suptitle(f'Layer {args.layer}, Head {args.head} position-bucketed q{args.percentile:g}(|dim|)')
    fig.tight_layout()
    out_png = Path(args.output_dir) / f'pair_bucket_q{int(args.percentile)}_L{args.layer}_H{args.head}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

    import csv
    out_csv = Path(args.output_dir) / f'pair_bucket_q{int(args.percentile)}_L{args.layer}_H{args.head}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['pair','bucket_center','post_dim0_q','post_dim1_q','post_ratio_minmax'])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    print(f'Saved: {out_png}')
    print(f'Saved: {out_csv}')


if __name__ == '__main__':
    main()
