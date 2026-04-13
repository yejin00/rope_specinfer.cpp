#!/usr/bin/env python3
"""
Calibrate a conservative residual gamma on top of base RPN scales, while keeping
rotation fixed.

This variant is intended as a refinement of gamma_util_full:
- base RPN remains s_base = base_alpha / m_p
- rotated post-RoPE channel stats are measured after the fixed rotation
- a raw gamma is computed against a separate channel target (e.g. 6.5 or 7.0)
- only sufficiently balanced pairs receive gamma
- gamma is shrunk toward 1 and then clamped conservatively

The output CSV is consumable by fuse_rpn_alpha.py via --gamma-csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from typing import Dict, List

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def load_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            row['_layer'] = int(row['layer'])
            row['_head'] = int(row['head'])
            row['_pair'] = int(row['pair'])
            row['_alpha_rad'] = float(row['suggested_alpha_rad'])
            rows.append(row)
    return fieldnames, rows


def summarize_pair_l2(l2_per_token: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    l2_max = l2_per_token.max(axis=0)
    if stat_mode == 'max':
        return l2_max.astype(np.float32, copy=False)

    q = np.percentile(l2_per_token, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == 'percentile':
        return q
    if stat_mode == 'blend':
        return (q + tail_lambda * (l2_max - q)).astype(np.float32, copy=False)

    raise ValueError(f'Unsupported stat mode: {stat_mode}')


def summarize_abs_channels(abs_vals: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    ch_max = abs_vals.max(axis=0)
    if stat_mode == 'max':
        return ch_max.astype(np.float32, copy=False)

    q = np.percentile(abs_vals, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == 'percentile':
        return q
    if stat_mode == 'blend':
        return (q + tail_lambda * (ch_max - q)).astype(np.float32, copy=False)

    raise ValueError(f'Unsupported stat mode: {stat_mode}')


def read_chunk_f32(f, n_floats: int) -> np.ndarray:
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f'Expected {n_floats} float32 values, got {got}')
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


def build_output_fieldnames(base_fieldnames: list[str]) -> list[str]:
    extra = [
        'base_alpha',
        'channel_target',
        'base_pair_stat',
        'base_k_scale',
        'base_q_scale',
        'rot_dim_stat0',
        'rot_dim_stat1',
        'base_scaled_dim0',
        'base_scaled_dim1',
        'base_channel_ratio',
        'apply_gamma',
        'gamma_raw',
        'shrink_lambda',
        'residual_gamma',
        'final_k_scale',
        'final_q_scale',
        'final_dim0',
        'final_dim1',
        'dominant_channel',
        'dominant_before',
        'target_after',
    ]
    out = list(base_fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Calibrate conservative balanced-pair residual gamma on top of fixed base RPN'
    )
    parser.add_argument('baseline_csv', help='Baseline late-pair rotation CSV (angles fixed)')
    parser.add_argument('ropv_path', help='Original ROPV dump with pre/post values')
    parser.add_argument('output_csv', help='Output CSV with conservative residual_gamma')
    parser.add_argument('--analysis-csv', help='Optional compact analysis CSV')
    parser.add_argument('--base-alpha', type=float, default=8.0,
                        help='Base RPN alpha used for s_base = base_alpha / m_p (default: 8.0)')
    parser.add_argument('--channel-target', type=float, default=7.0,
                        help='Target for dominant rotated channel before shrink (default: 7.0)')
    parser.add_argument('--max-tokens', type=int, default=20480, help='Max ROPV tokens to load per layer')
    parser.add_argument('--pair-stat', choices=['max', 'percentile', 'blend'], default='max',
                        help='Statistic for pre-RoPE pair L2 used in base RPN scale')
    parser.add_argument('--pair-percentile', type=float, default=99.9,
                        help='Percentile used when --pair-stat is percentile/blend')
    parser.add_argument('--pair-tail-lambda', type=float, default=0.1,
                        help='Blend factor when --pair-stat is blend')
    parser.add_argument('--dim-stat', choices=['max', 'percentile', 'blend'], default='max',
                        help='Statistic for rotated post-RoPE per-channel magnitudes')
    parser.add_argument('--dim-percentile', type=float, default=99.9,
                        help='Percentile used when --dim-stat is percentile/blend')
    parser.add_argument('--dim-tail-lambda', type=float, default=0.1,
                        help='Blend factor when --dim-stat is blend')
    parser.add_argument('--balance-threshold', type=float, default=0.90,
                        help='Require min/max >= threshold to apply gamma (default: 0.90)')
    parser.add_argument('--shrink-lambda', type=float, default=0.10,
                        help='Use gamma_eff = 1 + lambda * (gamma_raw - 1) (default: 0.10)')
    parser.add_argument('--upward-only', action='store_true',
                        help='If set, gamma_raw below 1.0 is forced to 1.0 before shrink')
    parser.add_argument('--min-gamma', type=float, default=1.0, help='Lower clamp for residual gamma')
    parser.add_argument('--max-gamma', type=float, default=1.08, help='Upper clamp for residual gamma')
    args = parser.parse_args()

    base_fieldnames, rows = load_rows(args.baseline_csv)
    rows_by_layer: Dict[int, List[dict]] = {}
    for row in rows:
        rows_by_layer.setdefault(row['_layer'], []).append(row)

    output_fieldnames = build_output_fieldnames(base_fieldnames)
    merged_rows: list[dict] = []
    analysis_rows: list[dict] = []
    all_gamma: list[float] = []
    apply_count = 0

    with open(args.ropv_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f'Invalid magic: {hex(magic)}')

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        n_pairs = n_dims // 2
        stride = n_heads * n_dims

        print(
            f'ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, '
            f'tokens={n_tokens}'
        )
        print(
            f'  base_alpha={args.base_alpha}, channel_target={args.channel_target}, '
            f'balance_threshold={args.balance_threshold}, shrink_lambda={args.shrink_lambda}'
        )
        print(
            f'  pair_stat={args.pair_stat}, dim_stat={args.dim_stat}, '
            f'gamma_clip=[{args.min_gamma}, {args.max_gamma}], upward_only={args.upward_only}'
        )

        for layer in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
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

            post_count = struct.unpack('I', f.read(4))[0]
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

            layer_rows = rows_by_layer.get(layer, [])
            if not layer_rows:
                continue
            if pre_loaded == 0 or post_loaded == 0:
                raise ValueError(f'Layer {layer} has no loaded tokens')

            l2_per_token = np.sqrt(np.sum(pre * pre, axis=3, dtype=np.float32), dtype=np.float32)
            pair_stats = summarize_pair_l2(
                l2_per_token,
                stat_mode=args.pair_stat,
                percentile=args.pair_percentile,
                tail_lambda=args.pair_tail_lambda,
            )

            for row in layer_rows:
                head = row['_head']
                pair = row['_pair']
                alpha = row['_alpha_rad']

                pair_stat = float(max(pair_stats[head, pair], 1e-8))
                base_k_scale = args.base_alpha / pair_stat
                base_q_scale = 1.0 / base_k_scale

                rotated = rotate_pair(post[:, head, pair, :], alpha)
                rot_dim_stats = summarize_abs_channels(
                    np.abs(rotated, dtype=np.float32),
                    stat_mode=args.dim_stat,
                    percentile=args.dim_percentile,
                    tail_lambda=args.dim_tail_lambda,
                )
                stat0 = float(rot_dim_stats[0])
                stat1 = float(rot_dim_stats[1])

                base_scaled_dim0 = base_k_scale * stat0
                base_scaled_dim1 = base_k_scale * stat1
                dominant_before = max(base_scaled_dim0, base_scaled_dim1, 1e-8)
                ratio = float(min(base_scaled_dim0, base_scaled_dim1) / dominant_before)

                gamma_raw = args.channel_target / dominant_before
                if args.upward_only:
                    gamma_raw = max(gamma_raw, 1.0)

                apply_gamma = ratio >= args.balance_threshold
                if apply_gamma:
                    gamma_eff = 1.0 + args.shrink_lambda * (gamma_raw - 1.0)
                    gamma_eff = float(np.clip(gamma_eff, args.min_gamma, args.max_gamma))
                    apply_count += 1
                else:
                    gamma_eff = 1.0

                final_k_scale = base_k_scale * gamma_eff
                final_q_scale = 1.0 / final_k_scale
                final_dim0 = final_k_scale * stat0
                final_dim1 = final_k_scale * stat1
                dominant_channel = 0 if base_scaled_dim0 >= base_scaled_dim1 else 1

                merged = dict(row)
                for k in list(merged.keys()):
                    if k.startswith('_'):
                        del merged[k]
                merged['base_alpha'] = f'{args.base_alpha:.10f}'
                merged['channel_target'] = f'{args.channel_target:.10f}'
                merged['base_pair_stat'] = f'{pair_stat:.10f}'
                merged['base_k_scale'] = f'{base_k_scale:.10f}'
                merged['base_q_scale'] = f'{base_q_scale:.10f}'
                merged['rot_dim_stat0'] = f'{stat0:.10f}'
                merged['rot_dim_stat1'] = f'{stat1:.10f}'
                merged['base_scaled_dim0'] = f'{base_scaled_dim0:.10f}'
                merged['base_scaled_dim1'] = f'{base_scaled_dim1:.10f}'
                merged['base_channel_ratio'] = f'{ratio:.10f}'
                merged['apply_gamma'] = '1' if apply_gamma else '0'
                merged['gamma_raw'] = f'{gamma_raw:.10f}'
                merged['shrink_lambda'] = f'{args.shrink_lambda:.10f}'
                merged['residual_gamma'] = f'{gamma_eff:.10f}'
                merged['final_k_scale'] = f'{final_k_scale:.10f}'
                merged['final_q_scale'] = f'{final_q_scale:.10f}'
                merged['final_dim0'] = f'{final_dim0:.10f}'
                merged['final_dim1'] = f'{final_dim1:.10f}'
                merged['dominant_channel'] = str(dominant_channel)
                merged['dominant_before'] = f'{dominant_before:.10f}'
                merged['target_after'] = f'{max(final_dim0, final_dim1):.10f}'
                merged_rows.append(merged)

                analysis_rows.append({
                    'layer': layer,
                    'head': head,
                    'pair': pair,
                    'suggested_alpha_deg': math.degrees(alpha),
                    'base_pair_stat': pair_stat,
                    'base_k_scale': base_k_scale,
                    'rot_dim_stat0': stat0,
                    'rot_dim_stat1': stat1,
                    'base_scaled_dim0': base_scaled_dim0,
                    'base_scaled_dim1': base_scaled_dim1,
                    'base_channel_ratio': ratio,
                    'apply_gamma': int(apply_gamma),
                    'gamma_raw': gamma_raw,
                    'residual_gamma': gamma_eff,
                    'final_dim0': final_dim0,
                    'final_dim1': final_dim1,
                    'dominant_channel': dominant_channel,
                })
                all_gamma.append(gamma_eff)

            if (layer + 1) % 8 == 0:
                print(f'  processed {layer + 1}/{n_layers} layers')

    with open(args.output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, '') for k in output_fieldnames})

    if args.analysis_csv and analysis_rows:
        with open(args.analysis_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(analysis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(analysis_rows)

    g = np.array(all_gamma, dtype=np.float32)
    print(f'Saved gamma CSV: {args.output_csv}')
    if args.analysis_csv and analysis_rows:
        print(f'Saved analysis CSV: {args.analysis_csv}')
    if g.size:
        print(
            f'Residual gamma stats: min={g.min():.6f} mean={g.mean():.6f} '
            f'median={np.median(g):.6f} max={g.max():.6f}'
        )
        print(f'Applied gamma to {apply_count} / {len(merged_rows)} rows')


if __name__ == '__main__':
    main()
