#!/usr/bin/env python3
"""
Calibrate late-bucket target alpha values on top of a fixed rotation CSV by
minimizing token-wise full-head q4_0_head fake quant/dequant error.

The intended use is:
- keep rotation fixed from the input CSV
- keep the base RPN family s = alpha / m_p
- but allow late-pair buckets to use different alpha targets
- express the chosen bucket alpha as residual_gamma = alpha_bucket / base_alpha
  so the output CSV can be consumed by fuse_rpn_alpha.py --gamma-csv
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


def parse_csv_grid(text: str) -> np.ndarray:
    vals = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError('Alpha grid is empty')
    return np.array(vals, dtype=np.float32)


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


def fake_quant_dequant_q4_0_head(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f'Expected 2D array, got shape {x.shape}')
    n_tokens, n_dims = x.shape
    if n_dims % QK4_0_HEAD != 0:
        raise ValueError(f'Expected n_dims multiple of {QK4_0_HEAD}, got {n_dims}')

    n_blocks = n_dims // QK4_0_HEAD
    out = np.empty_like(x, dtype=np.float32)

    for bi in range(n_blocks):
        lo = bi * QK4_0_HEAD
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


def fake_quant_dequant_q4_0_head_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f'Expected 3D array, got shape {x.shape}')
    batch, n_tokens, n_dims = x.shape
    if n_dims % QK4_0_HEAD != 0:
        raise ValueError(f'Expected n_dims multiple of {QK4_0_HEAD}, got {n_dims}')

    n_blocks = n_dims // QK4_0_HEAD
    out = np.empty_like(x, dtype=np.float32)

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

    return out


def sse_per_token(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    diff = x.astype(np.float32, copy=False) - x_hat.astype(np.float32, copy=False)
    return np.sum(diff * diff, axis=1, dtype=np.float64)


def sse_per_token_batch(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    diff = x.astype(np.float32, copy=False) - x_hat.astype(np.float32, copy=False)
    return np.sum(diff * diff, axis=2, dtype=np.float64)


def build_output_fieldnames(base_fieldnames: Sequence[str]) -> list[str]:
    extra = [
        'base_alpha',
        'bucket_alpha',
        'bucket_index',
        'bucket_pair_lo',
        'bucket_pair_hi',
        'base_pair_stat',
        'base_k_scale',
        'base_q_scale',
        'residual_gamma',
        'final_k_scale',
        'final_q_scale',
        'bucket_head_fakeq_err_before',
        'bucket_head_fakeq_err_after',
        'bucket_head_fakeq_err_delta',
        'bucket_head_fakeq_err_gain_pct',
        'alpha_grid',
        'alpha_search_mode',
    ]
    out = list(base_fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def bucket_index_for_pair(pair: int, pair_start: int, bucket_size: int) -> int:
    if pair < pair_start:
        return -1
    return (pair - pair_start) // bucket_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Calibrate per-bucket target alpha by minimizing full-head q4_0_head fake quant error'
    )
    parser.add_argument('rotation_csv', help='Rotation CSV with fixed suggested alpha per pair')
    parser.add_argument('ropv_path', help='Input ROPV dump with pre/post activations')
    parser.add_argument('output_csv', help='Output CSV with residual_gamma chosen from bucket alpha')
    parser.add_argument('--analysis-csv', help='Optional per-bucket analysis CSV')
    parser.add_argument('--base-alpha', type=float, default=8.0, help='Base alpha used outside the bucket override (default: 8.0)')
    parser.add_argument('--alpha-grid', default='5.6,6.4,7.2,8.0', help='Comma-separated candidate bucket alpha values')
    parser.add_argument('--pair-start', type=int, default=32, help='First late pair index (default: 32)')
    parser.add_argument('--bucket-size', type=int, default=8, help='Pairs per bucket (default: 8)')
    parser.add_argument('--max-tokens', type=int, default=20480, help='Max tokens per layer to load')
    parser.add_argument('--pair-stat', choices=['max', 'percentile', 'blend'], default='max',
                        help='Statistic for pre-RoPE pair L2 used in base RPN scale')
    parser.add_argument('--pair-percentile', type=float, default=99.9,
                        help='Percentile used when --pair-stat is percentile/blend')
    parser.add_argument('--pair-tail-lambda', type=float, default=0.1,
                        help='Blend factor when --pair-stat is blend')
    parser.add_argument('--layers', type=int, nargs='*', help='Optional layer filter')
    parser.add_argument('--heads', type=int, nargs='*', help='Optional head filter')
    args = parser.parse_args()

    alpha_grid = parse_csv_grid(args.alpha_grid)
    layers_filter = set(args.layers) if args.layers else None
    heads_filter = set(args.heads) if args.heads else None

    base_fieldnames, rows = load_rows(args.rotation_csv)
    rows = [r for r in rows if (layers_filter is None or r['_layer'] in layers_filter) and (heads_filter is None or r['_head'] in heads_filter)]
    if not rows:
        raise ValueError('No rows left after filters')

    rows_by_layer_head: Dict[tuple[int, int], List[dict]] = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row['_layer'], row['_head'])].append(row)

    output_fieldnames = build_output_fieldnames(base_fieldnames)
    out_rows: list[dict] = []
    bucket_analysis: list[dict] = []
    chosen_alpha_counter: Counter = Counter()

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

        print(f'ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}')
        print(f'  base_alpha={args.base_alpha}, alpha_grid={alpha_grid.tolist()}, pair_start={args.pair_start}, bucket_size={args.bucket_size}')
        print(f'  filtered rows={len(rows)}')

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

            layer_heads = sorted({head for lyr, head in rows_by_layer_head if lyr == layer})
            if not layer_heads:
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

            print(f'  Layer {layer}: evaluating heads {layer_heads} with {post_loaded} tokens')

            for head in layer_heads:
                rows_for_head = sorted(rows_by_layer_head[(layer, head)], key=lambda r: r['_pair'])
                post_pairs = post[:, head, :, :].astype(np.float32, copy=True)

                for row in rows_for_head:
                    pair = row['_pair']
                    post_pairs[:, pair, :] = rotate_pair(post_pairs[:, pair, :], row['_alpha_rad'])

                rotated_head = post_pairs.reshape(post_loaded, n_dims)
                base_pair_stats = np.maximum(pair_stats[head], 1e-8).astype(np.float32, copy=False)
                base_pair_scales = (args.base_alpha / base_pair_stats).astype(np.float32, copy=False)
                base_scale_vec = np.repeat(base_pair_scales, 2, axis=0)
                base_scaled_head = rotated_head * base_scale_vec[None, :]

                base_deq = fake_quant_dequant_q4_0_head(base_scaled_head)
                base_sse = sse_per_token(base_scaled_head, base_deq)
                base_mean_sse = float(np.mean(base_sse))

                # group candidate rows by late bucket for this head
                bucket_to_rows: Dict[int, List[dict]] = defaultdict(list)
                for row in rows_for_head:
                    bi = bucket_index_for_pair(row['_pair'], args.pair_start, args.bucket_size)
                    if bi >= 0:
                        bucket_to_rows[bi].append(row)

                chosen_alpha_by_pair: Dict[int, float] = {}
                bucket_metrics_by_pair: Dict[int, tuple[float,float,float,float,int,int]] = {}

                for bi, bucket_rows in sorted(bucket_to_rows.items()):
                    bucket_pairs = sorted(r['_pair'] for r in bucket_rows)
                    dims = []
                    for p in bucket_pairs:
                        dims.extend([2 * p, 2 * p + 1])
                    dims = np.array(dims, dtype=np.int64)

                    gammas = alpha_grid / args.base_alpha
                    cand = np.broadcast_to(base_scaled_head, (len(alpha_grid),) + base_scaled_head.shape).copy()
                    cand[:, :, dims] *= gammas[:, None, None]

                    cand_deq = fake_quant_dequant_q4_0_head_batch(cand)
                    cand_sse = sse_per_token_batch(cand, cand_deq)
                    cand_mean_sse = cand_sse.mean(axis=1)

                    tie_break = cand_mean_sse + 1e-12 * np.abs(alpha_grid - args.base_alpha)
                    best_idx = int(np.argmin(tie_break))
                    best_alpha = float(alpha_grid[best_idx])
                    best_gamma = float(gammas[best_idx])
                    best_mean_sse = float(cand_mean_sse[best_idx])
                    delta = float(base_mean_sse - best_mean_sse)
                    gain_pct = float(100.0 * delta / max(base_mean_sse, 1e-12))
                    pair_lo = min(bucket_pairs)
                    pair_hi = max(bucket_pairs)

                    bucket_analysis.append({
                        'layer': layer,
                        'head': head,
                        'bucket_index': bi,
                        'bucket_pair_lo': pair_lo,
                        'bucket_pair_hi': pair_hi,
                        'bucket_pairs': ','.join(str(p) for p in bucket_pairs),
                        'base_alpha': float(args.base_alpha),
                        'chosen_bucket_alpha': best_alpha,
                        'residual_gamma': best_gamma,
                        'head_fakeq_err_before': base_mean_sse,
                        'head_fakeq_err_after': best_mean_sse,
                        'head_fakeq_err_delta': delta,
                        'head_fakeq_err_gain_pct': gain_pct,
                    })
                    chosen_alpha_counter[round(best_alpha, 4)] += 1

                    for row in bucket_rows:
                        chosen_alpha_by_pair[row['_pair']] = best_alpha
                        bucket_metrics_by_pair[row['_pair']] = (best_gamma, base_mean_sse, best_mean_sse, delta, pair_lo, pair_hi)

                for row in rows_for_head:
                    pair = row['_pair']
                    if pair not in chosen_alpha_by_pair:
                        continue
                    chosen_alpha = chosen_alpha_by_pair[pair]
                    gamma = chosen_alpha / args.base_alpha
                    pair_stat = float(base_pair_stats[pair])
                    base_k_scale = float(base_pair_scales[pair])
                    final_k_scale = float(chosen_alpha / pair_stat)
                    base_q_scale = float(1.0 / base_k_scale)
                    final_q_scale = float(1.0 / final_k_scale)
                    bi = bucket_index_for_pair(pair, args.pair_start, args.bucket_size)
                    gamma2, err_before, err_after, delta, pair_lo, pair_hi = bucket_metrics_by_pair[pair]
                    gain_pct = 100.0 * delta / max(err_before, 1e-12)

                    out = dict(row)
                    for k in list(out.keys()):
                        if k.startswith('_'):
                            del out[k]
                    out['base_alpha'] = f'{args.base_alpha:.10f}'
                    out['bucket_alpha'] = f'{chosen_alpha:.10f}'
                    out['bucket_index'] = str(bi)
                    out['bucket_pair_lo'] = str(pair_lo)
                    out['bucket_pair_hi'] = str(pair_hi)
                    out['base_pair_stat'] = f'{pair_stat:.10f}'
                    out['base_k_scale'] = f'{base_k_scale:.10f}'
                    out['base_q_scale'] = f'{base_q_scale:.10f}'
                    out['residual_gamma'] = f'{gamma:.10f}'
                    out['final_k_scale'] = f'{final_k_scale:.10f}'
                    out['final_q_scale'] = f'{final_q_scale:.10f}'
                    out['bucket_head_fakeq_err_before'] = f'{err_before:.10f}'
                    out['bucket_head_fakeq_err_after'] = f'{err_after:.10f}'
                    out['bucket_head_fakeq_err_delta'] = f'{delta:.10f}'
                    out['bucket_head_fakeq_err_gain_pct'] = f'{gain_pct:.6f}'
                    out['alpha_grid'] = ','.join(f'{float(a):.6f}' for a in alpha_grid)
                    out['alpha_search_mode'] = 'bucket_full_head_q4_0_error'
                    out_rows.append(out)

    with open(args.output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, '') for k in output_fieldnames})
    print(f'Saved bucket-alpha gamma CSV: {args.output_csv}')

    if args.analysis_csv and bucket_analysis:
        with open(args.analysis_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(bucket_analysis[0].keys()))
            writer.writeheader()
            writer.writerows(bucket_analysis)
        print(f'Saved bucket analysis CSV: {args.analysis_csv}')

    if chosen_alpha_counter:
        print('Chosen alpha counts:', dict(sorted(chosen_alpha_counter.items())))


if __name__ == '__main__':
    main()
