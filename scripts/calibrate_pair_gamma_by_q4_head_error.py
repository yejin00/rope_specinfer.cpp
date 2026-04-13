#!/usr/bin/env python3
"""
Calibrate per-pair residual gamma on top of a fixed rotation CSV and fixed base
RPN scale by minimizing token-wise q4_0_head fake quant/dequant error.

For each (layer, head, pair) row in the input rotation CSV:
  1. Build the full head after applying all fixed pair rotations from the CSV
  2. Apply the base pre-RoPE RPN scale s_base,p = alpha / m_p to all pairs
  3. Sweep candidate gamma values only for the target pair
  4. Fake-quant/dequant the full 128-d head with q4_0_head
  5. Choose the gamma that minimizes mean token-wise full-head SSE

The output CSV is intended to be consumed by fuse_rpn_alpha.py via --gamma-csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

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
        raise ValueError('Gamma grid is empty')
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
        'target_alpha',
        'base_pair_stat',
        'base_k_scale',
        'base_q_scale',
        'base_scaled_dim0_absmax',
        'base_scaled_dim1_absmax',
        'selected_scaled_dim0_absmax',
        'selected_scaled_dim1_absmax',
        'base_channel_ratio',
        'selected_channel_ratio',
        'residual_gamma',
        'final_k_scale',
        'final_q_scale',
        'head_fakeq_err_before',
        'head_fakeq_err_after',
        'head_fakeq_err_delta',
        'head_fakeq_err_gain_pct',
        'gamma_grid',
        'gamma_search_mode',
    ]
    out = list(base_fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def row_allowed(row: dict, layers: set[int] | None, heads: set[int] | None, pairs: set[int] | None) -> bool:
    if layers is not None and row['_layer'] not in layers:
        return False
    if heads is not None and row['_head'] not in heads:
        return False
    if pairs is not None and row['_pair'] not in pairs:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Calibrate residual gamma per pair by minimizing full-head q4_0_head fake quant error'
    )
    parser.add_argument('rotation_csv', help='Rotation CSV with fixed suggested alpha per pair')
    parser.add_argument('ropv_path', help='Input ROPV dump with pre/post activations')
    parser.add_argument('output_csv', help='Output CSV with residual_gamma chosen by q4 error')
    parser.add_argument('--analysis-csv', help='Optional analysis CSV (same rows, extra metrics)')
    parser.add_argument('--target-alpha', type=float, default=8.0, help='Base RPN target alpha (default: 8.0)')
    parser.add_argument('--max-tokens', type=int, default=20480, help='Max tokens per layer to load (default: 20480)')
    parser.add_argument('--pair-stat', choices=['max', 'percentile', 'blend'], default='max',
                        help='Statistic for pre-RoPE pair L2 used in base RPN scale')
    parser.add_argument('--pair-percentile', type=float, default=99.9,
                        help='Percentile used when --pair-stat is percentile/blend')
    parser.add_argument('--pair-tail-lambda', type=float, default=0.1,
                        help='Blend factor when --pair-stat is blend')
    parser.add_argument('--gamma-grid', default='0.90,0.95,1.00,1.05,1.10',
                        help='Comma-separated candidate gamma values (default: 0.90,0.95,1.00,1.05,1.10)')
    parser.add_argument('--layers', type=int, nargs='*', help='Optional layer filter')
    parser.add_argument('--heads', type=int, nargs='*', help='Optional head filter')
    parser.add_argument('--pairs', type=int, nargs='*', help='Optional pair filter')
    args = parser.parse_args()

    gamma_grid = parse_csv_grid(args.gamma_grid)
    base_fieldnames, rows = load_rows(args.rotation_csv)

    layers_filter = set(args.layers) if args.layers else None
    heads_filter = set(args.heads) if args.heads else None
    pairs_filter = set(args.pairs) if args.pairs else None

    rows = [r for r in rows if row_allowed(r, layers_filter, heads_filter, pairs_filter)]
    if not rows:
        raise ValueError('No rows left after filters')

    rows_by_layer_head: Dict[tuple[int, int], List[dict]] = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row['_layer'], row['_head'])].append(row)

    output_fieldnames = build_output_fieldnames(base_fieldnames)
    out_rows: list[dict] = []
    all_gammas: list[float] = []
    all_deltas: list[float] = []

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
        print(f'  target_alpha={args.target_alpha}, pair_stat={args.pair_stat}, gamma_grid={gamma_grid.tolist()}')
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

                # Apply all fixed rotations from the input CSV for this head.
                for row in rows_for_head:
                    pair = row['_pair']
                    post_pairs[:, pair, :] = rotate_pair(post_pairs[:, pair, :], row['_alpha_rad'])

                rotated_head = post_pairs.reshape(post_loaded, n_dims)
                base_pair_stats = np.maximum(pair_stats[head], 1e-8).astype(np.float32, copy=False)
                base_pair_scales = (args.target_alpha / base_pair_stats).astype(np.float32, copy=False)
                base_scale_vec = np.repeat(base_pair_scales, 2, axis=0)
                base_scaled_head = rotated_head * base_scale_vec[None, :]

                base_deq = fake_quant_dequant_q4_0_head(base_scaled_head)
                base_sse = sse_per_token(base_scaled_head, base_deq)
                base_mean_sse = float(np.mean(base_sse))

                for row in rows_for_head:
                    pair = row['_pair']
                    d0 = 2 * pair
                    d1 = d0 + 1

                    cand = np.broadcast_to(base_scaled_head, (len(gamma_grid),) + base_scaled_head.shape).copy()
                    cand[:, :, d0:d1 + 1] *= gamma_grid[:, None, None]

                    cand_deq = fake_quant_dequant_q4_0_head_batch(cand)
                    cand_sse = sse_per_token_batch(cand, cand_deq)
                    cand_mean_sse = cand_sse.mean(axis=1)

                    tie_break = cand_mean_sse + 1e-12 * np.abs(gamma_grid - 1.0)
                    best_idx = int(np.argmin(tie_break))
                    best_gamma = float(gamma_grid[best_idx])
                    best_mean_sse = float(cand_mean_sse[best_idx])

                    base_dim0 = float(np.max(np.abs(base_scaled_head[:, d0])))
                    base_dim1 = float(np.max(np.abs(base_scaled_head[:, d1])))
                    sel_dim0 = float(np.max(np.abs(cand[best_idx, :, d0])))
                    sel_dim1 = float(np.max(np.abs(cand[best_idx, :, d1])))

                    base_ratio = float(min(base_dim0, base_dim1) / max(base_dim0, base_dim1, 1e-8))
                    sel_ratio = float(min(sel_dim0, sel_dim1) / max(sel_dim0, sel_dim1, 1e-8))

                    pair_stat = float(base_pair_stats[pair])
                    base_k_scale = float(base_pair_scales[pair])
                    final_k_scale = float(base_k_scale * best_gamma)
                    base_q_scale = float(1.0 / base_k_scale)
                    final_q_scale = float(1.0 / final_k_scale)
                    delta = float(base_mean_sse - best_mean_sse)
                    gain_pct = float(100.0 * delta / max(base_mean_sse, 1e-12))

                    out = dict(row)
                    for k in list(out.keys()):
                        if k.startswith('_'):
                            del out[k]
                    out['target_alpha'] = f'{args.target_alpha:.10f}'
                    out['base_pair_stat'] = f'{pair_stat:.10f}'
                    out['base_k_scale'] = f'{base_k_scale:.10f}'
                    out['base_q_scale'] = f'{base_q_scale:.10f}'
                    out['base_scaled_dim0_absmax'] = f'{base_dim0:.10f}'
                    out['base_scaled_dim1_absmax'] = f'{base_dim1:.10f}'
                    out['selected_scaled_dim0_absmax'] = f'{sel_dim0:.10f}'
                    out['selected_scaled_dim1_absmax'] = f'{sel_dim1:.10f}'
                    out['base_channel_ratio'] = f'{base_ratio:.10f}'
                    out['selected_channel_ratio'] = f'{sel_ratio:.10f}'
                    out['residual_gamma'] = f'{best_gamma:.10f}'
                    out['final_k_scale'] = f'{final_k_scale:.10f}'
                    out['final_q_scale'] = f'{final_q_scale:.10f}'
                    out['head_fakeq_err_before'] = f'{base_mean_sse:.10f}'
                    out['head_fakeq_err_after'] = f'{best_mean_sse:.10f}'
                    out['head_fakeq_err_delta'] = f'{delta:.10f}'
                    out['head_fakeq_err_gain_pct'] = f'{gain_pct:.6f}'
                    out['gamma_grid'] = ','.join(f'{float(g):.6f}' for g in gamma_grid)
                    out['gamma_search_mode'] = 'full_head_q4_0_error'
                    out_rows.append(out)
                    all_gammas.append(best_gamma)
                    all_deltas.append(delta)

    with open(args.output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, '') for k in output_fieldnames})
    print(f'Saved gamma CSV: {args.output_csv}')

    if args.analysis_csv:
        with open(args.analysis_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow({k: row.get(k, '') for k in output_fieldnames})
        print(f'Saved analysis CSV: {args.analysis_csv}')

    if all_gammas:
        arr_gamma = np.array(all_gammas, dtype=np.float64)
        arr_delta = np.array(all_deltas, dtype=np.float64)
        print(
            'Residual gamma stats: '
            f'min={arr_gamma.min():.6f} mean={arr_gamma.mean():.6f} '
            f'median={np.median(arr_gamma):.6f} max={arr_gamma.max():.6f}'
        )
        print(
            'Head fakeq delta stats: '
            f'min={arr_delta.min():.10f} mean={arr_delta.mean():.10f} '
            f'median={np.median(arr_delta):.10f} max={arr_delta.max():.10f}'
        )


if __name__ == '__main__':
    main()
