#!/usr/bin/env python3
"""
Analyze sampled Q / dense K activation dumps with direct QK logit metrics.

Expected input files:
- QK_DIST_Q_PATH: sampled final-Q activations written by the model
- QK_DIST_K_PATH: dense final-K activations written by the model

This first-pass analyzer focuses on the highest-signal metrics:
- global mean logit MSE
- global p99 logit squared error
- sink / local / long-range logit MSE
- top-1 match rate
- mean / p99 attention KL
- worst heads by mean logit MSE
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

MAGIC_ACTV = 0x41435456  # "ACTV"


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
            raise EOFError(f"{self.path}: no more layers to read")

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


def parse_test_arg(s: str) -> Tuple[str, str, str]:
    if "=" not in s or ":" not in s:
        raise argparse.ArgumentTypeError("Expected NAME=Q_PATH:K_PATH")
    name, rest = s.split("=", 1)
    q_path, k_path = rest.split(":", 1)
    return name, q_path, k_path


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


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a), len(b))
    if k <= 0:
        return 1.0
    a_idx = np.argpartition(a, -k)[-k:]
    b_idx = np.argpartition(b, -k)[-k:]
    a_set = set(int(x) for x in a_idx)
    b_set = set(int(x) for x in b_idx)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 1.0


def recall_topk(reference: np.ndarray, test: np.ndarray, k: int) -> float:
    k = min(k, len(reference), len(test))
    if k <= 0:
        return 1.0
    ref_idx = set(int(x) for x in np.argpartition(reference, -k)[-k:])
    test_idx = set(int(x) for x in np.argpartition(test, -k)[-k:])
    return len(ref_idx & test_idx) / len(ref_idx) if ref_idx else 1.0


def accumulate_region_stats(store: Dict[str, float], sq_err: np.ndarray, mask: np.ndarray, prefix: str) -> None:
    if mask is None:
        return
    count = int(mask.sum())
    if count <= 0:
        return
    store[f"{prefix}_sum"] += float(sq_err[mask].sum())
    store[f"{prefix}_count"] += count


def ensure_same_positions(kind: str, ref: np.ndarray, cur: np.ndarray, path: str) -> None:
    if ref.shape != cur.shape or not np.array_equal(ref, cur):
        raise ValueError(f"{kind} positions mismatch in {path}")


def init_method_accumulators(name: str) -> Dict[str, object]:
    return {
        "method": name,
        "sq_errors": [],
        "total_sum": 0.0,
        "total_count": 0,
        "sink_sum": 0.0,
        "sink_count": 0,
        "local_sum": 0.0,
        "local_count": 0,
        "long_sum": 0.0,
        "long_count": 0,
        "top1_match": 0,
        "query_count": 0,
        "top5_jaccard_sum": 0.0,
        "top10_recall_sum": 0.0,
        "attention_kl": [],
        "attention_l1": [],
        "margin_drop": [],
        "head_stats": {},
    }


def finalize_head_stats(head_stats: Dict[Tuple[int, int], Dict[str, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (layer, head), v in head_stats.items():
        total_count = int(v["total_count"])
        if total_count == 0:
            continue
        query_count = int(v["query_count"])
        rows.append({
            "layer": layer,
            "head": head,
            "mean_logit_mse": float(v["total_sum"]) / total_count,
            "sink_mse": float(v["sink_sum"]) / v["sink_count"] if v["sink_count"] else np.nan,
            "local_mse": float(v["local_sum"]) / v["local_count"] if v["local_count"] else np.nan,
            "long_mse": float(v["long_sum"]) / v["long_count"] if v["long_count"] else np.nan,
            "top1_match_rate": float(v["top1_match"]) / query_count if query_count else np.nan,
            "mean_attention_kl": float(v["attention_kl_sum"]) / query_count if query_count else np.nan,
            "mean_attention_l1": float(v["attention_l1_sum"]) / query_count if query_count else np.nan,
            "mean_margin_drop": float(v["margin_drop_sum"]) / query_count if query_count else np.nan,
            "query_count": query_count,
            "logit_count": total_count,
        })
    rows.sort(key=lambda x: x["mean_logit_mse"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sampled-Q / dense-K ACTV dumps with QK logit metrics")
    parser.add_argument("--orig-q", required=True, help="Reference/original sampled Q ACTV file")
    parser.add_argument("--orig-k", required=True, help="Reference/original dense K ACTV file")
    parser.add_argument(
        "--test",
        required=True,
        action="append",
        type=parse_test_arg,
        help="Method to compare, formatted as NAME=Q_PATH:K_PATH",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save CSV outputs")
    parser.add_argument("--max-layers", type=int, default=None, help="Optional layer cap")
    parser.add_argument(
        "--max-queries-per-layer",
        type=int,
        default=8,
        help="Limit sampled queries per layer to keep compute manageable (default: 8)",
    )
    parser.add_argument("--local-window", type=int, default=64, help="Distance threshold for local keys")
    parser.add_argument("--long-window", type=int, default=512, help="Distance threshold for long-range keys")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    orig_q = ActvReader(args.orig_q)
    orig_k = ActvReader(args.orig_k)
    test_readers = []
    for name, q_path, k_path in args.test:
        test_readers.append((name, ActvReader(q_path), ActvReader(k_path)))

    if orig_q.n_layers != orig_k.n_layers:
        raise ValueError("Original Q/K layer mismatch")
    if orig_q.n_dims != orig_k.n_dims:
        raise ValueError("Original Q/K dim mismatch")
    if orig_q.n_heads % orig_k.n_heads != 0:
        raise ValueError(
            f"Original Q/K head mismatch not divisible: q_heads={orig_q.n_heads}, k_heads={orig_k.n_heads}"
        )

    n_layers = orig_q.n_layers
    if args.max_layers is not None:
        n_layers = min(n_layers, args.max_layers)

    q_heads = orig_q.n_heads
    k_heads = orig_k.n_heads
    q_per_k = q_heads // k_heads

    method_acc = {name: init_method_accumulators(name) for name, _, _ in test_readers}

    for layer in range(n_layers):
        q_orig_layer = orig_q.read_next_layer()
        k_orig_layer = orig_k.read_next_layer()

        layer_tests = []
        for name, q_reader, k_reader in test_readers:
            q_layer = q_reader.read_next_layer()
            k_layer = k_reader.read_next_layer()
            ensure_same_positions("Q", q_orig_layer.positions, q_layer.positions, q_reader.path)
            ensure_same_positions("K", k_orig_layer.positions, k_layer.positions, k_reader.path)
            layer_tests.append((name, q_layer, k_layer))

        if q_orig_layer.values.shape[0] == 0 or k_orig_layer.values.shape[0] == 0:
            continue

        q_sel_idx = select_evenly_spaced_indices(q_orig_layer.values.shape[0], args.max_queries_per_layer)
        q_positions = q_orig_layer.positions[q_sel_idx]
        q_orig_sel = q_orig_layer.values[q_sel_idx]
        k_positions = k_orig_layer.positions
        k_orig_vals = k_orig_layer.values

        for local_query_idx, token_pos in enumerate(q_positions):
            key_end = int(np.searchsorted(k_positions, token_pos, side="right"))
            if key_end <= 0:
                continue

            key_positions = k_positions[:key_end].astype(np.int64)
            sink_mask = key_positions < 4
            dist = int(token_pos) - key_positions
            local_mask = (~sink_mask) & (dist >= 0) & (dist <= args.local_window)
            long_mask = (~sink_mask) & (dist > args.long_window)

            orig_q_heads = q_orig_sel[local_query_idx]

            # Precompute original logits once per Q head.
            orig_logits_by_head: List[np.ndarray] = []
            orig_attn_by_head: List[np.ndarray] = []
            orig_top1_idx: List[int] = []
            orig_top5: List[np.ndarray] = []
            orig_top10: List[np.ndarray] = []
            orig_margin: List[float] = []
            for q_head in range(q_heads):
                kv_head = min(q_head // q_per_k, k_heads - 1)
                orig_logits = k_orig_vals[:key_end, kv_head, :] @ orig_q_heads[q_head]
                orig_logits_by_head.append(orig_logits)
                orig_attn = stable_softmax(orig_logits)
                orig_attn_by_head.append(orig_attn)
                orig_top1_idx.append(int(np.argmax(orig_logits)))
                orig_margin.append(float(np.partition(orig_logits, -1)[-1] - np.partition(orig_logits, -2)[-2]) if len(orig_logits) >= 2 else 0.0)

            for name, q_test_layer, k_test_layer in layer_tests:
                acc = method_acc[name]
                q_test_heads = q_test_layer.values[q_sel_idx[local_query_idx]]
                k_test_vals = k_test_layer.values

                for q_head in range(q_heads):
                    kv_head = min(q_head // q_per_k, k_heads - 1)
                    test_logits = k_test_vals[:key_end, kv_head, :] @ q_test_heads[q_head]
                    orig_logits = orig_logits_by_head[q_head]
                    sq_err = (orig_logits - test_logits) ** 2

                    acc["sq_errors"].append(sq_err.astype(np.float32))
                    acc["total_sum"] += float(sq_err.sum())
                    acc["total_count"] += int(sq_err.size)
                    accumulate_region_stats(acc, sq_err, sink_mask, "sink")
                    accumulate_region_stats(acc, sq_err, local_mask, "local")
                    accumulate_region_stats(acc, sq_err, long_mask, "long")

                    test_top1 = int(np.argmax(test_logits))
                    acc["top1_match"] += int(test_top1 == orig_top1_idx[q_head])
                    acc["query_count"] += 1
                    acc["top5_jaccard_sum"] += jaccard_topk(orig_logits, test_logits, 5)
                    acc["top10_recall_sum"] += recall_topk(orig_logits, test_logits, 10)

                    orig_attn = orig_attn_by_head[q_head]
                    test_attn = stable_softmax(test_logits)
                    kl = float(np.sum(orig_attn * (np.log(orig_attn + 1e-12) - np.log(test_attn + 1e-12))))
                    l1 = float(np.sum(np.abs(orig_attn - test_attn)))
                    acc["attention_kl"].append(kl)
                    acc["attention_l1"].append(l1)

                    if len(test_logits) >= 2:
                        top2 = np.partition(test_logits, -2)[-2:]
                        test_margin = float(top2[-1] - top2[-2])
                    else:
                        test_margin = 0.0
                    acc["margin_drop"].append(orig_margin[q_head] - test_margin)

                    head_key = (layer, q_head)
                    hs = acc["head_stats"].setdefault(
                        head_key,
                        {
                            "total_sum": 0.0,
                            "total_count": 0,
                            "sink_sum": 0.0,
                            "sink_count": 0,
                            "local_sum": 0.0,
                            "local_count": 0,
                            "long_sum": 0.0,
                            "long_count": 0,
                            "top1_match": 0,
                            "query_count": 0,
                            "attention_kl_sum": 0.0,
                            "attention_l1_sum": 0.0,
                            "margin_drop_sum": 0.0,
                        },
                    )
                    hs["total_sum"] += float(sq_err.sum())
                    hs["total_count"] += int(sq_err.size)
                    accumulate_region_stats(hs, sq_err, sink_mask, "sink")
                    accumulate_region_stats(hs, sq_err, local_mask, "local")
                    accumulate_region_stats(hs, sq_err, long_mask, "long")
                    hs["top1_match"] += int(test_top1 == orig_top1_idx[q_head])
                    hs["query_count"] += 1
                    hs["attention_kl_sum"] += kl
                    hs["attention_l1_sum"] += l1
                    hs["margin_drop_sum"] += orig_margin[q_head] - test_margin

    orig_q.close()
    orig_k.close()
    for _, q_reader, k_reader in test_readers:
        q_reader.close()
        k_reader.close()

    summary_rows = []
    head_rows = []
    for name, acc in method_acc.items():
        sq_all = np.concatenate(acc["sq_errors"]) if acc["sq_errors"] else np.zeros(0, dtype=np.float32)
        kl_all = np.array(acc["attention_kl"], dtype=np.float32)
        l1_all = np.array(acc["attention_l1"], dtype=np.float32)
        margin_all = np.array(acc["margin_drop"], dtype=np.float32)

        head_metrics = finalize_head_stats(acc["head_stats"])
        for row in head_metrics:
            row["method"] = name
            head_rows.append(row)

        summary = {
            "method": name,
            "global_mean_logit_mse": acc["total_sum"] / acc["total_count"] if acc["total_count"] else np.nan,
            "global_p99_logit_se": float(np.percentile(sq_all, 99)) if sq_all.size else np.nan,
            "sink_logit_mse": acc["sink_sum"] / acc["sink_count"] if acc["sink_count"] else np.nan,
            "local_logit_mse": acc["local_sum"] / acc["local_count"] if acc["local_count"] else np.nan,
            "long_logit_mse": acc["long_sum"] / acc["long_count"] if acc["long_count"] else np.nan,
            "top1_match_rate": acc["top1_match"] / acc["query_count"] if acc["query_count"] else np.nan,
            "top5_jaccard": acc["top5_jaccard_sum"] / acc["query_count"] if acc["query_count"] else np.nan,
            "top10_recall": acc["top10_recall_sum"] / acc["query_count"] if acc["query_count"] else np.nan,
            "mean_attention_kl": float(np.mean(kl_all)) if kl_all.size else np.nan,
            "p99_attention_kl": float(np.percentile(kl_all, 99)) if kl_all.size else np.nan,
            "mean_attention_l1": float(np.mean(l1_all)) if l1_all.size else np.nan,
            "mean_margin_drop": float(np.mean(margin_all)) if margin_all.size else np.nan,
            "p99_margin_drop": float(np.percentile(margin_all, 99)) if margin_all.size else np.nan,
            "query_heads_evaluated": acc["query_count"],
            "logit_count_evaluated": acc["total_count"],
        }
        summary_rows.append(summary)

    summary_csv = os.path.join(args.output_dir, "qk_logit_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    head_csv = os.path.join(args.output_dir, "qk_logit_head_metrics.csv")
    with open(head_csv, "w", newline="") as f:
        fieldnames = [
            "method",
            "layer",
            "head",
            "mean_logit_mse",
            "sink_mse",
            "local_mse",
            "long_mse",
            "top1_match_rate",
            "mean_attention_kl",
            "mean_attention_l1",
            "mean_margin_drop",
            "query_count",
            "logit_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(head_rows)

    print(f"Saved summary: {summary_csv}")
    print(f"Saved head metrics: {head_csv}")
    for row in summary_rows:
        print(
            f"{row['method']}: mean_logit_mse={row['global_mean_logit_mse']:.6g} | "
            f"p99_logit_se={row['global_p99_logit_se']:.6g} | "
            f"sink={row['sink_logit_mse']:.6g} | local={row['local_logit_mse']:.6g} | "
            f"long={row['long_logit_mse']:.6g} | top1={row['top1_match_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
