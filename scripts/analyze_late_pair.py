#!/usr/bin/env python3
"""
Analyze late-dim pair imbalance & "minor killing" risk under Q4_0-like block quant.

Given a ROPV activation dump (pre/post per layer), this script analyzes POST-RoPE values
and focuses on late blocks (default dims 64-95 and 96-127).

It reports and plots:
1) MR distribution: MR = major/minor for each RoPE pair (even,odd)
2) minor/Δ distribution: |minor| / Δ(t), where Δ(t) ~ block_absmax(t) / qmax
3) P(|minor| < k*Δ(t)) : "zero-like" fraction for minor channel under a threshold k (default 0.5)

Important:
- Q4_0 in llama.cpp uses 32-d blocks and step proportional to block absmax (amax).
- Here Δ(t) is modeled as block_absmax(t)/qmax, where qmax is typically ~7 (or 7.5).
- The 0.5*Δ threshold corresponds to round-to-nearest mapping to 0 in mid-tread quantization.

We compute stats in two modes:
A) Unconditional over all tokens/pairs in the selected blocks
B) Conditional on the pair being the block argmax contributor for that token ("dominant-pair tokens")
   which is more aligned with the scenario where that pair determines the block scale.

Usage:
  python3 analyze_late_minor_vs_step.py rope_values_nofuse_wikitest.bin \
    --output-dir out --plot --qmax 7.0 --zero-thresh 0.5

"""

import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_ropv_header(f):
    magic = struct.unpack('I', f.read(4))[0]
    if magic != MAGIC_ROPV:
        raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{MAGIC_ROPV:08X}")
    version = struct.unpack('I', f.read(4))[0]
    n_layers = struct.unpack('I', f.read(4))[0]
    n_heads = struct.unpack('I', f.read(4))[0]
    n_dims = struct.unpack('I', f.read(4))[0]
    n_tokens = struct.unpack('I', f.read(4))[0]
    return version, n_layers, n_heads, n_dims, n_tokens


def cdf_vals(x: np.ndarray):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.linspace(0, 1, xs.size, endpoint=True)
    return xs, ys


def maybe_subsample(x: np.ndarray, max_samples: int, rng: np.random.Generator):
    if x.size <= max_samples:
        return x
    idx = rng.choice(x.size, size=max_samples, replace=False)
    return x[idx]


def analyze_file(
    file_path: str,
    blocks,
    qmax: float,
    zero_thresh: float,
    max_samples: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    # Accumulators
    out = {
        "mr_all": [],
        "minor_over_delta_all": [],
        "zero_like_all": [],

        "mr_dom": [],
        "minor_over_delta_dom": [],
        "zero_like_dom": [],

        "meta": {},
        "per_block": {}  # optional summaries per (block_start,block_end)
    }

    with open(file_path, "rb") as f:
        version, n_layers, n_heads, n_dims, n_tokens = read_ropv_header(f)
        out["meta"] = {
            "file": file_path,
            "version": version,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_dims": n_dims,
            "n_tokens_header": n_tokens,
            "blocks": blocks,
            "qmax": qmax,
            "zero_thresh": zero_thresh,
            "max_samples": max_samples,
        }

        for layer_idx in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            # ✅ 여기 pre_values를 읽어서 사용
            vals = np.frombuffer(f.read(pre_count * 4), dtype=np.float32) if pre_count > 0 else None

            post_count = struct.unpack('I', f.read(4))[0]
            # ✅ post_values는 스킵(내용은 있지만 쓸 필요 없음)
            if post_count > 0:
                f.seek(post_count * 4, 1)

            if vals is None or vals.size == 0:
                continue

            n_tokens_layer = vals.size // (n_heads * n_dims)
            if n_tokens_layer == 0:
                continue

            post = vals.reshape(n_tokens_layer, n_heads, n_dims)  # ✅ 변수명은 post로 둬도 됨(실제 post-RoPE)
            post_abs = np.abs(post)

            for head_idx in range(n_heads):
                head_vals = post_abs[:, head_idx, :]  # [T, D]
                T = head_vals.shape[0]

                for (b_start, b_end) in blocks:
                    block = head_vals[:, b_start:b_end]  # [T, 32]
                    if block.size == 0:
                        continue

                    # token-wise block absmax and modeled quant step
                    block_absmax = block.max(axis=1)  # [T]
                    # avoid zeros
                    block_absmax = np.where(block_absmax > 1e-12, block_absmax, 1e-12)
                    delta = block_absmax / qmax  # [T]

                    # reshape to pairs inside block: [T, n_pairs, 2]
                    # b_start should be even for clean pairing; if odd, we drop the last dim
                    width = b_end - b_start
                    n_pairs = width // 2
                    if n_pairs <= 0:
                        continue
                    pair = block[:, : (n_pairs * 2)].reshape(T, n_pairs, 2)  # [T, P, 2]
                    even = pair[:, :, 0]
                    odd = pair[:, :, 1]
                    major = np.maximum(even, odd)
                    minor = np.minimum(even, odd)

                    # Magnitude Ratio MR = major/minor (safe)
                    safe_minor = np.where(minor > 1e-12, minor, 1e-12)
                    mr = major / safe_minor  # [T, P]

                    # minor / Δ(t)  (Δ depends on token, broadcast)
                    minor_over_delta = minor / delta[:, None]  # [T, P]

                    # "zero-like" under threshold k*Δ
                    zero_like = minor < (zero_thresh * delta[:, None])  # [T, P] bool

                    # Condition: pair is dominant contributor to block absmax for that token
                    # i.e., its major equals block_absmax (within tolerance).
                    # Use relative tolerance to handle float noise.
                    dom = np.isclose(major, block_absmax[:, None], rtol=1e-6, atol=1e-6)  # [T, P]

                    # Flatten for accumulation (subsample to keep memory bounded)
                    mr_flat = mr.reshape(-1)
                    mod_flat = minor_over_delta.reshape(-1)
                    zl_flat = zero_like.reshape(-1).astype(np.float32)

                    dom_mr_flat = mr[dom]
                    dom_mod_flat = minor_over_delta[dom]
                    dom_zl_flat = zero_like[dom].astype(np.float32)

                    mr_flat = maybe_subsample(mr_flat, max_samples, rng)
                    mod_flat = maybe_subsample(mod_flat, max_samples, rng)
                    zl_flat = maybe_subsample(zl_flat, max_samples, rng)

                    dom_mr_flat = maybe_subsample(dom_mr_flat, max_samples, rng)
                    dom_mod_flat = maybe_subsample(dom_mod_flat, max_samples, rng)
                    dom_zl_flat = maybe_subsample(dom_zl_flat, max_samples, rng)

                    out["mr_all"].append(mr_flat)
                    out["minor_over_delta_all"].append(mod_flat)
                    out["zero_like_all"].append(zl_flat)

                    out["mr_dom"].append(dom_mr_flat)
                    out["minor_over_delta_dom"].append(dom_mod_flat)
                    out["zero_like_dom"].append(dom_zl_flat)

                    key = f"L{layer_idx}_H{head_idx}_B{b_start}-{b_end}"
                    # Lightweight per-block summary (no sampling needed)
                    per = out["per_block"].setdefault(key, {})
                    per["mean_block_absmax"] = float(np.mean(block_absmax))
                    per["p99_block_absmax"] = float(np.percentile(block_absmax, 99))
                    per["mean_delta"] = float(np.mean(delta))
                    per["dom_token_frac"] = float(np.mean(np.any(dom, axis=1)))  # token fraction with some dominant pair (should be ~1)
                    # Zero-like among dominant-pair entries
                    if dom_zl_flat.size > 0:
                        per["zero_like_dom_frac"] = float(np.mean(dom_zl_flat))
                    else:
                        per["zero_like_dom_frac"] = float("nan")

    # Concatenate
    def cat(xs):
        xs = [x for x in xs if x is not None and np.size(x) > 0]
        return np.concatenate(xs) if xs else np.array([], dtype=np.float32)

    out["mr_all"] = cat(out["mr_all"])
    out["minor_over_delta_all"] = cat(out["minor_over_delta_all"])
    out["zero_like_all"] = cat(out["zero_like_all"])

    out["mr_dom"] = cat(out["mr_dom"])
    out["minor_over_delta_dom"] = cat(out["minor_over_delta_dom"])
    out["zero_like_dom"] = cat(out["zero_like_dom"])

    return out


def summarize(arr: np.ndarray, name: str):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{name}: (empty)"
    return (
        f"{name}: count={arr.size:,}, "
        f"mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
        f"p90={np.percentile(arr,90):.4f}, p99={np.percentile(arr,99):.4f}"
    )


def plot_and_save(out, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    mr_all = out["mr_all"]
    mod_all = out["minor_over_delta_all"]
    zl_all = out["zero_like_all"]

    mr_dom = out["mr_dom"]
    mod_dom = out["minor_over_delta_dom"]
    zl_dom = out["zero_like_dom"]

    # Print headline stats
    print("\n=== GLOBAL SUMMARY (late blocks) ===")
    print(summarize(mr_all, "MR (all pairs/tokens)"))
    print(summarize(mod_all, "|minor|/Δ (all pairs/tokens)"))
    if zl_all.size:
        print(f"P(|minor| < {out['meta']['zero_thresh']}Δ) (all) = {np.mean(zl_all):.4f}")
    print()
    print(summarize(mr_dom, "MR (dominant-pair tokens)"))
    print(summarize(mod_dom, "|minor|/Δ (dominant-pair tokens)"))
    if zl_dom.size:
        print(f"P(|minor| < {out['meta']['zero_thresh']}Δ) (dominant) = {np.mean(zl_dom):.4f}")

    # 1) MR histogram (log-x helps)
    plt.figure(figsize=(10, 6))
    if mr_dom.size:
        plt.hist(np.clip(mr_dom, 1, 1e3), bins=80, alpha=0.6, label="dominant-pair tokens")
    if mr_all.size:
        plt.hist(np.clip(mr_all, 1, 1e3), bins=80, alpha=0.4, label="all pairs/tokens")
    plt.xscale("log")
    plt.xlabel("MR = major/minor (clipped to [1, 1e3], log scale)")
    plt.ylabel("Count")
    plt.title("Late blocks: Magnitude Ratio (MR) distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mr_hist.png", dpi=150)
    plt.close()

    # 2) |minor|/Δ histogram
    plt.figure(figsize=(10, 6))
    if mod_dom.size:
        plt.hist(np.clip(mod_dom, 0, 20), bins=80, alpha=0.6, label="dominant-pair tokens")
    if mod_all.size:
        plt.hist(np.clip(mod_all, 0, 20), bins=80, alpha=0.4, label="all pairs/tokens")
    plt.xlabel("|minor| / Δ  (clipped to [0, 20])")
    plt.ylabel("Count")
    plt.title("Late blocks: minor relative to quant step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "minor_over_delta_hist.png", dpi=150)
    plt.close()

    # 3) CDF of |minor|/Δ
    plt.figure(figsize=(10, 6))
    xs, ys = cdf_vals(mod_dom)
    if xs.size:
        plt.plot(xs, ys, label="dominant-pair tokens", linewidth=2)
    xs2, ys2 = cdf_vals(mod_all)
    if xs2.size:
        plt.plot(xs2, ys2, label="all pairs/tokens", linewidth=2, alpha=0.7)
    plt.xlabel("|minor| / Δ")
    plt.ylabel("CDF")
    plt.title("Late blocks: CDF of |minor|/Δ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "minor_over_delta_cdf.png", dpi=150)
    plt.close()

    # 4) Zero-like fraction bar
    plt.figure(figsize=(7, 5))
    vals = []
    labels = []
    if zl_dom.size:
        vals.append(float(np.mean(zl_dom)))
        labels.append("dominant")
    if zl_all.size:
        vals.append(float(np.mean(zl_all)))
        labels.append("all")
    plt.bar(labels, [v * 100 for v in vals])
    plt.ylabel(f"P(|minor| < {out['meta']['zero_thresh']}Δ) (%)")
    plt.title("Late blocks: zero-like fraction of minor channel")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "zero_like_frac.png", dpi=150)
    plt.close()

    # Save json summary (compact)
    summary = {
        "meta": out["meta"],
        "global": {
            "mr_all": {
                "mean": float(np.mean(mr_all)) if mr_all.size else None,
                "median": float(np.median(mr_all)) if mr_all.size else None,
                "p99": float(np.percentile(mr_all, 99)) if mr_all.size else None,
            },
            "minor_over_delta_all": {
                "mean": float(np.mean(mod_all)) if mod_all.size else None,
                "median": float(np.median(mod_all)) if mod_all.size else None,
                "p10": float(np.percentile(mod_all, 10)) if mod_all.size else None,
            },
            "zero_like_all": float(np.mean(zl_all)) if zl_all.size else None,
            "mr_dom": {
                "mean": float(np.mean(mr_dom)) if mr_dom.size else None,
                "median": float(np.median(mr_dom)) if mr_dom.size else None,
                "p99": float(np.percentile(mr_dom, 99)) if mr_dom.size else None,
            },
            "minor_over_delta_dom": {
                "mean": float(np.mean(mod_dom)) if mod_dom.size else None,
                "median": float(np.median(mod_dom)) if mod_dom.size else None,
                "p10": float(np.percentile(mod_dom, 10)) if mod_dom.size else None,
            },
            "zero_like_dom": float(np.mean(zl_dom)) if zl_dom.size else None,
        },
    }
    import json
    with open(output_dir / "late_minor_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved plots + summary JSON to: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_path", help="ROPV file path (post-RoPE values)")
    ap.add_argument("--output-dir", default="late_minor_analysis_out", help="output directory")
    ap.add_argument("--qmax", type=float, default=7.0, help="Q4_0 max integer (approx). Try 7.0 or 7.5")
    ap.add_argument("--zero-thresh", type=float, default=0.5, help="threshold k in |minor| < k*Δ for 'zero-like'")
    ap.add_argument("--max-samples", type=int, default=500000, help="max samples kept per metric (for memory)")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for subsampling")
    ap.add_argument("--plot", action="store_true", help="save plots")
    args = ap.parse_args()

    # late blocks (dims 64-95, 96-127)
    blocks = [(64, 96), (96, 128)]

    out = analyze_file(
        file_path=args.file_path,
        blocks=blocks,
        qmax=args.qmax,
        zero_thresh=args.zero_thresh,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    if args.plot:
        plot_and_save(out, output_dir)
    else:
        # print quick summary only
        print(summarize(out["mr_dom"], "MR (dominant-pair tokens)"))
        print(summarize(out["minor_over_delta_dom"], "|minor|/Δ (dominant-pair tokens)"))
        if out["zero_like_dom"].size:
            print(f"P(|minor| < {args.zero_thresh}Δ) (dominant) = {np.mean(out['zero_like_dom']):.4f}")


if __name__ == "__main__":
    main()