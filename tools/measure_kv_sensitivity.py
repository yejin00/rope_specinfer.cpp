#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

CASE_TABLE = [
    {"name": "bf16_to_q4",   "baseline_type": "bf16",      "probe_type": "q4_0_head"},
    {"name": "bf16_to_q2q4", "baseline_type": "bf16",      "probe_type": "q2_0_q4_0_head"},
    {"name": "bf16_to_q2",   "baseline_type": "bf16",      "probe_type": "q2_0_head"},
    {"name": "q4_to_q2q4",   "baseline_type": "q4_0_head", "probe_type": "q2_0_q4_0_head"},
    {"name": "q4_to_q2",     "baseline_type": "q4_0_head", "probe_type": "q2_0_head"},
]

TYPE_TAGS = {
    "bf16": "bf16",
    "q4_0_head": "q4",
    "q3_0_head": "q3",
    "q2_0_head": "q2",
    "q2_0_q4_0_head": "q2q4",
    "q4_0_q2_0_head": "q4q2",
}


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Stage-A KVTuner-style layer-wise KV sensitivity measurement with llama-cli instrumentation."
    )
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--repo-root", default=str(repo_root_default), help="repo root containing build-cuda/bin/llama-cli")
    parser.add_argument("--llama-cli-path", default=None, help="explicit llama-cli path")
    parser.add_argument("--gsm8k-path", default=None, help="local GSM8K JSON/JSONL path; if omitted, tries datasets.load_dataset('gsm8k')")
    parser.add_argument("--gsm8k-split", default="train", help="dataset split name for HF datasets or top-level JSON key")
    parser.add_argument("--gsm8k-subset-size", type=int, default=100, help="number of GSM8K samples for calibration in each case")
    parser.add_argument("--seed", type=int, default=0, help="deterministic subset seed")
    parser.add_argument("--baseline-type", default="bf16", help="legacy single-case baseline KV type; used only when --probe-type is set")
    parser.add_argument("--probe-type", default=None, help="legacy single-case probe KV type; if set, only one case is run")
    parser.add_argument("--measure-v-sensitivity", action="store_true", help="measure V-only sensitivity with fixed K and eo-centered ranking")
    parser.add_argument("--k-fixed-type", default="q4_0_head", help="fixed K type for V-only sensitivity mode")
    parser.add_argument("--v-baseline-type", default="q4_0_head", help="baseline V type for V-only sensitivity mode")
    parser.add_argument("--v-probe-type", default="q2_0_head", help="probe V type for V-only sensitivity mode")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--ctx-size", type=int, default=4096, help="ctx size for llama-cli")
    parser.add_argument("--batch-size", type=int, default=4096, help="batch size for llama-cli")
    parser.add_argument("--ubatch-size", type=int, default=4096, help="ubatch size for llama-cli")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="ngl value for llama-cli")
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on", help="flash attention mode")
    parser.add_argument("--n-layers", type=int, default=None, help="override model layer count")
    parser.add_argument("--prompt-prefix", default="Solve the following grade school math problem carefully.\n\nQuestion:\n", help="prefix added before each GSM8K question")
    parser.add_argument("--prompt-suffix", default="\n\nAnswer:\nLet's think step by step.\n", help="suffix added after each GSM8K question")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra llama-cli arg; may be repeated")
    return parser.parse_args()


def load_gsm8k_local(path: Path, split: str):
    if not path.exists():
        raise FileNotFoundError(f"GSM8K path does not exist: {path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fin:
            return [json.loads(line) for line in fin if line.strip()]

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fin:
            data = json.load(fin)
        if isinstance(data, dict):
            if split not in data:
                raise KeyError(f"split '{split}' not found in {path}")
            return data[split]
        if isinstance(data, list):
            return data
        raise ValueError(f"unsupported JSON structure in {path}")

    raise ValueError(f"unsupported GSM8K file format: {path}")


def load_gsm8k(path: Optional[str], split: str):
    if path:
        return load_gsm8k_local(Path(path), split)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not available and --gsm8k-path was not provided. "
            "Install datasets or pass a local GSM8K JSON/JSONL path."
        ) from exc

    ds = load_dataset("gsm8k", "main", split=split)
    return [dict(row) for row in ds]


def normalize_samples(samples):
    normalized = []
    for i, sample in enumerate(samples):
        question = sample.get("question")
        answer = sample.get("answer", "")
        system_prompt = sample.get("system_prompt", "")

        if question is None and "messages" in sample:
            messages = sample["messages"]
            if not isinstance(messages, list):
                raise ValueError(f"sample {i} has non-list 'messages'")

            system_parts = []
            user_parts = []
            assistant_parts = []

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "system":
                    system_parts.append(content)
                elif role == "user":
                    user_parts.append(content)
                elif role == "assistant":
                    assistant_parts.append(content)

            if user_parts:
                question = user_parts[-1]
            if not answer:
                answer = sample.get("rendered_answer") or (assistant_parts[-1] if assistant_parts else "")
            if system_parts:
                system_prompt = "\n\n".join(system_parts)

        if not question:
            raise ValueError(f"sample {i} is missing 'question' or usable 'messages'")
        normalized.append({
            "id": sample.get("id", i),
            "question": question,
            "answer": answer,
            "system_prompt": system_prompt,
        })
    return normalized


def select_subset(samples, subset_size: int, seed: int):
    if subset_size <= 0:
        raise ValueError("--gsm8k-subset-size must be positive")
    if subset_size > len(samples):
        raise ValueError(f"requested {subset_size} samples, but dataset has only {len(samples)}")

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(samples)), subset_size))
    return [(idx, samples[idx]) for idx in indices]


def format_prompt(question: str, prefix: str, suffix: str, system_prompt: str = "") -> str:
    if system_prompt:
        return f"{system_prompt}\n\nQuestion:\n{question}{suffix}"
    return f"{prefix}{question}{suffix}"


def infer_llama_cli_path(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = repo_root / "build-cuda" / "bin" / "llama-cli"

    if not path.exists():
        raise FileNotFoundError(f"llama-cli not found: {path}")

    return path


def infer_n_layers(args, llama_cli: Path) -> int:
    if args.n_layers is not None:
        return args.n_layers

    cmd = [
        str(llama_cli),
        "-m", args.model,
        "-p", "hi",
        "-n", "0",
        "-no-cnv",
        "--no-display-prompt",
        "--simple-io",
        "--no-warmup",
        "--ctx-size", "32",
        "-b", "32",
        "-ub", "32",
        "-ngl", str(args.n_gpu_layers),
    ]

    proc = subprocess.run(
        cmd,
        cwd=args.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to infer n_layers:\n{proc.stdout}")

    match = re.search(r"n_layer\s*=\s*(\d+)", proc.stdout)
    if not match:
        raise RuntimeError(f"could not infer n_layers from llama-cli output:\n{proc.stdout}")

    return int(match.group(1))


def type_tag(type_name: str) -> str:
    if type_name in TYPE_TAGS:
        return TYPE_TAGS[type_name]
    return re.sub(r"[^a-z0-9]+", "", type_name.lower())


def case_metadata(case):
    mode = case.get("mode", "legacy_kv")
    if mode == "v_only":
        return {
            "mode": mode,
            "k_fixed_type": case["k_fixed_type"],
            "baseline_v_type": case["baseline_v_type"],
            "probe_v_type": case["probe_v_type"],
        }

    return {
        "mode": mode,
        "baseline_type": case["baseline_type"],
        "probe_type": case["probe_type"],
    }


def run_layer_probe(
    args,
    llama_cli: Path,
    prompt_path: Path,
    dump_path: Path,
    layer: int,
    case,
):
    cmd = [
        str(llama_cli),
        "-m", args.model,
        "-f", str(prompt_path),
        "-n", "1",
        "-no-cnv",
        "--no-display-prompt",
        "--simple-io",
        "--no-warmup",
        "--ctx-size", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ngl", str(args.n_gpu_layers),
        "-fa", args.flash_attn,
        "--measure-kv-sensitivity",
        "--sensitivity-layer", str(layer),
        "--dump-attn-error", str(dump_path),
    ]

    if case.get("mode") == "v_only":
        cmd.extend([
            "-ctk", case["k_fixed_type"],
            "-ctv", case["baseline_v_type"],
            "--kv-layer-v-types", f"{layer}:{case['probe_v_type']}",
            "--sensitivity-baseline-k-type", case["k_fixed_type"],
            "--sensitivity-baseline-v-type", case["baseline_v_type"],
            "--sensitivity-probe-k-type", case["k_fixed_type"],
            "--sensitivity-probe-v-type", case["probe_v_type"],
        ])
    else:
        layer_spec = f"{layer}:{case['probe_type']}"
        cmd.extend([
            "-ctk", case["baseline_type"],
            "-ctv", case["baseline_type"],
            "--kv-layer-k-types", layer_spec,
            "--kv-layer-v-types", layer_spec,
            "--sensitivity-baseline-type", case["baseline_type"],
            "--sensitivity-probe-type", case["probe_type"],
        ])

    cmd.extend(args.extra_arg)

    proc = subprocess.run(
        cmd,
        cwd=args.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed for layer {layer} and prompt {prompt_path.name}:\n{proc.stdout}"
        )

    if not dump_path.exists():
        raise RuntimeError(
            f"expected KV sensitivity dump was not created: {dump_path}\nllama-cli output:\n{proc.stdout}"
        )

    with dump_path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)

    if data.get("score_captures", 0) <= 0 or data.get("output_captures", 0) <= 0:
        raise RuntimeError(
            "KV sensitivity instrumentation produced zero captures. "
            "This usually means the prompt was split across multiple prefill passes. "
            "Increase --batch-size/--ubatch-size or shorten the prompt."
        )

    return data, proc.stdout


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for Stage-A sensitivity plots. "
            "Install matplotlib in the active Python environment."
        ) from exc

    return plt


def aggregate_by_layer(records, n_layers: int):
    grouped = {layer: [] for layer in range(n_layers)}
    for rec in records:
        grouped[rec["layer"]].append(rec)

    summary = []
    for layer in range(n_layers):
        rows = grouped[layer]
        if not rows:
            continue

        def mean(key):
            return sum(r[key] for r in rows) / len(rows)

        def std(key, mu):
            return math.sqrt(sum((r[key] - mu) ** 2 for r in rows) / len(rows))

        delta_eos = sorted(r["delta_eo"] for r in rows)
        mid = len(delta_eos) // 2
        median_delta_eo = delta_eos[mid] if len(delta_eos) % 2 == 1 else 0.5 * (delta_eos[mid - 1] + delta_eos[mid])
        mu_delta_eo = mean("delta_eo")
        summary.append({
            "layer": layer,
            "count": len(rows),
            "mean_baseline_ea": mean("baseline_ea"),
            "mean_probe_ea": mean("probe_ea"),
            "mean_delta_ea": mean("delta_ea"),
            "mean_baseline_eo": mean("baseline_eo"),
            "mean_probe_eo": mean("probe_eo"),
            "mean_delta_eo": mu_delta_eo,
            "std_delta_eo": std("delta_eo", mu_delta_eo),
            "median_delta_eo": median_delta_eo,
            "min_delta_eo": delta_eos[0],
            "max_delta_eo": delta_eos[-1],
        })

    return summary


def plot_layerwise_probe_eo(records, summary, out_path: Path):
    plt = load_pyplot()
    by_sample = {}
    for rec in records:
        by_sample.setdefault(rec["sample_id"], {})[rec["layer"]] = rec["probe_eo"]

    plt.figure(figsize=(10, 4.5))
    for sample_id, layer_map in sorted(by_sample.items()):
        xs = sorted(layer_map.keys())
        ys = [layer_map[x] for x in xs]
        plt.plot(xs, ys, linewidth=1.0, alpha=0.35)

    mean_x = [row["layer"] for row in summary]
    mean_y = [row["mean_probe_eo"] for row in summary]
    plt.plot(mean_x, mean_y, color="black", linewidth=2.4, label="mean probe eo")
    plt.xlabel("Layer")
    plt.ylabel("Attention output relative error (eo)")
    plt.title("Layer-wise probe eo across GSM8K calibration prompts")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_delta_eo_bar(summary, out_path: Path):
    plt = load_pyplot()
    xs = [row["layer"] for row in summary]
    ys = [row["mean_delta_eo"] for row in summary]
    plt.figure(figsize=(10, 4.5))
    plt.bar(xs, ys, color="#4C78A8")
    plt.xlabel("Layer")
    plt.ylabel("Mean delta eo")
    plt.title("Layer-wise delta eo (probe - baseline)")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ranking(summary, out_path: Path):
    plt = load_pyplot()
    ranked = sorted(summary, key=lambda row: row["mean_delta_eo"], reverse=True)
    labels = [str(row["layer"]) for row in ranked]
    ys = [row["mean_delta_eo"] for row in ranked]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(ranked)), ys, color="#F58518")
    plt.xticks(range(len(ranked)), labels, rotation=90)
    plt.xlabel("Layer (sorted)")
    plt.ylabel("Mean delta eo")
    plt.title("Layer sensitivity ranking")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def determine_cases(args):
    if args.measure_v_sensitivity:
        if args.probe_type:
            raise ValueError("--measure-v-sensitivity cannot be combined with legacy --probe-type")

        return [{
            "name": f"v_{type_tag(args.v_baseline_type)}to{type_tag(args.v_probe_type)}",
            "mode": "v_only",
            "k_fixed_type": args.k_fixed_type,
            "baseline_v_type": args.v_baseline_type,
            "probe_v_type": args.v_probe_type,
        }]

    if args.probe_type:
        case_name = f"{args.baseline_type}_to_{args.probe_type}".replace("/", "_")
        return [{
            "name": case_name,
            "mode": "legacy_kv",
            "baseline_type": args.baseline_type,
            "probe_type": args.probe_type,
        }]

    return [{"mode": "legacy_kv", **case} for case in CASE_TABLE]


def write_case_outputs(
    args,
    case,
    output_dir: Path,
    raw_records,
    raw_json_records,
    summary,
    ranking,
    n_layers: int,
):
    layer_errors_csv = output_dir / "layer_errors.csv"
    layer_errors_json = output_dir / "layer_errors.json"
    layer_ranking_csv = output_dir / "layer_ranking.csv"
    sensitivity_plot_png = output_dir / "sensitivity_plot.png"
    delta_eo_bar_png = output_dir / "delta_eo_bar.png"
    ranking_plot_png = output_dir / "layer_ranking_bar.png"
    summary_json = output_dir / "summary.json"
    measure_config_json = output_dir / "measure_config.json"

    write_csv(
        layer_errors_csv,
        raw_records,
        [
            "sample_id",
            "dataset_index",
            "layer",
            "baseline_ea",
            "probe_ea",
            "delta_ea",
            "baseline_eo",
            "probe_eo",
            "delta_eo",
            "score_captures",
            "output_captures",
        ],
    )
    write_csv(
        layer_ranking_csv,
        ranking,
        [
            "layer",
            "count",
            "mean_baseline_ea",
            "mean_probe_ea",
            "mean_delta_ea",
            "mean_baseline_eo",
            "mean_probe_eo",
            "mean_delta_eo",
            "std_delta_eo",
            "median_delta_eo",
            "min_delta_eo",
            "max_delta_eo",
        ],
    )

    metadata = case_metadata(case)
    with layer_errors_json.open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "model": args.model,
                "case": case["name"],
                "subset_size": args.gsm8k_subset_size,
                "seed": args.seed,
                **metadata,
                "n_layers": n_layers,
                "records": raw_json_records,
                "summary": summary,
            },
            fout,
            indent=2,
        )

    case_summary = {
        "case": case["name"],
        "model": args.model,
        "subset_size": args.gsm8k_subset_size,
        "seed": args.seed,
        **metadata,
        "n_layers": n_layers,
        "top_10_layers": [
            {
                "rank": rank + 1,
                "layer": row["layer"],
                "mean_delta_eo": row["mean_delta_eo"],
                "std_delta_eo": row["std_delta_eo"],
                "median_delta_eo": row["median_delta_eo"],
            }
            for rank, row in enumerate(ranking[:10])
        ],
        "files": {
            "layer_errors_csv": layer_errors_csv.name,
            "layer_errors_json": layer_errors_json.name,
            "layer_ranking_csv": layer_ranking_csv.name,
            "sensitivity_plot_png": sensitivity_plot_png.name,
            "delta_eo_bar_png": delta_eo_bar_png.name,
            "layer_ranking_bar_png": ranking_plot_png.name,
        },
    }

    with summary_json.open("w", encoding="utf-8") as fout:
        json.dump(case_summary, fout, indent=2)

    with measure_config_json.open("w", encoding="utf-8") as fout:
        json.dump(
            {
                **vars(args),
                "case": case["name"],
                **metadata,
            },
            fout,
            indent=2,
        )

    plot_layerwise_probe_eo(raw_records, summary, sensitivity_plot_png)
    plot_delta_eo_bar(summary, delta_eo_bar_png)
    plot_ranking(summary, ranking_plot_png)

    print(f"[measure_kv_sensitivity] wrote {layer_errors_csv}")
    print(f"[measure_kv_sensitivity] wrote {layer_errors_json}")
    print(f"[measure_kv_sensitivity] wrote {layer_ranking_csv}")
    print(f"[measure_kv_sensitivity] wrote {sensitivity_plot_png}")
    print(f"[measure_kv_sensitivity] wrote {delta_eo_bar_png}")
    print(f"[measure_kv_sensitivity] wrote {ranking_plot_png}")
    print(f"[measure_kv_sensitivity] wrote {summary_json}")

    return case_summary


def run_case(
    args,
    llama_cli: Path,
    subset,
    n_layers: int,
    case,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    prompt_dir = output_dir / "prompts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    raw_records = []
    raw_json_records = []

    for sample_pos, (dataset_idx, sample) in enumerate(subset):
        prompt = format_prompt(
            sample["question"],
            args.prompt_prefix,
            args.prompt_suffix,
            sample.get("system_prompt", ""),
        )
        prompt_path = prompt_dir / f"prompt_{sample_pos:03d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        for layer in range(n_layers):
            dump_path = raw_dir / f"sample_{sample_pos:03d}_layer_{layer:03d}.json"
            metrics, llama_output = run_layer_probe(
                args,
                llama_cli,
                prompt_path,
                dump_path,
                layer,
                case,
            )

            record = {
                "sample_id": sample_pos,
                "dataset_index": dataset_idx,
                "layer": layer,
                "baseline_ea": metrics["baseline_ea"],
                "probe_ea": metrics["probe_ea"],
                "delta_ea": metrics["delta_ea"],
                "baseline_eo": metrics["baseline_eo"],
                "probe_eo": metrics["probe_eo"],
                "delta_eo": metrics["delta_eo"],
                "score_captures": metrics["score_captures"],
                "output_captures": metrics["output_captures"],
            }
            raw_records.append(record)

            raw_json_records.append({
                **record,
                "question": sample["question"],
                "answer": sample["answer"],
                **case_metadata(case),
                "llama_output_tail": llama_output.splitlines()[-20:],
            })

            print(
                f"[measure_kv_sensitivity] case={case['name']} "
                f"sample={sample_pos + 1}/{len(subset)} "
                f"layer={layer + 1}/{n_layers} "
                f"delta_eo={record['delta_eo']:.6f}"
            )

    summary = aggregate_by_layer(raw_records, n_layers)
    ranking = sorted(summary, key=lambda row: row["mean_delta_eo"], reverse=True)
    return write_case_outputs(args, case, output_dir, raw_records, raw_json_records, summary, ranking, n_layers)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    llama_cli = infer_llama_cli_path(repo_root, args.llama_cli_path)
    samples = normalize_samples(load_gsm8k(args.gsm8k_path, args.gsm8k_split))
    subset = select_subset(samples, args.gsm8k_subset_size, args.seed)
    n_layers = infer_n_layers(args, llama_cli)
    cases = determine_cases(args)
    for case in cases:
        (output_root / case["name"]).mkdir(parents=True, exist_ok=True)

    master_summary = {
        "model": args.model,
        "subset_size": args.gsm8k_subset_size,
        "seed": args.seed,
        "auto_case_mode": (not args.measure_v_sensitivity) and args.probe_type is None,
        "measure_v_sensitivity": args.measure_v_sensitivity,
        "n_layers": n_layers,
        "cases": [],
    }

    for case_index, case in enumerate(cases):
        print(
            f"[measure_kv_sensitivity] starting case {case_index + 1}/{len(cases)}: "
            f"{case['name']} ({json.dumps(case_metadata(case), sort_keys=True)})"
        )
        case_output_dir = output_root / case["name"]
        case_summary = run_case(args, llama_cli, subset, n_layers, case, case_output_dir)
        master_summary["cases"].append({
            **case_summary,
            "output_dir": str(case_output_dir),
        })

    master_summary_path = output_root / "master_summary.json"
    with master_summary_path.open("w", encoding="utf-8") as fout:
        json.dump(master_summary, fout, indent=2)

    print(f"[measure_kv_sensitivity] wrote {master_summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"measure_kv_sensitivity.py: error: {exc}", file=sys.stderr)
        sys.exit(1)
