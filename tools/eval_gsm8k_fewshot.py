#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional


DEFAULT_SYSTEM_PROMPT = (
    'Answer the grade school math word problem below, using step-by-step '
    'problem-solving process. Print the final answer after "####".'
)

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
HASH_ANSWER_RE = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K few-shot accuracy via llama-cli with optional KV mixed-precision presets."
    )
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--repo-root", default=str(repo_root_default), help="repo root containing llama-cli builds")
    parser.add_argument("--llama-cli-path", default=None, help="explicit llama-cli path")
    parser.add_argument("--gsm8k-path", default=None, help="local GSM8K JSON/JSONL path; if omitted, tries datasets.load_dataset('gsm8k')")
    parser.add_argument("--gsm8k-split", default="train", help="dataset split name for HF datasets or top-level JSON key")
    parser.add_argument("--fewshot-path", default=None, help="local GSM8K JSON/JSONL path used only for few-shot exemplars")
    parser.add_argument("--fewshot-split", default="train", help="dataset split name for few-shot exemplars")
    parser.add_argument("--eval-path", default=None, help="local GSM8K JSON/JSONL path used only for evaluation samples")
    parser.add_argument("--eval-split", default="test", help="dataset split name for evaluation samples")
    parser.add_argument("--shots", nargs="+", type=int, default=[4, 8, 16], help="few-shot counts to evaluate")
    parser.add_argument("--num-samples", type=int, default=200, help="number of evaluation samples")
    parser.add_argument("--seed", type=int, default=42, help="seed for exemplar and evaluation subset selection")
    parser.add_argument(
        "--prompt-format",
        choices=["auto", "plain", "chat-template"],
        default="auto",
        help="few-shot prompt formatting style; auto prefers chat-template for instruct/chat models",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="max generated tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature for llama-cli")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p for llama-cli")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--ctx-size", type=int, default=4096, help="ctx size for llama-cli")
    parser.add_argument("--batch-size", type=int, default=4096, help="batch size for llama-cli")
    parser.add_argument("--ubatch-size", type=int, default=4096, help="ubatch size for llama-cli")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="ngl value for llama-cli")
    parser.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on", help="flash attention mode")
    parser.add_argument("--hadamard", action="store_true", help="enable Hadamard rotation in llama-cli")
    parser.add_argument("--hadamard-seed", type=int, default=0, help="Hadamard seed passed to llama-cli")
    parser.add_argument(
        "--hadamard-granularity",
        choices=["layer", "head"],
        default="head",
        help="Hadamard granularity passed to llama-cli",
    )
    parser.add_argument("--ctk", default=None, help="KV cache type for K")
    parser.add_argument("--ctv", default=None, help="KV cache type for V")
    parser.add_argument("--kv-layer-k-types", default=None, help="layer-wise KV cache overrides for K")
    parser.add_argument("--kv-layer-v-types", default=None, help="layer-wise KV cache overrides for V")
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


def normalize_numeric_token(token: str) -> Optional[str]:
    if token is None:
        return None

    cleaned = token.strip()
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.rstrip(".")

    if not cleaned:
        return None

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def extract_final_numeric_answer(text: str):
    if not text:
        return None, "none"

    cleaned = text.replace("\u2212", "-")

    hash_matches = HASH_ANSWER_RE.findall(cleaned)
    if hash_matches:
        answer = normalize_numeric_token(hash_matches[0])
        if answer is not None:
            return answer, "hash"

    matches = NUMBER_RE.findall(cleaned)
    if matches:
        answer = normalize_numeric_token(matches[-1])
        if answer is not None:
            return answer, "last_number"

    return None, "missing"


def normalize_samples(samples):
    normalized = []
    for i, sample in enumerate(samples):
        question = sample.get("question")
        if question is None:
            question = sample.get("problem")
        if question is None:
            question = sample.get("input")

        answer_text = sample.get("answer", "")
        if not answer_text:
            answer_text = sample.get("response", "")
        if not answer_text:
            answer_text = sample.get("output", "")
        rendered_answer = sample.get("rendered_answer", "")
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
            if assistant_parts:
                answer_text = assistant_parts[-1]
            if system_parts:
                system_prompt = "\n\n".join(system_parts)

        if not question:
            raise ValueError(f"sample {i} is missing 'question' or usable 'messages'")

        gold_answer, gold_parse_strategy = extract_final_numeric_answer(rendered_answer or answer_text)
        if gold_answer is None:
            raise ValueError(f"sample {i} is missing a parseable numeric answer")

        normalized.append({
            "id": sample.get("id", i),
            "dataset_index": i,
            "question": question.strip(),
            "answer_text": answer_text.strip(),
            "rendered_answer": rendered_answer.strip(),
            "system_prompt": system_prompt.strip(),
            "gold_answer": gold_answer,
            "gold_parse_strategy": gold_parse_strategy,
        })

    return normalized


def resolve_llama_cli_path(repo_root: Path, explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.append(repo_root / "build-cuda" / "bin" / "llama-cli")
        candidates.append(repo_root / "build" / "bin" / "llama-cli")

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "llama-cli not found. Tried: " + ", ".join(str(path) for path in candidates)
    )


def normalize_shots(shots):
    if not shots:
        raise ValueError("--shots must contain at least one non-negative integer")

    result = []
    seen = set()
    for shot in shots:
        if shot < 0:
            raise ValueError(f"invalid shot count: {shot}")
        if shot not in seen:
            result.append(shot)
            seen.add(shot)
    return result


def select_demo_and_eval_sets(samples, max_shots: int, num_samples: int, seed: int):
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    if len(samples) < max_shots + num_samples:
        raise ValueError(
            f"dataset has {len(samples)} samples, but need at least {max_shots + num_samples} "
            f"for max_shots={max_shots} and num_samples={num_samples}"
        )

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    demo_indices = indices[:max_shots]
    eval_indices = indices[max_shots:max_shots + num_samples]

    demos = [samples[idx] for idx in demo_indices]
    eval_subset = [samples[idx] for idx in eval_indices]
    return demos, eval_subset


def select_subset(samples, count: int, seed: int):
    if count < 0:
        raise ValueError("subset size must be non-negative")
    if count == 0:
        return []

    if len(samples) < count:
        raise ValueError(f"dataset has {len(samples)} samples, but need at least {count}")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    chosen_indices = indices[:count]
    return [samples[idx] for idx in chosen_indices]


def resolve_dataset_config(args):
    if args.fewshot_path is None and args.eval_path is None:
        source = {
            "path": args.gsm8k_path,
            "split": args.gsm8k_split,
        }
        return {
            "mode": "shared",
            "fewshot": source,
            "eval": dict(source),
            "same_source": True,
        }

    fewshot = {
        "path": args.fewshot_path if args.fewshot_path is not None else args.gsm8k_path,
        "split": args.fewshot_split,
    }
    eval_source = {
        "path": args.eval_path if args.eval_path is not None else args.gsm8k_path,
        "split": args.eval_split,
    }

    if fewshot["path"] is None and eval_source["path"] is None:
        return {
            "mode": "split",
            "fewshot": fewshot,
            "eval": eval_source,
            "same_source": fewshot["split"] == eval_source["split"],
        }

    if fewshot["path"] is None or eval_source["path"] is None:
        same_source = (
            fewshot["path"] is None
            and eval_source["path"] is None
            and fewshot["split"] == eval_source["split"]
        )
    else:
        same_source = (
            Path(fewshot["path"]).resolve() == Path(eval_source["path"]).resolve()
            and fewshot["split"] == eval_source["split"]
        )

    return {
        "mode": "split",
        "fewshot": fewshot,
        "eval": eval_source,
        "same_source": same_source,
    }


def load_dataset_sources(args, max_shots: int):
    config = resolve_dataset_config(args)

    if config["same_source"]:
        shared_samples = normalize_samples(load_gsm8k(config["eval"]["path"], config["eval"]["split"]))
        if max_shots == 0:
            return config, [], shared_samples
        return config, shared_samples, shared_samples

    fewshot_samples = []
    if max_shots > 0:
        fewshot_samples = normalize_samples(load_gsm8k(config["fewshot"]["path"], config["fewshot"]["split"]))

    eval_samples = normalize_samples(load_gsm8k(config["eval"]["path"], config["eval"]["split"]))
    return config, fewshot_samples, eval_samples


def select_fewshot_and_eval_sets(fewshot_samples, eval_samples, max_shots: int, num_samples: int, seed: int, same_source: bool):
    if max_shots == 0:
        return [], select_subset(eval_samples, num_samples, seed)

    if same_source:
        return select_demo_and_eval_sets(fewshot_samples, max_shots, num_samples, seed)

    demos = select_subset(fewshot_samples, max_shots, seed)
    eval_subset = select_subset(eval_samples, num_samples, seed)
    return demos, eval_subset


def choose_system_prompt(examples, target_sample) -> str:
    prompts = [sample["system_prompt"] for sample in examples + [target_sample] if sample["system_prompt"]]
    if prompts:
        return Counter(prompts).most_common(1)[0][0]
    return DEFAULT_SYSTEM_PROMPT


def build_fewshot_prompt(examples, target_sample) -> str:
    system_prompt = choose_system_prompt(examples, target_sample)
    if not examples:
        parts = [
            system_prompt,
            "",
            "Solve the following GSM8K problem step by step and end with '#### <number>'.",
            "",
            f"Question: {target_sample['question']}",
            "Answer:",
            "Let's think step by step.",
        ]
        return "\n".join(parts).strip() + "\n"

    parts = [system_prompt, "", "Follow the format of the solved examples and end with '#### <number>'.", ""]

    for i, example in enumerate(examples, start=1):
        parts.append(f"Example {i}")
        parts.append(f"Question: {example['question']}")
        parts.append("Answer:")
        parts.append(example["answer_text"])
        parts.append("")

    parts.append("Now solve the next problem.")
    parts.append(f"Question: {target_sample['question']}")
    parts.append("Answer:")
    parts.append("Let's think step by step.")
    return "\n".join(parts).strip() + "\n"


def build_chat_user_message(examples, target_sample) -> str:
    if not examples:
        parts = [
            "Solve the following GSM8K problem.",
            "Show step-by-step reasoning and end with exactly one final line in the form '#### <number>'.",
            "",
            f"Question: {target_sample['question']}",
            "Answer:",
            "Let's think step by step.",
        ]
        return "\n".join(parts).strip() + "\n"

    parts = [
        "Solve the final GSM8K problem using the style of the solved examples.",
        "Show step-by-step reasoning and end with exactly one final line in the form '#### <number>'.",
        "",
        "Solved examples:",
        "",
    ]

    for i, example in enumerate(examples, start=1):
        parts.append(f"Example {i}")
        parts.append(f"Question: {example['question']}")
        parts.append("Answer:")
        parts.append(example["answer_text"])
        parts.append("")

    parts.append("Now solve the target problem.")
    parts.append(f"Question: {target_sample['question']}")
    parts.append("Answer:")
    parts.append("Let's think step by step.")
    return "\n".join(parts).strip() + "\n"


def resolve_prompt_format(args) -> str:
    if args.prompt_format != "auto":
        return args.prompt_format

    model_name = Path(args.model).name.lower()
    if "instruct" in model_name or "chat" in model_name:
        return "chat-template"
    return "plain"


def get_kv_preset(args):
    return {
        "hadamard": args.hadamard,
        "hadamard_seed": args.hadamard_seed,
        "hadamard_granularity": args.hadamard_granularity,
        "ctk": args.ctk,
        "ctv": args.ctv,
        "kv_layer_k_types": args.kv_layer_k_types,
        "kv_layer_v_types": args.kv_layer_v_types,
    }


def build_llama_cli_command(
    args,
    llama_cli: Path,
    prompt_path: Path,
    system_prompt_path: Optional[Path],
    prompt_format: str,
    sample_seed: int,
):
    cmd = [
        str(llama_cli),
        "-m", args.model,
        "-n", str(args.max_tokens),
        "-s", str(sample_seed),
        "--no-display-prompt",
        "--simple-io",
        "--no-warmup",
        "--no-perf",
        "--ctx-size", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ngl", str(args.n_gpu_layers),
        "-fa", args.flash_attn,
        "--temp", str(args.temperature),
        "--top-p", str(args.top_p),
    ]

    if prompt_format == "chat-template":
        cmd.extend(["-cnv", "-st", "-f", str(prompt_path)])
        if system_prompt_path is not None:
            cmd.extend(["-sysf", str(system_prompt_path)])
    else:
        cmd.extend(["-no-cnv", "-f", str(prompt_path)])

    if args.hadamard:
        cmd.extend([
            "--hadamard",
            "--hadamard-seed", str(args.hadamard_seed),
            "--hadamard-granularity", args.hadamard_granularity,
        ])

    if args.ctk:
        cmd.extend(["-ctk", args.ctk])
    if args.ctv:
        cmd.extend(["-ctv", args.ctv])
    if args.kv_layer_k_types:
        cmd.extend(["--kv-layer-k-types", args.kv_layer_k_types])
    if args.kv_layer_v_types:
        cmd.extend(["--kv-layer-v-types", args.kv_layer_v_types])

    cmd.extend(args.extra_arg)
    return cmd


def build_llama_cli_env():
    # Avoid inheriting LLAMA_ARG_* defaults from the user's shell. Empty values such
    # as LLAMA_ARG_KV_LAYER_V_TYPES="" can override explicit evaluator settings and
    # make runs non-reproducible.
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LLAMA_ARG_")
    }


def run_llama_cli(args, llama_cli: Path, prompt: str, system_prompt: Optional[str], prompt_format: str, sample_seed: int):
    prompt_path = None
    system_prompt_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as fout:
            fout.write(prompt)
            prompt_path = Path(fout.name)

        if prompt_format == "chat-template" and system_prompt:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as fout:
                fout.write(system_prompt)
                system_prompt_path = Path(fout.name)

        cmd = build_llama_cli_command(args, llama_cli, prompt_path, system_prompt_path, prompt_format, sample_seed)
        proc = subprocess.run(
            cmd,
            cwd=args.repo_root,
            env=build_llama_cli_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        if prompt_path is not None and prompt_path.exists():
            prompt_path.unlink()
        if system_prompt_path is not None and system_prompt_path.exists():
            system_prompt_path.unlink()

    if proc.returncode != 0:
        raise RuntimeError(
            "llama-cli failed:\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    # Parse answers only from stdout. stderr may contain CUDA / loader logs such as
    # "compute capability 8.6", which can otherwise be mistaken for the model's numeric answer.
    raw_output = proc.stdout
    raw_output = raw_output.replace(" [end of text]\n", "\n")
    raw_output = raw_output.replace(" [end of text]", "")
    raw_output = raw_output.strip()

    return {
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "raw_output": raw_output,
    }


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_txt(path: Path, result):
    lines = [
        f"model: {result['model']}",
        f"shot: {result['shot']}",
        f"num_samples: {result['num_samples']}",
        f"correct: {result['correct']}",
        f"accuracy: {result['accuracy']:.6f}",
        f"seed: {result['seed']}",
        f"hadamard: {result['kv_preset']['hadamard']}",
        f"hadamard-seed: {result['kv_preset']['hadamard_seed']}",
        f"hadamard-granularity: {result['kv_preset']['hadamard_granularity']}",
        f"ctk: {result['kv_preset']['ctk']}",
        f"ctv: {result['kv_preset']['ctv']}",
        f"kv-layer-k-types: {result['kv_preset']['kv_layer_k_types']}",
        f"kv-layer-v-types: {result['kv_preset']['kv_layer_v_types']}",
        f"fewshot_path: {result['fewshot_source']['path']}",
        f"fewshot_split: {result['fewshot_source']['split']}",
        f"eval_path: {result['eval_source']['path']}",
        f"eval_split: {result['eval_source']['split']}",
        f"fewshot_example_ids: {', '.join(str(x) for x in result['fewshot_example_ids'])}",
        f"fewshot_example_dataset_indices: {', '.join(str(x) for x in result['fewshot_example_dataset_indices'])}",
        "files:",
        f"  predictions.jsonl: {result['files']['predictions_jsonl']}",
        f"  results.json: {result['files']['results_json']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate accuracy_comparison.png. "
            "Install matplotlib in the active Python environment."
        ) from exc

    return plt


def plot_accuracy_comparison(results, out_path: Path):
    plt = load_pyplot()
    labels = [f"{result['shot']}-shot" for result in results]
    accuracies = [result["accuracy"] for result in results]

    plt.figure(figsize=(6.5, 4.0))
    plt.bar(labels, accuracies, color="#4C78A8")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Few-shot setting")
    plt.ylabel("EM accuracy")
    plt.title("GSM8K Few-shot Accuracy")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def evaluate_shot(args, llama_cli: Path, shot: int, fewshot_examples, eval_subset, output_dir: Path, dataset_config):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.txt"

    kv_preset = get_kv_preset(args)
    prompt_format = resolve_prompt_format(args)
    predictions = []
    correct = 0

    for sample_pos, sample in enumerate(eval_subset):
        system_prompt = choose_system_prompt(fewshot_examples, sample)
        if prompt_format == "chat-template":
            prompt = build_chat_user_message(fewshot_examples, sample)
        else:
            prompt = build_fewshot_prompt(fewshot_examples, sample)
            system_prompt = None

        sample_seed = args.seed + shot * 100000 + sample_pos
        llama_result = run_llama_cli(args, llama_cli, prompt, system_prompt, prompt_format, sample_seed)
        parsed_answer, parse_strategy = extract_final_numeric_answer(llama_result["raw_output"])
        is_correct = parsed_answer == sample["gold_answer"]
        correct += int(is_correct)

        record = {
            "question_id": sample["id"],
            "dataset_index": sample["dataset_index"],
            "shot": shot,
            "question": sample["question"],
            "prompt_format": prompt_format,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "raw_model_output": llama_result["raw_output"],
            "stdout": llama_result["stdout"],
            "stderr": llama_result["stderr"],
            "parsed_answer": parsed_answer,
            "parse_strategy": parse_strategy,
            "gold_answer": sample["gold_answer"],
            "gold_parse_strategy": sample["gold_parse_strategy"],
            "correct": is_correct,
            "sample_seed": sample_seed,
            "fewshot_example_ids": [example["id"] for example in fewshot_examples],
            "fewshot_example_dataset_indices": [example["dataset_index"] for example in fewshot_examples],
            "fewshot_source": dataset_config["fewshot"],
            "eval_source": dataset_config["eval"],
            "kv_preset": kv_preset,
            "llama_cli_command": llama_result["command"],
        }
        predictions.append(record)

        print(
            f"[eval_gsm8k_fewshot] {shot}-shot "
            f"sample={sample_pos + 1}/{len(eval_subset)} "
            f"pred={parsed_answer} gold={sample['gold_answer']} "
            f"correct={is_correct}",
            flush=True,
        )

    accuracy = correct / len(eval_subset) if eval_subset else 0.0

    write_jsonl(predictions_path, predictions)

    result = {
        "shot": shot,
        "model": args.model,
        "num_samples": len(eval_subset),
        "seed": args.seed,
        "prompt_format": prompt_format,
        "accuracy": accuracy,
        "correct": correct,
        "fewshot_source": dataset_config["fewshot"],
        "eval_source": dataset_config["eval"],
        "fewshot_example_ids": [sample["id"] for sample in fewshot_examples],
        "fewshot_example_dataset_indices": [sample["dataset_index"] for sample in fewshot_examples],
        "kv_preset": kv_preset,
        "files": {
            "predictions_jsonl": predictions_path.name,
            "results_json": results_path.name,
            "summary_txt": summary_path.name,
        },
    }

    with results_path.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2)

    write_summary_txt(summary_path, result)
    return result


def main():
    args = parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())

    shots = normalize_shots(args.shots)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root)
    llama_cli = resolve_llama_cli_path(repo_root, args.llama_cli_path)
    dataset_config, fewshot_samples, eval_samples = load_dataset_sources(args, max(shots))

    max_shots = max(shots)
    fewshot_pool, eval_subset = select_fewshot_and_eval_sets(
        fewshot_samples,
        eval_samples,
        max_shots,
        args.num_samples,
        args.seed,
        dataset_config["same_source"],
    )

    shot_results = []
    for shot in shots:
        shot_dir = output_root / f"{shot}shot"
        shot_examples = fewshot_pool[:shot]
        result = evaluate_shot(args, llama_cli, shot, shot_examples, eval_subset, shot_dir, dataset_config)
        shot_results.append(result)

    master_summary = {
        "model_path": args.model,
        "repo_root": args.repo_root,
        "llama_cli_path": str(llama_cli),
        "gsm8k_path": args.gsm8k_path,
        "gsm8k_split": args.gsm8k_split,
        "fewshot_path": dataset_config["fewshot"]["path"],
        "fewshot_split": dataset_config["fewshot"]["split"],
        "eval_path": dataset_config["eval"]["path"],
        "eval_split": dataset_config["eval"]["split"],
        "dataset_mode": dataset_config["mode"],
        "same_source": dataset_config["same_source"],
        "shots": shots,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "prompt_format": args.prompt_format,
        "resolved_prompt_format": resolve_prompt_format(args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ctx_size": args.ctx_size,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "n_gpu_layers": args.n_gpu_layers,
        "flash_attn": args.flash_attn,
        "hadamard": args.hadamard,
        "hadamard_seed": args.hadamard_seed,
        "hadamard_granularity": args.hadamard_granularity,
        "ctk": args.ctk,
        "ctv": args.ctv,
        "kv_layer_k_types": args.kv_layer_k_types,
        "kv_layer_v_types": args.kv_layer_v_types,
        "shot_results": [
            {
                "shot": result["shot"],
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "num_samples": result["num_samples"],
                "output_dir": str(output_root / f"{result['shot']}shot"),
            }
            for result in shot_results
        ],
    }

    master_summary_path = output_root / "master_summary.json"
    with master_summary_path.open("w", encoding="utf-8") as fout:
        json.dump(master_summary, fout, indent=2)

    accuracy_plot_path = output_root / "accuracy_comparison.png"
    plot_accuracy_comparison(shot_results, accuracy_plot_path)

    print(f"[eval_gsm8k_fewshot] wrote {master_summary_path}", flush=True)
    print(f"[eval_gsm8k_fewshot] wrote {accuracy_plot_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"eval_gsm8k_fewshot.py: error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
