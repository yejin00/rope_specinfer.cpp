#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


LONG_BENCH_E_TASKS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

TASK_CATEGORIES = {
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "gov_report": "summarization",
    "multi_news": "summarization",
    "trec": "few_shot",
    "triviaqa": "few_shot",
    "samsum": "few_shot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}

TASK_METRICS = {
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "gov_report": "rouge_l",
    "multi_news": "rouge_l",
    "trec": "accuracy",
    "triviaqa": "f1",
    "samsum": "rouge_l",
    "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy",
    "lcc": "edit_sim",
    "repobench-p": "edit_sim",
}

TASK_MAX_NEW_TOKENS = {
    "qasper": 64,
    "multifieldqa_en": 64,
    "hotpotqa": 64,
    "2wikimqa": 64,
    "gov_report": 256,
    "multi_news": 256,
    "trec": 16,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 16,
    "passage_retrieval_en": 16,
    "lcc": 96,
    "repobench-p": 96,
}


@dataclass
class PredictionRecord:
    task: str
    category: str
    metric: str
    sample_index: int
    sample_id: str
    prediction_raw: str
    prediction_text: str
    references: List[str]
    score: float
    correct: Optional[bool]
    prompt_tokens: int
    prompt_chars: int
    response_tokens: int
    latency_ms: float
    stop_type: str
    ctx_length: Any


class ServerClient:
    def __init__(self, base_url: str, timeout: float, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def request_json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        body = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc

    def wait_until_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            try:
                self.request_json("GET", "/health")
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"llama-server did not become ready within {timeout:.1f}s: {last_error}")

    def apply_template(self, system_prompt: str, user_content: str) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        response = self.request_json("POST", "/apply-template", {"messages": messages})
        prompt = response.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("/apply-template response did not contain a string prompt")
        return prompt

    def tokenize_count(self, content: str, add_special: bool = True) -> int:
        response = self.request_json(
            "POST",
            "/tokenize",
            {"content": content, "add_special": add_special},
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("/tokenize response did not contain a token list")
        return len(tokens)

    def completion(
        self,
        prompt: str,
        seed: int,
        n_predict: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
            "cache_prompt": False,
        }
        return self.request_json("POST", "/completion", payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate LongBench-E tasks through llama-server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument("--tasks", nargs="+", default=LONG_BENCH_E_TASKS, help="LongBench-E tasks to evaluate")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV outputs")
    parser.add_argument("--setting-name", default="default", help="Label for this server/model setting")
    parser.add_argument("--format", choices=["chat", "completion"], default="chat", help="Prompt formatting mode")
    parser.add_argument(
        "--system-prompt",
        default="You are a careful assistant. Follow the task instruction exactly and answer concisely.",
        help="System prompt used for chat formatting",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p")
    parser.add_argument("--seed", type=int, default=42, help="Base generation seed")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument("--ready-timeout", type=float, default=60.0, help="Time to wait for /health")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional per-task sample limit for debugging")
    parser.add_argument("--api-key", default="", help="Optional llama-server API key")
    parser.add_argument("--hf-repo", default="THUDM/LongBench", help="HF dataset repo for LongBench")
    parser.add_argument("--verbose", action="store_true", help="Print every sample as it finishes")
    parser.add_argument("--repo-root", default=str(repo_root_default), help="Unused placeholder for symmetry with other scripts")
    return parser.parse_args(argv)


def load_task_dataset(hf_repo: str, task: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not available. Install it with `pip install datasets`."
        ) from exc

    dataset_name = f"{task}_e"
    ds = load_dataset(hf_repo, dataset_name, split="test")
    return [dict(row) for row in ds]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_references(raw_answers: Any) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        return [normalize_text(item) for item in raw_answers if normalize_text(item)]
    normalized = normalize_text(raw_answers)
    return [normalized] if normalized else []


def task_instruction(task: str, sample: Dict[str, Any]) -> str:
    input_text = normalize_text(sample.get("input"))
    context = normalize_text(sample.get("context"))
    classes = sample.get("all_classes") or []
    classes_text = ", ".join(classes) if classes else ""

    if task in {"qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "triviaqa"}:
        return (
            "Read the following context and answer the question. "
            "Give a short answer only.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{input_text}\n\n"
            "Answer:"
        )
    if task in {"gov_report", "multi_news"}:
        return (
            "Read the following source and produce a concise summary.\n\n"
            f"Source:\n{context}\n\n"
            f"Instruction:\n{input_text}\n\n"
            "Summary:"
        )
    if task == "samsum":
        return (
            "Summarize the following dialogue concisely.\n\n"
            f"Dialogue:\n{context}\n\n"
            f"Instruction:\n{input_text}\n\n"
            "Summary:"
        )
    if task == "trec":
        return (
            "Classify the question into the correct category. "
            "Answer with only one label from the candidate classes.\n\n"
            f"Candidate classes: {classes_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{input_text}\n\n"
            "Label:"
        )
    if task == "passage_count":
        return (
            "Count how many distinct passages appear in the following context. "
            "Answer with only the number.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{input_text}\n\n"
            "Answer:"
        )
    if task == "passage_retrieval_en":
        return (
            "Given the passages below, identify which passage matches the query. "
            "Answer with only the passage identifier or index.\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{input_text}\n\n"
            "Answer:"
        )
    if task in {"lcc", "repobench-p"}:
        return (
            "Complete the code snippet with the next line or short continuation. "
            "Do not explain.\n\n"
            f"Context:\n{context}\n\n"
            f"Instruction:\n{input_text}\n\n"
            "Completion:"
        )
    raise ValueError(f"unsupported task: {task}")


def render_prompt(client: ServerClient, args: argparse.Namespace, user_content: str) -> str:
    if args.format == "chat":
        return client.apply_template(args.system_prompt, user_content)
    return user_content


def strip_special_tokens(text: str) -> str:
    text = normalize_text(text)
    text = text.replace(" [end of text]", "")
    return text.strip()


def normalize_qa_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_qa_answer(prediction).split()
    ref_tokens = normalize_qa_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = defaultdict(int)
    for token in ref_tokens:
        common[token] += 1

    overlap = 0
    for token in pred_tokens:
        if common[token] > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(xs: List[str], ys: List[str]) -> int:
    if not xs or not ys:
        return 0
    dp = [0] * (len(ys) + 1)
    for x in xs:
        prev = 0
        for j, y in enumerate(ys, start=1):
            temp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def normalized_exact_match(prediction: str, reference: str) -> bool:
    pred_norm = normalize_qa_answer(prediction)
    ref_norm = normalize_qa_answer(reference)
    return pred_norm == ref_norm


def code_edit_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, prediction.strip(), reference.strip()).ratio()


def score_prediction(task: str, prediction: str, references: List[str]) -> float:
    metric = TASK_METRICS[task]
    if not references:
        return 0.0

    if metric == "f1":
        return max(token_f1(prediction, ref) for ref in references)
    if metric == "rouge_l":
        return max(rouge_l_f1(prediction, ref) for ref in references)
    if metric == "accuracy":
        return 1.0 if any(normalized_exact_match(prediction, ref) for ref in references) else 0.0
    if metric == "edit_sim":
        return max(code_edit_similarity(prediction, ref) for ref in references)

    raise ValueError(f"unsupported metric: {metric}")


def score_is_binary(task: str) -> bool:
    return TASK_METRICS[task] == "accuracy"


def write_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_task(client: ServerClient, args: argparse.Namespace, task: str, output_root: Path):
    samples = load_task_dataset(args.hf_repo, task)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    records: List[PredictionRecord] = []
    category = TASK_CATEGORIES[task]
    metric = TASK_METRICS[task]

    for idx, sample in enumerate(samples):
        user_content = task_instruction(task, sample)
        prompt = render_prompt(client, args, user_content)
        prompt_tokens = client.tokenize_count(prompt, add_special=True)
        max_new_tokens = TASK_MAX_NEW_TOKENS[task]

        started = time.time()
        response = client.completion(
            prompt=prompt,
            seed=args.seed + idx,
            n_predict=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        latency_ms = (time.time() - started) * 1000.0

        raw_text = normalize_text(response.get("content", ""))
        prediction = strip_special_tokens(raw_text)
        references = normalize_references(sample.get("answers"))
        score = score_prediction(task, prediction, references)

        timings = response.get("timings", {})
        if not isinstance(timings, dict):
            timings = {}

        record = PredictionRecord(
            task=task,
            category=category,
            metric=metric,
            sample_index=idx,
            sample_id=str(sample.get("_id", idx)),
            prediction_raw=raw_text,
            prediction_text=prediction,
            references=references,
            score=score,
            correct=bool(score) if score_is_binary(task) else None,
            prompt_tokens=prompt_tokens,
            prompt_chars=len(prompt),
            response_tokens=int(timings.get("predicted_n", 0) or 0),
            latency_ms=latency_ms,
            stop_type=str(response.get("stop_type", "")),
            ctx_length=sample.get("length"),
        )
        records.append(record)

        if args.verbose:
            print(
                f"[longbench-e] task={task} sample={idx + 1}/{len(samples)} "
                f"score={score:.4f} pred={prediction!r}",
                flush=True,
            )

    task_score = sum(record.score for record in records) / len(records) if records else 0.0
    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = [asdict(record) for record in records]
    write_jsonl(task_dir / "predictions.jsonl", prediction_rows)
    write_csv(task_dir / "predictions.csv", prediction_rows)

    task_result = {
        "task": task,
        "category": category,
        "metric": metric,
        "setting_name": args.setting_name,
        "num_samples": len(records),
        "score": task_score,
        "files": {
            "predictions_jsonl": "predictions.jsonl",
            "predictions_csv": "predictions.csv",
        },
    }
    write_json(task_dir / "results.json", task_result)
    return task_result, records


def aggregate(task_results: List[Dict[str, Any]]):
    by_category: Dict[str, List[float]] = defaultdict(list)
    for result in task_results:
        by_category[result["category"]].append(result["score"])

    category_results = {
        category: sum(scores) / len(scores)
        for category, scores in sorted(by_category.items())
    }
    overall = sum(result["score"] for result in task_results) / len(task_results) if task_results else 0.0
    return category_results, overall


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    client = ServerClient(args.base_url, args.timeout, args.api_key)
    client.wait_until_ready(args.ready_timeout)

    task_results: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    for task in args.tasks:
        result, records = evaluate_task(client, args, task, output_root)
        task_results.append(result)
        all_predictions.extend(asdict(record) for record in records)

    category_results, overall = aggregate(task_results)
    summary = {
        "setting_name": args.setting_name,
        "base_url": args.base_url,
        "tasks": args.tasks,
        "task_results": task_results,
        "category_average": category_results,
        "overall_average": overall,
    }

    write_json(output_root / "summary.json", summary)
    write_csv(output_root / "task_results.csv", task_results)
    write_jsonl(output_root / "all_predictions.jsonl", all_predictions)

    print("")
    print("Task Scores")
    for result in task_results:
        print(
            f"{result['task']:<22} metric={result['metric']:<8} "
            f"score={result['score']:.4f} n={result['num_samples']}"
        )
    print("")
    print("Category Averages")
    for category, score in category_results.items():
        print(f"{category:<16} score={score:.4f}")
    print("")
    print(f"Overall Average: {overall:.4f}")
    print(f"wrote {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"eval_longbench_e.py: error: {exc}", file=sys.stderr)
        raise
