#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shlex
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TRIAL_RE = re.compile(
    r"^\[\s*(?P<index>\d+)/(?P<total>\d+)\]\s+"
    r"ctx=(?P<ctx>\d+)\s+"
    r"depth=(?P<depth>[0-9.]+)\s+"
    r"seed=(?P<seed>\d+)\s+"
    r"correct=(?P<correct>[01])\s+"
    r"pred=(?P<pred>'(?:[^'\\]|\\.)*'|\S+)\s+"
    r"answer=(?P<answer>'(?:[^'\\]|\\.)*'|\S+)\s+"
    r"truncated=(?P<truncated>[01])\s*$"
)


@dataclass
class TrialResult:
    ctx_len: int
    requested_depth: float
    actual_depth: float
    seed: int
    needle_kind: str
    answer: str
    prediction_raw: str
    prediction_extracted: str
    correct: bool
    truncated: bool
    prompt_tokens: int
    tokens_evaluated: int
    tokens_cached: int
    predicted_tokens: int
    latency_ms: float
    prompt_ms: float
    predicted_ms: float
    prompt_tps: float
    predicted_tps: float
    stop_type: str
    filler_chunks: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover NIAH JSON/CSV outputs from a run.log file."
    )
    parser.add_argument("--run-log", required=True, help="Path to NIAH run.log")
    parser.add_argument("--output-json", default="", help="Override output JSON path")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path")
    return parser.parse_args(argv)


def parse_literal(text: str) -> str:
    text = text.strip()
    if text.startswith("'") or text.startswith('"'):
        try:
            value = ast.literal_eval(text)
            return str(value)
        except Exception:
            return text.strip("'\"")
    return text


def parse_command_args(run_log_text: str) -> Dict[str, Any]:
    command_line = None
    for line in run_log_text.splitlines():
        if line.startswith("NIAH command: "):
            command_line = line[len("NIAH command: ") :].strip()
            break

    parsed: Dict[str, Any] = {}
    if not command_line:
        return parsed

    argv = shlex.split(command_line)
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                parsed[key] = argv[i + 1]
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            i += 1
    return parsed


def infer_output_paths(run_log_path: Path, cmd_args: Dict[str, Any], output_json: str, output_csv: str) -> Tuple[Path, Optional[Path]]:
    if output_json:
        json_path = Path(output_json)
    else:
        output_dir = Path(cmd_args.get("output_dir", str(run_log_path.parent)))
        stem = cmd_args.get("output_stem", run_log_path.parent.name)
        json_path = output_dir / f"{stem}.json"

    csv_path: Optional[Path]
    if output_csv:
        csv_path = Path(output_csv)
    elif "output_dir" in cmd_args and "output_stem" in cmd_args:
        csv_path = Path(cmd_args["output_dir"]) / f"{cmd_args['output_stem']}.csv"
    else:
        csv_path = None

    return json_path, csv_path


def parse_trials(run_log_text: str, cmd_args: Dict[str, Any]) -> Tuple[List[TrialResult], int]:
    needle_kind = str(cmd_args.get("needle_kind", "passkey"))
    expected_total = 0
    results: List[TrialResult] = []

    for line in run_log_text.splitlines():
        match = TRIAL_RE.match(line.strip())
        if not match:
            continue

        total = int(match.group("total"))
        if total > expected_total:
            expected_total = total

        requested_depth = float(match.group("depth"))
        prediction = parse_literal(match.group("pred"))
        answer = parse_literal(match.group("answer"))

        results.append(
            TrialResult(
                ctx_len=int(match.group("ctx")),
                requested_depth=requested_depth,
                actual_depth=requested_depth,
                seed=int(match.group("seed")),
                needle_kind=needle_kind,
                answer=answer,
                prediction_raw=prediction,
                prediction_extracted=prediction,
                correct=bool(int(match.group("correct"))),
                truncated=bool(int(match.group("truncated"))),
                prompt_tokens=0,
                tokens_evaluated=0,
                tokens_cached=0,
                predicted_tokens=0,
                latency_ms=0.0,
                prompt_ms=0.0,
                predicted_ms=0.0,
                prompt_tps=0.0,
                predicted_tps=0.0,
                stop_type="",
                filler_chunks=0,
            )
        )

    return results, expected_total


def aggregate_results(results: Sequence[TrialResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, float], List[TrialResult]] = defaultdict(list)
    for result in results:
        grouped[(result.ctx_len, result.requested_depth)].append(result)

    summary: List[Dict[str, Any]] = []
    for (ctx_len, requested_depth), bucket in sorted(grouped.items()):
        n = len(bucket)
        n_correct = sum(1 for item in bucket if item.correct)
        n_truncated = sum(1 for item in bucket if item.truncated)
        summary.append(
            {
                "ctx_len": ctx_len,
                "requested_depth": requested_depth,
                "accuracy": n_correct / n if n else 0.0,
                "n": n,
                "n_correct": n_correct,
                "n_truncated": n_truncated,
                "avg_latency_ms": 0.0,
                "avg_prompt_tokens": 0.0,
                "avg_actual_depth": sum(item.actual_depth for item in bucket) / n if n else 0.0,
            }
        )
    return summary


def write_csv(path: Path, results: Sequence[TrialResult]) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else list(TrialResult.__annotations__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_log_path = Path(args.run_log)
    run_log_text = run_log_path.read_text(encoding="utf-8", errors="replace")

    cmd_args = parse_command_args(run_log_text)
    results, expected_total = parse_trials(run_log_text, cmd_args)
    if not results:
        raise RuntimeError(f"no NIAH trial lines found in {run_log_path}")

    json_path, csv_path = infer_output_paths(run_log_path, cmd_args, args.output_json, args.output_csv)
    summary = aggregate_results(results)

    config = {
        "base_url": cmd_args.get("base_url", ""),
        "ctx_lens": [int(part) for part in str(cmd_args.get("ctx_lens", "")).split(",") if part],
        "depths": [float(part) for part in str(cmd_args.get("depths", "")).split(",") if part],
        "seeds": [int(part) for part in str(cmd_args.get("seeds", "")).split(",") if part],
        "needle_kind": cmd_args.get("needle_kind", "passkey"),
        "digits": int(cmd_args.get("digits", "0") or 0),
        "format": cmd_args.get("format", ""),
        "system_prompt": cmd_args.get("system_prompt", ""),
        "intro": "",
        "filler_file": cmd_args.get("filler_file", ""),
        "max_output_tokens": int(cmd_args.get("max_output_tokens", "0") or 0),
        "temperature": float(cmd_args.get("temperature", "0") or 0.0),
        "api_key": False,
        "cache_prompt": bool(cmd_args.get("cache_prompt", False)),
        "server_ctx": None,
        "recovered_from_run_log": True,
        "run_log": str(run_log_path),
        "expected_trials": expected_total,
        "recovered_trials": len(results),
        "complete": (expected_total == len(results)) if expected_total else True,
    }

    payload = {
        "config": config,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {json_path}")

    if csv_path is not None:
        write_csv(csv_path, results)
        print(f"wrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
