#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_FILLER_CHUNKS = [
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
    "A quick note from the archive says the report was copied, reviewed, and stored without incident.",
    "Several analysts compared routine observations, found nothing urgent, and moved on to the next checklist.",
    "This paragraph exists only as filler text and contains no answer to the final question at all.",
]

DEFAULT_INTRO = (
    "There is an important fact hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will ask you about the important fact later."
)


@dataclass
class NeedleSpec:
    kind: str
    answer: str
    needle_text: str
    question: str
    answer_pattern: str


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

    def get_server_ctx(self) -> Optional[int]:
        try:
            slots = self.request_json("GET", "/slots")
        except RuntimeError:
            return None
        if not isinstance(slots, list):
            return None
        sizes = [int(slot.get("n_ctx", 0)) for slot in slots if isinstance(slot, dict)]
        return max(sizes) if sizes else None

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
        cache_prompt: bool,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "cache_prompt": cache_prompt,
        }
        return self.request_json("POST", "/completion", payload)


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def load_filler_chunks(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_FILLER_CHUNKS)
    text = Path(path).read_text(encoding="utf-8")
    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    raise ValueError(f"no usable filler text found in {path}")


def build_needle(kind: str, seed: int, digits: int, format_mode: str) -> NeedleSpec:
    rng = random.Random(seed)
    if kind == "passkey":
        answer = "".join(str(rng.randint(0, 9)) for _ in range(digits))
        question = (
            "What is the pass key? The pass key is"
            if format_mode == "completion"
            else "What is the pass key? Answer with digits only."
        )
        return NeedleSpec(
            kind=kind,
            answer=answer,
            needle_text=f"The pass key is {answer}. Remember it. {answer} is the pass key.",
            question=question,
            answer_pattern=rf"\b\d{{{digits}}}\b",
        )
    if kind == "word":
        words = [
            "marigold",
            "cobalt",
            "zephyr",
            "walnut",
            "harbor",
            "lantern",
            "velvet",
            "solstice",
        ]
        answer = words[rng.randrange(len(words))]
        question = (
            "What is the secret word? The secret word is"
            if format_mode == "completion"
            else "What is the secret word? Answer with the single word only."
        )
        return NeedleSpec(
            kind=kind,
            answer=answer,
            needle_text=f"The secret word is {answer}. Remember it. {answer} is the secret word.",
            question=question,
            answer_pattern=r"\b[a-zA-Z]+\b",
        )
    if kind == "kv":
        projects = [
            "Project Atlas",
            "Project Beacon",
            "Project Cypress",
            "Project Delta",
            "Project Ember",
            "Project Nimbus",
        ]
        owners = [
            "mina",
            "jules",
            "nora",
            "owen",
            "tariq",
            "selene",
            "lucas",
            "aria",
        ]
        project = projects[rng.randrange(len(projects))]
        answer = owners[rng.randrange(len(owners))]
        question = (
            f"The owner of {project} is"
            if format_mode == "completion"
            else f"Who is the owner of {project}? Answer with the single word only."
        )
        return NeedleSpec(
            kind=kind,
            answer=answer,
            needle_text=f"The owner of {project} is {answer}. Remember it. {answer} is the owner of {project}.",
            question=question,
            answer_pattern=r"\b[a-zA-Z]+\b",
        )
    raise ValueError(f"unsupported needle kind: {kind}")


def extract_answer(text: str, spec: NeedleSpec) -> str:
    if spec.kind == "passkey":
        normalized = text.replace(",", "")
        match = re.search(spec.answer_pattern, normalized)
        if match:
            return match.group(0)
        groups = re.findall(r"\d+", normalized)
        if groups:
            return max(groups, key=len)
        return ""
    match = re.search(spec.answer_pattern, text)
    return match.group(0).lower() if match else ""


def build_filler_sequence(chunks: Sequence[str], total: int, seed: int) -> List[str]:
    if total <= 0:
        return []
    offset = seed % len(chunks)
    return [chunks[(offset + i) % len(chunks)] for i in range(total)]


def build_user_content(
    intro: str,
    filler_chunks: Sequence[str],
    requested_depth: float,
    spec: NeedleSpec,
) -> Tuple[str, str]:
    total_chunks = len(filler_chunks)
    before_count = int(round(total_chunks * requested_depth))
    before_count = max(0, min(total_chunks, before_count))
    before = "\n\n".join(filler_chunks[:before_count]).strip()
    after = "\n\n".join(filler_chunks[before_count:]).strip()

    pieces = [intro.strip()]
    if before:
        pieces.append(before)
    prefix_content = "\n\n".join(pieces)

    pieces_with_needle = list(pieces)
    pieces_with_needle.append(spec.needle_text)
    if after:
        pieces_with_needle.append(after)
    pieces_with_needle.append(spec.question)
    full_content = "\n\n".join(piece for piece in pieces_with_needle if piece)

    return full_content, prefix_content


def render_prompt(
    client: ServerClient,
    format_mode: str,
    system_prompt: str,
    user_content: str,
) -> str:
    if format_mode == "chat":
        return client.apply_template(system_prompt, user_content)
    return user_content


def prompt_token_count(
    client: ServerClient,
    format_mode: str,
    system_prompt: str,
    user_content: str,
) -> int:
    prompt = render_prompt(client, format_mode, system_prompt, user_content)
    return client.tokenize_count(prompt, add_special=True)


def fit_filler_chunks(
    client: ServerClient,
    format_mode: str,
    system_prompt: str,
    intro: str,
    filler_corpus: Sequence[str],
    requested_depth: float,
    spec: NeedleSpec,
    seed: int,
    target_prompt_tokens: int,
) -> Tuple[str, str, int, int]:
    empty_content, _ = build_user_content(intro, [], requested_depth, spec)
    empty_tokens = prompt_token_count(client, format_mode, system_prompt, empty_content)
    if empty_tokens > target_prompt_tokens:
        raise RuntimeError(
            f"static prompt already uses {empty_tokens} tokens, above target prompt budget {target_prompt_tokens}"
        )

    low = 0
    high = 1
    best_content = empty_content
    best_tokens = empty_tokens

    while True:
        filler = build_filler_sequence(filler_corpus, high, seed)
        user_content, _ = build_user_content(intro, filler, requested_depth, spec)
        tokens = prompt_token_count(client, format_mode, system_prompt, user_content)
        if tokens > target_prompt_tokens:
            break
        low = high
        best_content = user_content
        best_tokens = tokens
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        filler = build_filler_sequence(filler_corpus, mid, seed)
        user_content, _ = build_user_content(intro, filler, requested_depth, spec)
        tokens = prompt_token_count(client, format_mode, system_prompt, user_content)
        if tokens <= target_prompt_tokens:
            low = mid
            best_content = user_content
            best_tokens = tokens
        else:
            high = mid

    final_filler = build_filler_sequence(filler_corpus, low, seed)
    final_content, prefix_content = build_user_content(intro, final_filler, requested_depth, spec)
    final_tokens = prompt_token_count(client, format_mode, system_prompt, final_content)

    if final_tokens > target_prompt_tokens:
        raise AssertionError("binary search returned an over-budget prompt")
    if final_tokens != best_tokens:
        best_content = final_content
        best_tokens = final_tokens

    return best_content, prefix_content, low, best_tokens


def compute_actual_depth(
    client: ServerClient,
    format_mode: str,
    system_prompt: str,
    full_content: str,
    prefix_content: str,
) -> float:
    full_prompt = render_prompt(client, format_mode, system_prompt, full_content)
    prefix_prompt = render_prompt(client, format_mode, system_prompt, prefix_content)
    full_tokens = client.tokenize_count(full_prompt, add_special=True)
    prefix_tokens = client.tokenize_count(prefix_prompt, add_special=True)
    if full_tokens <= 0:
        return 0.0
    return min(1.0, max(0.0, prefix_tokens / full_tokens))


def aggregate_results(results: Sequence[TrialResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, float], List[TrialResult]] = defaultdict(list)
    for result in results:
        grouped[(result.ctx_len, result.requested_depth)].append(result)

    summary: List[Dict[str, Any]] = []
    for (ctx_len, requested_depth), bucket in sorted(grouped.items()):
        n = len(bucket)
        n_correct = sum(1 for item in bucket if item.correct)
        truncated = sum(1 for item in bucket if item.truncated)
        summary.append(
            {
                "ctx_len": ctx_len,
                "requested_depth": requested_depth,
                "accuracy": n_correct / n,
                "n": n,
                "n_correct": n_correct,
                "n_truncated": truncated,
                "avg_latency_ms": sum(item.latency_ms for item in bucket) / n,
                "avg_prompt_tokens": sum(item.prompt_tokens for item in bucket) / n,
                "avg_actual_depth": sum(item.actual_depth for item in bucket) / n,
            }
        )
    return summary


def write_csv(path: Path, results: Sequence[TrialResult]) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else list(TrialResult.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def print_summary(summary: Sequence[Dict[str, Any]]) -> None:
    print("")
    print("ctx_len depth  acc    correct/total  truncated  avg_prompt_tok  avg_latency_ms")
    for row in summary:
        print(
            f"{row['ctx_len']:>7d} "
            f"{row['requested_depth']:<5.2f} "
            f"{row['accuracy']:<6.3f} "
            f"{row['n_correct']:>3d}/{row['n']:<7d} "
            f"{row['n_truncated']:>3d} "
            f"{row['avg_prompt_tokens']:>15.1f} "
            f"{row['avg_latency_ms']:>14.1f}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Needle In A Haystack retrieval sweep against llama-server.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument("--ctx-lens", default="8192,16384,32768,65536", help="Comma-separated context lengths")
    parser.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated insertion depths")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated per-trial seeds")
    parser.add_argument("--needle-kind", choices=("passkey", "word", "kv"), default="passkey")
    parser.add_argument("--digits", type=int, default=5, help="Number of digits for passkey mode")
    parser.add_argument("--format", choices=("chat", "completion"), default="chat")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt for chat mode")
    parser.add_argument("--intro", default=DEFAULT_INTRO, help="Instruction text placed before the haystack")
    parser.add_argument("--filler-file", default="", help="Optional text file to use as haystack filler")
    parser.add_argument("--max-output-tokens", type=int, default=16, help="Generation budget reserved for the answer")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--api-key", default="", help="Optional server API key")
    parser.add_argument("--ready-timeout", type=float, default=30.0, help="Time to wait for /health")
    parser.add_argument("--cache-prompt", action="store_true", help="Enable llama-server prompt caching")
    parser.add_argument("--output-dir", default="results/niah", help="Directory to write CSV and JSON results")
    parser.add_argument("--output-stem", default="", help="Optional filename prefix override")
    parser.add_argument("--verbose", action="store_true", help="Print every trial result as it finishes")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ctx_lens = parse_int_list(args.ctx_lens)
    depths = parse_float_list(args.depths)
    seeds = parse_int_list(args.seeds)

    if not ctx_lens:
        raise ValueError("--ctx-lens must not be empty")
    if not depths:
        raise ValueError("--depths must not be empty")
    if not seeds:
        raise ValueError("--seeds must not be empty")
    for depth in depths:
        if not 0.0 <= depth <= 1.0:
            raise ValueError(f"depth must be between 0 and 1, got {depth}")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")

    filler_chunks = load_filler_chunks(args.filler_file or None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ServerClient(args.base_url, args.timeout, args.api_key)
    client.wait_until_ready(args.ready_timeout)
    server_ctx = client.get_server_ctx()
    if server_ctx is not None and max(ctx_lens) > server_ctx:
        print(
            f"warning: requested ctx_len up to {max(ctx_lens)} but server reports n_ctx={server_ctx}",
            file=sys.stderr,
        )

    stem = args.output_stem
    if not stem:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        stem = f"niah_{args.needle_kind}_{timestamp}"

    results: List[TrialResult] = []
    total_trials = len(ctx_lens) * len(depths) * len(seeds)
    trial_index = 0

    for ctx_len in ctx_lens:
        target_prompt_tokens = ctx_len - args.max_output_tokens
        if target_prompt_tokens <= 0:
            raise ValueError(
                f"ctx_len={ctx_len} is smaller than reserved generation budget {args.max_output_tokens}"
            )
        for depth in depths:
            for seed in seeds:
                trial_index += 1
                spec = build_needle(args.needle_kind, seed, args.digits, args.format)
                user_content, prefix_content, filler_count, prompt_tokens = fit_filler_chunks(
                    client=client,
                    format_mode=args.format,
                    system_prompt=args.system_prompt,
                    intro=args.intro,
                    filler_corpus=filler_chunks,
                    requested_depth=depth,
                    spec=spec,
                    seed=seed,
                    target_prompt_tokens=target_prompt_tokens,
                )
                actual_depth = compute_actual_depth(
                    client,
                    args.format,
                    args.system_prompt,
                    user_content,
                    prefix_content,
                )
                prompt = render_prompt(client, args.format, args.system_prompt, user_content)

                started = time.time()
                response = client.completion(
                    prompt=prompt,
                    seed=seed,
                    n_predict=args.max_output_tokens,
                    temperature=args.temperature,
                    cache_prompt=args.cache_prompt,
                )
                elapsed_ms = (time.time() - started) * 1000.0

                content = response.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                prediction = extract_answer(content, spec)
                timings = response.get("timings", {})
                if not isinstance(timings, dict):
                    timings = {}

                result = TrialResult(
                    ctx_len=ctx_len,
                    requested_depth=depth,
                    actual_depth=actual_depth,
                    seed=seed,
                    needle_kind=spec.kind,
                    answer=spec.answer,
                    prediction_raw=content,
                    prediction_extracted=prediction,
                    correct=(prediction == spec.answer.lower()),
                    truncated=bool(response.get("truncated", False)),
                    prompt_tokens=prompt_tokens,
                    tokens_evaluated=int(response.get("tokens_evaluated", 0) or 0),
                    tokens_cached=int(response.get("tokens_cached", 0) or 0),
                    predicted_tokens=int(timings.get("predicted_n", 0) or 0),
                    latency_ms=elapsed_ms,
                    prompt_ms=float(timings.get("prompt_ms", 0.0) or 0.0),
                    predicted_ms=float(timings.get("predicted_ms", 0.0) or 0.0),
                    prompt_tps=float(timings.get("prompt_per_second", 0.0) or 0.0),
                    predicted_tps=float(timings.get("predicted_per_second", 0.0) or 0.0),
                    stop_type=str(response.get("stop_type", "")),
                    filler_chunks=filler_count,
                )
                results.append(result)

                if args.verbose:
                    print(
                        f"[{trial_index:>3d}/{total_trials}] ctx={ctx_len} depth={depth:.2f} "
                        f"seed={seed} correct={int(result.correct)} pred={prediction!r} "
                        f"answer={spec.answer!r} truncated={int(result.truncated)}"
                    )

    summary = aggregate_results(results)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    write_csv(csv_path, results)
    payload = {
        "config": {
            "base_url": args.base_url,
            "ctx_lens": ctx_lens,
            "depths": depths,
            "seeds": seeds,
            "needle_kind": args.needle_kind,
            "digits": args.digits,
            "format": args.format,
            "system_prompt": args.system_prompt,
            "intro": args.intro,
            "filler_file": args.filler_file,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "api_key": bool(args.api_key),
            "cache_prompt": args.cache_prompt,
            "server_ctx": server_ctx,
        },
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_summary(summary)
    print("")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
