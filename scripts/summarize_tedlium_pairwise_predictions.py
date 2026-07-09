#!/usr/bin/env python3
"""Summarize TED-LIUM pairwise real-vs-synthetic A/B predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_lookup(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {row["request_id"]: row for row in read_jsonl(path)}


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "nan"
    return f"{numerator / denominator:.3f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.3f}"


def merged_prediction(
    prediction: dict[str, Any],
    requests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    request = requests.get(prediction["request_id"], {})
    merged = dict(request)
    merged.update(prediction)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="pairwise prediction JSONL path")
    parser.add_argument("--requests", help="request JSONL path used for the run")
    parser.add_argument("--prompt-mode", help="only summarize this prompt_mode")
    parser.add_argument("--split", choices=["dev", "test"], help="only summarize this split")
    args = parser.parse_args()

    prediction_path = Path(args.predictions).expanduser().resolve()
    request_path = Path(args.requests).expanduser().resolve() if args.requests else None
    requests = request_lookup(request_path)

    rows = []
    for prediction in read_jsonl(prediction_path):
        row = merged_prediction(prediction, requests)
        if args.prompt_mode and row.get("prompt_mode") != args.prompt_mode:
            continue
        if args.split and row.get("split") != args.split:
            continue
        rows.append(row)

    parsed = [row for row in rows if row.get("raw_parsed_choice") in {"A", "B"}]
    parse_failures = len(rows) - len(parsed)
    correct = [row for row in parsed if row.get("is_correct") is True]
    choices = Counter(str(row["raw_parsed_choice"]) for row in parsed)

    print(f"predictions: {prediction_path}")
    if request_path:
        print(f"requests: {request_path}")
    print(f"considered: {len(rows)}")
    print(f"parsed: {len(parsed)}/{len(rows)}")
    print(f"parse_failures: {parse_failures}")
    print(f"accuracy: {pct(len(correct), len(parsed))}")
    print(f"choice_a_rate: {pct(choices['A'], len(parsed))}")
    print(f"choice_b_rate: {pct(choices['B'], len(parsed))}")

    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_direction[str(row.get("direction", "unknown"))].append(row)
    for direction in sorted(by_direction):
        direction_rows = by_direction[direction]
        direction_correct = sum(row.get("is_correct") is True for row in direction_rows)
        direction_choices = Counter(str(row["raw_parsed_choice"]) for row in direction_rows)
        print(f"{direction}_n: {len(direction_rows)}")
        print(f"{direction}_accuracy: {pct(direction_correct, len(direction_rows))}")
        print(f"{direction}_choice_a_rate: {pct(direction_choices['A'], len(direction_rows))}")
        print(f"{direction}_choice_b_rate: {pct(direction_choices['B'], len(direction_rows))}")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_pair[str(row["pair_id"])].append(row)

    complete_pair_scores = []
    both_correct = 0
    one_correct = 0
    both_wrong = 0
    for pair_rows in by_pair.values():
        directions = {str(row.get("direction")) for row in pair_rows}
        if directions != {"real_a_synthetic_b", "synthetic_a_real_b"}:
            continue
        score = mean([1.0 if row.get("is_correct") is True else 0.0 for row in pair_rows])
        complete_pair_scores.append(score)
        if score == 1.0:
            both_correct += 1
        elif score == 0.5:
            one_correct += 1
        elif score == 0.0:
            both_wrong += 1

    print(f"complete_pairs: {len(complete_pair_scores)}")
    print(f"pair_score_mean: {fmt(mean(complete_pair_scores))}")
    print(f"both_correct_pairs: {both_correct}")
    print(f"one_correct_pairs: {one_correct}")
    print(f"both_wrong_pairs: {both_wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
