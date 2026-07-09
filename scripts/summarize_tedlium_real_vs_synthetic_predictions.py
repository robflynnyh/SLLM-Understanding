#!/usr/bin/env python3
"""Summarize TED-LIUM real-vs-synthetic quality prediction distributions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def score_suffix_for_row(row: dict[str, Any], key: str | None) -> str:
    if key and key.endswith("_1_10"):
        return "1_10"
    scale = row.get("raw_score_scale")
    if isinstance(scale, dict):
        try:
            return f"{int(scale['min'])}_{int(scale['max'])}"
        except (KeyError, TypeError, ValueError):
            pass
    if "raw_parsed_score_1_10" in row:
        return "1_10"
    return "1_10"


def prediction_score(row: dict[str, Any], key: str | None) -> tuple[float | None, str]:
    suffix = score_suffix_for_row(row, key)
    if key:
        value = row.get(key)
    elif "raw_parsed_score_1_10" in row:
        value = row.get("raw_parsed_score_1_10")
    else:
        value = row.get("raw_parsed_score")
    if value is None:
        return None, suffix
    return float(value), suffix


def request_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {row["request_id"]: row for row in read_jsonl(path)}


def metric_line(prefix: str, scores: list[float]) -> None:
    if not scores:
        print(f"{prefix}_n: 0")
        return
    print(f"{prefix}_n: {len(scores)}")
    print(f"{prefix}_mean: {fmt(average(scores))}")
    print(f"{prefix}_min: {fmt(min(scores))}")
    print(f"{prefix}_max: {fmt(max(scores))}")


def paired_deltas(valid: list[tuple[dict[str, Any], float]]) -> list[float]:
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for request, score in valid:
        by_pair[str(request["pair_id"])][str(request["label"])] = score
    return [
        pair_scores["synthetic"] - pair_scores["real"]
        for pair_scores in by_pair.values()
        if "synthetic" in pair_scores and "real" in pair_scores
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="prediction JSONL path")
    parser.add_argument("--requests", required=True, help="request JSONL path used for the run")
    parser.add_argument("--score-key", help="override parsed prediction score field")
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions).expanduser().resolve())
    requests = request_lookup(Path(args.requests).expanduser().resolve())

    valid: list[tuple[dict[str, Any], float]] = []
    parse_failures = 0
    for prediction in predictions:
        score, _suffix = prediction_score(prediction, args.score_key)
        request = requests.get(prediction["request_id"])
        if score is None or request is None or prediction.get("raw_parse_error") is not None:
            parse_failures += 1
            continue
        valid.append((request, score))

    print(f"predictions: {Path(args.predictions)}")
    print(f"requests: {Path(args.requests)}")
    print(f"parsed: {len(valid)}/{len(predictions)}")
    print(f"parse_failures: {parse_failures}")

    by_label: dict[str, list[float]] = defaultdict(list)
    for request, score in valid:
        by_label[str(request["label"])].append(score)
    for label in ("real", "synthetic"):
        metric_line(label, by_label[label])

    deltas = paired_deltas(valid)
    metric_line("synthetic_minus_real_pair_delta", deltas)
    if deltas:
        lower = sum(delta < 0 for delta in deltas)
        equal = sum(delta == 0 for delta in deltas)
        higher = sum(delta > 0 for delta in deltas)
        print(f"synthetic_lower_than_real_pairs: {lower}")
        print(f"synthetic_equal_to_real_pairs: {equal}")
        print(f"synthetic_higher_than_real_pairs: {higher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
