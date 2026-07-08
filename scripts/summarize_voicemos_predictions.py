#!/usr/bin/env python3
"""Summarize VoiceMOS MOS prediction metrics."""

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


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mean_x = average(xs)
    mean_y = average(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranked = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            ranked[order[rank_index]] = average_rank
        index = end
    return ranked


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    manifest = {}
    for row in read_jsonl(path):
        manifest[int(row["row_id"])] = row
    return manifest


def score_suffix_for_row(row: dict[str, Any], key: str | None) -> str:
    if key and key.endswith("_1_10"):
        return "1_10"
    if key and key.endswith("_1_5"):
        return "1_5"
    if key and key.endswith("_0_10"):
        return "0_10"
    scale = row.get("raw_score_scale")
    if isinstance(scale, dict):
        try:
            return f"{int(scale['min'])}_{int(scale['max'])}"
        except (KeyError, TypeError, ValueError):
            pass
    if "raw_parsed_score_1_5" in row:
        return "1_5"
    if "raw_parsed_score_1_10" in row:
        return "1_10"
    if "raw_parsed_score_0_10" in row:
        return "0_10"
    return "1_5"


def prediction_score(row: dict[str, Any], key: str | None) -> tuple[float | None, str]:
    suffix = score_suffix_for_row(row, key)
    if key:
        value = row.get(key)
    elif "raw_parsed_score_1_5" in row:
        value = row.get("raw_parsed_score_1_5")
    elif "raw_parsed_score_0_10" in row:
        value = row.get("raw_parsed_score_0_10")
    elif "raw_parsed_score_1_10" in row:
        value = row.get("raw_parsed_score_1_10")
    else:
        value = row.get("raw_parsed_score")
    if value is None:
        return None, suffix
    return float(value), suffix


def to_mos_1_5(score: float, suffix: str) -> float:
    if suffix == "1_10":
        return 1.0 + ((score - 1.0) / 9.0) * 4.0
    if suffix == "0_10":
        return 1.0 + (score / 10.0) * 4.0
    return score


def metric_block(prefix: str, model_scores: list[float], human_scores: list[float]) -> None:
    errors = [model - human for model, human in zip(model_scores, human_scores)]
    abs_errors = [abs(error) for error in errors]
    sq_errors = [error * error for error in errors]
    print(f"{prefix}_n: {len(model_scores)}")
    print(f"{prefix}_pearson: {fmt(pearson(model_scores, human_scores))}")
    print(f"{prefix}_spearman: {fmt(spearman(model_scores, human_scores))}")
    print(f"{prefix}_mae: {fmt(average(abs_errors))}")
    print(f"{prefix}_mse: {fmt(average(sq_errors))}")
    print(f"{prefix}_rmse: {fmt(math.sqrt(average(sq_errors)))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="prediction JSONL path")
    parser.add_argument("--manifest", required=True, help="VoiceMOS manifest JSONL path")
    parser.add_argument("--score-key", help="override parsed prediction score field")
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions).expanduser().resolve())
    manifest = load_manifest(Path(args.manifest).expanduser().resolve())

    valid: list[tuple[dict[str, Any], float]] = []
    for prediction in predictions:
        score, suffix = prediction_score(prediction, args.score_key)
        row = manifest.get(int(prediction["row_id"]))
        if score is None or row is None or row.get("mos") is None:
            continue
        if prediction.get("raw_parse_error") is not None:
            continue
        valid.append((row, to_mos_1_5(score, suffix)))

    if not valid:
        raise SystemExit("no valid parsed predictions with manifest MOS labels found")

    model_scores = [score for _, score in valid]
    human_scores = [float(row["mos"]) for row, _ in valid]

    print(f"predictions: {Path(args.predictions)}")
    print(f"manifest: {Path(args.manifest)}")
    print(f"parsed: {len(valid)}/{len(predictions)}")
    metric_block("utterance", model_scores, human_scores)

    by_system: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, model_score in valid:
        by_system[str(row["system_id"])].append((model_score, float(row["mos"])))
    system_model = [average([pair[0] for pair in pairs]) for pairs in by_system.values()]
    system_human = [average([pair[1] for pair in pairs]) for pairs in by_system.values()]
    metric_block("system", system_model, system_human)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
