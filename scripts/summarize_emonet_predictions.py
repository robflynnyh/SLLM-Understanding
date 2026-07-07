#!/usr/bin/env python3
"""Summarize EmoNet prediction JSONL metrics.

By default, correlation metrics are computed on the raw 0-2 scale while MAE and
RMSE are reported on the paper's 0-10 scale. Pearson and Spearman are unchanged
by this positive linear rescaling, but error magnitudes are not.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "emonet_data.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_manifest_path() -> Path:
    config = load_json(DATA_CONFIG_PATH)
    return Path(config["default_data_root"]) / config["manifest_dir"] / "train.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def majority_label(row: dict[str, Any]) -> int:
    values = [int(value) for value in row["annotator_scores_raw_0_2"].values()]
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    mean_score = float(row["mean_score_raw_0_2"])
    return min(candidates, key=lambda label: (abs(label - mean_score), label))


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    manifest: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = int(row["row_id"])
        manifest[row_id] = {
            "mean": float(row["mean_score_raw_0_2"]),
            "rounded_mean": int(round(float(row["mean_score_raw_0_2"]))),
            "majority": majority_label(row),
        }
    return manifest


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


def prediction_score(row: dict[str, Any], key: str | None) -> float | None:
    if key:
        value = row.get(key)
    elif "raw_parsed_score_0_2" in row:
        value = row.get("raw_parsed_score_0_2")
    elif "raw_parsed_emotion_score_0_2" in row:
        value = row.get("raw_parsed_emotion_score_0_2")
    else:
        value = row.get("raw_parsed_score")
    if value is None:
        return None
    return float(value)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="prediction JSONL path")
    parser.add_argument(
        "--manifest",
        default=str(default_manifest_path()),
        help="manifest JSONL path; defaults to configured train manifest",
    )
    parser.add_argument("--score-key", help="override parsed prediction score field")
    parser.add_argument(
        "--error-scale",
        choices=["paper_0_10", "raw_0_2"],
        default="paper_0_10",
        help="scale for MAE/RMSE reporting; correlations are unchanged by paper_0_10 scaling",
    )
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions).expanduser().resolve())
    manifest = load_manifest(Path(args.manifest).expanduser().resolve())

    valid_rows: list[tuple[dict[str, Any], float]] = []
    for row in predictions:
        score = prediction_score(row, args.score_key)
        if row.get("raw_parse_error") is None and score is not None:
            valid_rows.append((row, score))

    if not valid_rows:
        raise SystemExit("no valid parsed predictions found")

    model_scores = [score for _, score in valid_rows]
    human_means = [manifest[int(row["row_id"])]["mean"] for row, _ in valid_rows]
    human_majorities = [manifest[int(row["row_id"])]["majority"] for row, _ in valid_rows]
    human_rounded = [manifest[int(row["row_id"])]["rounded_mean"] for row, _ in valid_rows]

    scale_factor = 5.0 if args.error_scale == "paper_0_10" else 1.0
    scaled_model_scores = [score * scale_factor for score in model_scores]
    scaled_human_means = [score * scale_factor for score in human_means]

    mae = average([abs(model - human) for model, human in zip(scaled_model_scores, scaled_human_means)])
    rmse = math.sqrt(
        average([(model - human) ** 2 for model, human in zip(scaled_model_scores, scaled_human_means)])
    )
    acc_majority = average(
        [float(model == human) for model, human in zip(model_scores, human_majorities)]
    )
    acc_rounded = average([float(model == human) for model, human in zip(model_scores, human_rounded)])

    print(f"predictions: {Path(args.predictions)}")
    print(f"manifest: {Path(args.manifest)}")
    print(f"parsed: {len(valid_rows)}/{len(predictions)}")
    print(f"error_scale: {args.error_scale}")
    print(f"accuracy_vs_majority: {acc_majority:.1%}")
    print(f"accuracy_vs_rounded_mean: {acc_rounded:.1%}")
    print(f"pearson: {fmt(pearson(model_scores, human_means))}")
    print(f"spearman: {fmt(spearman(model_scores, human_means))}")
    print(f"mae: {fmt(mae)}")
    print(f"rmse: {fmt(rmse)}")
    print(
        "score_distribution: "
        + ", ".join(f"{label:g}={count}" for label, count in sorted(Counter(model_scores).items()))
    )
    print(
        "human_majority_distribution: "
        + ", ".join(f"{label:g}={count}" for label, count in sorted(Counter(human_majorities).items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
