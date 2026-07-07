#!/usr/bin/env python3
"""Summarize leave-one-annotator-out EmoNet agreement metrics.

Each human rating is treated as the prediction once. The remaining ratings for
the same row are treated as the reference target. Error metrics use the paper
0-10 scale by default so they are directly comparable to model summaries.
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


def read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def majority(values: list[int]) -> int:
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    mean_score = average([float(value) for value in values])
    return min(candidates, key=lambda label: (abs(label - mean_score), label))


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


def rounded_mean(values: list[int]) -> int:
    return int(round(average([float(value) for value in values])))


def build_rotations(rows: list[dict[str, Any]], reference: str) -> list[dict[str, Any]]:
    rotations: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("annotator_scores_raw_0_2", {})
        if not isinstance(scores, dict) or len(scores) < 2:
            continue
        annotator_scores = [(str(key), int(value)) for key, value in sorted(scores.items())]
        for held_out, prediction in annotator_scores:
            if reference == "all_annotators":
                reference_scores = [score for _, score in annotator_scores]
            else:
                reference_scores = [
                    score for annotator, score in annotator_scores if annotator != held_out
                ]
            reference_mean = average([float(score) for score in reference_scores])
            rotations.append(
                {
                    "row_id": int(row["row_id"]),
                    "held_out_annotator": held_out,
                    "prediction_raw_0_2": prediction,
                    "prediction_0_10": prediction * 5.0,
                    "reference_mean_raw_0_2": reference_mean,
                    "reference_mean_0_10": reference_mean * 5.0,
                    "reference_majority_raw_0_2": majority(reference_scores),
                    "reference_rounded_mean_raw_0_2": rounded_mean(reference_scores),
                }
            )
    return rotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(default_manifest_path()),
        help="manifest JSONL path; defaults to configured train manifest",
    )
    parser.add_argument("--limit", type=int, help="limit manifest rows from the start")
    parser.add_argument(
        "--reference",
        choices=["leave_one_out", "all_annotators"],
        default="leave_one_out",
        help="reference aggregation for the human target; all_annotators includes the held-out score",
    )
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest).expanduser().resolve(), args.limit)
    rotations = build_rotations(rows, args.reference)
    if not rotations:
        raise SystemExit("no rows with at least two annotators found")

    model_scores = [row["prediction_0_10"] for row in rotations]
    reference_means = [row["reference_mean_0_10"] for row in rotations]
    absolute_errors = [abs(model - reference) for model, reference in zip(model_scores, reference_means)]
    squared_errors = [(model - reference) ** 2 for model, reference in zip(model_scores, reference_means)]
    acc_majority = average(
        [
            float(row["prediction_raw_0_2"] == row["reference_majority_raw_0_2"])
            for row in rotations
        ]
    )
    acc_rounded = average(
        [
            float(row["prediction_raw_0_2"] == row["reference_rounded_mean_raw_0_2"])
            for row in rotations
        ]
    )

    by_row: dict[int, list[dict[str, Any]]] = {}
    for rotation in rotations:
        by_row.setdefault(rotation["row_id"], []).append(rotation)
    row_mae = average(
        [
            average([abs(row["prediction_0_10"] - row["reference_mean_0_10"]) for row in row_rotations])
            for row_rotations in by_row.values()
        ]
    )
    row_rmse = math.sqrt(
        average(
            [
                average(
                    [
                        (row["prediction_0_10"] - row["reference_mean_0_10"]) ** 2
                        for row in row_rotations
                    ]
                )
                for row_rotations in by_row.values()
            ]
        )
    )
    row_acc_majority = average(
        [
            average(
                [
                    float(row["prediction_raw_0_2"] == row["reference_majority_raw_0_2"])
                    for row in row_rotations
                ]
            )
            for row_rotations in by_row.values()
        ]
    )
    row_acc_rounded = average(
        [
            average(
                [
                    float(row["prediction_raw_0_2"] == row["reference_rounded_mean_raw_0_2"])
                    for row in row_rotations
                ]
            )
            for row_rotations in by_row.values()
        ]
    )

    print(f"manifest: {Path(args.manifest)}")
    print(f"rows: {len(rows)}")
    print(f"rows_with_2plus_annotators: {len(by_row)}")
    print(f"held_out_ratings: {len(rotations)}")
    print(f"reference: {args.reference}")
    print("prediction_scale: 0_10")
    print("error_scale: paper_0_10")
    print(f"pooled_accuracy_vs_majority: {acc_majority:.1%}")
    print(f"pooled_accuracy_vs_rounded_mean: {acc_rounded:.1%}")
    print(f"pooled_pearson: {fmt(pearson(model_scores, reference_means))}")
    print(f"pooled_spearman: {fmt(spearman(model_scores, reference_means))}")
    print(f"pooled_mae: {fmt(average(absolute_errors))}")
    print(f"pooled_rmse: {fmt(math.sqrt(average(squared_errors)))}")
    print(f"sample_weighted_accuracy_vs_majority: {row_acc_majority:.1%}")
    print(f"sample_weighted_accuracy_vs_rounded_mean: {row_acc_rounded:.1%}")
    print(f"sample_weighted_mae: {fmt(row_mae)}")
    print(f"sample_weighted_rmse: {fmt(row_rmse)}")
    print(
        "held_out_score_distribution_raw_0_2: "
        + ", ".join(
            f"{label:g}={count}"
            for label, count in sorted(Counter(row["prediction_raw_0_2"] for row in rotations).items())
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
