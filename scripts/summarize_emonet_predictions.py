#!/usr/bin/env python3
"""Summarize EmoNet prediction JSONL metrics.

By default, MAE and RMSE are reported on the paper's 0-10 scale. The script
accepts prediction files produced on either a raw 0-2 scale or the paper 0-10
scale. Pearson and Spearman are unchanged by positive linear rescaling, but
error magnitudes are not.
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
        mean_raw = float(row["mean_score_raw_0_2"])
        manifest[row_id] = {
            "mean": mean_raw,
            "mean_0_10": float(row.get("mean_score_0_10", mean_raw * 5.0)),
            "rounded_mean": int(round(mean_raw)),
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


def score_suffix_for_row(row: dict[str, Any], key: str | None) -> str:
    if key and key.endswith("_0_10"):
        return "0_10"
    if key and key.endswith("_0_2"):
        return "0_2"
    scale = row.get("raw_score_scale")
    if isinstance(scale, dict):
        try:
            return f"{int(scale['min'])}_{int(scale['max'])}"
        except (KeyError, TypeError, ValueError):
            pass
    if "raw_parsed_score_0_10" in row or "raw_parsed_emotion_score_0_10" in row:
        return "0_10"
    return "0_2"


def prediction_score(row: dict[str, Any], key: str | None) -> tuple[float | None, str]:
    suffix = score_suffix_for_row(row, key)
    if key:
        value = row.get(key)
    elif "raw_parsed_score_0_10" in row:
        value = row.get("raw_parsed_score_0_10")
    elif "raw_parsed_emotion_score_0_10" in row:
        value = row.get("raw_parsed_emotion_score_0_10")
    elif "raw_parsed_score_0_2" in row:
        value = row.get("raw_parsed_score_0_2")
    elif "raw_parsed_emotion_score_0_2" in row:
        value = row.get("raw_parsed_emotion_score_0_2")
    else:
        value = row.get("raw_parsed_score")
    if value is None:
        return None, suffix
    return float(value), suffix


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
    parser.add_argument(
        "--prediction-scale",
        choices=["auto", "paper_0_10", "raw_0_2"],
        default="auto",
        help="scale used by parsed prediction scores; defaults to raw_score_scale metadata",
    )
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions).expanduser().resolve())
    manifest = load_manifest(Path(args.manifest).expanduser().resolve())

    valid_rows: list[tuple[dict[str, Any], float, str]] = []
    for row in predictions:
        score, inferred_suffix = prediction_score(row, args.score_key)
        if row.get("raw_parse_error") is None and score is not None:
            if args.prediction_scale == "paper_0_10":
                suffix = "0_10"
            elif args.prediction_scale == "raw_0_2":
                suffix = "0_2"
            else:
                suffix = inferred_suffix
            valid_rows.append((row, score, suffix))

    if not valid_rows:
        raise SystemExit("no valid parsed predictions found")

    prediction_scales = sorted({suffix for _, _, suffix in valid_rows})
    if len(prediction_scales) != 1:
        raise SystemExit(f"mixed prediction scales are not supported: {prediction_scales}")
    prediction_scale = prediction_scales[0]

    model_scores = [score for _, score, _ in valid_rows]
    human_means = [manifest[int(row["row_id"])]["mean"] for row, _, _ in valid_rows]
    human_means_0_10 = [manifest[int(row["row_id"])]["mean_0_10"] for row, _, _ in valid_rows]
    human_majorities = [manifest[int(row["row_id"])]["majority"] for row, _, _ in valid_rows]
    human_rounded = [manifest[int(row["row_id"])]["rounded_mean"] for row, _, _ in valid_rows]

    if prediction_scale == "0_10":
        model_scores_0_10 = model_scores
        model_scores_0_2 = [score / 5.0 for score in model_scores]
        comparable_majorities = [score * 5.0 for score in human_majorities]
        comparable_rounded = [score * 5.0 for score in human_rounded]
    else:
        model_scores_0_2 = model_scores
        model_scores_0_10 = [score * 5.0 for score in model_scores]
        comparable_majorities = [float(score) for score in human_majorities]
        comparable_rounded = [float(score) for score in human_rounded]

    if args.error_scale == "paper_0_10":
        error_model_scores = model_scores_0_10
        error_human_means = human_means_0_10
    else:
        error_model_scores = model_scores_0_2
        error_human_means = human_means

    mae = average([abs(model - human) for model, human in zip(error_model_scores, error_human_means)])
    rmse = math.sqrt(
        average([(model - human) ** 2 for model, human in zip(error_model_scores, error_human_means)])
    )
    acc_majority = average(
        [float(model == human) for model, human in zip(model_scores, comparable_majorities)]
    )
    acc_rounded = average([float(model == human) for model, human in zip(model_scores, comparable_rounded)])

    print(f"predictions: {Path(args.predictions)}")
    print(f"manifest: {Path(args.manifest)}")
    print(f"parsed: {len(valid_rows)}/{len(predictions)}")
    print(f"prediction_scale: {prediction_scale}")
    print(f"error_scale: {args.error_scale}")
    print(f"accuracy_vs_majority: {acc_majority:.1%}")
    print(f"accuracy_vs_rounded_mean: {acc_rounded:.1%}")
    print(f"pearson: {fmt(pearson(model_scores_0_10, human_means_0_10))}")
    print(f"spearman: {fmt(spearman(model_scores_0_10, human_means_0_10))}")
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
