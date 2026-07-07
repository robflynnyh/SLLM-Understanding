#!/usr/bin/env python3
"""Analyze EmoNet human inter-annotator agreement."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "emonet_data.json"
LABELS = (0, 1, 2)


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


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def annotator_scores(row: dict[str, Any]) -> list[tuple[str, int]]:
    scores = row.get("annotator_scores_raw_0_2", {})
    if not isinstance(scores, dict):
        return []
    return [(str(annotator), int(score)) for annotator, score in sorted(scores.items())]


def coincidence_matrix(units: list[list[int]]) -> dict[tuple[int, int], float]:
    matrix: dict[tuple[int, int], float] = defaultdict(float)
    for values in units:
        counts = Counter(values)
        n_values = sum(counts.values())
        if n_values < 2:
            continue
        for c in LABELS:
            for k in LABELS:
                value = counts[c] * (counts[k] - int(c == k)) / (n_values - 1)
                matrix[(c, k)] += value
    return matrix


def distance_matrix(metric: str, marginals: dict[int, float]) -> dict[tuple[int, int], float]:
    distances: dict[tuple[int, int], float] = {}
    for c in LABELS:
        for k in LABELS:
            if c == k:
                distances[(c, k)] = 0.0
            elif metric == "nominal":
                distances[(c, k)] = 1.0
            elif metric == "interval":
                distances[(c, k)] = float((c - k) ** 2)
            elif metric == "ordinal":
                lo, hi = sorted((c, k))
                cumulative = sum(marginals[label] for label in LABELS if lo <= label <= hi)
                distances[(c, k)] = (cumulative - (marginals[c] + marginals[k]) / 2.0) ** 2
            else:
                raise ValueError(f"unknown distance metric: {metric}")
    return distances


def krippendorff_alpha(units: list[list[int]], metric: str) -> float:
    coincidences = coincidence_matrix(units)
    marginals = {label: sum(coincidences[(label, other)] for other in LABELS) for label in LABELS}
    total = sum(marginals.values())
    if total <= 1:
        return float("nan")
    distances = distance_matrix(metric, marginals)
    observed = sum(coincidences[(c, k)] * distances[(c, k)] for c in LABELS for k in LABELS)
    expected = sum(
        marginals[c] * marginals[k] * distances[(c, k)] / (total - 1.0)
        for c in LABELS
        for k in LABELS
    )
    if expected == 0:
        return float("nan")
    return 1.0 - observed / expected


def pairwise_kappas(rows: list[dict[str, Any]], weights: str | None) -> dict[str, float]:
    pair_scores: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        scores = annotator_scores(row)
        for (left_id, left_score), (right_id, right_score) in combinations(scores, 2):
            pair_scores[(left_id, right_id)].append((left_score, right_score))

    kappas: list[float] = []
    supports: list[int] = []
    for values in pair_scores.values():
        if len(values) < 2:
            continue
        left = [score for score, _ in values]
        right = [score for _, score in values]
        score = cohen_kappa_score(left, right, labels=list(LABELS), weights=weights)
        if not math.isnan(score):
            kappas.append(float(score))
            supports.append(len(values))

    if not kappas:
        return {"mean": float("nan"), "support_weighted_mean": float("nan"), "pairs": 0.0}
    weighted = sum(score * support for score, support in zip(kappas, supports)) / sum(supports)
    return {
        "mean": average(kappas),
        "support_weighted_mean": weighted,
        "pairs": float(len(kappas)),
    }


def split_half(rows: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    pearsons: list[float] = []
    spearmans: list[float] = []
    corrected_pearsons: list[float] = []
    corrected_spearmans: list[float] = []

    score_rows = [annotator_scores(row) for row in rows]
    score_rows = [scores for scores in score_rows if len(scores) >= 2]
    for _ in range(iterations):
        left_means: list[float] = []
        right_means: list[float] = []
        for scores in score_rows:
            values = [score for _, score in scores]
            rng.shuffle(values)
            split_at = len(values) // 2
            left = values[:split_at]
            right = values[split_at:]
            left_means.append(average([float(score) for score in left]))
            right_means.append(average([float(score) for score in right]))
        pearson_score = pearson(left_means, right_means)
        spearman_score = float(spearmanr(left_means, right_means).correlation)
        pearsons.append(pearson_score)
        spearmans.append(spearman_score)
        corrected_pearsons.append(2 * pearson_score / (1 + pearson_score))
        corrected_spearmans.append(2 * spearman_score / (1 + spearman_score))

    return {
        "pearson": average(pearsons),
        "spearman": average(spearmans),
        "spearman_brown_pearson": average(corrected_pearsons),
        "spearman_brown_spearman": average(corrected_spearmans),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(default_manifest_path()),
        help="manifest JSONL path; defaults to configured train manifest",
    )
    parser.add_argument("--limit", type=int, help="limit manifest rows from the start")
    parser.add_argument("--split-half-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest).expanduser().resolve(), args.limit)
    units = [[score for _, score in annotator_scores(row)] for row in rows]
    units = [scores for scores in units if len(scores) >= 2]
    if not units:
        raise SystemExit("no rows with at least two annotators found")

    all_scores = [score for scores in units for score in scores]
    exact_rows = [float(len(set(scores)) == 1) for scores in units]
    pair_agreements: list[float] = []
    pair_abs_diffs: list[float] = []
    for scores in units:
        for left, right in combinations(scores, 2):
            pair_agreements.append(float(left == right))
            pair_abs_diffs.append(abs(left - right))

    unweighted_kappa = pairwise_kappas(rows, weights=None)
    linear_kappa = pairwise_kappas(rows, weights="linear")
    quadratic_kappa = pairwise_kappas(rows, weights="quadratic")
    split = split_half(rows, iterations=args.split_half_iterations, seed=args.seed)

    print(f"manifest: {Path(args.manifest)}")
    print(f"rows: {len(rows)}")
    print(f"rows_with_2plus_annotators: {len(units)}")
    print(f"ratings: {len(all_scores)}")
    print(
        "annotators_per_row: "
        + ", ".join(f"{count}={n_rows}" for count, n_rows in sorted(Counter(map(len, units)).items()))
    )
    print(
        "rating_distribution_raw_0_2: "
        + ", ".join(f"{label}={count}" for label, count in sorted(Counter(all_scores).items()))
    )
    print(f"exact_row_agreement: {average(exact_rows):.1%}")
    print(f"pairwise_exact_agreement: {average(pair_agreements):.1%}")
    print(f"pairwise_abs_diff_mean_raw_0_2: {fmt(average(pair_abs_diffs))}")
    print(f"krippendorff_alpha_nominal: {fmt(krippendorff_alpha(units, 'nominal'))}")
    print(f"krippendorff_alpha_ordinal: {fmt(krippendorff_alpha(units, 'ordinal'))}")
    print(f"krippendorff_alpha_interval: {fmt(krippendorff_alpha(units, 'interval'))}")
    print(f"pairwise_cohen_kappa_pairs: {int(unweighted_kappa['pairs'])}")
    print(f"pairwise_cohen_kappa_unweighted_mean: {fmt(unweighted_kappa['mean'])}")
    print(f"pairwise_cohen_kappa_unweighted_support_weighted: {fmt(unweighted_kappa['support_weighted_mean'])}")
    print(f"pairwise_cohen_kappa_linear_mean: {fmt(linear_kappa['mean'])}")
    print(f"pairwise_cohen_kappa_linear_support_weighted: {fmt(linear_kappa['support_weighted_mean'])}")
    print(f"pairwise_cohen_kappa_quadratic_mean: {fmt(quadratic_kappa['mean'])}")
    print(f"pairwise_cohen_kappa_quadratic_support_weighted: {fmt(quadratic_kappa['support_weighted_mean'])}")
    print(f"split_half_iterations: {args.split_half_iterations}")
    print(f"split_half_pearson_mean: {fmt(split['pearson'])}")
    print(f"split_half_spearman_mean: {fmt(split['spearman'])}")
    print(f"split_half_spearman_brown_pearson: {fmt(split['spearman_brown_pearson'])}")
    print(f"split_half_spearman_brown_spearman: {fmt(split['spearman_brown_spearman'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
