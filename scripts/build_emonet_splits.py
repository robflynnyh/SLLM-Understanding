#!/usr/bin/env python3
"""Build deterministic smoke and quick EmoNet manifests.

This script reads a prepared full manifest and writes smaller stratified
manifests. It does not copy audio; rows keep the same relative audio paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "emonet_data.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_json(DATA_CONFIG_PATH)
    value = raw_value or os.environ.get("EMONET_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def group_by_label(rows: list[dict[str, Any]], label_set: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if label_set == "official40" and not row.get("is_official_40", False):
            continue
        label = str(row["target_label"])
        grouped.setdefault(label, []).append(row)
    return grouped


def stratified_sample(
    grouped: dict[str, list[dict[str, Any]]],
    per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for label in sorted(grouped):
        label_rows = list(grouped[label])
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(label_rows)
        picked = label_rows[:per_label]
        if len(picked) < per_label:
            raise RuntimeError(
                f"label {label!r} has only {len(picked)} rows, requested {per_label}"
            )
        sampled.extend(picked)
    return sorted(sampled, key=lambda row: int(row["row_id"]))


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["target_label"])
        counts[label] = counts.get(label, 0) + 1
    payload = {
        "source_manifest": str(args.manifest),
        "label_set": args.label_set,
        "seed": args.seed,
        "num_rows": len(rows),
        "num_labels": len(counts),
        "label_counts": dict(sorted(counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="dataset root; defaults to EMONET_DATA_ROOT or config")
    parser.add_argument("--manifest", help="full manifest path; defaults to data-root/manifests/train.jsonl")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--label-set",
        choices=["official40", "all42"],
        default="official40",
        help="official40 excludes Authenticity/Arousal; all42 keeps every HF label",
    )
    parser.add_argument("--smoke-per-label", type=int, default=1)
    parser.add_argument("--quick-per-label", type=int, default=10)
    parser.add_argument("--smoke-name", default="smoke.jsonl")
    parser.add_argument("--quick-name", default="quick.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    config = load_json(DATA_CONFIG_PATH)
    manifest_dir = data_root / config["manifest_dir"]
    full_manifest = Path(args.manifest).expanduser().resolve() if args.manifest else manifest_dir / "train.jsonl"
    if not full_manifest.exists():
        raise SystemExit(f"manifest not found: {full_manifest}")

    smoke_path = manifest_dir / args.smoke_name
    quick_path = manifest_dir / args.quick_name
    for path in [smoke_path, quick_path]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} already exists; pass --overwrite to replace it")

    rows = read_jsonl(full_manifest)
    grouped = group_by_label(rows, args.label_set)
    smoke_rows = stratified_sample(grouped, args.smoke_per_label, args.seed)
    quick_rows = stratified_sample(grouped, args.quick_per_label, args.seed)

    write_jsonl(smoke_path, smoke_rows)
    write_jsonl(quick_path, quick_rows)
    write_summary(manifest_dir / "smoke.summary.json", smoke_rows, args)
    write_summary(manifest_dir / "quick.summary.json", quick_rows, args)

    print(f"source: {full_manifest}")
    print(f"label_set: {args.label_set}")
    print(f"smoke: {smoke_path} rows={len(smoke_rows)} labels={len(grouped)}")
    print(f"quick: {quick_path} rows={len(quick_rows)} labels={len(grouped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

