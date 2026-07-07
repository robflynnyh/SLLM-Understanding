#!/usr/bin/env python3
"""Build JSONL request files for EmoNet model evaluation.

The default mode is one-by-one scoring: for each manifest row, emit one request
per emotion using the prompt template configured in configs/emonet_eval.json.
The all-at-once mode emits one request per audio asking for every emotion score
in a strict JSON object. The resulting request files are model-agnostic and
preserve enough metadata to store raw outputs before any calibration is applied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "emonet_data.json"
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "emonet_eval.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_json(DATA_CONFIG_PATH)
    value = raw_value or os.environ.get("EMONET_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def manifest_path(data_root: Path, raw_manifest: str | None) -> Path:
    if raw_manifest:
        return Path(raw_manifest).expanduser().resolve()
    config = load_json(DATA_CONFIG_PATH)
    return data_root / config["manifest_dir"] / "train.jsonl"


def read_manifest(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def official_emotions() -> tuple[list[str], list[str]]:
    from prepare_emonet import EXTRA_LABELS, OFFICIAL_40

    return list(OFFICIAL_40), list(EXTRA_LABELS)


def emotion_list(rows: list[dict[str, Any]], emotion_set: str) -> list[str]:
    official_40, extra_labels = official_emotions()
    if emotion_set == "official40":
        return official_40
    if emotion_set == "all42":
        return official_40 + extra_labels
    if emotion_set == "manifest":
        return sorted({str(row["target_label"]) for row in rows})
    raise ValueError(f"unknown emotion set: {emotion_set}")


def audio_path(data_root: Path, row: dict[str, Any]) -> str | None:
    relative = row.get("audio_path")
    if not relative:
        return None
    return str((data_root / relative).resolve())


def slug(value: str) -> str:
    return "-".join(value.lower().replace("/", "-").replace("&", "and").split())


def score_scale(config: dict[str, Any]) -> dict[str, int]:
    return {
        "min": int(config["score_min"]),
        "max": int(config["score_max"]),
    }


def build_one_by_one_requests(
    data_root: Path,
    rows: list[dict[str, Any]],
    emotions: list[str],
    output_path: Path,
    mode: str = "one_by_one",
    config_key: str = "one_by_one",
) -> None:
    eval_config = load_json(EVAL_CONFIG_PATH)
    one_by_one = eval_config[config_key]
    prompt_template = one_by_one["prompt_template"]
    raw_score_scale = score_scale(one_by_one)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = int(row["row_id"])
            for emotion in emotions:
                request = {
                    "request_id": f"row-{row_id:06d}__emotion-{slug(emotion)}",
                    "mode": mode,
                    "row_id": row_id,
                    "audio_path": audio_path(data_root, row),
                    "target_label": row["target_label"],
                    "scored_emotion": emotion,
                    "prompt": prompt_template.format(emotion=emotion),
                    "raw_score_scale": raw_score_scale,
                    "human_mean_score_raw_0_2": row.get("mean_score_raw_0_2"),
                    "human_mean_score_0_10": row.get("mean_score_0_10"),
                    "preserve_raw_scores": True,
                }
                handle.write(json.dumps(request, sort_keys=True) + "\n")


def build_all_at_once_requests(
    data_root: Path,
    rows: list[dict[str, Any]],
    emotions: list[str],
    output_path: Path,
) -> None:
    eval_config = load_json(EVAL_CONFIG_PATH)
    all_at_once = eval_config["all_at_once"]
    prompt_template = all_at_once["prompt_template"]
    raw_score_scale = score_scale(all_at_once)
    emotion_lines = "\n".join(f"- {emotion}" for emotion in emotions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = int(row["row_id"])
            request = {
                "request_id": f"row-{row_id:06d}__all-emotions",
                "mode": "all_at_once",
                "row_id": row_id,
                "audio_path": audio_path(data_root, row),
                "target_label": row["target_label"],
                "scored_emotions": emotions,
                "prompt": prompt_template.format(emotion_lines=emotion_lines),
                "raw_score_scale": raw_score_scale,
                "human_mean_score_raw_0_2": row.get("mean_score_raw_0_2"),
                "human_mean_score_0_10": row.get("mean_score_0_10"),
                "preserve_raw_scores": True,
            }
            handle.write(json.dumps(request, sort_keys=True) + "\n")


def build_requests(
    mode: str,
    data_root: Path,
    rows: list[dict[str, Any]],
    emotions: list[str],
    output_path: Path,
) -> None:
    if mode == "one_by_one":
        build_one_by_one_requests(data_root, rows, emotions, output_path)
        return
    if mode == "one_by_one_human_rubric":
        build_one_by_one_requests(
            data_root,
            rows,
            emotions,
            output_path,
            mode="one_by_one_human_rubric",
            config_key="one_by_one_human_rubric",
        )
        return
    if mode == "all_at_once":
        build_all_at_once_requests(data_root, rows, emotions, output_path)
        return
    raise ValueError(f"unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="dataset root; defaults to EMONET_DATA_ROOT or config")
    parser.add_argument("--manifest", help="manifest JSONL path; defaults to data-root/manifests/train.jsonl")
    parser.add_argument(
        "--output",
        default="runs/emonet_one_by_one_requests.jsonl",
        help="output request JSONL path",
    )
    parser.add_argument(
        "--emotion-set",
        choices=["official40", "all42", "manifest"],
        default="official40",
        help="which emotions to score for each audio",
    )
    parser.add_argument(
        "--mode",
        choices=["one_by_one", "one_by_one_human_rubric", "all_at_once"],
        default="one_by_one",
        help="one request per audio/emotion pair, the 0-2 human-rubric variant, or one request per audio for every emotion",
    )
    parser.add_argument("--limit", type=int, help="limit manifest rows for smoke tests")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    rows = list(read_manifest(manifest_path(data_root, args.manifest), args.limit))
    emotions = emotion_list(rows, args.emotion_set)
    output_path = Path(args.output).expanduser().resolve()
    build_requests(args.mode, data_root, rows, emotions, output_path)
    print(f"wrote: {output_path}")
    print(f"mode: {args.mode}")
    print(f"rows: {len(rows)}")
    print(f"emotions_per_row: {len(emotions)}")
    request_count = len(rows) if args.mode == "all_at_once" else len(rows) * len(emotions)
    print(f"requests: {request_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
