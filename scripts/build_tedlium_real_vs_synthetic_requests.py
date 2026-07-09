#!/usr/bin/env python3
"""Build MOSS judge requests for TED-LIUM real-vs-synthetic quality scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "tedlium_real_vs_synthetic.json"
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "tedlium_real_vs_synthetic_eval.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_json(DATA_CONFIG_PATH)
    value = raw_value or os.environ.get("TEDLIUM_SYNTH_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def pairs_manifest_path(data_root: Path, raw_manifest: str | None, split: str) -> Path:
    if raw_manifest:
        return Path(raw_manifest).expanduser().resolve()
    return data_root / "manifests" / f"pairs_{split}.jsonl"


def read_pairs(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def score_scale(config: dict[str, Any]) -> dict[str, int]:
    return {"min": int(config["score_min"]), "max": int(config["score_max"])}


def format_prompt(template: str, row: dict[str, Any]) -> str:
    return template.format(transcript=str(row["text"]))


def build_requests(
    data_root: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
    config_key: str,
) -> None:
    eval_mode = load_json(EVAL_CONFIG_PATH)[config_key]
    prompt_template = eval_mode["prompt_template"]
    raw_score_scale = score_scale(eval_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row_id, row in enumerate(rows):
            audio_path = (data_root / str(row["audio_path"])).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"audio not found for {row['sample_id']}: {audio_path}")
            request = {
                "request_id": f"tedlium_rvs__{row['split']}__{config_key}__row-{row_id:06d}__{row['label']}",
                "mode": "one_by_one",
                "row_id": row_id,
                "audio_path": str(audio_path),
                "target_label": "speech_quality_naturalness",
                "scored_emotion": "speech_quality_naturalness",
                "prompt": format_prompt(prompt_template, row),
                "raw_score_scale": raw_score_scale,
                "dataset": "tedlium-moss-real-vs-synthetic",
                "split": row["split"],
                "label": row["label"],
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "talk_id": row["talk_id"],
                "speaker_id": row["speaker_id"],
                "transcript": row["text"],
                "prompt_mode": config_key,
                "preserve_raw_scores": True,
            }
            handle.write(json.dumps(request, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="dataset root; defaults to TEDLIUM_SYNTH_DATA_ROOT or config")
    parser.add_argument("--manifest", help="pairs manifest JSONL path; defaults to data-root/manifests/pairs_<split>.jsonl")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", default="runs/tedlium_rvs_dev_quality_1_10_requests.jsonl")
    parser.add_argument("--limit", type=int, help="limit emitted requests for smoke tests")
    parser.add_argument(
        "--mode",
        choices=["quality_1_10", "quality_1_10_with_transcript"],
        default="quality_1_10",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    manifest = pairs_manifest_path(data_root, args.manifest, args.split)
    rows = list(read_pairs(manifest, args.limit))
    output_path = Path(args.output).expanduser().resolve()
    build_requests(data_root, rows, output_path, args.mode)
    print(f"wrote: {output_path}")
    print(f"manifest: {manifest}")
    print(f"rows: {len(rows)}")
    print(f"mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
