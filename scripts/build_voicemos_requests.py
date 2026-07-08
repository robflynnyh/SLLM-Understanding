#!/usr/bin/env python3
"""Build JSONL request files for VoiceMOS 2022 MOS evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "voicemos_data.json"
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "voicemos_eval.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_json(DATA_CONFIG_PATH)
    value = raw_value or os.environ.get("VOICEMOS_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def manifest_path(data_root: Path, raw_manifest: str | None, split: str) -> Path:
    if raw_manifest:
        return Path(raw_manifest).expanduser().resolve()
    config = load_json(DATA_CONFIG_PATH)
    return data_root / config["manifest_dir"] / f"{split}.jsonl"


def read_manifest(path: Path, limit: int | None, require_audio: bool) -> Iterable[dict[str, Any]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if require_audio and not row.get("audio_path"):
                continue
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def audio_path(data_root: Path, row: dict[str, Any]) -> str | None:
    relative = row.get("audio_path")
    if not relative:
        return None
    return str((data_root / relative).resolve())


def score_scale(config: dict[str, Any]) -> dict[str, int]:
    return {"min": int(config["score_min"]), "max": int(config["score_max"])}


def build_requests(
    data_root: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
    config_key: str,
) -> None:
    eval_mode = load_json(EVAL_CONFIG_PATH)[config_key]
    prompt = eval_mode["prompt_template"]
    raw_score_scale = score_scale(eval_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            request = {
                "request_id": (
                    f"voicemos2022__{row['split_key']}__row-{int(row['row_id']):06d}"
                ),
                "mode": "one_by_one",
                "row_id": int(row["row_id"]),
                "audio_path": audio_path(data_root, row),
                "target_label": "naturalness_mos",
                "scored_emotion": "naturalness_mos",
                "prompt": prompt,
                "raw_score_scale": raw_score_scale,
                "human_mos_1_5": row.get("mos"),
                "preserve_raw_scores": True,
            }
            handle.write(json.dumps(request, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="dataset root; defaults to VOICEMOS_DATA_ROOT or config")
    parser.add_argument("--manifest", help="manifest JSONL path; defaults to data-root/manifests/<split>.jsonl")
    parser.add_argument("--split", default="main_dev", help="manifest split key, e.g. main_train, main_dev, main_test")
    parser.add_argument("--output", default="runs/voicemos2022_main_dev_requests.jsonl")
    parser.add_argument("--limit", type=int, help="limit emitted requests for smoke tests")
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="include rows without audio_path; runners will fail on these unless handled separately",
    )
    parser.add_argument("--mode", choices=["mos_1_5"], default="mos_1_5")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    manifest = manifest_path(data_root, args.manifest, args.split)
    rows = list(read_manifest(manifest, args.limit, require_audio=not args.allow_missing_audio))
    output_path = Path(args.output).expanduser().resolve()
    build_requests(data_root, rows, output_path, args.mode)
    print(f"wrote: {output_path}")
    print(f"manifest: {manifest}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
