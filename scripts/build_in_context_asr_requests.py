#!/usr/bin/env python3
"""Build MOSS-Audio request JSONL for the in-context-asr probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_text_info(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    full_text = lines[0].split("full: ", 1)[1].strip()
    targets = [target.strip() for target in lines[1].split("target: ", 1)[1].split("|")]
    separators = [separator.strip() for separator in lines[2].split("separators: ", 1)[1].split(",")]
    return {"full_text": full_text, "targets": targets, "separators": separators}


def prompt_for(condition: str, targets: list[str], separators: list[str]) -> str:
    target_text = " | ".join(targets)
    if condition == "without_repeat":
        return (
            "You are evaluating speech recognition on a short audio clip. "
            "Transcribe only what is clearly audible; do not infer missing words from context.\n\n"
            f"Target word or accepted variant: {target_text}\n\n"
            "Return exactly one JSON object with this schema:\n"
            "{\"target_present\": true, \"transcript\": \"...\"}\n\n"
            "Set target_present to true only if the target word is clearly audible and recognizable."
        )

    separator_text = " | ".join(separators)
    return (
        "You are evaluating speech recognition on an audio clip that may contain an initial unclear "
        "sentence, a spoken request to repeat it, and then a clearer repeated sentence. Transcribe only "
        "what is clearly audible; do not infer missing words from context.\n\n"
        f"Target word or accepted variant: {target_text}\n"
        f"Possible repeat-request phrases: {separator_text}\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\"target_before_repeat\": false, \"target_after_repeat\": true, "
        "\"before_repeat_transcript\": \"...\", \"after_repeat_transcript\": \"...\"}\n\n"
        "Set each boolean to true only if the target word is clearly audible and recognizable in that segment."
    )


def build_requests(data_root: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    item_dirs = sorted((path for path in data_root.iterdir() if path.is_dir()), key=lambda path: int(path.name))
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for item_dir in item_dirs:
            row_id = int(item_dir.name)
            text_info = parse_text_info(item_dir / "text.txt")
            base = {
                "dataset": "in-context-asr",
                "row_id": row_id,
                "full_text": text_info["full_text"],
                "targets": text_info["targets"],
                "separators": text_info["separators"],
                "target_label": text_info["targets"][0],
                "mode": "in_context_asr_target_probe",
            }
            for condition, filename in [
                ("without_repeat", "sentence_without_repeat.wav"),
                ("with_repeat", "sentence_with_repeat.wav"),
            ]:
                request = {
                    **base,
                    "request_id": f"in_context_asr__row-{row_id:03d}__{condition}",
                    "condition": condition,
                    "audio_path": str((item_dir / filename).resolve()),
                    "prompt": prompt_for(condition, text_info["targets"], text_info["separators"]),
                }
                handle.write(json.dumps(request, sort_keys=True) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="../in-context-asr/data",
        help="path to the in-context-asr data directory",
    )
    parser.add_argument(
        "--output",
        default="runs/in_context_asr_moss4b_target_probe_requests.jsonl",
        help="output request JSONL path",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"data root does not exist: {data_root}")
    output_path = Path(args.output).expanduser().resolve()
    count = build_requests(data_root, output_path)
    print(f"wrote: {output_path}")
    print(f"data_root: {data_root}")
    print(f"requests: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
