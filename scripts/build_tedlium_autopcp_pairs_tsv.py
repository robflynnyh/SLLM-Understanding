#!/usr/bin/env python3
"""Build AutoPCP src/tgt TSVs for TED-LIUM real-vs-synthetic audio."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "tedlium_real_vs_synthetic.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_config()
    value = raw_value or os.environ.get("TEDLIUM_SYNTH_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--data-root", help="dataset root; defaults to TEDLIUM_SYNTH_DATA_ROOT or config")
    parser.add_argument(
        "--output",
        required=True,
        help="output TSV with src_audio and tgt_audio columns",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    pairs_path = data_root / "manifests" / f"pairs_{args.split}.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"pair manifest not found: {pairs_path}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with pairs_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        fieldnames = ["pair_id", "split", "src_audio", "tgt_audio"]
        writer = csv.DictWriter(target, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") != "real":
                continue
            real_audio = data_root / row["audio_path"]
            synthetic_audio = data_root / row["other_audio_path"]
            if not real_audio.exists() or not synthetic_audio.exists():
                missing = [
                    str(path)
                    for path in (real_audio, synthetic_audio)
                    if not path.exists()
                ]
                raise FileNotFoundError(
                    f"missing audio for {pairs_path}:{line_no}: {missing}"
                )
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "split": row["split"],
                    "src_audio": str(real_audio.resolve()),
                    "tgt_audio": str(synthetic_audio.resolve()),
                }
            )
            count += 1

    print(f"wrote {count} AutoPCP pairs to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
