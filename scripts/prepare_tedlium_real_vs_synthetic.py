#!/usr/bin/env python3
"""Prepare TED-LIUM real-vs-synthetic manifests for MOSS-TTS generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "tedlium_real_vs_synthetic.json"


@dataclass(frozen=True)
class Utterance:
    split: str
    talk_id: str
    speaker_id: str
    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def utterance_id(self) -> str:
        return f"{self.split}__{safe_id(self.talk_id)}__{safe_id(self.speaker_id)}__{self.start_s:.3f}_{self.end_s:.3f}"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")


def moss_output_filename(idx: int, item_id: str) -> str:
    cleaned = safe_id(item_id)
    stem = cleaned[:96] if cleaned else f"item_{idx:04d}"
    return f"{idx:04d}_{stem}.wav"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_root(raw_value: str | None, env_name: str, config_key: str) -> Path:
    config = load_config()
    value = raw_value or os.environ.get(env_name) or config[config_key]
    return Path(value).expanduser().resolve()


def legacy_split_root(tedlium_root: Path, split: str) -> Path:
    return tedlium_root / load_config()["source_variant"] / split


def sph_path(tedlium_root: Path, utt: Utterance) -> Path:
    return legacy_split_root(tedlium_root, utt.split) / "sph" / f"{utt.talk_id}.sph"


def read_split(tedlium_root: Path, split: str) -> list[Utterance]:
    stm_dir = legacy_split_root(tedlium_root, split) / "stm"
    if not stm_dir.exists():
        raise FileNotFoundError(f"STM directory not found: {stm_dir}")

    utterances: list[Utterance] = []
    for stm_path in sorted(stm_dir.glob("*.stm")):
        with stm_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=6)
                if len(parts) != 7:
                    raise ValueError(f"Malformed STM line {stm_path}:{line_no}: {raw_line!r}")
                talk_id, _channel, speaker_id, start, end, _labels, text = parts
                text = text.strip()
                if text == "ignore_time_segment_in_scoring" or speaker_id == "inter_segment_gap":
                    continue
                utterance = Utterance(
                    split=split,
                    talk_id=talk_id,
                    speaker_id=speaker_id,
                    start_s=float(start),
                    end_s=float(end),
                    text=normalize_transcript(text),
                )
                if utterance.duration_s > 0 and utterance.text:
                    utterances.append(utterance)
    return utterances


def normalize_transcript(text: str) -> str:
    text = text.replace(" '", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def choose_prompt(
    target: Utterance,
    candidates: list[Utterance],
    min_duration_s: float,
    target_duration_s: float,
) -> Utterance | None:
    same_speaker = [
        utt
        for utt in candidates
        if utt.talk_id == target.talk_id
        and utt.speaker_id == target.speaker_id
        and utt != target
        and utt.duration_s >= min_duration_s
    ]
    if not same_speaker:
        same_speaker = [
            utt
            for utt in candidates
            if utt.talk_id == target.talk_id and utt.speaker_id == target.speaker_id and utt != target
        ]
    if not same_speaker:
        return None
    return min(
        same_speaker,
        key=lambda utt: (
            abs(utt.duration_s - target_duration_s),
            abs(utt.start_s - target.start_s),
            utt.start_s,
        ),
    )


def split_limit(utterances: list[Utterance], max_items: int | None) -> list[Utterance]:
    if max_items is None:
        return utterances
    return utterances[:max_items]


def relative_to_data_root(data_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def extract_clip(
    source_sph: Path,
    output_wav: Path,
    start_s: float,
    duration_s: float,
    sample_rate: int,
    overwrite: bool,
) -> None:
    if output_wav.exists() and not overwrite:
        return
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "sox",
        str(source_sph),
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-b",
        "16",
        str(output_wav),
        "trim",
        f"{start_s:.3f}",
        f"{duration_s:.3f}",
    ]
    subprocess.run(command, check=True)


def build_split(
    tedlium_root: Path,
    data_root: Path,
    split: str,
    max_items: int | None,
    overwrite_audio: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config()
    sample_rate = int(config["sample_rate"])
    prompt_min_duration_s = float(config["prompt_min_duration_s"])
    prompt_target_duration_s = float(config["prompt_target_duration_s"])
    all_utterances = read_split(tedlium_root, split)
    targets = split_limit(all_utterances, max_items)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_id, target in enumerate(targets):
        prompt = choose_prompt(target, all_utterances, prompt_min_duration_s, prompt_target_duration_s)
        if prompt is None:
            skipped.append(
                {
                    "split": split,
                    "utterance_id": target.utterance_id,
                    "talk_id": target.talk_id,
                    "speaker_id": target.speaker_id,
                    "start_s": target.start_s,
                    "end_s": target.end_s,
                    "duration_s": target.duration_s,
                    "text": target.text,
                    "reason": "no_different_same_speaker_prompt_utterance",
                }
            )
            continue
        output_row_id = len(records)
        real_wav = data_root / "real" / split / f"{target.utterance_id}.wav"
        prompt_wav = data_root / "prompts" / split / f"{prompt.utterance_id}.wav"
        synthetic_wav = (
            data_root
            / "synthetic"
            / "moss-tts-realtime"
            / split
            / moss_output_filename(output_row_id, target.utterance_id)
        )

        extract_clip(
            sph_path(tedlium_root, target),
            real_wav,
            target.start_s,
            target.duration_s,
            sample_rate,
            overwrite_audio,
        )
        extract_clip(
            sph_path(tedlium_root, prompt),
            prompt_wav,
            prompt.start_s,
            prompt.duration_s,
            sample_rate,
            overwrite_audio,
        )

        records.append(
            {
                "dataset": "tedlium-moss-real-vs-synthetic",
                "row_id": output_row_id,
                "split": split,
                "utterance_id": target.utterance_id,
                "talk_id": target.talk_id,
                "speaker_id": target.speaker_id,
                "text": target.text,
                "start_s": target.start_s,
                "end_s": target.end_s,
                "duration_s": target.duration_s,
                "real_audio_path": relative_to_data_root(data_root, real_wav),
                "synthetic_audio_path": relative_to_data_root(data_root, synthetic_wav),
                "synthetic_label": "synthetic",
                "real_label": "real",
                "tts_model": "OpenMOSS-Team/MOSS-TTS-Realtime",
                "prompt_utterance_id": prompt.utterance_id,
                "prompt_start_s": prompt.start_s,
                "prompt_end_s": prompt.end_s,
                "prompt_duration_s": prompt.duration_s,
                "prompt_audio_path": relative_to_data_root(data_root, prompt_wav),
                "prompt_is_same_utterance": prompt.utterance_id == target.utterance_id,
            }
        )
    return records, skipped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_split_outputs(
    data_root: Path,
    split: str,
    records: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    manifest_dir = data_root / "manifests"
    write_jsonl(manifest_dir / f"{split}.jsonl", records)
    write_jsonl(manifest_dir / f"skipped_{split}.jsonl", skipped)

    moss_rows = [
        {
            "id": row["utterance_id"],
            "text": row["text"],
            "prompt_wav": str((data_root / row["prompt_audio_path"]).resolve()),
            "real_wav": str((data_root / row["real_audio_path"]).resolve()),
            "synthetic_wav": str((data_root / row["synthetic_audio_path"]).resolve()),
            "speaker": row["speaker_id"],
            "talk_id": row["talk_id"],
            "split": split,
        }
        for row in records
    ]
    write_jsonl(manifest_dir / f"moss_texts_{split}.jsonl", moss_rows)

    pair_rows = []
    for row in records:
        pair_rows.extend(
            [
                {
                    "sample_id": f"{row['utterance_id']}__real",
                    "pair_id": row["utterance_id"],
                    "label": "real",
                    "audio_path": row["real_audio_path"],
                    "other_audio_path": row["synthetic_audio_path"],
                    "split": split,
                    "text": row["text"],
                    "speaker_id": row["speaker_id"],
                    "talk_id": row["talk_id"],
                },
                {
                    "sample_id": f"{row['utterance_id']}__synthetic",
                    "pair_id": row["utterance_id"],
                    "label": "synthetic",
                    "audio_path": row["synthetic_audio_path"],
                    "other_audio_path": row["real_audio_path"],
                    "split": split,
                    "text": row["text"],
                    "speaker_id": row["speaker_id"],
                    "talk_id": row["talk_id"],
                },
            ]
        )
    write_jsonl(manifest_dir / f"pairs_{split}.jsonl", pair_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tedlium-root", help="TED-LIUM release root")
    parser.add_argument("--data-root", help="output dataset root")
    parser.add_argument("--splits", nargs="+", default=load_config()["splits"])
    parser.add_argument("--max-items-per-split", type=int, help="limit targets per split for smoke tests")
    parser.add_argument("--overwrite-audio", action="store_true")
    args = parser.parse_args()

    tedlium_root = resolve_root(args.tedlium_root, "TEDLIUM_ROOT", "default_tedlium_root")
    data_root = resolve_root(args.data_root, "TEDLIUM_SYNTH_DATA_ROOT", "default_data_root")
    data_root.mkdir(parents=True, exist_ok=True)

    all_records = []
    for split in args.splits:
        records, skipped = build_split(
            tedlium_root=tedlium_root,
            data_root=data_root,
            split=split,
            max_items=args.max_items_per_split,
            overwrite_audio=args.overwrite_audio,
        )
        write_split_outputs(data_root, split, records, skipped)
        all_records.extend(records)
        print(f"{split}: {len(records)} rows, skipped {len(skipped)}")

    metadata = {
        "dataset": "tedlium-moss-real-vs-synthetic",
        "tedlium_root": str(tedlium_root),
        "data_root": str(data_root),
        "splits": args.splits,
        "rows": len(all_records),
        "sample_rate": int(load_config()["sample_rate"]),
        "tts_model": "OpenMOSS-Team/MOSS-TTS-Realtime",
    }
    (data_root / "metadata").mkdir(parents=True, exist_ok=True)
    (data_root / "metadata" / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote manifests: {data_root / 'manifests'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
