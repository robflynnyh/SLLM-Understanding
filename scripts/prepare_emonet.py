#!/usr/bin/env python3
"""Download and prepare EmoNet-Voice Bench.

This script keeps large artifacts under a configurable data root. It downloads
the HF Parquet shards and converts each row into:
  - an extracted audio file
  - one JSONL manifest entry with normalized human labels
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "emonet_data.json"


OFFICIAL_40 = [
    "Amusement",
    "Elation",
    "Pleasure",
    "Contentment",
    "Thankfulness",
    "Affection",
    "Infatuation",
    "Hope",
    "Triumph",
    "Pride",
    "Interest",
    "Awe",
    "Astonishment",
    "Concentration",
    "Contemplation",
    "Relief",
    "Longing",
    "Teasing",
    "Impatience and Irritability",
    "Sexual Lust",
    "Doubt",
    "Fear",
    "Distress",
    "Confusion",
    "Embarrassment",
    "Shame",
    "Disappointment",
    "Sadness",
    "Bitterness",
    "Contempt",
    "Disgust",
    "Anger",
    "Malevolence",
    "Sourness",
    "Pain",
    "Helplessness",
    "Fatigue",
    "Emotional Numbness",
    "Intoxication",
    "Jealousy / Envy",
]

EXTRA_LABELS = ["Authenticity", "Arousal"]

LABEL_ALIASES = {
    "Pleasure/Ecstasy": "Pleasure",
    "Thankfulness/Gratitude": "Thankfulness",
    "Hope/Enthusiasm/Optimism": "Hope",
    "Astonishment/Surprise": "Astonishment",
    "Malevolence/Malice": "Malevolence",
    "Fatigue/Exhaustion": "Fatigue",
    "Intoxication/Altered States of Consciousness": "Intoxication",
    "Jealousy & Envy": "Jealousy / Envy",
    "Jealousy and Envy": "Jealousy / Envy",
}


def load_config() -> dict[str, str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_config()
    value = raw_value or os.environ.get("EMONET_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def request_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "sllm-understanding/emonet-prep"})
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def hf_parquet_files(dataset_id: str) -> list[dict[str, Any]]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_id}"
    payload = request_json(url)
    if payload.get("failed"):
        raise RuntimeError(f"HF parquet endpoint reports failures: {payload['failed']}")
    files = payload.get("parquet_files", [])
    if not files:
        raise RuntimeError(f"No parquet files found for {dataset_id}")
    return files


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, dest: Path, expected_size: int | None, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        if expected_size is None or dest.stat().st_size == expected_size:
            print(f"exists: {dest} ({human_bytes(dest.stat().st_size)})")
            return
        raise RuntimeError(
            f"{dest} exists but size is {dest.stat().st_size}, expected {expected_size}; "
            "pass --overwrite to replace it"
        )

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    print(f"download: {url}")
    print(f"     -> {dest}")
    request = Request(url, headers={"User-Agent": "sllm-understanding/emonet-prep"})
    started = time.time()
    downloaded = 0
    try:
        with urlopen(request, timeout=300) as response, tmp_dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded % (64 * 1024 * 1024) < len(chunk):
                    elapsed = max(time.time() - started, 1e-6)
                    rate = downloaded / elapsed
                    print(f"        {human_bytes(downloaded)} at {human_bytes(int(rate))}/s")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc

    if expected_size is not None and tmp_dest.stat().st_size != expected_size:
        raise RuntimeError(
            f"downloaded size mismatch for {dest}: got {tmp_dest.stat().st_size}, "
            f"expected {expected_size}"
        )
    tmp_dest.replace(dest)


def download_dataset(data_root: Path, overwrite: bool) -> list[Path]:
    config = load_config()
    raw_dir = data_root / config["raw_dir"]
    files = hf_parquet_files(config["dataset_id"])
    paths: list[Path] = []
    total_size = sum(int(item.get("size", 0)) for item in files)
    print(f"dataset: {config['dataset_id']}")
    print(f"raw_dir: {raw_dir}")
    print(f"files: {len(files)}, total: {human_bytes(total_size)}")
    for item in files:
        filename = item["filename"]
        dest = raw_dir / filename
        download_file(item["url"], dest, int(item["size"]), overwrite)
        paths.append(dest)
    write_metadata(data_root, {"parquet_files": files})
    return paths


def write_metadata(data_root: Path, extra: dict[str, Any]) -> None:
    config = load_config()
    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": config["dataset_id"],
        "split": config["split"],
        "official_40": OFFICIAL_40,
        "extra_labels": EXTRA_LABELS,
        **extra,
    }
    with (metadata_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Preparing manifests requires pyarrow to read Parquet.\n"
            "Install it in a project-specific environment, for example:\n"
            "  python -m pip install pyarrow\n"
            "No packages were installed by this script."
        ) from exc
    return pq


def canonical_label(label: str) -> str:
    label = " ".join(label.strip().split())
    return LABEL_ALIASES.get(label, label)


def parse_label(raw_label: str) -> tuple[str, dict[str, int]]:
    try:
        records = ast.literal_eval(raw_label)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"could not parse label string: {raw_label!r}") from exc

    annotator_scores: dict[str, int] = {}
    target_labels: set[str] = set()
    if not isinstance(records, list):
        raise ValueError(f"expected label list, got {type(records).__name__}: {raw_label!r}")

    for record in records:
        if not isinstance(record, dict) or len(record) != 1:
            raise ValueError(f"bad annotator record: {record!r}")
        annotator, value = next(iter(record.items()))
        if not isinstance(value, dict) or len(value) != 1:
            raise ValueError(f"bad emotion score record: {record!r}")
        label, score = next(iter(value.items()))
        label = canonical_label(str(label))
        score = int(score)
        if score not in (0, 1, 2):
            raise ValueError(f"unexpected human score {score} in {record!r}")
        target_labels.add(label)
        annotator_scores[str(annotator)] = score

    if len(target_labels) != 1:
        raise ValueError(f"expected one target label per row, got {sorted(target_labels)}")
    return next(iter(target_labels)), annotator_scores


def audio_extension(audio: Any) -> str:
    path = ""
    if isinstance(audio, dict):
        path = str(audio.get("path") or "")
    suffix = Path(path).suffix.lower()
    if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        return suffix
    data = audio_bytes(audio)
    if data.startswith(b"RIFF"):
        return ".wav"
    if data.startswith(b"ID3") or data[:2] == b"\xff\xfb":
        return ".mp3"
    if data.startswith(b"fLaC"):
        return ".flac"
    return ".bin"


def audio_bytes(audio: Any) -> bytes:
    if isinstance(audio, dict):
        data = audio.get("bytes")
    else:
        data = audio
    if data is None:
        raise ValueError(f"audio row has no embedded bytes: {audio!r}")
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, bytes):
        return data
    raise ValueError(f"unsupported audio bytes type: {type(data).__name__}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def prepare_dataset(
    data_root: Path,
    max_rows: int | None,
    write_audio: bool,
    overwrite: bool,
) -> Path:
    pq = require_pyarrow()
    config = load_config()
    raw_dir = data_root / config["raw_dir"]
    audio_dir = data_root / config["audio_dir"]
    manifest_dir = data_root / config["manifest_dir"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if write_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = sorted(raw_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(
            f"No Parquet shards found in {raw_dir}. Run:\n"
            f"  python scripts/prepare_emonet.py download --data-root {data_root}"
        )

    manifest_path = manifest_dir / "train.jsonl"
    tmp_manifest_path = manifest_path.with_suffix(".jsonl.part")
    if manifest_path.exists() and not overwrite:
        raise SystemExit(f"{manifest_path} already exists; pass --overwrite to rebuild")

    label_counts: dict[str, int] = {}
    rows_written = 0
    official_set = set(OFFICIAL_40)
    all_scores_10: list[float] = []

    with tmp_manifest_path.open("w", encoding="utf-8") as manifest:
        for shard_idx, parquet_path in enumerate(parquet_paths):
            table = pq.read_table(parquet_path, columns=["audioId", "label"])
            rows = table.to_pylist()
            print(f"prepare: {parquet_path.name} rows={len(rows)}")
            for local_row_idx, row in enumerate(rows):
                if max_rows is not None and rows_written >= max_rows:
                    break

                target_label, annotator_scores_raw = parse_label(row["label"])
                raw_scores = list(annotator_scores_raw.values())
                scores_10 = [score * 5 for score in raw_scores]
                all_scores_10.extend(scores_10)

                global_row_id = rows_written
                relative_audio_path: str | None = None
                audio_sha256: str | None = None
                audio_size = 0
                if write_audio:
                    data = audio_bytes(row["audioId"])
                    audio_sha256 = hashlib.sha256(data).hexdigest()
                    ext = audio_extension(row["audioId"])
                    label_slug = target_label.lower().replace("/", " ").replace("&", "and")
                    label_slug = "-".join(label_slug.split())
                    audio_path = audio_dir / f"{global_row_id:06d}_{label_slug}{ext}"
                    if not audio_path.exists() or overwrite:
                        audio_path.write_bytes(data)
                    audio_size = len(data)
                    relative_audio_path = str(audio_path.relative_to(data_root))

                entry = {
                    "row_id": global_row_id,
                    "dataset_id": config["dataset_id"],
                    "split": config["split"],
                    "source_parquet": str(parquet_path.relative_to(data_root)),
                    "source_row": local_row_idx,
                    "audio_path": relative_audio_path,
                    "audio_sha256": audio_sha256,
                    "audio_num_bytes": audio_size,
                    "target_label": target_label,
                    "is_official_40": target_label in official_set,
                    "annotator_scores_raw_0_2": annotator_scores_raw,
                    "annotator_scores_0_10": {
                        annotator: score * 5 for annotator, score in annotator_scores_raw.items()
                    },
                    "n_annotators": len(raw_scores),
                    "mean_score_raw_0_2": mean([float(score) for score in raw_scores]),
                    "mean_score_0_10": mean([float(score) for score in scores_10]),
                    "median_score_0_10": median([float(score) for score in scores_10]),
                    "std_score_0_10": stddev([float(score) for score in scores_10]),
                }
                manifest.write(json.dumps(entry, sort_keys=True) + "\n")
                label_counts[target_label] = label_counts.get(target_label, 0) + 1
                rows_written += 1

            if max_rows is not None and rows_written >= max_rows:
                break

    tmp_manifest_path.replace(manifest_path)
    write_metadata(
        data_root,
        {
            "prepared_rows": rows_written,
            "label_counts": dict(sorted(label_counts.items())),
            "manifest": str(manifest_path.relative_to(data_root)),
            "audio_extracted": write_audio,
            "score_0_10_mean": mean(all_scores_10) if all_scores_10 else None,
        },
    )
    print(f"wrote: {manifest_path}")
    print(f"rows: {rows_written}")
    print(f"labels: {len(label_counts)}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["download", "prepare", "all"],
        help="download raw shards, prepare manifest/audio, or run both",
    )
    parser.add_argument("--data-root", help="dataset root; defaults to EMONET_DATA_ROOT or config")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    parser.add_argument("--max-rows", type=int, help="prepare only this many rows")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="write manifests without extracting audio files",
    )
    args = parser.parse_args(argv)

    data_root = resolve_data_root(args.data_root)
    print(f"data_root: {data_root}")

    if args.command in {"download", "all"}:
        download_dataset(data_root, overwrite=args.overwrite)
    if args.command in {"prepare", "all"}:
        prepare_dataset(
            data_root,
            max_rows=args.max_rows,
            write_audio=not args.no_audio,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

