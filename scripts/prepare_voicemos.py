#!/usr/bin/env python3
"""Download and prepare VoiceMOS Challenge 2022 manifests.

The public Zenodo archive contains BVCC/main and OOD track metadata plus the
redistributable audio. Blizzard Challenge samples are not redistributed; rows
whose WAV files are unavailable are kept in the manifests with ``audio_path`` set
to null so downstream request builders can skip them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "voicemos_data.json"

ARCHIVES = {
    "main.tar.gz": {
        "url": "https://zenodo.org/records/6572573/files/main.tar.gz?download=1",
        "size": 286_700_000,
    },
    "ood.tar.gz": {
        "url": "https://zenodo.org/records/6572573/files/ood.tar.gz?download=1",
        "size": 152_600,
    },
    "scoring_program_distribute.tar.gz": {
        "url": "https://zenodo.org/records/6572573/files/scoring_program_distribute.tar.gz?download=1",
        "size": 34_000,
    },
}

SPLIT_FILES = {
    "main": {
        "train": "train_mos_list.txt",
        "dev": "val_mos_list.txt",
        "test": "test_mos_list.txt",
    },
    "ood": {
        "train": "train_mos_list.txt",
        "dev": "val_mos_list.txt",
        "test": "test_mos_list.txt",
        "unlabeled": "unlabeled_mos_list.txt",
    },
}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_config()
    value = raw_value or os.environ.get("VOICEMOS_DATA_ROOT") or config["default_data_root"]
    return Path(value).expanduser().resolve()


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, dest: Path, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"exists: {dest} ({human_bytes(dest.stat().st_size)})")
        return

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    print(f"download: {url}")
    print(f"     -> {dest}")
    request = Request(url, headers={"User-Agent": "sllm-understanding/voicemos-prep"})
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
    tmp_dest.replace(dest)


def download_dataset(data_root: Path, overwrite: bool) -> None:
    raw_dir = data_root / load_config()["raw_dir"]
    for filename, metadata in ARCHIVES.items():
        download_file(str(metadata["url"]), raw_dir / filename, overwrite=overwrite)


def safe_extract_tar(archive_path: Path, dest_dir: Path, overwrite: bool) -> None:
    marker = dest_dir / f".{archive_path.name}.extracted"
    if marker.exists() and not overwrite:
        print(f"already extracted: {archive_path}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        dest_root = dest_dir.resolve()
        for member in archive.getmembers():
            target = (dest_dir / member.name).resolve()
            if dest_root not in [target, *target.parents]:
                raise RuntimeError(f"refusing to extract path outside data root: {member.name}")
        archive.extractall(dest_dir)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"extracted: {archive_path}")


def extract_dataset(data_root: Path, overwrite: bool) -> None:
    raw_dir = data_root / load_config()["raw_dir"]
    for filename in ARCHIVES:
        archive_path = raw_dir / filename
        if not archive_path.exists():
            raise SystemExit(f"missing archive: {archive_path}; run download first")
        safe_extract_tar(archive_path, data_root, overwrite=overwrite)


def read_system_means(track_dir: Path) -> dict[str, float]:
    path = track_dir / "DATA" / "mydata_system.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["system_ID"]: float(row["mean"]) for row in csv.DictReader(handle)}


def read_listener_scores(track_dir: Path, split: str) -> dict[str, list[int]]:
    split_file = {
        "train": "TRAINSET",
        "dev": "DEVSET",
        "test": "TESTSET",
        "unlabeled": "UNLABELEDSET",
    }.get(split)
    if split_file is None:
        return {}
    path = track_dir / "DATA" / "sets" / split_file
    if not path.exists():
        return {}

    scores: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            filename = row[1]
            try:
                score = int(row[2])
            except ValueError:
                continue
            scores.setdefault(filename, []).append(score)
    return scores


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def system_id_from_filename(filename: str) -> str:
    return filename.split("-utt", 1)[0]


def utterance_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(data_root: Path, overwrite: bool) -> None:
    config = load_config()
    manifest_dir = data_root / config["manifest_dir"]
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    row_id = 0
    missing_audio = 0

    for track in config["tracks"]:
        track_dir = data_root / track
        sets_dir = track_dir / "DATA" / "sets"
        wav_dir = track_dir / "DATA" / "wav"
        if not sets_dir.exists():
            print(f"skip missing track: {track_dir}")
            continue

        system_means = read_system_means(track_dir)
        listener_cache = {
            split: read_listener_scores(track_dir, split) for split in SPLIT_FILES[track]
        }
        for split, filename in SPLIT_FILES[track].items():
            mos_path = sets_dir / filename
            if not mos_path.exists():
                print(f"skip missing split file: {mos_path}")
                continue
            split_key = f"{track}_{split}"
            split_rows: list[dict[str, Any]] = []
            with mos_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                for source_row, fields in enumerate(reader):
                    if len(fields) < 1 or (split != "unlabeled" and len(fields) < 2):
                        continue
                    wav_name = fields[0]
                    mos_value = None if split == "unlabeled" else float(fields[1])
                    system_id = system_id_from_filename(wav_name)
                    wav_path = wav_dir / wav_name
                    audio_path = None
                    audio_num_bytes = 0
                    audio_sha256 = None
                    if wav_path.exists():
                        audio_path = str(wav_path.relative_to(data_root))
                        audio_num_bytes = wav_path.stat().st_size
                        audio_sha256 = sha256(wav_path)
                    else:
                        missing_audio += 1

                    listener_scores = listener_cache.get(split, {}).get(wav_name, [])
                    entry = {
                        "row_id": row_id,
                        "dataset_id": config["dataset_id"],
                        "track": track,
                        "split": split,
                        "split_key": split_key,
                        "source_file": str(mos_path.relative_to(data_root)),
                        "source_row": source_row,
                        "wav_name": wav_name,
                        "audio_path": audio_path,
                        "audio_exists": audio_path is not None,
                        "audio_num_bytes": audio_num_bytes,
                        "audio_sha256": audio_sha256,
                        "system_id": system_id,
                        "utterance_id": utterance_id_from_filename(wav_name),
                        "mos": mos_value,
                        "system_mean_mos": system_means.get(system_id),
                        "listener_scores": listener_scores,
                        "n_listeners": len(listener_scores),
                        "listener_mean_mos": mean([float(score) for score in listener_scores]),
                    }
                    split_rows.append(entry)
                    all_rows.append(entry)
                    row_id += 1
            rows_by_split[split_key] = split_rows

    if not all_rows:
        raise SystemExit(f"no VoiceMOS rows found under {data_root}; run download/extract first")

    for split_key, rows in rows_by_split.items():
        path = manifest_dir / f"{split_key}.jsonl"
        if path.exists() and not overwrite:
            raise SystemExit(f"{path} already exists; pass --overwrite to rebuild")
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote: {path} rows={len(rows)} audio={sum(1 for row in rows if row['audio_exists'])}")

    all_path = manifest_dir / "all.jsonl"
    if all_path.exists() and not overwrite:
        raise SystemExit(f"{all_path} already exists; pass --overwrite to rebuild")
    with all_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metadata = {
        "dataset_id": config["dataset_id"],
        "zenodo_record": config["zenodo_record"],
        "rows": len(all_rows),
        "rows_with_audio": sum(1 for row in all_rows if row["audio_exists"]),
        "missing_audio_rows": missing_audio,
        "splits": {
            split: {
                "rows": len(rows),
                "rows_with_audio": sum(1 for row in rows if row["audio_exists"]),
            }
            for split, rows in sorted(rows_by_split.items())
        },
    }
    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote: {all_path} rows={len(all_rows)} audio={metadata['rows_with_audio']}")
    print(f"missing_audio_rows: {missing_audio}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "extract", "prepare", "all"])
    parser.add_argument("--data-root", help="dataset root; defaults to VOICEMOS_DATA_ROOT or config")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files/manifests")
    args = parser.parse_args(argv)

    data_root = resolve_data_root(args.data_root)
    print(f"data_root: {data_root}")
    if args.command in {"download", "all"}:
        download_dataset(data_root, overwrite=args.overwrite)
    if args.command in {"extract", "all"}:
        extract_dataset(data_root, overwrite=args.overwrite)
    if args.command in {"prepare", "all"}:
        prepare_dataset(data_root, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
