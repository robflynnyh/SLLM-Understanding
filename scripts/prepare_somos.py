#!/usr/bin/env python3
"""Download and prepare SOMOS manifests.

The Zenodo release is a single outer ZIP containing metadata, training split
files, and a nested ``audios.zip``. The prepare step builds manifests for both
the SOMOS-clean and SOMOS-full split variants. Rows are kept even when audio has
not been extracted yet; downstream request builders skip rows without
``audio_path`` by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "somos_data.json"

ARCHIVE_NAME = "somos.zip"
ARCHIVE_URL = "https://zenodo.org/records/7378801/files/somos.zip?download=1"
ARCHIVE_SIZE = 3_971_669_902
ARCHIVE_MD5 = "bdfde4cae256549dfab05d713136e4af"

SPLIT_FILE_NAMES = {
    "train": "train_mos_list.txt",
    "valid": "valid_mos_list.txt",
    "test": "test_mos_list.txt",
}

RAW_SCORE_FILE_NAMES = {
    "train": "raw_scores_removed_excess_gt_trainset.tsv",
    "valid": "raw_scores_removed_excess_gt_validset.tsv",
    "test": "raw_scores_removed_excess_gt_testset.tsv",
}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_config()
    value = raw_value or os.environ.get("SOMOS_DATA_ROOT") or config["default_data_root"]
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
    request = Request(url, headers={"User-Agent": "sllm-understanding/somos-prep"})
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
                if downloaded % (256 * 1024 * 1024) < len(chunk):
                    elapsed = max(time.time() - started, 1e-6)
                    rate = downloaded / elapsed
                    print(f"        {human_bytes(downloaded)} at {human_bytes(int(rate))}/s")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    tmp_dest.replace(dest)


def download_dataset(data_root: Path, overwrite: bool) -> None:
    raw_dir = data_root / load_config()["raw_dir"]
    download_file(ARCHIVE_URL, raw_dir / ARCHIVE_NAME, overwrite=overwrite)


def safe_extract_zip(
    archive_path: Path,
    dest_dir: Path,
    overwrite: bool,
    skip_names: set[str] | None = None,
) -> None:
    marker = dest_dir / f".{archive_path.name}.extracted"
    if marker.exists() and not overwrite:
        print(f"already extracted: {archive_path}")
        return

    skip_names = skip_names or set()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        dest_root = dest_dir.resolve()
        for member in archive.infolist():
            if member.filename in skip_names:
                continue
            target = (dest_dir / member.filename).resolve()
            if dest_root not in [target, *target.parents]:
                raise RuntimeError(f"refusing to extract path outside data root: {member.filename}")
            archive.extract(member, dest_dir)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"extracted: {archive_path}")


def extract_audios_zip(outer_archive: Path, data_root: Path, overwrite: bool) -> None:
    audios_zip_path = data_root / "audios.zip"
    audio_marker = data_root / ".audios.zip.extracted"
    if audio_marker.exists() and not overwrite:
        print("already extracted: audios.zip")
        return

    if not audios_zip_path.exists() or overwrite:
        print(f"extract nested archive: {outer_archive}::audios.zip -> {audios_zip_path}")
        with zipfile.ZipFile(outer_archive) as archive:
            with archive.open("audios.zip") as source, audios_zip_path.open("wb") as dest:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    dest.write(chunk)

    safe_extract_zip(audios_zip_path, data_root / "audios", overwrite=overwrite)
    audio_marker.write_text("ok\n", encoding="utf-8")


def extract_dataset(data_root: Path, overwrite: bool, metadata_only: bool) -> None:
    raw_dir = data_root / load_config()["raw_dir"]
    archive_path = raw_dir / ARCHIVE_NAME
    if not archive_path.exists():
        raise SystemExit(f"missing archive: {archive_path}; run download first")

    safe_extract_zip(archive_path, data_root, overwrite=overwrite, skip_names={"audios.zip"})
    if not metadata_only:
        extract_audios_zip(archive_path, data_root, overwrite=overwrite)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def system_id_from_utterance(filename: str) -> str:
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[-1]


def sentence_id_from_utterance(filename: str) -> str:
    stem = Path(filename).stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def read_system_means(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["systemId"]: float(row["mean"]) for row in csv.DictReader(handle)}


def read_listener_scores(data_root: Path, split: str, clean_only: bool) -> dict[str, list[int]]:
    path = data_root / "raw_scores_with_metadata" / "split1" / RAW_SCORE_FILE_NAMES[split]
    if not path.exists():
        return {}
    scores: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if clean_only and row.get("clean") != "1":
                continue
            try:
                score = int(row["choice"])
            except (KeyError, ValueError):
                continue
            scores.setdefault(row["utteranceId"], []).append(score)
    return scores


def build_audio_index(data_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in [data_root / "audios", data_root]:
        if not root.exists():
            continue
        for path in root.rglob("*.wav"):
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def prepare_dataset(data_root: Path, overwrite: bool) -> None:
    config = load_config()
    manifest_dir = data_root / config["manifest_dir"]
    manifest_dir.mkdir(parents=True, exist_ok=True)

    audio_index = build_audio_index(data_root)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    row_id = 0

    for split_variant in config["split_variants"]:
        clean_only = split_variant == "clean"
        variant_dir = data_root / "training_files" / "split1" / split_variant
        if not variant_dir.exists():
            print(f"skip missing split variant: {variant_dir}")
            continue

        for split in config["splits"]:
            mos_path = variant_dir / SPLIT_FILE_NAMES[split]
            system_path = variant_dir / f"{split}_system.csv"
            if not mos_path.exists():
                print(f"skip missing split file: {mos_path}")
                continue

            system_means = read_system_means(system_path)
            listener_scores = read_listener_scores(data_root, split, clean_only=clean_only)
            split_key = f"{split_variant}_{split}"
            split_rows: list[dict[str, Any]] = []

            with mos_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for source_row, row in enumerate(reader):
                    wav_name = row["utteranceId"]
                    utterance_stem = Path(wav_name).stem
                    system_id = system_id_from_utterance(wav_name)
                    audio_file = audio_index.get(wav_name) or audio_index.get(utterance_stem)
                    audio_path = None
                    audio_num_bytes = 0
                    audio_sha256 = None
                    if audio_file is not None:
                        audio_path = str(audio_file.resolve().relative_to(data_root))
                        audio_num_bytes = audio_file.stat().st_size
                        audio_sha256 = sha256(audio_file)
                    scores = listener_scores.get(utterance_stem, [])
                    entry = {
                        "row_id": row_id,
                        "dataset_id": config["dataset_id"],
                        "split_variant": split_variant,
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
                        "utterance_id": utterance_stem,
                        "sentence_id": sentence_id_from_utterance(wav_name),
                        "mos": float(row["mean"]),
                        "system_mean_mos": system_means.get(system_id),
                        "listener_scores": scores,
                        "n_listeners": len(scores),
                        "listener_mean_mos": mean([float(score) for score in scores]),
                    }
                    split_rows.append(entry)
                    all_rows.append(entry)
                    row_id += 1

            rows_by_split[split_key] = split_rows

    if not all_rows:
        raise SystemExit(f"no SOMOS rows found under {data_root}; run download/extract first")

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
        "archive": {
            "url": ARCHIVE_URL,
            "size_bytes": ARCHIVE_SIZE,
            "md5": ARCHIVE_MD5,
        },
        "rows": len(all_rows),
        "rows_with_audio": sum(1 for row in all_rows if row["audio_exists"]),
        "splits": {
            split: {
                "rows": len(rows),
                "rows_with_audio": sum(1 for row in rows if row["audio_exists"]),
                "systems": len({row["system_id"] for row in rows}),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "extract", "prepare", "all"])
    parser.add_argument("--data-root", help="dataset root; defaults to SOMOS_DATA_ROOT or config")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files/manifests")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="extract only metadata from somos.zip, leaving audios.zip unexpanded",
    )
    args = parser.parse_args(argv)

    data_root = resolve_data_root(args.data_root)
    print(f"data_root: {data_root}")
    if args.command in {"download", "all"}:
        download_dataset(data_root, overwrite=args.overwrite)
    if args.command in {"extract", "all"}:
        extract_dataset(data_root, overwrite=args.overwrite, metadata_only=args.metadata_only)
    if args.command in {"prepare", "all"}:
        prepare_dataset(data_root, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
