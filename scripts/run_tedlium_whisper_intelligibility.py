#!/usr/bin/env python3
"""Run Whisper ASR on TED-LIUM real-vs-synthetic audio and compute WER deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "tedlium_real_vs_synthetic.json"
DEFAULT_MODEL_PATH = (
    "/store/store5/data/acp21rjf/models/Kimi-Audio-7B-Instruct/whisper-large-v3"
)
DEFAULT_OPENAI_MODEL = "large-v3"
DEFAULT_OPENAI_DOWNLOAD_ROOT = "/store/store5/data/acp21rjf/models/whisper"
DEFAULT_DATA_ROOT = "/store/store5/data/acp21rjf/data/tedlium-moss-real-vs-synthetic"
DEFAULT_SAMPLE_RATE = 16_000


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def resolve_data_root(raw_value: str | None) -> Path:
    config = load_config()
    value = (
        raw_value
        or os.environ.get("TEDLIUM_SYNTH_DATA_ROOT")
        or config.get("default_data_root")
        or DEFAULT_DATA_ROOT
    )
    return Path(value).expanduser().resolve()


def read_manifest(data_root: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    manifest = data_root / "manifests" / f"{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_word != hyp_word)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current[j] = min(substitution, insertion, deletion)
        previous = current
    return previous[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> tuple[float, int, int]:
    reference_words = normalize_text(reference_text).split()
    hypothesis_words = normalize_text(hypothesis_text).split()
    edits = edit_distance(reference_words, hypothesis_words)
    if not reference_words:
        return (math.nan if hypothesis_words else 0.0), edits, 0
    return edits / len(reference_words), edits, len(reference_words)


def load_audio(path: Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        audio = waveform.mean(dim=0)
        if sample_rate != target_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
        return audio.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        import soundfile as sf
        from scipy.signal import resample_poly

        data, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
        audio = np.asarray(data)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != target_sample_rate:
            divisor = math.gcd(sample_rate, target_sample_rate)
            audio = resample_poly(audio, target_sample_rate // divisor, sample_rate // divisor)
        return audio.astype(np.float32, copy=False)


def build_records(data_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        common = {
            "split": row["split"],
            "utterance_id": row["utterance_id"],
            "pair_id": row["utterance_id"],
            "reference_text": row["text"],
        }
        records.append(
            {
                **common,
                "sample_type": "source",
                "audio_path": row["real_audio_path"],
                "absolute_audio_path": str((data_root / row["real_audio_path"]).resolve()),
            }
        )
        records.append(
            {
                **common,
                "sample_type": "synthetic",
                "audio_path": row["synthetic_audio_path"],
                "absolute_audio_path": str((data_root / row["synthetic_audio_path"]).resolve()),
            }
        )
    return records


def transcribe_records(
    records: list[dict[str, Any]],
    backend: str,
    model_path: Path,
    openai_model: str,
    openai_download_root: Path,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    dtype: str,
) -> list[dict[str, Any]]:
    if backend == "openai":
        return transcribe_records_openai(
            records,
            model_name=openai_model,
            download_root=openai_download_root,
            device=device,
            dtype=dtype,
        )
    return transcribe_records_transformers(
        records,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
    )


@torch.inference_mode()
def transcribe_records_transformers(
    records: list[dict[str, Any]],
    model_path: Path,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    dtype: str,
) -> list[dict[str, Any]]:
    torch_dtype = {
        "auto": torch.float16 if torch.cuda.is_available() else torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved_device.type == "cpu":
        torch_dtype = torch.float32

    processor = WhisperProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = load_whisper_model(model_path, torch_dtype=torch_dtype).to(resolved_device)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    output_rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Whisper ASR", unit="batch"):
        batch = records[start : start + batch_size]
        audios = [load_audio(Path(row["absolute_audio_path"])) for row in batch]
        inputs = processor(
            audios,
            sampling_rate=DEFAULT_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device=resolved_device, dtype=torch_dtype)
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for row, prediction in zip(batch, predictions):
            wer, edits, ref_words = word_error_rate(row["reference_text"], prediction)
            output_rows.append(
                {
                    "split": row["split"],
                    "utterance_id": row["utterance_id"],
                    "pair_id": row["pair_id"],
                    "sample_type": row["sample_type"],
                    "audio_path": row["audio_path"],
                    "reference_text": row["reference_text"],
                    "prediction_text": prediction.strip(),
                    "normalized_reference": normalize_text(row["reference_text"]),
                    "normalized_prediction": normalize_text(prediction),
                    "reference_words": ref_words,
                    "edit_distance": edits,
                    "wer": f"{wer:.8f}" if not math.isnan(wer) else "nan",
                }
            )
    return output_rows


@torch.inference_mode()
def transcribe_records_openai(
    records: list[dict[str, Any]],
    model_name: str,
    download_root: Path,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    import whisper

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    download_root.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name, device=resolved_device, download_root=str(download_root))
    use_fp16 = resolved_device.startswith("cuda") and dtype != "float32"

    output_rows: list[dict[str, Any]] = []
    for row in tqdm(records, desc="Whisper ASR", unit="file"):
        result = model.transcribe(
            row["absolute_audio_path"],
            language="en",
            task="transcribe",
            fp16=use_fp16,
            temperature=0.0,
            verbose=False,
            condition_on_previous_text=False,
        )
        prediction = str(result.get("text", "")).strip()
        wer, edits, ref_words = word_error_rate(row["reference_text"], prediction)
        output_rows.append(
            {
                "split": row["split"],
                "utterance_id": row["utterance_id"],
                "pair_id": row["pair_id"],
                "sample_type": row["sample_type"],
                "audio_path": row["audio_path"],
                "reference_text": row["reference_text"],
                "prediction_text": prediction,
                "normalized_reference": normalize_text(row["reference_text"]),
                "normalized_prediction": normalize_text(prediction),
                "reference_words": ref_words,
                "edit_distance": edits,
                "wer": f"{wer:.8f}" if not math.isnan(wer) else "nan",
            }
        )
    return output_rows


def load_whisper_model(model_path: Path, torch_dtype: torch.dtype) -> WhisperForConditionalGeneration:
    try:
        return WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'get'" not in str(exc):
            raise

    from safetensors.torch import load_file

    config = WhisperConfig.from_pretrained(str(model_path), local_files_only=True)
    model = WhisperForConditionalGeneration(config)
    state_dict = load_file(str(model_path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"proj_out.weight"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"unexpected Whisper checkpoint keys: missing={missing}, unexpected={unexpected}"
        )
    model.tie_weights()
    model.generation_config = GenerationConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    return model.to(dtype=torch_dtype)


def pair_rows(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in transcripts:
        grouped.setdefault((row["split"], row["pair_id"]), {})[row["sample_type"]] = row

    pairs: list[dict[str, Any]] = []
    for (split, pair_id), rows in sorted(grouped.items()):
        source = rows.get("source")
        synthetic = rows.get("synthetic")
        if source is None or synthetic is None:
            continue
        source_wer = float(source["wer"])
        synthetic_wer = float(synthetic["wer"])
        pairs.append(
            {
                "split": split,
                "pair_id": pair_id,
                "source_audio": source["audio_path"],
                "synthetic_audio": synthetic["audio_path"],
                "reference_text": source["reference_text"],
                "source_prediction": source["prediction_text"],
                "synthetic_prediction": synthetic["prediction_text"],
                "reference_words": source["reference_words"],
                "source_edit_distance": source["edit_distance"],
                "synthetic_edit_distance": synthetic["edit_distance"],
                "source_wer": f"{source_wer:.8f}",
                "synthetic_wer": f"{synthetic_wer:.8f}",
                "wer_delta_synthetic_minus_source": f"{synthetic_wer - source_wer:.8f}",
            }
        )
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for split in sorted({row["split"] for row in pairs} | {"all"}):
        selected = pairs if split == "all" else [row for row in pairs if row["split"] == split]
        if not selected:
            continue
        source = [float(row["source_wer"]) for row in selected]
        synthetic = [float(row["synthetic_wer"]) for row in selected]
        delta = [float(row["wer_delta_synthetic_minus_source"]) for row in selected]
        out.append(
            {
                "split": split,
                "pairs": str(len(selected)),
                "source_wer_mean": f"{sum(source) / len(source):.8f}",
                "synthetic_wer_mean": f"{sum(synthetic) / len(synthetic):.8f}",
                "wer_delta_mean": f"{sum(delta) / len(delta):.8f}",
                "source_wer_median": f"{median(source):.8f}",
                "synthetic_wer_median": f"{median(synthetic):.8f}",
                "wer_delta_median": f"{median(delta):.8f}",
            }
        )
    return out


def median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="TED-LIUM real-vs-synthetic dataset root")
    parser.add_argument("--backend", choices=["openai", "transformers"], default="openai")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-download-root", default=DEFAULT_OPENAI_DOWNLOAD_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    parser.add_argument("--limit", type=int, help="limit rows per split for smoke tests")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    model_path = Path(args.model_path).expanduser().resolve()
    openai_download_root = Path(args.openai_download_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    manifest_rows: list[dict[str, Any]] = []
    for split in args.splits:
        manifest_rows.extend(read_manifest(data_root, split, args.limit))
    records = build_records(data_root, manifest_rows)
    transcripts = transcribe_records(
        records,
        backend=args.backend,
        model_path=model_path,
        openai_model=args.openai_model,
        openai_download_root=openai_download_root,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    pairs = pair_rows(transcripts)

    transcript_fields = [
        "split",
        "utterance_id",
        "pair_id",
        "sample_type",
        "audio_path",
        "reference_text",
        "prediction_text",
        "normalized_reference",
        "normalized_prediction",
        "reference_words",
        "edit_distance",
        "wer",
    ]
    pair_fields = [
        "split",
        "pair_id",
        "source_audio",
        "synthetic_audio",
        "reference_text",
        "source_prediction",
        "synthetic_prediction",
        "reference_words",
        "source_edit_distance",
        "synthetic_edit_distance",
        "source_wer",
        "synthetic_wer",
        "wer_delta_synthetic_minus_source",
    ]
    summary_fields = [
        "split",
        "pairs",
        "source_wer_mean",
        "synthetic_wer_mean",
        "wer_delta_mean",
        "source_wer_median",
        "synthetic_wer_median",
        "wer_delta_median",
    ]

    for split in args.splits:
        write_csv(
            output_dir / f"{split}_whisper_transcripts.csv",
            [row for row in transcripts if row["split"] == split],
            transcript_fields,
        )
        write_csv(
            output_dir / f"{split}_whisper_wer_delta.csv",
            [row for row in pairs if row["split"] == split],
            pair_fields,
        )
    write_csv(output_dir / "all_whisper_transcripts.csv", transcripts, transcript_fields)
    write_csv(output_dir / "all_whisper_wer_delta.csv", pairs, pair_fields)
    summary = summarize(pairs)
    write_csv(output_dir / "whisper_wer_summary.csv", summary, summary_fields)

    for row in summary:
        print(
            "split={split} pairs={pairs} source_wer={source_wer_mean} "
            "synthetic_wer={synthetic_wer_mean} delta={wer_delta_mean}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
