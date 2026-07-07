#!/usr/bin/env python3
"""Run Kimi-Audio on one-by-one EmoNet request JSONL files.

This runner preserves raw model text for calibration experiments. It writes one
prediction JSONL row per request and never overwrites existing outputs unless
--overwrite is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
KIMI_CONFIG_PATH = REPO_ROOT / "configs" / "kimi_audio.json"


def configure_hf_cache() -> None:
    with KIMI_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    hf_home = Path(os.environ.get("HF_HOME") or config["default_hf_home"]).expanduser().resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ.get(
        "HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")
    )
    hf_home.mkdir(parents=True, exist_ok=True)


def parse_score(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+)", text)
    if not match:
        return None
    value = float(match.group(0))
    if 1.0 <= value <= 10.0:
        return value
    return None


def read_requests(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="request JSONL from build_emonet_requests.py")
    parser.add_argument("--output", required=True, help="prediction JSONL path")
    parser.add_argument(
        "--model-path",
        required=True,
        help="local Kimi model path. Pass a local path, not the HF ID, to avoid full snapshot surprises.",
    )
    parser.add_argument("--limit", type=int, help="limit requests for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    request_path = Path(args.requests).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it")

    configure_hf_cache()

    from kimia_infer.api.kimia import KimiAudio

    model = KimiAudio(model_path=str(model_path), load_detokenizer=False)
    sampling_params = {
        "audio_temperature": 0.0,
        "audio_top_k": 1,
        "text_temperature": 0.0,
        "text_top_k": 1,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
        "max_new_tokens": args.max_new_tokens,
    }

    requests = read_requests(request_path, args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for request in requests:
            messages = [
                {"role": "user", "message_type": "text", "content": request["prompt"]},
                {"role": "user", "message_type": "audio", "content": request["audio_path"]},
            ]
            _, text = model.generate(messages, **sampling_params, output_type="text")
            prediction = {
                "request_id": request["request_id"],
                "row_id": request["row_id"],
                "target_label": request["target_label"],
                "scored_emotion": request["scored_emotion"],
                "prompt": request["prompt"],
                "raw_response_text": text,
                "raw_parsed_score_1_10": parse_score(text),
                "raw_score_scale": request["raw_score_scale"],
                "human_mean_score_0_10": request.get("human_mean_score_0_10"),
                "model": "moonshotai/Kimi-Audio-7B-Instruct",
            }
            out.write(json.dumps(prediction, sort_keys=True) + "\n")
            out.flush()

    print(f"wrote: {output_path}")
    print(f"requests: {len(requests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
