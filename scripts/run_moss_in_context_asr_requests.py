#!/usr/bin/env python3
"""Run MOSS-Audio on in-context-asr request JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from run_moss_emonet_requests import (
    configure_hf_cache,
    generate_text,
    infer_model_name,
    load_moss_audio,
    parse_json_object,
    read_requests,
)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return None


def parse_prediction(text: str, condition: str) -> dict[str, Any]:
    payload = parse_json_object(text)
    if payload is not None:
        if condition == "without_repeat":
            target_present = parse_bool(payload.get("target_present"))
            return {
                "target_present": target_present,
                "transcript": payload.get("transcript"),
                "raw_parse_error": None if target_present is not None else "missing boolean target_present",
            }

        before = parse_bool(payload.get("target_before_repeat"))
        after = parse_bool(payload.get("target_after_repeat"))
        parse_error = None
        if before is None or after is None:
            parse_error = "missing boolean target_before_repeat or target_after_repeat"
        return {
            "target_before_repeat": before,
            "target_after_repeat": after,
            "before_repeat_transcript": payload.get("before_repeat_transcript"),
            "after_repeat_transcript": payload.get("after_repeat_transcript"),
            "raw_parse_error": parse_error,
        }

    lowered = text.lower()
    if condition == "without_repeat":
        match = re.search(r"target_present\s*[:=]\s*(true|false|yes|no)", lowered)
        target_present = parse_bool(match.group(1)) if match else None
        return {
            "target_present": target_present,
            "transcript": None,
            "raw_parse_error": None if target_present is not None else "response did not contain a JSON object",
        }

    before_match = re.search(r"target_before_repeat\s*[:=]\s*(true|false|yes|no)", lowered)
    after_match = re.search(r"target_after_repeat\s*[:=]\s*(true|false|yes|no)", lowered)
    before = parse_bool(before_match.group(1)) if before_match else None
    after = parse_bool(after_match.group(1)) if after_match else None
    return {
        "target_before_repeat": before,
        "target_after_repeat": after,
        "before_repeat_transcript": None,
        "after_repeat_transcript": None,
        "raw_parse_error": (
            None
            if before is not None and after is not None
            else "response did not contain a JSON object"
        ),
    }


def build_prediction(request: dict[str, Any], text: str, model_name: str) -> dict[str, Any]:
    condition = request["condition"]
    parsed = parse_prediction(text, condition)
    return {
        "request_id": request["request_id"],
        "dataset": request.get("dataset", "in-context-asr"),
        "mode": request["mode"],
        "condition": condition,
        "row_id": request["row_id"],
        "target_label": request["target_label"],
        "targets": request["targets"],
        "separators": request["separators"],
        "audio_path": request["audio_path"],
        "prompt": request["prompt"],
        "raw_response_text": text,
        "model": model_name,
        **parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="local MOSS-Audio model path")
    parser.add_argument("--model-name", help="model name recorded in prediction JSONL")
    parser.add_argument("--requests", required=True, help="request JSONL path")
    parser.add_argument("--output", required=True, help="raw prediction JSONL path")
    parser.add_argument("--limit", type=int, help="limit requests for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output if it exists")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device-map", help="transformers device_map, e.g. cuda:0")
    parser.add_argument("--dtype", default="auto", help="dtype passed to from_pretrained")
    parser.add_argument("--max-gpu-memory", help="per-GPU max memory when using device_map")
    parser.add_argument("--max-primary-gpu-memory", help="GPU 0 max memory when using device_map")
    parser.add_argument("--max-cpu-memory", help="CPU max memory when using device_map")
    args = parser.parse_args()

    configure_hf_cache()

    model_path = Path(args.model_path).expanduser().resolve()
    request_path = Path(args.requests).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requests = read_requests(request_path, args.limit)
    model_name = args.model_name or infer_model_name(model_path)
    model, processor = load_moss_audio(
        model_path=model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        max_gpu_memory=args.max_gpu_memory,
        max_primary_gpu_memory=args.max_primary_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            text = generate_text(
                model=model,
                processor=processor,
                audio_path=request["audio_path"],
                prompt=request["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            prediction = build_prediction(request, text, model_name)
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")
            handle.flush()

    print(f"wrote: {output_path}")
    print(f"requests: {len(requests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
