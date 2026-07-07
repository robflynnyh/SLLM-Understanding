#!/usr/bin/env python3
"""Run Kimi-Audio on EmoNet request JSONL files.

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


def score_field_suffix(score_scale: dict[str, Any]) -> str:
    min_score = int(score_scale["min"])
    max_score = int(score_scale["max"])
    return f"{min_score}_{max_score}"


def parse_score(text: str, score_scale: dict[str, Any]) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+)", text)
    if not match:
        return None
    value = float(match.group(0))
    if float(score_scale["min"]) <= value <= float(score_scale["max"]):
        return value
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    start = stripped.find("{")
    if start == -1:
        return None

    in_string = False
    escaped = False
    depth = 0
    for index, char in enumerate(stripped[start:], start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def parse_numeric_score(value: Any, score_scale: dict[str, Any]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if float(score_scale["min"]) <= score <= float(score_scale["max"]):
        return score
    return None


def parse_all_at_once_scores(
    text: str,
    expected_emotions: list[str],
    score_scale: dict[str, Any],
) -> dict[str, Any]:
    payload = parse_json_object(text)
    if payload is None:
        return {
            "scores": {},
            "missing_emotions": expected_emotions,
            "extra_emotions": [],
            "invalid_scores": {},
            "parse_error": "response did not contain a JSON object",
        }

    raw_scores = payload.get("scores", payload)
    if not isinstance(raw_scores, dict):
        return {
            "scores": {},
            "missing_emotions": expected_emotions,
            "extra_emotions": [],
            "invalid_scores": {},
            "parse_error": "JSON scores field was not an object",
        }

    expected = set(expected_emotions)
    parsed_scores: dict[str, float] = {}
    invalid_scores: dict[str, Any] = {}
    for emotion in expected_emotions:
        if emotion not in raw_scores:
            continue
        score = parse_numeric_score(raw_scores[emotion], score_scale)
        if score is None:
            invalid_scores[emotion] = raw_scores[emotion]
        else:
            parsed_scores[emotion] = score

    extra_emotions = sorted(str(emotion) for emotion in raw_scores if emotion not in expected)
    missing_emotions = [emotion for emotion in expected_emotions if emotion not in parsed_scores]
    parse_error = None
    if missing_emotions or extra_emotions or invalid_scores:
        parse_error = "JSON scores did not exactly match expected emotions and numeric scale"

    return {
        "scores": parsed_scores,
        "missing_emotions": missing_emotions,
        "extra_emotions": extra_emotions,
        "invalid_scores": invalid_scores,
        "parse_error": parse_error,
    }


def parse_contrastive_scores(text: str, score_scale: dict[str, Any]) -> dict[str, Any]:
    payload = parse_json_object(text)
    if payload is None:
        return {
            "emotion_score": None,
            "opposite_score": None,
            "parse_error": "response did not contain a JSON object",
        }

    emotion_score = parse_numeric_score(payload.get("emotion_score"), score_scale)
    opposite_score = parse_numeric_score(payload.get("opposite_score"), score_scale)
    parse_error = None
    if emotion_score is None or opposite_score is None:
        parse_error = "JSON did not contain numeric emotion_score and opposite_score within scale"

    return {
        "emotion_score": emotion_score,
        "opposite_score": opposite_score,
        "parse_error": parse_error,
    }


def read_requests(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cuda_max_memory(
    max_gpu_memory: str | None,
    max_primary_gpu_memory: str | None,
    max_cpu_memory: str | None,
) -> dict[int | str, str] | None:
    import torch

    if not any([max_gpu_memory, max_primary_gpu_memory, max_cpu_memory]):
        return None

    max_memory: dict[int | str, str] = {}
    if max_gpu_memory or max_primary_gpu_memory:
        for index in range(torch.cuda.device_count()):
            max_memory[index] = max_gpu_memory or max_primary_gpu_memory or "18GiB"
        if max_primary_gpu_memory:
            max_memory[0] = max_primary_gpu_memory
    if max_cpu_memory:
        max_memory["cpu"] = max_cpu_memory
    return max_memory


def patch_transformers_empty_safetensors_metadata() -> None:
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils, "_kimi_safetensors_metadata_patch", False):
        return

    original_safe_open = modeling_utils.safe_open

    class SafeOpenWithMetadata:
        def __init__(self, *args, **kwargs):
            self._context = original_safe_open(*args, **kwargs)
            self._handle = None

        def __enter__(self):
            self._handle = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def metadata(self):
            return self._handle.metadata() or {"format": "pt"}

        def __getattr__(self, name):
            return getattr(self._handle, name)

    modeling_utils.safe_open = SafeOpenWithMetadata
    modeling_utils._kimi_safetensors_metadata_patch = True


def load_kimi_audio(
    model_path: Path,
    device_map: str | None,
    max_gpu_memory: str | None,
    max_primary_gpu_memory: str | None,
    max_cpu_memory: str | None,
):
    """Load Kimi-Audio with optional multi-GPU placement for small cards."""
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from kimia_infer.api.kimia import KimiAudio
    from kimia_infer.api.prompt_manager import KimiAPromptManager

    patch_transformers_empty_safetensors_metadata()

    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if not hasattr(config, "rope_theta"):
        config.rope_theta = raw_config.get("rope_theta", 10000.0)

    model = KimiAudio.__new__(KimiAudio)
    model_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
        max_memory = cuda_max_memory(max_gpu_memory, max_primary_gpu_memory, max_cpu_memory)
        if max_memory:
            model_kwargs["max_memory"] = max_memory

    model_class = get_class_from_dynamic_module(
        raw_config["auto_map"]["AutoModelForCausalLM"],
        str(model_path),
    )
    model_class._no_split_modules = ["MoonshotDecoderLayer"]
    model.alm = model_class.from_pretrained(str(model_path), **model_kwargs)
    if not device_map:
        model.alm = model.alm.to(torch.cuda.current_device())

    model_config = model.alm.config
    model.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
    model.kimia_token_offset = model_config.kimia_token_offset
    model.prompt_manager = KimiAPromptManager(
        model_path=str(model_path),
        kimia_token_offset=model.kimia_token_offset,
        kimia_text_audiodelaytokens=model.kimia_text_audiodelaytokens,
    )
    model.detokenizer = None
    model.extra_tokens = model.prompt_manager.extra_tokens
    model.eod_ids = [model.extra_tokens.msg_end, model.extra_tokens.media_end]
    return model


def build_prediction(request: dict[str, Any], text: str) -> dict[str, Any]:
    mode = request.get("mode", "one_by_one")
    prediction = {
        "request_id": request["request_id"],
        "mode": mode,
        "row_id": request["row_id"],
        "target_label": request["target_label"],
        "prompt": request["prompt"],
        "raw_response_text": text,
        "raw_score_scale": request["raw_score_scale"],
        "human_mean_score_raw_0_2": request.get("human_mean_score_raw_0_2"),
        "human_mean_score_0_10": request.get("human_mean_score_0_10"),
        "model": "moonshotai/Kimi-Audio-7B-Instruct",
    }
    if mode in {"one_by_one", "one_by_one_paper_0_10", "one_by_one_human_rubric"}:
        parsed_score = parse_score(text, request["raw_score_scale"])
        prediction.update(
            {
                "scored_emotion": request["scored_emotion"],
                "raw_parsed_score": parsed_score,
                f"raw_parsed_score_{score_field_suffix(request['raw_score_scale'])}": parsed_score,
            }
        )
        return prediction

    if mode == "one_by_one_contrastive_rubric":
        parsed = parse_contrastive_scores(text, request["raw_score_scale"])
        prediction.update(
            {
                "scored_emotion": request["scored_emotion"],
                "opposite_emotion": request["opposite_emotion"],
                "raw_parsed_emotion_score": parsed["emotion_score"],
                f"raw_parsed_emotion_score_{score_field_suffix(request['raw_score_scale'])}": parsed[
                    "emotion_score"
                ],
                "raw_parsed_opposite_score": parsed["opposite_score"],
                f"raw_parsed_opposite_score_{score_field_suffix(request['raw_score_scale'])}": parsed[
                    "opposite_score"
                ],
                "raw_parse_error": parsed["parse_error"],
            }
        )
        return prediction

    if mode == "all_at_once":
        parsed = parse_all_at_once_scores(
            text=text,
            expected_emotions=list(request["scored_emotions"]),
            score_scale=request["raw_score_scale"],
        )
        prediction.update(
            {
                "scored_emotions": request["scored_emotions"],
                "raw_parsed_scores_0_10": parsed["scores"],
                "raw_parse_error": parsed["parse_error"],
                "missing_emotions": parsed["missing_emotions"],
                "extra_emotions": parsed["extra_emotions"],
                "invalid_scores": parsed["invalid_scores"],
            }
        )
        return prediction

    raise ValueError(f"unknown request mode: {mode}")


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
    parser.add_argument("--audio-temperature", type=float, default=0.0)
    parser.add_argument("--audio-top-k", type=int, default=1)
    parser.add_argument("--text-temperature", type=float, default=0.0)
    parser.add_argument("--text-top-k", type=int, default=1)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--audio-repetition-window-size", type=int, default=64)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--text-repetition-window-size", type=int, default=16)
    parser.add_argument(
        "--device-map",
        help="optional transformers device_map for the main Kimi model, e.g. auto",
    )
    parser.add_argument(
        "--max-gpu-memory",
        help="per-visible-GPU max_memory for device_map loading, e.g. 18GiB",
    )
    parser.add_argument(
        "--max-primary-gpu-memory",
        help="GPU 0 max_memory for device_map loading; leave room for audio tokenizers",
    )
    parser.add_argument(
        "--max-cpu-memory",
        help="optional CPU max_memory for device_map loading, e.g. 64GiB",
    )
    args = parser.parse_args()

    request_path = Path(args.requests).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite to replace it")

    configure_hf_cache()

    model = load_kimi_audio(
        model_path=model_path,
        device_map=args.device_map,
        max_gpu_memory=args.max_gpu_memory,
        max_primary_gpu_memory=args.max_primary_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
    )
    sampling_params = {
        "audio_temperature": args.audio_temperature,
        "audio_top_k": args.audio_top_k,
        "text_temperature": args.text_temperature,
        "text_top_k": args.text_top_k,
        "audio_repetition_penalty": args.audio_repetition_penalty,
        "audio_repetition_window_size": args.audio_repetition_window_size,
        "text_repetition_penalty": args.text_repetition_penalty,
        "text_repetition_window_size": args.text_repetition_window_size,
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
            prediction = build_prediction(request, text)
            out.write(json.dumps(prediction, sort_keys=True) + "\n")
            out.flush()

    print(f"wrote: {output_path}")
    print(f"requests: {len(requests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
