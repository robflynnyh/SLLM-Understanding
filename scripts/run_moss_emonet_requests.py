#!/usr/bin/env python3
"""Run MOSS-Audio on EmoNet request JSONL files.

The script accepts the same request JSONL format as the Kimi runner and writes
one raw prediction row per request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MOSS_CONFIG_PATH = REPO_ROOT / "configs" / "moss_audio.json"
MOSS_AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"


def configure_hf_cache() -> None:
    with MOSS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    hf_home = Path(os.environ.get("HF_HOME") or config["default_hf_home"]).expanduser().resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ.get(
        "HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")
    )
    hf_home.mkdir(parents=True, exist_ok=True)


def score_field_suffix(score_scale: dict[str, Any]) -> str:
    return f"{int(score_scale['min'])}_{int(score_scale['max'])}"


def parse_score(text: str, score_scale: dict[str, Any]) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+)", text)
    if not match:
        return None
    value = float(match.group(0))
    if float(score_scale["min"]) <= value <= float(score_scale["max"]):
        return value
    return None


def parse_final_score(text: str, score_scale: dict[str, Any]) -> float | None:
    payload = parse_json_object(text)
    if payload is not None:
        for key in ("score", "final_score"):
            score = parse_numeric_score(payload.get(key), score_scale)
            if score is not None:
                return score

    patterns = [
        r"final\s*score\s*[:=]\s*([-+]?(?:\d+\.\d+|\d+))",
        r"score\s*[:=]\s*([-+]?(?:\d+\.\d+|\d+))",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for raw_value in reversed(matches):
            value = float(raw_value)
            if float(score_scale["min"]) <= value <= float(score_scale["max"]):
                return value
    return parse_score(text, score_scale)


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


def parse_choice(text: str) -> str | None:
    payload = parse_json_object(text)
    if payload is not None:
        for key in ("choice", "answer", "synthetic", "synthetic_audio"):
            value = payload.get(key)
            if isinstance(value, str):
                parsed = parse_choice(value)
                if parsed is not None:
                    return parsed

    stripped = text.strip().upper()
    if stripped in {"A", "B"}:
        return stripped
    match = re.match(r"\s*([AB])(?:\b|[^A-Z])", stripped)
    if match:
        return match.group(1)
    match = re.search(r"\b([AB])\b", stripped)
    if match:
        return match.group(1)
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

    expected = set(expected_emotions)
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


def parse_quality_feature_scores(
    text: str,
    expected_features: list[str],
    score_scale: dict[str, Any],
) -> dict[str, Any]:
    payload = parse_json_object(text)
    if payload is None:
        return {
            "feature_scores": {},
            "feature_observations": {},
            "overall_score": None,
            "overall_reason": None,
            "missing_fields": [*expected_features, "overall_score"],
            "extra_fields": [],
            "invalid_scores": {},
            "structure_warning": None,
            "parse_error": "response did not contain a JSON object",
        }

    raw_scores = payload.get("scores")
    raw_observations = payload.get("observations")
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    if not isinstance(raw_observations, dict):
        raw_observations = {}

    expected_fields = [*expected_features, "overall_score"]
    parsed_scores: dict[str, float] = {}
    invalid_scores: dict[str, Any] = {}
    for field in expected_features:
        if field not in raw_scores:
            continue
        score = parse_numeric_score(raw_scores[field], score_scale)
        if score is None:
            invalid_scores[field] = raw_scores[field]
        else:
            parsed_scores[field] = score
    overall_score = parse_numeric_score(payload.get("overall_score"), score_scale)
    if overall_score is not None:
        parsed_scores["overall_score"] = overall_score
    elif "overall_score" in payload:
        invalid_scores["overall_score"] = payload["overall_score"]

    feature_observations = {
        feature: raw_observations[feature]
        for feature in expected_features
        if isinstance(raw_observations.get(feature), str) and raw_observations[feature].strip()
    }
    missing_observations = [
        feature for feature in expected_features if feature not in feature_observations
    ]
    overall_reason = payload.get("overall_reason")
    if not isinstance(overall_reason, str) or not overall_reason.strip():
        overall_reason = None

    missing_score_fields = [field for field in expected_fields if field not in parsed_scores]
    missing_fields = list(missing_score_fields)
    missing_fields.extend(f"observations.{field}" for field in missing_observations)
    if overall_reason is None:
        missing_fields.append("overall_reason")
    expected_top_level = {"observations", "scores", "overall_score", "overall_reason"}
    extra_fields = sorted(str(field) for field in payload if field not in expected_top_level)
    extra_fields.extend(
        f"scores.{field}" for field in raw_scores if field not in set(expected_features)
    )
    extra_fields.extend(
        f"observations.{field}"
        for field in raw_observations
        if field not in set(expected_features)
    )
    parse_error = None
    if missing_score_fields or invalid_scores:
        parse_error = "JSON did not contain every expected numeric score within scale"
    structure_warning = None
    if missing_observations or overall_reason is None or extra_fields:
        structure_warning = "JSON evidence fields did not exactly match the requested structure"

    return {
        "feature_scores": {
            feature: parsed_scores[feature]
            for feature in expected_features
            if feature in parsed_scores
        },
        "feature_observations": feature_observations,
        "overall_score": parsed_scores.get("overall_score"),
        "overall_reason": overall_reason,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "invalid_scores": invalid_scores,
        "structure_warning": structure_warning,
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


def infer_model_name(model_path: Path) -> str:
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        name_or_path = config.get("_name_or_path")
        if isinstance(name_or_path, str) and name_or_path:
            return name_or_path
    return model_path.name


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


def load_moss_audio(
    model_path: Path,
    device_map: str | None,
    dtype: str,
    max_gpu_memory: str | None,
    max_primary_gpu_memory: str | None,
    max_cpu_memory: str | None,
):
    import torch
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    if device_map is None:
        if torch.cuda.is_available():
            device_map = "cuda:0"
        elif torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "device_map": device_map,
    }
    max_memory = cuda_max_memory(max_gpu_memory, max_primary_gpu_memory, max_cpu_memory)
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    model = MossAudioModel.from_pretrained(str(model_path), **model_kwargs)
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        enable_time_marker=True,
    )
    return model, processor


def moss_prompt_with_audio_placeholders(prompt: str, audio_count: int) -> str:
    if audio_count <= 1 or MOSS_AUDIO_PLACEHOLDER in prompt:
        return prompt

    content = prompt
    if audio_count == 2 and "[first audio]" in content and "[second audio]" in content:
        content = content.replace("[first audio]", MOSS_AUDIO_PLACEHOLDER, 1)
        content = content.replace("[second audio]", MOSS_AUDIO_PLACEHOLDER, 1)
    else:
        placeholders = "\n".join(
            f"Audio {index + 1}: {MOSS_AUDIO_PLACEHOLDER}" for index in range(audio_count)
        )
        content = f"{placeholders}\n\n{content}"

    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prediction(request: dict[str, Any], text: str, model_name: str) -> dict[str, Any]:
    mode = request.get("mode", "one_by_one")
    prediction = {
        "request_id": request["request_id"],
        "mode": mode,
        "row_id": request["row_id"],
        "target_label": request["target_label"],
        "prompt": request["prompt"],
        "raw_response_text": text,
        "raw_score_scale": request.get("raw_score_scale"),
        "raw_choice_set": request.get("raw_choice_set"),
        "human_mean_score_raw_0_2": request.get("human_mean_score_raw_0_2"),
        "human_mean_score_0_10": request.get("human_mean_score_0_10"),
        "model": model_name,
    }
    if mode in {
        "one_by_one",
        "one_by_one_paper_0_10",
        "one_by_one_paper_0_10_thinking",
        "one_by_one_human_rubric",
    }:
        if mode == "one_by_one_paper_0_10_thinking":
            parsed_score = parse_final_score(text, request["raw_score_scale"])
        else:
            parsed_score = parse_score(text, request["raw_score_scale"])
        prediction.update(
            {
                "scored_emotion": request["scored_emotion"],
                "raw_parsed_score": parsed_score,
                f"raw_parsed_score_{score_field_suffix(request['raw_score_scale'])}": parsed_score,
                "raw_parse_error": None if parsed_score is not None else "no numeric score within scale",
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

    if mode == "quality_features_1_10":
        parsed = parse_quality_feature_scores(
            text,
            request["scored_features"],
            request["raw_score_scale"],
        )
        suffix = score_field_suffix(request["raw_score_scale"])
        prediction.update(
            {
                "scored_features": request["scored_features"],
                "raw_parsed_feature_scores": parsed["feature_scores"],
                f"raw_parsed_feature_scores_{suffix}": parsed["feature_scores"],
                "raw_feature_observations": parsed["feature_observations"],
                "raw_parsed_score": parsed["overall_score"],
                f"raw_parsed_score_{suffix}": parsed["overall_score"],
                "raw_overall_reason": parsed["overall_reason"],
                "raw_missing_fields": parsed["missing_fields"],
                "raw_extra_fields": parsed["extra_fields"],
                "raw_invalid_scores": parsed["invalid_scores"],
                "raw_structure_warning": parsed["structure_warning"],
                "raw_parse_error": parsed["parse_error"],
            }
        )
        return prediction

    if mode == "pairwise_real_vs_synthetic":
        parsed_choice = parse_choice(text)
        correct_choice = request["correct_choice"]
        prediction.update(
            {
                "pair_id": request["pair_id"],
                "prompt_mode": request.get("prompt_mode"),
                "direction": request["direction"],
                "question_target": request.get("question_target"),
                "audio_a_label": request["audio_a_label"],
                "audio_b_label": request["audio_b_label"],
                "correct_choice": correct_choice,
                "raw_parsed_choice": parsed_choice,
                "is_correct": None if parsed_choice is None else parsed_choice == correct_choice,
                "raw_parse_error": None if parsed_choice is not None else "no A/B choice parsed",
            }
        )
        return prediction

    if mode == "all_at_once":
        parsed = parse_all_at_once_scores(
            text,
            request["scored_emotions"],
            request["raw_score_scale"],
        )
        suffix = score_field_suffix(request["raw_score_scale"])
        prediction.update(
            {
                "scored_emotions": request["scored_emotions"],
                "raw_parsed_scores": parsed["scores"],
                f"raw_parsed_scores_{suffix}": parsed["scores"],
                "raw_missing_emotions": parsed["missing_emotions"],
                "raw_extra_emotions": parsed["extra_emotions"],
                "raw_invalid_scores": parsed["invalid_scores"],
                "raw_parse_error": parsed["parse_error"],
            }
        )
        return prediction

    raise ValueError(f"unknown request mode: {mode}")


def generate_text(
    model,
    processor,
    audio_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None = None,
) -> str:
    import torch
    from src.audio_io import load_audio

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    raw_audios = [load_audio(audio_path, sample_rate=processor.config.mel_sr) for audio_path in audio_paths]
    processor_prompt = moss_prompt_with_audio_placeholders(prompt, len(raw_audios))
    inputs = processor(text=processor_prompt, audios=raw_audios, return_tensors="pt")
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="local MOSS-Audio model path")
    parser.add_argument("--model-name", help="model name recorded in prediction JSONL")
    parser.add_argument("--requests", required=True, help="request JSONL from build_emonet_requests.py")
    parser.add_argument("--output", required=True, help="raw prediction JSONL path")
    parser.add_argument("--limit", type=int, help="limit requests for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output if it exists")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1, help="number of generations per request")
    parser.add_argument("--seed", type=int, help="base random seed; actual seed is offset by request and repeat")
    parser.add_argument("--device-map", help="transformers device_map, e.g. auto or cuda:0")
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

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    with output_path.open("w", encoding="utf-8") as handle:
        for request_index, request in enumerate(requests):
            if request.get("audio_paths"):
                audio_paths = list(request["audio_paths"])
            elif request.get("audio_path"):
                audio_paths = [request["audio_path"]]
            else:
                raise ValueError(f"request has no audio path(s): {request['request_id']}")
            for repeat_index in range(args.repeats):
                generation_seed = None
                if args.seed is not None:
                    generation_seed = args.seed + request_index * args.repeats + repeat_index
                text = generate_text(
                    model=model,
                    processor=processor,
                    audio_paths=audio_paths,
                    prompt=request["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=generation_seed,
                )
                prediction = build_prediction(request, text, model_name)
                prediction["repeat_index"] = repeat_index
                prediction["repeat_count"] = args.repeats
                prediction["generation_seed"] = generation_seed
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
                handle.flush()

    print(f"wrote: {output_path}")
    print(f"requests: {len(requests)}")
    print(f"repeats: {args.repeats}")
    print(f"generations: {len(requests) * args.repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
