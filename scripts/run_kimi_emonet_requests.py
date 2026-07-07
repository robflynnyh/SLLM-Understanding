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
