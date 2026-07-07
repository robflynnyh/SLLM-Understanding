#!/usr/bin/env python3
"""Download Kimi-Audio model files to project storage.

By default this downloads the text-output subset needed for audio understanding
experiments and skips the large audio detokenizer/vocoder files. Use --full only
when speech generation is needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "kimi_audio.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", help="local model directory")
    parser.add_argument("--hf-home", help="HF cache directory")
    parser.add_argument("--full", action="store_true", help="include detokenizer and vocoder files")
    parser.add_argument("--revision", help="HF model revision")
    args = parser.parse_args()

    config = load_config()
    model_root = Path(args.model_root or config["default_model_root"]).expanduser().resolve()
    hf_home = Path(args.hf_home or os.environ.get("HF_HOME") or config["default_hf_home"]).expanduser().resolve()
    model_id = config["model_id"]
    revision = args.revision

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    hf_home.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for downloads. Install it in the Kimi env first."
        ) from exc

    allow_patterns = list(config["text_only_allow_patterns"])
    if args.full:
        allow_patterns.extend(config["full_extra_allow_patterns"])

    print(f"model_id: {model_id}")
    print(f"model_root: {model_root}")
    print(f"hf_home: {hf_home}")
    print(f"mode: {'full' if args.full else 'text-only'}")
    print("allow_patterns:")
    for pattern in allow_patterns:
        print(f"  - {pattern}")

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(model_root),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f"downloaded_to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

