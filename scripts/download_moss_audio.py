#!/usr/bin/env python3
"""Download MOSS-Audio model files to project storage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "moss_audio.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", help="HF model id")
    parser.add_argument("--model-root", help="local model directory")
    parser.add_argument("--hf-home", help="HF cache directory")
    parser.add_argument("--revision", help="HF model revision")
    args = parser.parse_args()

    config = load_config()
    model_id = args.model_id or config["model_id"]
    model_root = Path(args.model_root or config["default_model_root"]).expanduser().resolve()
    hf_home = Path(args.hf_home or os.environ.get("HF_HOME") or config["default_hf_home"]).expanduser().resolve()

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    hf_home.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for downloads. Install the MOSS env first."
        ) from exc

    print(f"model_id: {model_id}")
    print(f"model_root: {model_root}")
    print(f"hf_home: {hf_home}")

    path = snapshot_download(
        repo_id=model_id,
        revision=args.revision,
        local_dir=str(model_root),
        local_dir_use_symlinks=False,
    )
    print(f"downloaded_to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
