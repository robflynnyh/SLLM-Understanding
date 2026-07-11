#!/usr/bin/env python3
"""Score audio pairs with the standalone AutoPCP metric."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm_understanding.metrics.autopcp import AutoPCP


def read_tsv(path: Path) -> tuple[list[str], list[str]]:
    src_paths: list[str] = []
    tgt_paths: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty TSV: {path}")
        missing = {"src_audio", "tgt_audio"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV is missing required column(s): {sorted(missing)}")
        for row in reader:
            src_paths.append(row["src_audio"])
            tgt_paths.append(row["tgt_audio"])
    return src_paths, tgt_paths


def write_scores(scores: list[float], output: str | None) -> None:
    lines = [f"{score:.8f}" for score in scores]
    if output is None or output == "-":
        for line in lines:
            print(line)
        return
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--tsv", help="TSV with src_audio and tgt_audio columns")
    input_group.add_argument(
        "--src",
        help="single source audio path; requires --tgt",
    )
    parser.add_argument("--tgt", help="single target audio path")
    parser.add_argument(
        "--output",
        "-o",
        help="score output path; defaults to stdout. Mean is printed to stderr.",
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, cuda:0, or auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", help="AutoPCP comparator cache directory")
    parser.add_argument("--comparator-path", help="existing .config file or checkpoint directory")
    parser.add_argument(
        "--encoder-path",
        default="facebook/wav2vec2-large-xlsr-53",
        help="Wav2Vec2 encoder path or Hugging Face id",
    )
    parser.add_argument("--pick-layer", type=int, default=9)
    parser.add_argument(
        "--no-symmetrize",
        action="store_true",
        help="disable source->target and target->source score averaging",
    )
    parser.add_argument("--progress", action="store_true", help="show encoder progress")
    args = parser.parse_args()

    if args.tsv:
        src_paths, tgt_paths = read_tsv(Path(args.tsv).expanduser().resolve())
    else:
        if not args.tgt:
            parser.error("--src requires --tgt")
        src_paths = [args.src]
        tgt_paths = [args.tgt]

    scorer = AutoPCP(
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        comparator_path=args.comparator_path,
        encoder_path=args.encoder_path,
        pick_layer=args.pick_layer,
        symmetrize=not args.no_symmetrize,
        progress=args.progress,
    )
    scores = scorer.score_pairs(src_paths, tgt_paths)
    write_scores(scores, args.output)
    mean = statistics.fmean(scores) if scores else float("nan")
    print(f"mean: {mean:.8f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
