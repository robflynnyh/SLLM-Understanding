#!/usr/bin/env python3
"""Summarize MOSS-Audio predictions for the in-context-asr probe."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    EnglishTextNormalizer = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(count: int, total: int) -> float:
    return (count / total) * 100 if total else 0.0


def bool_value(row: dict[str, Any], key: str) -> bool | None:
    value = row.get(key)
    return value if isinstance(value, bool) else None


def normalize_text(text: str) -> str:
    if EnglishTextNormalizer is not None:
        return EnglishTextNormalizer()(text)
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def transcript_hit(text: Any, targets: list[str]) -> bool:
    normalized = normalize_text(str(text or ""))
    return any(normalize_text(target) in normalized for target in targets)


def split_on_separator(text: Any, separators: list[str]) -> tuple[str | None, str | None, str | None]:
    normalized = normalize_text(str(text or ""))
    for separator in separators:
        normalized_separator = normalize_text(separator)
        if normalized_separator in normalized:
            before, after = normalized.split(normalized_separator, 1)
            return before, after, normalized_separator
    return None, None, None


def transcript_for(row: dict[str, Any], key: str = "transcript") -> Any:
    return row.get(key) if row.get(key) is not None else row.get("raw_response_text")


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="raw prediction JSONL path")
    parser.add_argument("--show-raw", type=int, default=0, help="print this many raw response snippets")
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions).expanduser().resolve())
    without = [row for row in predictions if row.get("condition") == "without_repeat"]
    with_repeat = [row for row in predictions if row.get("condition") == "with_repeat"]

    bool_available = any(
        key in row
        for row in predictions
        for key in ("target_present", "target_before_repeat", "target_after_repeat")
    )
    correct_without = sum(bool_value(row, "target_present") is True for row in without)
    correct_clear = sum(bool_value(row, "target_after_repeat") is True for row in with_repeat)
    correct_corrupt = sum(bool_value(row, "target_before_repeat") is True for row in with_repeat)

    transcript_without = sum(
        transcript_hit(transcript_for(row), list(row.get("targets") or [])) for row in without
    )
    transcript_clear = 0
    transcript_corrupt = 0
    separators_found = 0
    for row in with_repeat:
        targets = list(row.get("targets") or [])
        if row.get("before_repeat_transcript") is not None or row.get("after_repeat_transcript") is not None:
            before = row.get("before_repeat_transcript")
            after = row.get("after_repeat_transcript")
            separators_found += 1
        else:
            before, after, separator = split_on_separator(transcript_for(row), list(row.get("separators") or []))
            if separator is not None:
                separators_found += 1
        transcript_corrupt += transcript_hit(before, targets)
        transcript_clear += transcript_hit(after, targets)
    parse_errors = sum(row.get("raw_parse_error") is not None for row in predictions)

    print(f"predictions: {len(predictions)}")
    print(f"items: {len({row.get('row_id') for row in predictions})}")
    print(f"without_repeat_requests: {len(without)}")
    print(f"with_repeat_requests: {len(with_repeat)}")
    print(f"parsed: {len(predictions) - parse_errors}/{len(predictions)}")
    print(f"separators_found: {separators_found}/{len(with_repeat)}")
    print("Original-repo transcript-search metrics:")
    print(f"Correct in clear repeat: {fmt(pct(transcript_clear, len(with_repeat)))}")
    print(f"Correct in corrupt repeat: {fmt(pct(transcript_corrupt, len(with_repeat)))}")
    print(f"Correct without repeat: {fmt(pct(transcript_without, len(without)))}")
    print(f"raw_parse_errors: {parse_errors}")
    if bool_available:
        print("Boolean-field metrics:")
        print(f"Correct in clear repeat: {fmt(pct(correct_clear, len(with_repeat)))}")
        print(f"Correct in corrupt repeat: {fmt(pct(correct_corrupt, len(with_repeat)))}")
        print(f"Correct without repeat: {fmt(pct(correct_without, len(without)))}")
        mismatches = 0
        for row in without:
            mismatches += bool_value(row, "target_present") is not transcript_hit(
                transcript_for(row), list(row.get("targets") or [])
            )
        for row in with_repeat:
            mismatches += bool_value(row, "target_before_repeat") is not transcript_hit(
                row.get("before_repeat_transcript"), list(row.get("targets") or [])
            )
            mismatches += bool_value(row, "target_after_repeat") is not transcript_hit(
                row.get("after_repeat_transcript"), list(row.get("targets") or [])
            )
        print(f"boolean_transcript_mismatches: {mismatches}")
        print(
            "without_repeat_target_present: "
            + ", ".join(f"{key}={value}" for key, value in sorted(Counter(row.get("target_present") for row in without).items(), key=lambda item: str(item[0])))
        )
        print(
            "with_repeat_target_before: "
            + ", ".join(f"{key}={value}" for key, value in sorted(Counter(row.get("target_before_repeat") for row in with_repeat).items(), key=lambda item: str(item[0])))
        )
        print(
            "with_repeat_target_after: "
            + ", ".join(f"{key}={value}" for key, value in sorted(Counter(row.get("target_after_repeat") for row in with_repeat).items(), key=lambda item: str(item[0])))
        )

    if args.show_raw:
        print("\nRaw response snippets:")
        for row in predictions[: args.show_raw]:
            text = str(row.get("raw_response_text", "")).replace("\n", "\\n")
            if len(text) > 500:
                text = text[:497] + "..."
            print(f"{row['request_id']}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
