#!/usr/bin/env python3
"""Summarize MOSS-Audio predictions for the in-context-asr probe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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

    correct_without = sum(bool_value(row, "target_present") is True for row in without)
    correct_clear = sum(bool_value(row, "target_after_repeat") is True for row in with_repeat)
    correct_corrupt = sum(bool_value(row, "target_before_repeat") is True for row in with_repeat)
    parse_errors = sum(row.get("raw_parse_error") is not None for row in predictions)

    print(f"predictions: {len(predictions)}")
    print(f"items: {len({row.get('row_id') for row in predictions})}")
    print(f"without_repeat_requests: {len(without)}")
    print(f"with_repeat_requests: {len(with_repeat)}")
    print(f"parsed: {len(predictions) - parse_errors}/{len(predictions)}")
    print(f"Correct in clear repeat: {fmt(pct(correct_clear, len(with_repeat)))}")
    print(f"Correct in corrupt repeat: {fmt(pct(correct_corrupt, len(with_repeat)))}")
    print(f"Correct without repeat: {fmt(pct(correct_without, len(without)))}")
    print(f"raw_parse_errors: {parse_errors}")
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
