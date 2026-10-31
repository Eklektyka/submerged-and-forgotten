"""
analyze_results.py

Reads charley_results.jsonl, groups cases by vehicle_found_crashed classification,
prints a summary to stdout, saves crashed_vehicle_cases.jsonl and summary.txt.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

INPUT_FILE = "charley_results.jsonl"
CRASHED_FILE = "crashed_vehicle_cases.jsonl"
SUMMARY_FILE = "summary.txt"


def load_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno} skipped — JSON error: {e}", file=sys.stderr)
    return cases


def get_classification(case: dict) -> tuple[bool | None, str]:
    """Return (vehicle_found_crashed, confidence) from a case dict."""
    cr = case.get("classification_result") or {}
    crashed = cr.get("vehicle_found_crashed")
    confidence = (cr.get("confidence") or "unknown").lower()
    return crashed, confidence


def build_summary(cases: list[dict]) -> tuple[str, list[dict]]:
    """
    Returns (summary_text, crashed_cases_list).
    """
    total = len(cases)

    # Group cases
    crashed_by_confidence: dict[str, list[dict]] = defaultdict(list)
    not_crashed: list[dict] = []

    for case in cases:
        crashed, confidence = get_classification(case)
        if crashed is True:
            crashed_by_confidence[confidence].append(case)
        else:
            not_crashed.append(case)

    crashed_cases = [c for group in crashed_by_confidence.values() for c in group]
    total_crashed = len(crashed_cases)
    total_not_crashed = len(not_crashed)
    pct = (total_crashed / total * 100) if total else 0.0

    confidence_order = ["high", "medium", "low", "error", "unknown"]

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("CHARLEY PROJECT — VEHICLE CRASH CLASSIFICATION SUMMARY")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"Total cases analysed          : {total}")
    lines.append(f"Vehicle found crashed = True  : {total_crashed}  ({pct:.1f}%)")

    for conf in confidence_order:
        count = len(crashed_by_confidence.get(conf, []))
        if count:
            lines.append(f"    {conf:<8}               : {count}")

    lines.append(f"Vehicle found crashed = False/null : {total_not_crashed}")
    lines.append("")

    # High-confidence True cases with evidence
    high_cases = crashed_by_confidence.get("high", [])
    if high_cases:
        lines.append("-" * 64)
        lines.append(f"HIGH-CONFIDENCE CRASHED VEHICLE CASES ({len(high_cases)} total)")
        lines.append("-" * 64)
        for case in high_cases:
            cr = case.get("classification_result", {})
            name = case.get("name", "Unknown")
            location = case.get("missing_from", "Unknown location")
            evidence = cr.get("evidence", "No evidence quoted.")
            lines.append("")
            lines.append(f"  Name     : {name}")
            lines.append(f"  Location : {location}")
            lines.append(f"  Evidence : {evidence}")
    else:
        lines.append("No high-confidence crashed-vehicle cases found.")

    lines.append("")
    lines.append("=" * 64)

    return "\n".join(lines), crashed_cases


def main() -> None:
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        sys.exit(
            f"ERROR: '{INPUT_FILE}' not found in the current directory ({Path.cwd()}).\n"
            "Run charley_scraper.py first to generate it."
        )

    print(f"Loading {INPUT_FILE}...")
    cases = load_cases(INPUT_FILE)
    print(f"Loaded {len(cases)} cases.")

    summary_text, crashed_cases = build_summary(cases)

    # Print summary to stdout
    print()
    print(summary_text)

    # Save crashed_vehicle_cases.jsonl
    with open(CRASHED_FILE, "w", encoding="utf-8") as f:
        for case in crashed_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(crashed_cases)} crashed-vehicle cases to '{CRASHED_FILE}'.")

    # Save summary.txt
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Saved summary to '{SUMMARY_FILE}'.")


if __name__ == "__main__":
    main()
