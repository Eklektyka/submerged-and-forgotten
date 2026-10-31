"""
analyze_non_crashed.py

Pulls the 92 cases where vehicle_associated_with_disappearance=True
but vehicle_found_crashed=False, categorises them by likely reason,
and prints a full breakdown.
"""

import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT_FILE  = "vehicle_associated_cases.jsonl"
OUTPUT_FILE = "non_crashed_vehicle_cases.jsonl"

# ---------------------------------------------------------------------------
# Keyword-based categorisation rules (checked in order — first match wins)
# ---------------------------------------------------------------------------
CATEGORIES = [
    (
        "Homicide — vehicle used or linked to crime",
        [
            "murder", "homicide", "killed", "foul play", "shot", "stabbed",
            "strangled", "convicted", "arrested", "suspect", "abducted",
            "kidnap", "body was found", "remains were found",
            "charged with", "sentenced", "guilty",
        ],
    ),
    (
        "Suicide / likely self-harm — vehicle parked at scene",
        [
            "suicide", "took his own life", "took her own life",
            "self-inflicted", "jumped", "intentionally",
            "bridge", "railroad", "train", "jumped from",
            "believed to have walked", "believed to have gone into",
            "no foul play", "no indication of foul play",
            "drowned", "drowning",
        ],
    ),
    (
        "Vehicle abandoned — found parked or left at scene",
        [
            "abandoned", "parked", "parking lot", "left his car",
            "left her car", "left their car", "left his truck",
            "left her truck", "left his vehicle", "left her vehicle",
            "found his car", "found her car", "located his car",
            "located her car", "keys inside", "keys in the ignition",
            "found running", "engine running",
        ],
    ),
    (
        "Remains / person found away from vehicle (vehicle intact)",
        [
            "remains were found", "body was found", "skeletal remains",
            "bones were found", "identified", "found in a wooded",
            "found in the woods", "found in a field", "found near",
            "found off", "found along", "found on the side",
        ],
    ),
    (
        "Vehicle never recovered — still missing",
        [
            "never found", "never been found", "never recovered",
            "still missing", "whereabouts unknown", "vehicle has not been found",
            "car has not been found", "truck has not been found",
        ],
    ),
    (
        "Accident / medical — vehicle not crashed",
        [
            "accident", "medical", "seizure", "diabetic", "heart attack",
            "fell", "exposure", "hypothermia", "lost", "wandered",
            "dementia", "alzheimer",
        ],
    ),
]

CATEGORY_OTHER = "Other / unclear"


def categorise(narrative: str, evidence: str) -> str:
    combined = (narrative + " " + evidence).lower()
    for label, keywords in CATEGORIES:
        if any(kw in combined for kw in keywords):
            return label
    return CATEGORY_OTHER


def load_target_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            vc = c.get("vehicle_classification", {})
            if (
                vc.get("vehicle_associated_with_disappearance") is True
                and vc.get("vehicle_found_crashed") is False
            ):
                cases.append(c)
    return cases


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def main() -> None:
    if not Path(INPUT_FILE).exists():
        raise SystemExit(f"ERROR: '{INPUT_FILE}' not found.")

    cases = load_target_cases(INPUT_FILE)
    print(f"Found {len(cases)} cases: vehicle associated but NOT found crashed.\n")

    # Categorise
    by_category: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        vc        = case.get("vehicle_classification", {})
        narrative = clean(case.get("narrative", ""))
        evidence  = clean(vc.get("evidence", ""))
        category  = categorise(narrative, evidence)
        case["_category"] = category
        by_category[category].append(case)

    # --- Summary ---
    total = len(cases)
    lines = []
    lines.append("=" * 68)
    lines.append("VEHICLE-ASSOCIATED CASES WHERE VEHICLE WAS NOT FOUND CRASHED")
    lines.append("=" * 68)
    lines.append(f"\nTotal: {total}\n")

    cat_order = [label for label, _ in CATEGORIES] + [CATEGORY_OTHER]
    for cat in cat_order:
        group = by_category.get(cat, [])
        if not group:
            continue
        pct = len(group) / total * 100
        lines.append(f"  {cat}")
        lines.append(f"  {'─' * 60}")
        lines.append(f"  Count: {len(group)}  ({pct:.1f}%)\n")
        for c in group:
            vc       = c.get("vehicle_classification", {})
            name     = c.get("name", "Unknown")
            location = clean(c.get("missing_from", ""))
            since    = c.get("missing_since", "")
            evidence = clean(vc.get("evidence", ""))
            conf     = vc.get("confidence", "?")
            lines.append(f"    • {name}  |  {location}  |  missing {since}  [{conf}]")
            lines.append(f"      {evidence}")
        lines.append("")

    summary = "\n".join(lines)
    print(summary)

    # Save JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Saved {len(cases)} cases to '{OUTPUT_FILE}'.")

    with open("non_crashed_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("Saved summary to 'non_crashed_summary.txt'.")


if __name__ == "__main__":
    main()
