"""
deep_analysis.py

Three analyses on already-classified data — no scraping:

  Task 1: Compare active (vehicle_missing) vs resolved (vehicle_associated / crashed)
  Task 2: Second-pass on the 127 active nulls — keyword resolution + optional API re-run
  Task 4: Pattern analysis of the 94 active vehicle_missing=True cases

Usage:
    python deep_analysis.py              # tasks 1, 2 (keyword only), 4
    python deep_analysis.py --reclassify # task 2 also re-runs nulls through API
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ACTIVE_FILE   = "active_sample_classified.jsonl"
RESOLVED_FILE = "vehicle_associated_cases.jsonl"
NULL_RECLASS_FILE = "active_nulls_reclassified.jsonl"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

# ---------------------------------------------------------------------------
# TASK 1 — Active vs Resolved comparison
# ---------------------------------------------------------------------------

def task1_compare(active: list[dict], resolved: list[dict]) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("TASK 1 — ACTIVE vs RESOLVED: VEHICLE COMPARISON")
    lines.append("=" * 64)

    # --- Active stats ---
    act_total  = len(active)
    act_true   = [c for c in active if c["vehicle_classification"].get("vehicle_missing") is True]
    act_false  = [c for c in active if c["vehicle_classification"].get("vehicle_missing") is False]
    act_null   = [c for c in active if c["vehicle_classification"].get("vehicle_missing") is None]

    lines.append(f"\nACTIVE CASES  (n={act_total}, random sample across 1980s–2020s)")
    lines.append(f"  Vehicle still missing (True)  : {len(act_true):>4}  ({len(act_true)/act_total*100:.1f}%)")
    lines.append(f"  Vehicle accounted for (False) : {len(act_false):>4}  ({len(act_false)/act_total*100:.1f}%)")
    lines.append(f"  Ambiguous / null              : {len(act_null):>4}  ({len(act_null)/act_total*100:.1f}%)")

    # --- Resolved stats ---
    res_total = len(resolved)
    res_assoc = [c for c in resolved if c.get("vehicle_classification", {}).get("vehicle_associated_with_disappearance") is True]
    res_crashed = [c for c in res_assoc if c.get("vehicle_classification", {}).get("vehicle_found_crashed") is True]
    res_not_crashed = [c for c in res_assoc if c.get("vehicle_classification", {}).get("vehicle_found_crashed") is False]

    lines.append(f"\nRESOLVED CASES  (n={res_total}, full dataset A–Z)")
    lines.append(f"  Vehicle associated            : {len(res_assoc):>4}  ({len(res_assoc)/res_total*100:.1f}%)")
    lines.append(f"    → later found crashed       : {len(res_crashed):>4}  ({len(res_crashed)/len(res_assoc)*100:.1f}% of vehicle cases)")
    lines.append(f"    → vehicle not crashed       : {len(res_not_crashed):>4}  ({len(res_not_crashed)/len(res_assoc)*100:.1f}% of vehicle cases)")

    # --- Inference ---
    lines.append("\nINFERENCE")
    lines.append(f"  Of resolved vehicle-associated cases, {len(res_crashed)/len(res_assoc)*100:.0f}% ended in a crash.")
    est_eventual_crashes = round(len(act_true) * len(res_crashed) / len(res_assoc))
    lines.append(f"  Applying that rate to the {len(act_true)} active vehicle-missing cases:")
    lines.append(f"  → ~{est_eventual_crashes} may eventually be found as crash/submersion scenarios")
    lines.append(f"  → ~{len(act_true) - est_eventual_crashes} may resolve another way (homicide, voluntary, etc.)")

    # Decade breakdown for active
    lines.append("\nACTIVE vehicle_missing=True by decade:")
    for decade in ["1980s", "1990s", "2000s", "2010s", "2020s"]:
        grp = [c for c in active if c.get("decade") == decade]
        t   = sum(1 for c in grp if c["vehicle_classification"].get("vehicle_missing") is True)
        pct = t / len(grp) * 100 if grp else 0
        lines.append(f"  {decade}  {t:>3} / {len(grp)}  ({pct:.1f}%)")

    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# TASK 2 — Null resolution
# ---------------------------------------------------------------------------

VEHICLE_KEYWORDS = [
    "car", "truck", "van", "vehicle", "suv", "pickup", "automobile",
    "motorcycle", "motorbike", "bike", "bicycle",
    "boat", "kayak", "canoe", "vessel", "yacht",
    "plane", "aircraft", "airplane", "helicopter",
    "atv", "rv", "motorhome", "motor home",
    "driving", "drove", "driven", "last seen driving",
]
FOUND_KEYWORDS = [
    "found", "located", "recovered", "abandoned", "parked",
    "towed", "impounded", "pulled from", "pulled out",
    "submerged", "wrecked", "crashed", "discovered",
]

def keyword_resolve_null(narrative: str, evidence: str) -> str | None:
    """
    Try to resolve a null with keyword heuristics.
    Returns 'true', 'false', or None (still ambiguous).
    """
    combined = (narrative + " " + evidence).lower()
    has_vehicle = any(kw in combined for kw in VEHICLE_KEYWORDS)
    was_found   = any(kw in combined for kw in FOUND_KEYWORDS)

    if not has_vehicle:
        return "false"
    if was_found:
        return "false"   # vehicle mentioned and found → not missing
    # Vehicle mentioned, no indication it was found → lean true but uncertain
    return None


def task2_nulls(active: list[dict], client: anthropic.Anthropic | None) -> tuple[str, list[dict]]:
    nulls = [c for c in active if c["vehicle_classification"].get("vehicle_missing") is None]

    lines = []
    lines.append("=" * 64)
    lines.append(f"TASK 2 — NULL RESOLUTION  ({len(nulls)} cases)")
    lines.append("=" * 64)

    # Keyword pass
    kw_true  = []
    kw_false = []
    kw_still_null = []

    for case in nulls:
        vc        = case["vehicle_classification"]
        narrative = clean(case.get("narrative", ""))
        evidence  = clean(vc.get("evidence", ""))
        result    = keyword_resolve_null(narrative, evidence)
        if result == "true":
            kw_true.append(case)
        elif result == "false":
            kw_false.append(case)
        else:
            kw_still_null.append(case)

    lines.append(f"\nKeyword pass resolved {len(kw_true) + len(kw_false)} of {len(nulls)} nulls:")
    lines.append(f"  Resolved → vehicle_missing=False  : {len(kw_false)}")
    lines.append(f"  Resolved → vehicle_missing=True   : {len(kw_true)}")
    lines.append(f"  Still ambiguous after keywords    : {len(kw_still_null)}")

    # Print the still-ambiguous narratives
    lines.append(f"\nSTILL-AMBIGUOUS CASES ({len(kw_still_null)}) — narrative excerpts:")
    for c in kw_still_null:
        vc  = c["vehicle_classification"]
        narr = clean(c.get("narrative", ""))[:220]
        lines.append(f"\n  • {c['name']}  ({clean(c.get('missing_from',''))})")
        lines.append(f"    Narrative: {narr}...")
        lines.append(f"    Claude evidence: {vc.get('evidence','')}")

    reclassified = []

    # Optional API re-run on still-ambiguous cases
    if client and kw_still_null:
        lines.append(f"\nRe-classifying {len(kw_still_null)} remaining nulls via API...")
        reclassified = api_reclassify_nulls(kw_still_null, client)

        api_true  = [c for c in reclassified if c["vehicle_classification"].get("vehicle_missing") is True]
        api_false = [c for c in reclassified if c["vehicle_classification"].get("vehicle_missing") is False]
        api_null  = [c for c in reclassified if c["vehicle_classification"].get("vehicle_missing") is None]

        lines.append(f"  API resolved → True  : {len(api_true)}")
        lines.append(f"  API resolved → False : {len(api_false)}")
        lines.append(f"  Still null           : {len(api_null)}")

        with open(NULL_RECLASS_FILE, "w", encoding="utf-8") as f:
            for c in reclassified:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        lines.append(f"  Saved to '{NULL_RECLASS_FILE}'.")

    lines.append("")
    return "\n".join(lines), reclassified


NULL_RECLASSIFY_PROMPT = """\
You are reviewing a missing persons case narrative where a previous classification
was uncertain (null). Decide specifically:

vehicle_missing = TRUE if a vehicle disappeared with the person AND there is no
  indication in the narrative that it was subsequently found or accounted for.
vehicle_missing = FALSE if no vehicle is involved, OR if the vehicle was found
  in any state.
vehicle_missing = NULL only if the narrative is truly too sparse to decide.

Respond ONLY with valid JSON:
{
  "vehicle_missing": true | false | null,
  "confidence": "high" | "medium" | "low",
  "evidence": "<one sentence, or 'None'>"
}
"""


def api_reclassify_nulls(
    cases: list[dict], client: anthropic.Anthropic
) -> list[dict]:
    """Re-classify ambiguous nulls one at a time (small batch — no need for Batch API)."""
    results = []
    for i, case in enumerate(cases, 1):
        narrative = clean(case.get("narrative", "")) or "(No narrative.)"
        print(f"  [{i}/{len(cases)}] {case['name']}")
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=NULL_RECLASSIFY_PROMPT,
                messages=[{"role": "user", "content": f"Narrative:\n\n{narrative}"}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            cl = json.loads(raw.strip())
        except Exception as e:
            cl = {"vehicle_missing": None, "confidence": "error", "evidence": str(e)}
        updated = {**case, "vehicle_classification": cl, "reclassified": True}
        results.append(updated)
        time.sleep(0.3)
    return results

# ---------------------------------------------------------------------------
# TASK 4 — Patterns in vehicle_missing=True cases
# ---------------------------------------------------------------------------

VEHICLE_TYPE_PATTERNS = [
    ("Car / sedan",       ["car", "sedan", "coupe", "automobile", "vehicle"]),
    ("Truck / pickup",    ["truck", "pickup", "lorry"]),
    ("SUV / van",         ["suv", "van", "minivan", "motorhome", "rv", "motor home"]),
    ("Motorcycle",        ["motorcycle", "motorbike", "bike"]),
    ("Boat / watercraft", ["boat", "kayak", "canoe", "vessel", "yacht", "watercraft"]),
    ("Aircraft",          ["plane", "aircraft", "airplane", "helicopter"]),
    ("ATV / other",       ["atv", "quad", "four-wheeler"]),
]

CIRCUMSTANCE_PATTERNS = [
    ("Last seen driving alone",    ["last seen driving", "last seen in his car", "last seen in her car",
                                    "driving alone", "left alone", "drove away"]),
    ("Travelling / road trip",     ["road trip", "travelling", "traveling", "en route",
                                    "driving from", "driving to", "headed to", "heading to"]),
    ("Left home / work",           ["left home", "left his home", "left her home",
                                    "left work", "left his job", "left her job",
                                    "left the house", "left his apartment", "left her apartment"]),
    ("Argument / domestic",        ["argument", "dispute", "fight", "domestic", "divorce",
                                    "separated", "estranged"]),
    ("Mental health / crisis",     ["depression", "mental health", "suicidal", "suicide",
                                    "bipolar", "schizophrenia", "psychiatric"]),
    ("Unknown / few details",      ["few details", "little is known", "circumstances unknown",
                                    "details are unknown", "not known"]),
]

def detect_vehicle_type(narrative: str) -> str:
    lower = narrative.lower()
    for label, keywords in VEHICLE_TYPE_PATTERNS:
        if any(k in lower for k in keywords):
            return label
    return "Unknown"


def detect_circumstance(narrative: str) -> list[str]:
    lower = narrative.lower()
    matched = []
    for label, keywords in CIRCUMSTANCE_PATTERNS:
        if any(k in lower for k in keywords):
            matched.append(label)
    return matched or ["No pattern matched"]


def extract_state(missing_from: str) -> str:
    clean_loc = re.sub(r"\s+", " ", missing_from).strip()
    parts = [p.strip() for p in clean_loc.split(",")]
    return parts[-1] if parts else "Unknown"


def task4_patterns(active: list[dict]) -> str:
    true_cases = [c for c in active if c["vehicle_classification"].get("vehicle_missing") is True]

    lines = []
    lines.append("=" * 64)
    lines.append(f"TASK 4 — PATTERNS IN {len(true_cases)} VEHICLE-MISSING CASES")
    lines.append("=" * 64)

    # Vehicle type
    type_counter: Counter = Counter()
    for c in true_cases:
        vtype = detect_vehicle_type(c.get("narrative", ""))
        type_counter[vtype] += 1

    lines.append("\nVEHICLE TYPE")
    for vtype, count in type_counter.most_common():
        pct = count / len(true_cases) * 100
        lines.append(f"  {vtype:<28}: {count:>3}  ({pct:.1f}%)")

    # Circumstances
    circ_counter: Counter = Counter()
    for c in true_cases:
        for circ in detect_circumstance(c.get("narrative", "")):
            circ_counter[circ] += 1

    lines.append("\nCIRCUMSTANCES  (cases may match multiple)")
    for circ, count in circ_counter.most_common():
        pct = count / len(true_cases) * 100
        lines.append(f"  {circ:<36}: {count:>3}  ({pct:.1f}%)")

    # State
    state_counter: Counter = Counter()
    for c in true_cases:
        state_counter[extract_state(c.get("missing_from", ""))] += 1

    lines.append("\nTOP STATES / REGIONS")
    for state, count in state_counter.most_common(15):
        pct = count / len(true_cases) * 100
        lines.append(f"  {state:<30}: {count:>3}  ({pct:.1f}%)")

    # Decade
    lines.append("\nBY DECADE")
    for decade in ["1980s", "1990s", "2000s", "2010s", "2020s"]:
        grp = [c for c in true_cases if c.get("decade") == decade]
        lines.append(f"  {decade}  : {len(grp)}")

    # Full case list
    lines.append(f"\nFULL LIST — all {len(true_cases)} vehicle-missing cases")
    lines.append(f"  {'Name':<35} {'Decade':<7} {'Missing From'}")
    lines.append(f"  {'-'*35} {'-'*7} {'-'*30}")
    for c in sorted(true_cases, key=lambda x: x.get("decade", "")):
        name     = c.get("name", "Unknown")[:34]
        decade   = c.get("decade", "?")
        loc      = clean(c.get("missing_from", ""))[:40]
        vc       = c["vehicle_classification"]
        evidence = vc.get("evidence", "")[:100]
        lines.append(f"  {name:<35} {decade:<7} {loc}")
        lines.append(f"    Evidence: {evidence}")

    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reclassify", action="store_true",
                        help="Use API to re-classify ambiguous nulls in task 2.")
    args = parser.parse_args()

    for f in [ACTIVE_FILE, RESOLVED_FILE]:
        if not Path(f).exists():
            sys.exit(f"ERROR: '{f}' not found.")

    active   = load_jsonl(ACTIVE_FILE)
    resolved = load_jsonl(RESOLVED_FILE)
    print(f"Loaded {len(active)} active, {len(resolved)} resolved cases.\n")

    client = None
    if args.reclassify:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ERROR: ANTHROPIC_API_KEY not set. Needed for --reclassify.")
        client = anthropic.Anthropic(api_key=api_key)

    t1 = task1_compare(active, resolved)
    t2, _ = task2_nulls(active, client)
    t4 = task4_patterns(active)

    full_report = "\n".join([t1, t2, t4])
    print(full_report)

    with open("deep_analysis.txt", "w", encoding="utf-8") as f:
        f.write(full_report + "\n")
    print("Saved full report to 'deep_analysis.txt'.")


if __name__ == "__main__":
    main()
