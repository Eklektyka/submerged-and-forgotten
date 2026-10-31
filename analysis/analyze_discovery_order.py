"""
analyze_discovery_order.py

For the 190 resolved cases where the person disappeared with their OWN vehicle,
classifies each case along two axes:

  person_found:
    "yes"      — person's body / remains / person was found
    "no"       — person never found (case resolved another way, e.g. conviction)
    "unknown"  — unclear from narrative

  vehicle_found:
    "yes"      — vehicle was found at some point
    "no"       — vehicle was never found / still missing
    "unknown"  — unclear

  found_first:
    "person"        — person/remains found before (or without) the vehicle
    "vehicle"       — vehicle found first, person found later or inside
    "simultaneous"  — found at the same time (body inside car when car discovered)
    "person_only"   — person found, vehicle never found
    "vehicle_only"  — vehicle found, person never found
    "neither"       — neither definitively found
    "unknown"       — ordering unclear

Usage:
    python analyze_discovery_order.py            # submit batch + poll + analyze
    python analyze_discovery_order.py --poll     # resume prior batch
    python analyze_discovery_order.py --analyze  # analyze existing output file
"""

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT_FILE    = "own_vehicle_classified.jsonl"
OUTPUT_FILE   = "discovery_order_classified.jsonl"
BATCH_ID_FILE = ".discovery_order_batch_id"

SYSTEM_PROMPT = """\
You are reviewing a resolved missing persons case narrative.
The person disappeared alongside their own vehicle.

Classify TWO things based strictly on what the narrative says:

1. person_found — were the person's remains or body (or the person alive) found?
   "yes"     — yes, definitively found
   "no"      — never found (case resolved another way, e.g. suspect convicted)
   "unknown" — narrative does not say

2. vehicle_found — was the vehicle found at any point?
   "yes"     — yes, vehicle was found
   "no"      — vehicle was never found / still missing
   "unknown" — narrative does not say

3. found_first — what was found first?
   "person"        — person/remains were found before, or independently of, the vehicle
   "vehicle"       — vehicle was found first; person found inside or separately afterward
   "simultaneous"  — found at the same time (e.g. body was inside the car when car was found)
   "person_only"   — person found but vehicle never found
   "vehicle_only"  — vehicle found but person never found
   "neither"       — neither was found
   "unknown"       — ordering is genuinely unclear from the narrative

Respond ONLY with valid JSON:
{
  "person_found": "yes" | "no" | "unknown",
  "vehicle_found": "yes" | "no" | "unknown",
  "found_first": "person" | "vehicle" | "simultaneous" | "person_only" | "vehicle_only" | "neither" | "unknown",
  "confidence": "high" | "medium" | "low",
  "evidence": "<one sentence summarising the key discovery facts>"
}
"""


def load_own_vehicle_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("own_vehicle_classification", {}).get("own_vehicle") is True:
                cases.append(c)
    return cases


def build_requests(cases: list[dict]) -> list[dict]:
    reqs = []
    for i, case in enumerate(cases):
        narrative = (case.get("narrative") or "").strip() or "(No narrative.)"
        reqs.append({
            "custom_id": str(i),
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 250,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": f"Case narrative:\n\n{narrative}"}],
            },
        })
    return reqs


def parse_response(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"person_found": "unknown", "vehicle_found": "unknown",
                "found_first": "unknown", "confidence": "error", "evidence": raw[:100]}


def submit(client: anthropic.Anthropic, cases: list[dict]) -> str:
    print(f"Submitting batch of {len(cases)} cases...")
    batch = client.messages.batches.create(requests=build_requests(cases))
    Path(BATCH_ID_FILE).write_text(batch.id)
    print(f"Batch ID: {batch.id}")
    return batch.id


def poll(client: anthropic.Anthropic, batch_id: str) -> None:
    print(f"Polling {batch_id}...")
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"  {b.processing_status} — processing: {c.processing}, "
              f"succeeded: {c.succeeded}, errored: {c.errored}")
        if b.processing_status == "ended":
            break
        time.sleep(20)


def retrieve(client: anthropic.Anthropic, batch_id: str, cases: list[dict]) -> list[dict]:
    print("Retrieving results...")
    results = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            results[r.custom_id] = parse_response(r.result.message.content[0].text)
        else:
            results[r.custom_id] = {"person_found": "unknown", "vehicle_found": "unknown",
                                     "found_first": "unknown", "confidence": "error",
                                     "evidence": r.result.type}

    enriched = []
    for i, case in enumerate(cases):
        cl = results.get(str(i), {"person_found": "unknown", "vehicle_found": "unknown",
                                   "found_first": "unknown", "confidence": "missing",
                                   "evidence": "No result."})
        enriched.append({**case, "discovery_classification": cl})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(enriched)} records to '{OUTPUT_FILE}'.")
    return enriched


def analyze(enriched: list[dict]) -> None:
    total = len(enriched)

    def get(c, key):
        return c["discovery_classification"].get(key, "unknown")

    lines = [
        "",
        "=" * 64,
        f"DISCOVERY ORDER ANALYSIS  (n={total} own-vehicle resolved cases)",
        "=" * 64,
    ]

    # --- Person found ---
    pf = Counter(get(c, "person_found") for c in enriched)
    lines += [
        "\nPERSON FOUND?",
        f"  Yes      : {pf['yes']:>4}  ({pf['yes']/total*100:.1f}%)",
        f"  No       : {pf['no']:>4}  ({pf['no']/total*100:.1f}%)",
        f"  Unknown  : {pf['unknown']:>4}",
    ]

    # --- Vehicle found ---
    vf = Counter(get(c, "vehicle_found") for c in enriched)
    lines += [
        "\nVEHICLE FOUND?",
        f"  Yes      : {vf['yes']:>4}  ({vf['yes']/total*100:.1f}%)",
        f"  No       : {vf['no']:>4}  ({vf['no']/total*100:.1f}%)",
        f"  Unknown  : {vf['unknown']:>4}",
    ]

    # --- Found first ---
    ff_order = ["simultaneous", "vehicle", "person", "person_only",
                "vehicle_only", "neither", "unknown"]
    ff = Counter(get(c, "found_first") for c in enriched)

    lines += ["", "WHAT WAS FOUND FIRST?"]
    labels = {
        "simultaneous": "Body inside car when car found",
        "vehicle":      "Vehicle found first, person after",
        "person":       "Person found first, vehicle after",
        "person_only":  "Person found, vehicle never found",
        "vehicle_only": "Vehicle found, person never found",
        "neither":      "Neither found",
        "unknown":      "Order unclear",
    }
    for key in ff_order:
        n = ff[key]
        if n:
            pct = n / total * 100
            lines.append(f"  {labels[key]:<38}: {n:>4}  ({pct:.1f}%)")

    # --- Cross-tab: found_first × crashed ---
    lines += ["", "-" * 64,
              "CROSS-TAB: discovery order × vehicle crashed",
              "-" * 64]
    lines.append(f"  {'Category':<38} {'Total':>5}  {'Crashed':>7}  {'Not crashed':>11}")
    lines.append(f"  {'-'*38} {'-'*5}  {'-'*7}  {'-'*11}")

    for key in ff_order:
        group = [c for c in enriched if get(c, "found_first") == key]
        if not group:
            continue
        crashed     = sum(1 for c in group if c["vehicle_classification"].get("vehicle_found_crashed") is True)
        not_crashed = sum(1 for c in group if c["vehicle_classification"].get("vehicle_found_crashed") is False)
        lines.append(f"  {labels[key]:<38} {len(group):>5}  {crashed:>7}  {not_crashed:>11}")

    # --- Cases where person was found but vehicle was NOT ---
    person_only = [c for c in enriched if get(c, "found_first") == "person_only"]
    if person_only:
        lines += ["", f"PERSON FOUND, VEHICLE NEVER FOUND ({len(person_only)} cases)"]
        for c in person_only:
            dc = c["discovery_classification"]
            lines.append(f"  • {c['name']}  ({c.get('missing_from','').strip()[:40]})")
            lines.append(f"    {dc.get('evidence','')}")

    # --- Cases where vehicle found first (empty — person found separately) ---
    veh_first = [c for c in enriched if get(c, "found_first") == "vehicle"]
    if veh_first:
        lines += ["", f"VEHICLE FOUND FIRST, PERSON FOUND SEPARATELY ({len(veh_first)} cases)"]
        for c in veh_first:
            dc = c["discovery_classification"]
            lines.append(f"  • {c['name']}  ({c.get('missing_from','').strip()[:40]})")
            lines.append(f"    {dc.get('evidence','')}")

    lines.append("\n" + "=" * 64)

    report = "\n".join(lines)
    print(report)

    with open("discovery_order_summary.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("Saved to 'discovery_order_summary.txt'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll",    action="store_true")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing output file without API calls.")
    args = parser.parse_args()

    if args.analyze:
        if not Path(OUTPUT_FILE).exists():
            sys.exit(f"ERROR: '{OUTPUT_FILE}' not found.")
        enriched = [json.loads(l) for l in open(OUTPUT_FILE, encoding="utf-8") if l.strip()]
        print(f"Loaded {len(enriched)} records.")
        analyze(enriched)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=api_key)

    if not Path(INPUT_FILE).exists():
        sys.exit(f"ERROR: '{INPUT_FILE}' not found.")

    cases = load_own_vehicle_cases(INPUT_FILE)
    print(f"Loaded {len(cases)} own-vehicle cases from '{INPUT_FILE}'.")

    batch_id = args.batch_id
    if not batch_id and args.poll:
        if Path(BATCH_ID_FILE).exists():
            batch_id = Path(BATCH_ID_FILE).read_text().strip()
            print(f"Resuming batch {batch_id}.")
        else:
            sys.exit(f"No saved batch ID in '{BATCH_ID_FILE}'.")

    if not batch_id:
        batch_id = submit(client, cases)

    poll(client, batch_id)
    enriched = retrieve(client, batch_id, cases)
    analyze(enriched)


if __name__ == "__main__":
    main()
