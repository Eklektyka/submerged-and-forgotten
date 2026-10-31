"""
hypothesis_own_vehicle.py

Tests the hypothesis:
  "If a person disappears with their OWN vehicle, what is the probability
   that the vehicle is eventually found crashed?"

Method:
  1. Load the 249 vehicle-associated resolved cases
  2. Batch API: classify each as own_vehicle=True/False/null
       True  — the missing person's own car/truck/etc. vanished with them
       False — vehicle belongs to an abductor, stranger, or third party
       Null  — genuinely unclear
  3. Cross-reference with vehicle_found_crashed (already classified)
  4. Report the conditional probability P(crashed | own vehicle missing)

Usage:
    python hypothesis_own_vehicle.py            # submit batch + poll
    python hypothesis_own_vehicle.py --poll     # resume prior batch
    python hypothesis_own_vehicle.py --batch-id msgbatch_xxx
    python hypothesis_own_vehicle.py --analyze  # skip API, analyze existing output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

INPUT_FILE   = "vehicle_associated_cases.jsonl"
OUTPUT_FILE  = "own_vehicle_classified.jsonl"
BATCH_ID_FILE = ".own_vehicle_batch_id"

# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are reviewing a resolved missing persons case narrative.

Classify ONE thing:

own_vehicle:
  TRUE  — the missing person disappeared along with THEIR OWN vehicle
          (their own car, truck, van, motorcycle, boat, plane, ATV, etc.).
          The vehicle belonged to them or was regularly used by them, and it
          went missing at the same time they did.

  FALSE — any of these apply:
          • The person was last seen getting INTO someone else's vehicle
            (stranger, acquaintance, abductor, ride, etc.)
          • The person was abducted or forced into a vehicle they don't own
          • The only vehicle mentioned belongs to a suspect or third party
          • No vehicle is relevant to the disappearance

  NULL  — a vehicle is mentioned but ownership is genuinely unclear

Respond ONLY with valid JSON:
{
  "own_vehicle": true | false | null,
  "confidence": "high" | "medium" | "low",
  "evidence": "<one sentence quoting the key text, or 'None'>"
}
"""


def load_vehicle_cases(path: str) -> list[dict]:
    """Load only the vehicle-associated=True cases."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("vehicle_classification", {}).get("vehicle_associated_with_disappearance") is True:
                cases.append(c)
    return cases


def build_requests(cases: list[dict]) -> list[dict]:
    reqs = []
    for i, case in enumerate(cases):
        narrative = (case.get("narrative") or "").strip() or "(No narrative.)"
        evidence  = (case.get("vehicle_classification", {}).get("evidence") or "")
        content   = f"Case narrative:\n\n{narrative}"
        if evidence:
            content += f"\n\nPrior classification evidence: {evidence}"
        reqs.append({
            "custom_id": str(i),
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
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
        return {"own_vehicle": None, "confidence": "error", "evidence": raw[:100]}


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
            results[r.custom_id] = {"own_vehicle": None, "confidence": "error",
                                     "evidence": r.result.type}

    enriched = []
    for i, case in enumerate(cases):
        cl = results.get(str(i), {"own_vehicle": None, "confidence": "missing",
                                   "evidence": "No result."})
        enriched.append({**case, "own_vehicle_classification": cl})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(enriched)} records to '{OUTPUT_FILE}'.")
    return enriched


def analyze(enriched: list[dict]) -> None:
    total = len(enriched)

    own      = [c for c in enriched if c["own_vehicle_classification"].get("own_vehicle") is True]
    not_own  = [c for c in enriched if c["own_vehicle_classification"].get("own_vehicle") is False]
    unk      = [c for c in enriched if c["own_vehicle_classification"].get("own_vehicle") is None]

    # Of own-vehicle cases, how many were found crashed?
    own_crashed     = [c for c in own if c["vehicle_classification"].get("vehicle_found_crashed") is True]
    own_not_crashed = [c for c in own if c["vehicle_classification"].get("vehicle_found_crashed") is False]

    p_crashed = len(own_crashed) / len(own) * 100 if own else 0

    # Confidence breakdown for own=True
    high   = [c for c in own if c["own_vehicle_classification"].get("confidence") == "high"]
    p_high = len([c for c in high if c["vehicle_classification"].get("vehicle_found_crashed") is True]) / len(high) * 100 if high else 0

    lines = [
        "",
        "=" * 64,
        "HYPOTHESIS: P(crashed | own vehicle missing)",
        "=" * 64,
        f"\nInput: {total} vehicle-associated resolved cases",
        "",
        f"  Own vehicle (person's own car/truck/etc.)  : {len(own):>4}  ({len(own)/total*100:.1f}%)",
        f"  Not own (abductor's / third-party vehicle) : {len(not_own):>4}  ({len(not_own)/total*100:.1f}%)",
        f"  Unclear / null                             : {len(unk):>4}  ({len(unk)/total*100:.1f}%)",
        "",
        "-" * 64,
        "RESULT — own vehicle cases only:",
        "-" * 64,
        f"  Own vehicle cases                          : {len(own)}",
        f"  → Found crashed / submerged                : {len(own_crashed):>4}  ({p_crashed:.1f}%)",
        f"  → NOT found crashed                        : {len(own_not_crashed):>4}  ({100-p_crashed:.1f}%)",
        "",
        f"  High-confidence own-vehicle cases          : {len(high)}",
        f"  → Found crashed (high-conf subset)         : {len([c for c in high if c['vehicle_classification'].get('vehicle_found_crashed') is True]):>4}  ({p_high:.1f}%)",
        "",
        "CONCLUSION",
        f"  P(vehicle found crashed | own vehicle went missing) = {p_crashed:.1f}%",
        "",
        "NOT-OWN-VEHICLE cases — what happened to them:",
    ]

    not_own_crashed     = [c for c in not_own if c["vehicle_classification"].get("vehicle_found_crashed") is True]
    not_own_not_crashed = [c for c in not_own if c["vehicle_classification"].get("vehicle_found_crashed") is False]
    lines.append(f"  Found crashed                              : {len(not_own_crashed)}")
    lines.append(f"  Not crashed (homicide, abandoned, etc.)    : {len(not_own_not_crashed)}")

    lines += [
        "",
        "OWN VEHICLE — full case list",
        f"  {'Name':<35} {'Crashed':>7}  Evidence",
        f"  {'-'*35} {'-'*7}  {'-'*40}",
    ]
    for c in sorted(own, key=lambda x: x["vehicle_classification"].get("vehicle_found_crashed") is not True):
        name    = c.get("name", "Unknown")[:34]
        crashed = c["vehicle_classification"].get("vehicle_found_crashed")
        tag     = "CRASHED" if crashed else "no"
        ev      = (c["own_vehicle_classification"].get("evidence") or "")[:70]
        lines.append(f"  {name:<35} {tag:>7}  {ev}")

    lines.append("=" * 64)
    report = "\n".join(lines)
    print(report)

    with open("own_vehicle_hypothesis.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nSaved to 'own_vehicle_hypothesis.txt'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll",     action="store_true")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--analyze",  action="store_true",
                        help="Skip API — analyze existing own_vehicle_classified.jsonl")
    args = parser.parse_args()

    if args.analyze:
        if not Path(OUTPUT_FILE).exists():
            sys.exit(f"ERROR: '{OUTPUT_FILE}' not found. Run without --analyze first.")
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

    cases = load_vehicle_cases(INPUT_FILE)
    print(f"Loaded {len(cases)} vehicle-associated resolved cases.")

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
