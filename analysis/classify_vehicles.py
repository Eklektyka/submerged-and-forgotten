"""
classify_vehicles.py

For every case in charley_results.jsonl, uses the Anthropic Batch API to
classify two things from the narrative:

  1. vehicle_associated_with_disappearance
       Was a vehicle (car, truck, van, ATV, boat, plane, etc.) reported
       missing or last seen with the person at the time they disappeared?

  2. vehicle_found_crashed
       Was that vehicle (or any vehicle linked to the case) later found
       crashed, wrecked, submerged, or in an otherwise damaged/abnormal state?

Results saved to vehicle_associated_cases.jsonl.
A summary is printed and saved to vehicle_classification_summary.txt.

Usage:
    python classify_vehicles.py            # submit batch + poll until done
    python classify_vehicles.py --poll     # re-poll / retrieve a prior batch
    python classify_vehicles.py --batch-id msgbatch_xxx  # retrieve specific batch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

INPUT_FILE   = "charley_results.jsonl"
OUTPUT_FILE  = "vehicle_associated_cases.jsonl"
SUMMARY_FILE = "vehicle_classification_summary.txt"
BATCH_ID_FILE = ".last_batch_id"   # persists the batch ID between runs

SYSTEM_PROMPT = """\
You are an analyst reviewing resolved missing persons case narratives.
Classify two things based ONLY on what the narrative states:

1. vehicle_associated_with_disappearance
   TRUE if a vehicle (car, truck, van, motorcycle, ATV, boat, plane, etc.)
   was reported as missing along with the person, or was last seen with the
   person at the time of their disappearance.
   FALSE if no vehicle is mentioned in connection with the disappearance.

2. vehicle_found_crashed
   TRUE if a vehicle linked to the case was later found crashed, wrecked,
   submerged in water, driven off a road/cliff, or otherwise destroyed.
   FALSE if the vehicle was not found in such a state, or no vehicle applies.

Respond ONLY with valid JSON — no prose, no markdown — using exactly this schema:
{
  "vehicle_associated_with_disappearance": true | false,
  "vehicle_found_crashed": true | false,
  "confidence": "high" | "medium" | "low",
  "evidence": "<one sentence quoting or paraphrasing the key supporting text, or 'None'>"
}
"""


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
                print(f"[WARN] Line {lineno} skipped: {e}", file=sys.stderr)
    return cases


def build_batch_requests(cases: list[dict]) -> list[dict]:
    requests = []
    for i, case in enumerate(cases):
        narrative = case.get("narrative", "").strip()
        if not narrative:
            narrative = "(No narrative available.)"
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": f"Case narrative:\n\n{narrative}"}
                ],
            },
        })
    return requests


def submit_batch(client: anthropic.Anthropic, cases: list[dict]) -> str:
    print(f"Submitting batch of {len(cases)} cases to Anthropic Batch API...")
    requests = build_batch_requests(cases)
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"Batch submitted. ID: {batch_id}")
    Path(BATCH_ID_FILE).write_text(batch_id)
    print(f"Batch ID saved to '{BATCH_ID_FILE}'.")
    return batch_id


def poll_batch(client: anthropic.Anthropic, batch_id: str) -> None:
    """Poll until the batch is complete, printing progress."""
    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(
            f"  Status: {status} — "
            f"processing: {counts.processing}, "
            f"succeeded: {counts.succeeded}, "
            f"errored: {counts.errored}, "
            f"canceled: {counts.canceled}, "
            f"expired: {counts.expired}"
        )
        if status == "ended":
            print("Batch complete.")
            break
        time.sleep(30)


def parse_result_text(text: str) -> dict:
    """Parse Claude's JSON response, stripping any markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "vehicle_associated_with_disappearance": None,
            "vehicle_found_crashed": None,
            "confidence": "error",
            "evidence": f"Parse error: {raw[:120]}",
        }


def retrieve_and_save(client: anthropic.Anthropic, batch_id: str, cases: list[dict]) -> None:
    """Retrieve batch results, merge with original cases, save output."""
    print(f"Retrieving results for batch {batch_id}...")

    results_by_id: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            results_by_id[custom_id] = parse_result_text(text)
        else:
            results_by_id[custom_id] = {
                "vehicle_associated_with_disappearance": None,
                "vehicle_found_crashed": None,
                "confidence": "error",
                "evidence": f"API error: {result.result.type}",
            }

    print(f"Retrieved {len(results_by_id)} results.")

    enriched = []
    for i, case in enumerate(cases):
        classification = results_by_id.get(str(i), {
            "vehicle_associated_with_disappearance": None,
            "vehicle_found_crashed": None,
            "confidence": "missing",
            "evidence": "No result returned for this case.",
        })
        enriched.append({**case, "vehicle_classification": classification})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in enriched:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(enriched)} records to '{OUTPUT_FILE}'.")

    print_and_save_summary(enriched)


def print_and_save_summary(enriched: list[dict]) -> None:
    total = len(enriched)

    associated   = [c for c in enriched if c["vehicle_classification"].get("vehicle_associated_with_disappearance") is True]
    not_assoc    = [c for c in enriched if c["vehicle_classification"].get("vehicle_associated_with_disappearance") is False]
    assoc_errors = [c for c in enriched if c["vehicle_classification"].get("vehicle_associated_with_disappearance") is None]

    crashed_of_associated = [
        c for c in associated
        if c["vehicle_classification"].get("vehicle_found_crashed") is True
    ]

    pct_assoc   = len(associated) / total * 100 if total else 0
    pct_crashed = len(crashed_of_associated) / len(associated) * 100 if associated else 0
    pct_crashed_total = len(crashed_of_associated) / total * 100 if total else 0

    lines = []
    lines.append("=" * 64)
    lines.append("VEHICLE ASSOCIATION & CRASH CLASSIFICATION SUMMARY")
    lines.append("=" * 64)
    lines.append(f"\nTotal cases analysed                         : {total}")
    lines.append(f"\nVehicle associated with disappearance")
    lines.append(f"  Yes                                        : {len(associated):>5}  ({pct_assoc:.1f}%)")
    lines.append(f"  No                                         : {len(not_assoc):>5}  ({100 - pct_assoc - (len(assoc_errors)/total*100):.1f}%)")
    if assoc_errors:
        lines.append(f"  Error / unknown                            : {len(assoc_errors):>5}")
    lines.append(f"\nOf those with a vehicle associated:")
    lines.append(f"  Vehicle later found crashed                : {len(crashed_of_associated):>5}  ({pct_crashed:.1f}% of vehicle cases)")
    lines.append(f"  ({pct_crashed_total:.1f}% of all {total} cases)")
    lines.append("")
    lines.append("=" * 64)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to '{SUMMARY_FILE}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-classify vehicle association in Charley Project cases.")
    parser.add_argument("--poll",     action="store_true", help="Re-poll / retrieve a previously submitted batch.")
    parser.add_argument("--batch-id", default=None,        help="Specific batch ID to retrieve (skips submission).")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is not set.")

    if not Path(INPUT_FILE).exists():
        sys.exit(f"ERROR: '{INPUT_FILE}' not found.")

    client = anthropic.Anthropic(api_key=api_key)
    cases  = load_cases(INPUT_FILE)
    print(f"Loaded {len(cases)} cases from '{INPUT_FILE}'.")

    # Determine batch ID
    batch_id = args.batch_id
    if not batch_id and args.poll:
        if Path(BATCH_ID_FILE).exists():
            batch_id = Path(BATCH_ID_FILE).read_text().strip()
            print(f"Resuming batch {batch_id} from '{BATCH_ID_FILE}'.")
        else:
            sys.exit(f"ERROR: No batch ID found in '{BATCH_ID_FILE}'. Run without --poll first.")

    if not batch_id:
        # Fresh submission
        batch_id = submit_batch(client, cases)

    # Poll until done
    poll_batch(client, batch_id)

    # Retrieve and save
    retrieve_and_save(client, batch_id, cases)


if __name__ == "__main__":
    main()
