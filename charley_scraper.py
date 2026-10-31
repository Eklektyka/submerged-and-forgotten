"""
charley_scraper.py

Scrapes resolved missing persons cases from charleyproject.org,
extracts narrative text, and uses the Anthropic API to classify
whether each case involved a vehicle that was found crashed.

Results are saved to charley_results.jsonl.

Usage:
    python charley_scraper.py --max-pages 3
    python charley_scraper.py --letters A B C
    python charley_scraper.py --max-cases 10
"""

import argparse
import json
import os
import time
import sys
from datetime import datetime

import anthropic
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://charleyproject.org"
RESOLVED_URL = f"{BASE_URL}/case-updates/resolved"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
OUTPUT_FILE = "charley_results.jsonl"
REQUEST_DELAY = 1.5  # seconds between HTTP requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research-scraper; missing-persons-vehicle-crash-study) "
        "compatible with charleyproject.org"
    )
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get_page(url: str, session: requests.Session) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch {url}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def get_case_links_for_letter(letter: str, session: requests.Session) -> list[str]:
    """Return all individual case URLs listed for a given letter."""
    url = f"{RESOLVED_URL}?letter={letter}"
    soup = get_page(url, session)
    if soup is None:
        return []

    links = []
    for a_tag in soup.select("div#case-container div.case a"):
        href = a_tag.get("href", "").strip()
        if href:
            links.append(href)
    return links


def extract_case_data(url: str, session: requests.Session) -> dict | None:
    """
    Fetch an individual case page and extract:
      - name
      - missing_since
      - missing_from
      - classification
      - narrative  (Details of Disappearance text)
      - url
    Returns None if the page cannot be parsed.
    """
    soup = get_page(url, session)
    if soup is None:
        return None

    # Name
    name_tag = soup.select_one("h1.entry-title")
    name = name_tag.get_text(strip=True) if name_tag else "Unknown"

    # Metadata from the <ul> in case-top
    metadata: dict[str, str] = {}
    for li in soup.select("section#case-top ul li"):
        text = li.get_text(separator=" ", strip=True)
        if "Missing Since" in text:
            metadata["missing_since"] = text.replace("Missing Since", "").strip()
        elif "Missing From" in text:
            metadata["missing_from"] = text.replace("Missing From", "").strip()
        elif "Classification" in text:
            metadata["classification"] = text.replace("Classification", "").strip()

    # Narrative — first column of case-bottom contains "Details of Disappearance"
    narrative = ""
    case_bottom = soup.select_one("section#case-bottom")
    if case_bottom:
        first_col = case_bottom.select_one(".column")
        if first_col:
            paragraphs = [p.get_text(separator=" ", strip=True) for p in first_col.find_all("p")]
            narrative = " ".join(paragraphs)

    if not narrative:
        return None  # skip cases with no extractable text

    return {
        "url": url,
        "name": name,
        "missing_since": metadata.get("missing_since", ""),
        "missing_from": metadata.get("missing_from", ""),
        "classification": metadata.get("classification", ""),
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Anthropic classification
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an analyst reviewing resolved missing persons cases.
Your task: determine whether the case narrative describes a scenario where a vehicle
belonging to or associated with the missing person was found crashed or wrecked
(e.g., found in a ditch, off a road, submerged, hit a tree, involved in an accident).

Respond ONLY with valid JSON and nothing else, using exactly this schema:
{
  "vehicle_found_crashed": true | false,
  "confidence": "high" | "medium" | "low",
  "evidence": "<one sentence quoting or paraphrasing the key supporting text, or 'None' if not applicable>"
}
"""

def classify_narrative(narrative: str, client: anthropic.Anthropic) -> dict:
    """
    Call Claude to classify whether the narrative describes a crashed vehicle.
    Returns a dict with vehicle_found_crashed, confidence, evidence.
    """
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast + cheap for classification
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Case narrative:\n\n{narrative}",
                }
            ],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (anthropic.APIError, json.JSONDecodeError, IndexError) as e:
        print(f"  [WARN] Classification error: {e}", file=sys.stderr)
        return {
            "vehicle_found_crashed": None,
            "confidence": "error",
            "evidence": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape charleyproject.org resolved cases and classify vehicle crashes."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of letter-pages to scrape (1 page = 1 letter, e.g. --max-pages 3 → A, B, C).",
    )
    parser.add_argument(
        "--letters",
        nargs="+",
        metavar="LETTER",
        help="Specific letters to scrape (e.g. --letters A B C). Overrides --max-pages.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing this many cases total.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output JSONL file path (default: {OUTPUT_FILE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    session = requests.Session()

    # Determine which letters to scrape
    if args.letters:
        letters_to_scrape = [l.upper() for l in args.letters]
    elif args.max_pages:
        letters_to_scrape = LETTERS[: args.max_pages]
    else:
        letters_to_scrape = LETTERS

    print(f"Letters to scrape: {letters_to_scrape}")
    print(f"Output file: {args.output}")
    print("-" * 60)

    total_cases = 0
    written = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for letter in letters_to_scrape:
            print(f"\n[{letter}] Fetching case list...")
            time.sleep(REQUEST_DELAY)
            case_urls = get_case_links_for_letter(letter, session)
            print(f"  Found {len(case_urls)} cases.")

            for idx, case_url in enumerate(case_urls, 1):
                if args.max_cases and total_cases >= args.max_cases:
                    print(f"\nReached --max-cases {args.max_cases}. Stopping.")
                    break

                total_cases += 1
                print(f"  [{letter} {idx}/{len(case_urls)}] {case_url}")
                time.sleep(REQUEST_DELAY)

                case_data = extract_case_data(case_url, session)
                if case_data is None:
                    print("    -> Skipped (no narrative).")
                    continue

                print(f"    -> Classifying: {case_data['name']!r}...")
                classification = classify_narrative(case_data["narrative"], client)

                result = {
                    **case_data,
                    "classification_result": classification,
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1

                crashed = classification.get("vehicle_found_crashed")
                conf = classification.get("confidence", "?")
                print(f"    -> vehicle_found_crashed={crashed} ({conf})")

            else:
                continue  # inner loop completed normally
            break  # inner loop was broken by max_cases

    print("\n" + "=" * 60)
    print(f"Done. {written} cases written to {args.output}.")


if __name__ == "__main__":
    main()
