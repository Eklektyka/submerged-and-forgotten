"""
classify_location.py  — no API version

Classifies crash location type, days to discovery, and state for the
190 own-vehicle resolved cases using keyword matching and date parsing.
"""

import io
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT_FILE  = "discovery_order_classified.jsonl"
OUTPUT_FILE = "location_classified.jsonl"

# ---------------------------------------------------------------------------
# Location keywords
# ---------------------------------------------------------------------------
WATER_KW = [
    "river", "lake", "pond", "reservoir", "canal", "swamp", "bay", "ocean",
    "sea", "creek", "culvert", "drainage", "retention pond", "retention basin",
    "submerged", "underwater", "under water", "beneath the water",
    "pulled from", "pulled out of", "dragged from", "found in the water",
    "went into the water", "drove into the water", "drove into a",
    "driven into", "into the river", "into the lake", "into a lake",
    "into a pond", "into a canal", "into a creek", "into a reservoir",
    "into a retention", "into the ocean", "into the bay",
    "in the river", "in the lake", "in a lake", "in a pond",
    "in a canal", "in a creek", "in the creek", "in a reservoir",
    "in a swamp", "in a marsh", "marsh", "waterway", "inlet", "slough",
    "flood", "flooded", "dam", "levee", "bayou",
]

LAND_KW = [
    "ravine", "embankment", "cliff", "hillside", "canyon", "gulch", "gully",
    "ditch", "woods", "wooded", "forest", "field", "brush", "vegetation",
    "off the road", "off a road", "off the highway", "off a cliff",
    "off an embankment", "off the embankment", "off a bank",
    "rolled down", "went off", "gone off", "down a", "down an embankment",
    "down a ravine", "down a cliff", "down a hillside",
    "remote area", "remote location", "rugged", "off-road", "off road",
    "tree", "guardrail", "barrier", "overturned", "rolled over",
    "slope", "mountain", "trail", "terrain",
]

ROAD_KW = [
    "parking lot", "parking garage", "parked on", "parked at",
    "roadside", "road side", "on the road", "on the highway",
    "on the street", "on a road", "on a highway", "shoulder",
    "garage", "driveway", "abandoned on", "found on the side of",
    "side of the road", "side of a road",
]


def classify_location(narrative: str) -> tuple[str, str]:
    """Return (location_type, matched_evidence_snippet)."""
    lower = narrative.lower()

    # Score each category
    water_hits = [(kw, kw in lower) for kw in WATER_KW]
    land_hits  = [(kw, kw in lower) for kw in LAND_KW]
    road_hits  = [(kw, kw in lower) for kw in ROAD_KW]

    water_score = sum(1 for _, h in water_hits if h)
    land_score  = sum(1 for _, h in land_hits  if h)
    road_score  = sum(1 for _, h in road_hits  if h)

    # First matched keyword per category as evidence snippet
    water_ev = next((kw for kw, h in water_hits if h), None)
    land_ev  = next((kw for kw, h in land_hits  if h), None)
    road_ev  = next((kw for kw, h in road_hits  if h), None)

    scores = [
        ("water",        water_score, water_ev),
        ("land_offroad", land_score,  land_ev),
        ("roadside",     road_score,  road_ev),
    ]
    scores.sort(key=lambda x: -x[1])

    if scores[0][1] == 0:
        return "unknown", "no location keywords matched"

    loc_type = scores[0][0]
    evidence = scores[0][2] or ""
    return loc_type, f"keyword match: '{evidence}'"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# "In March 2006", "In October 2009", "in January 2008"
MONTH_YEAR_RE = re.compile(
    r'\bin\s+(' + '|'.join(MONTH_MAP.keys()) + r')[,\s]+(\d{4})\b',
    re.IGNORECASE
)
# Full dates: "March 15, 2006", "15 March 2006", "03/15/2006", "2006-03-15"
FULL_DATE_RE = re.compile(
    r'\b(?:(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'          # MM/DD/YYYY
    r'|(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})'              # YYYY-MM-DD
    r'|(' + '|'.join(MONTH_MAP.keys()) + r')\s+(\d{1,2}),?\s+(\d{4})'  # Month DD YYYY
    r'|\b(\d{1,2})\s+(' + '|'.join(MONTH_MAP.keys()) + r')\s+(\d{4}))\b',  # DD Month YYYY
    re.IGNORECASE
)
# "a week later", "two years later", "six months later", etc.
RELATIVE_RE = re.compile(
    r'\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)'
    r'\s+(day|days|week|weeks|month|months|year|years)\s+(later|after|afterward)',
    re.IGNORECASE
)
WORD_NUMS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7,
             "month": 30, "months": 30, "year": 365, "years": 365}


def parse_missing_date(missing_since: str) -> datetime | None:
    m = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', missing_since)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    m = re.search(r'(\d{4})', missing_since)
    if m:
        return datetime(int(m.group(1)), 7, 1)   # mid-year estimate
    return None


def extract_discovery_dates(narrative: str) -> list[datetime]:
    """Pull all parseable dates from the narrative."""
    dates = []
    for m in FULL_DATE_RE.finditer(narrative):
        try:
            g = m.groups()
            if g[0] and g[2]:          # MM/DD/YYYY
                dates.append(datetime(int(g[2]), int(g[0]), int(g[1])))
            elif g[3] and g[5]:        # YYYY-MM-DD
                dates.append(datetime(int(g[3]), int(g[4]), int(g[5])))
            elif g[6] and g[8]:        # Month DD YYYY
                mon = MONTH_MAP.get(g[6].lower())
                if mon:
                    dates.append(datetime(int(g[8]), mon, int(g[7])))
            elif g[9] and g[10]:       # DD Month YYYY
                mon = MONTH_MAP.get(g[10].lower())
                if mon:
                    dates.append(datetime(int(g[11]), mon, int(g[9])))
        except (ValueError, TypeError):
            pass

    for m in MONTH_YEAR_RE.finditer(narrative):
        try:
            mon = MONTH_MAP.get(m.group(1).lower())
            yr  = int(m.group(2))
            if mon:
                dates.append(datetime(yr, mon, 15))   # mid-month estimate
        except (ValueError, TypeError):
            pass

    return dates


def estimate_days(narrative: str, missing_date: datetime | None) -> int | None:
    """Try to estimate days between disappearance and discovery."""

    # Try relative expressions first ("six months later")
    for m in RELATIVE_RE.finditer(narrative):
        num_str  = m.group(1).lower()
        unit_str = m.group(2).lower()
        num = WORD_NUMS.get(num_str)
        if num is None:
            try:
                num = int(num_str)
            except ValueError:
                continue
        unit_days = UNIT_DAYS.get(unit_str, 1)
        return num * unit_days

    # Try absolute dates
    if missing_date is None:
        return None

    dates = extract_discovery_dates(narrative)
    # Filter to dates after disappearance
    future = [d for d in dates if d > missing_date]
    if not future:
        return None

    discovery = min(future)   # earliest date after disappearance
    delta = (discovery - missing_date).days
    return delta if 0 < delta < 36500 else None   # sanity-check: < 100 years


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------
US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
    "Puerto Rico",
}
CANADIAN = {
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland", "Nova Scotia", "Ontario", "Quebec", "Saskatchewan",
}


def extract_state(missing_from: str, narrative: str) -> str:
    clean = re.sub(r"\s+", " ", missing_from).strip()
    parts = [p.strip() for p in clean.split(",")]
    for part in reversed(parts):
        for state in US_STATES:
            if state.lower() == part.lower():
                return state
        for prov in CANADIAN:
            if prov.lower() == part.lower():
                return f"Canada – {prov}"
        if part.upper() in {
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
            "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
            "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
            "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
            "WI","WY","DC","PR",
        }:
            return part.upper()
    return "Unknown"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def analyze(enriched: list[dict]) -> None:
    total   = len(enriched)
    crashed = [c for c in enriched
               if c.get("vehicle_classification", {}).get("vehicle_found_crashed") is True]

    def lc(c):  return c["location_classification"]["location_type"]
    def d(c):   return c["location_classification"]["days_to_discovery"]

    loc_labels = {
        "water":        "Water (river / lake / pond / canal…)",
        "land_offroad": "Land off-road (ravine / cliff / woods…)",
        "roadside":     "Roadside / parking / on road",
        "unknown":      "Unknown / not determinable",
    }

    lines = [
        "", "=" * 64,
        f"CRASH LOCATION ANALYSIS  (n={total} own-vehicle cases)",
        "=" * 64,
        f"\nTotal own-vehicle cases         : {total}",
        f"Vehicle found crashed           : {len(crashed)}  ({len(crashed)/total*100:.1f}%)",
    ]

    # Location breakdown — all cases
    loc_all = Counter(lc(c) for c in enriched)
    lines += ["", "LOCATION TYPE — all 190 cases"]
    for key in ["water", "land_offroad", "roadside", "unknown"]:
        n = loc_all[key]
        lines.append(f"  {loc_labels[key]:<44}: {n:>4}  ({n/total*100:.1f}%)")

    # Location breakdown — crashed only
    loc_cr = Counter(lc(c) for c in crashed)
    lines += ["", f"LOCATION TYPE — crashed only (n={len(crashed)})"]
    for key in ["water", "land_offroad", "roadside", "unknown"]:
        n   = loc_cr[key]
        pct = n / len(crashed) * 100 if crashed else 0
        lines.append(f"  {loc_labels[key]:<44}: {n:>4}  ({pct:.1f}%)")

    # Average delay by location type
    lines += ["", "AVERAGE DAYS TO DISCOVERY by location type"]
    for key in ["water", "land_offroad", "roadside"]:
        grp  = [c for c in enriched if lc(c) == key]
        vals = [d(c) for c in grp if d(c) is not None]
        if vals:
            vals_s = sorted(vals)
            lines.append(f"  {loc_labels[key]}")
            lines.append(f"    n={len(vals)}  mean={avg(vals):.0f}d  "
                         f"median={vals_s[len(vals_s)//2]}d  "
                         f"min={vals_s[0]}d  max={vals_s[-1]}d")
        else:
            lines.append(f"  {loc_labels[key]:<44}: no date data")

    # Top 10 longest undiscovered
    with_days = [(c, d(c)) for c in enriched if d(c) is not None]
    with_days.sort(key=lambda x: -x[1])
    lines += ["", "TOP 10 LONGEST-UNDISCOVERED CASES"]
    lines.append(f"  {'Name':<35} {'Days':>6}  {'Type':<14}  State")
    lines.append(f"  {'-'*35} {'-'*6}  {'-'*14}  {'-'*20}")
    for case, days_val in with_days[:10]:
        name  = case.get("name", "")[:34]
        state = case["location_classification"]["state"]
        ev    = case["location_classification"]["evidence"][:70]
        lines.append(f"  {name:<35} {days_val:>6}  {lc(case):<14}  {state}")
        lines.append(f"    {ev}")

    # State breakdown
    state_ctr = Counter(c["location_classification"]["state"] for c in enriched)
    lines += ["", "TOP STATES (all 190 cases)"]
    for state, n in state_ctr.most_common(15):
        lines.append(f"  {state:<30}: {n}")

    lines.append("\n" + "=" * 64)
    report = "\n".join(lines)
    print(report)

    with open("location_summary.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("Saved to 'location_summary.txt'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not Path(INPUT_FILE).exists():
        sys.exit(f"ERROR: '{INPUT_FILE}' not found.")

    cases = [json.loads(l) for l in open(INPUT_FILE, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(cases)} cases. Classifying...")

    enriched = []
    for case in cases:
        narrative     = re.sub(r"\s+", " ", case.get("narrative", "")).strip()
        missing_since = case.get("missing_since", "")
        missing_from  = case.get("missing_from", "")

        loc_type, evidence    = classify_location(narrative)
        missing_date          = parse_missing_date(missing_since)
        days_val              = estimate_days(narrative, missing_date)
        state                 = extract_state(missing_from, narrative)

        enriched.append({
            **case,
            "location_classification": {
                "location_type":      loc_type,
                "days_to_discovery":  days_val,
                "state":              state,
                "evidence":           evidence,
            },
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(enriched)} records to '{OUTPUT_FILE}'.\n")

    analyze(enriched)


if __name__ == "__main__":
    main()
