"""
analyze_crashes.py

Reads crashed_vehicle_cases.jsonl and groups cases by:
  1. Vehicle found in water vs on land
  2. Estimated time until discovery (same year / months / 1-2 yrs / 3-5 yrs / 6+ yrs / unknown)
  3. State or country

Prints results and saves to crashed_vehicle_analysis.jsonl
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

INPUT_FILE  = "crashed_vehicle_cases.jsonl"
OUTPUT_FILE = "crashed_vehicle_analysis.jsonl"
SUMMARY_FILE = "crashed_vehicle_summary.txt"

# ---------------------------------------------------------------------------
# Water / land keyword sets
# ---------------------------------------------------------------------------
WATER_KEYWORDS = [
    "river", "creek", "lake", "pond", "reservoir", "submerged", "underwater",
    "canal", "bay", "ocean", "sea", "water", "flood", "swamp", "marsh",
    "inlet", "stream", "waterway", "culvert", "drainage", "levee", "dam",
    "underwater", "beneath the surface", "pulled from", "dragged from",
]
LAND_KEYWORDS = [
    "canyon", "ravine", "cliff", "embankment", "ditch", "gully", "gulch",
    "tree", "woods", "forest", "field", "brush", "vegetation", "hillside",
    "slope", "mountain", "trail", "road", "highway", "guardrail", "barrier",
    "overpass", "bridge", "off the road", "off-road", "off a road",
    "rolled over", "overturned", "crash", "wreck", "collision",
]


def classify_location(text: str) -> str:
    """Return 'water', 'land', or 'unknown' based on keyword matching."""
    lower = text.lower()
    water_hits = sum(1 for kw in WATER_KEYWORDS if kw in lower)
    land_hits  = sum(1 for kw in LAND_KEYWORDS  if kw in lower)
    if water_hits > land_hits:
        return "water"
    if land_hits > water_hits:
        return "land"
    if water_hits:
        return "water"   # tie — lean water (more distinctive)
    return "unknown"


# ---------------------------------------------------------------------------
# Time-to-discovery helpers
# ---------------------------------------------------------------------------
MONTH_WORDS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
]
MONTH_ABBR = [m[:3] for m in MONTH_WORDS]

def extract_years_from_text(text: str) -> list[int]:
    """Pull all 4-digit years (1950-2030) from text."""
    return [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', text)]


def parse_missing_year(missing_since: str) -> int | None:
    """Parse year from MM/DD/YYYY or similar."""
    m = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', missing_since)
    return int(m.group(1)) if m else None


def estimate_discovery_lag(missing_year: int | None, narrative: str) -> str:
    """
    Estimate time between disappearance and vehicle discovery.
    Returns a bucket label.
    """
    if missing_year is None:
        return "unknown"

    years_in_text = extract_years_from_text(narrative)
    # Discovery year is likely the largest year mentioned after the disappearance year
    future_years = [y for y in years_in_text if y > missing_year]

    # Check for explicit short-lag language first
    lower = narrative.lower()
    short_lag_patterns = [
        r'\b(days?|weeks?)\s+(later|after)',
        r'(the following|next)\s+(day|week|month)',
        r'shortly (after|later|thereafter)',
        r'\bsame (day|week|month)\b',
        r'\b(hours?)\s+(later|after)',
    ]
    for pat in short_lag_patterns:
        if re.search(pat, lower):
            return "days–weeks"

    month_lag_patterns = [
        r'\b(months?)\s+(later|after)',
        r'\ba\s+month\s+(later|after)',
        r'\bseveral\s+months\b',
    ]
    for pat in month_lag_patterns:
        if re.search(pat, lower):
            return "months"

    if not future_years:
        # No later year found — check if narrative mentions month words for same-year discovery
        if any(m in lower for m in MONTH_WORDS + MONTH_ABBR):
            return "same year (estimated)"
        return "unknown"

    discovery_year = min(future_years)  # earliest year after disappearance
    lag = discovery_year - missing_year

    if lag == 0:
        return "same year"
    if lag == 1:
        return "~1 year"
    if lag <= 2:
        return "1–2 years"
    if lag <= 5:
        return "3–5 years"
    if lag <= 10:
        return "6–10 years"
    return "10+ years"


# ---------------------------------------------------------------------------
# State/country extraction
# ---------------------------------------------------------------------------
US_STATES = {
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
    "District of Columbia",
}
STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}
CANADIAN_PROVINCES = {
    "Alberta","British Columbia","Manitoba","New Brunswick",
    "Newfoundland","Nova Scotia","Ontario","Prince Edward Island",
    "Quebec","Saskatchewan","Yukon","Northwest Territories","Nunavut",
}


def extract_region(missing_from: str) -> str:
    """Return state name, Canadian province, or 'Unknown'."""
    # Clean whitespace
    clean = re.sub(r'\s+', ' ', missing_from).strip()
    parts = [p.strip() for p in clean.split(',')]

    for part in reversed(parts):   # last segment is usually state/country
        # Full state name
        for state in US_STATES:
            if state.lower() == part.lower():
                return state
        # Abbreviation
        if part.upper() in STATE_ABBR:
            # Map abbr back to full name
            for state in US_STATES:
                if state[:2].upper() == part.upper() or state.upper().startswith(part.upper()):
                    pass
            return part.upper()  # keep abbr if no full match
        # Canadian province
        for prov in CANADIAN_PROVINCES:
            if prov.lower() == part.lower():
                return f"Canada – {prov}"
        # Catch "Canada" explicitly
        if "canada" in part.lower():
            return "Canada"

    return "Unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def main() -> None:
    if not Path(INPUT_FILE).exists():
        sys.exit(f"ERROR: '{INPUT_FILE}' not found. Run analyze_results.py first.")

    cases = load_cases(INPUT_FILE)
    print(f"Loaded {len(cases)} crashed-vehicle cases.\n")

    enriched = []
    by_location_type: dict[str, list] = defaultdict(list)
    by_lag: dict[str, list]           = defaultdict(list)
    by_region: dict[str, list]        = defaultdict(list)

    for case in cases:
        cr     = case.get("classification_result", {})
        name   = case.get("name", "Unknown")
        mfrom  = re.sub(r'\s+', ' ', case.get("missing_from", "")).strip()
        msince = case.get("missing_since", "")
        narrative = case.get("narrative", "")
        evidence  = cr.get("evidence", "")

        combined_text = f"{narrative} {evidence}"

        loc_type = classify_location(combined_text)
        lag      = estimate_discovery_lag(parse_missing_year(msince), combined_text)
        region   = extract_region(mfrom)

        record = {
            "name":          name,
            "missing_from":  mfrom,
            "missing_since": msince,
            "evidence":      evidence,
            "location_type": loc_type,
            "discovery_lag": lag,
            "region":        region,
            "url":           case.get("url", ""),
        }

        enriched.append(record)
        by_location_type[loc_type].append(record)
        by_lag[lag].append(record)
        by_region[region].append(record)

    # -----------------------------------------------------------------------
    # Build summary text
    # -----------------------------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("CRASHED VEHICLE CASES — DETAILED ANALYSIS")
    lines.append("=" * 64)
    lines.append(f"\nTotal crashed-vehicle cases: {len(enriched)}\n")

    # --- 1. Water vs land ---
    lines.append("-" * 64)
    lines.append("1. VEHICLE FOUND: WATER vs LAND")
    lines.append("-" * 64)
    lag_order = ["water", "land", "unknown"]
    for loc in lag_order:
        group = by_location_type.get(loc, [])
        pct = len(group) / len(enriched) * 100 if enriched else 0
        lines.append(f"  {loc.capitalize():<10}: {len(group):>4}  ({pct:.1f}%)")

    lines.append("\n  Water cases:")
    for r in by_location_type.get("water", []):
        lines.append(f"    • {r['name']} ({r['missing_from']}) — {r['evidence']}")

    lines.append("\n  Land cases:")
    for r in by_location_type.get("land", []):
        lines.append(f"    • {r['name']} ({r['missing_from']}) — {r['evidence']}")

    # --- 2. Time to discovery ---
    lines.append("")
    lines.append("-" * 64)
    lines.append("2. TIME UNTIL VEHICLE/REMAINS DISCOVERED")
    lines.append("-" * 64)
    lag_display_order = [
        "days–weeks", "months", "same year", "same year (estimated)",
        "~1 year", "1–2 years", "3–5 years", "6–10 years", "10+ years", "unknown",
    ]
    for lag_label in lag_display_order:
        group = by_lag.get(lag_label, [])
        if group:
            pct = len(group) / len(enriched) * 100
            lines.append(f"  {lag_label:<28}: {len(group):>4}  ({pct:.1f}%)")

    # --- 3. By state/region ---
    lines.append("")
    lines.append("-" * 64)
    lines.append("3. BY STATE / REGION")
    lines.append("-" * 64)
    sorted_regions = sorted(by_region.items(), key=lambda x: -len(x[1]))
    for region, group in sorted_regions:
        pct = len(group) / len(enriched) * 100
        lines.append(f"  {region:<30}: {len(group):>4}  ({pct:.1f}%)")

    lines.append("")
    lines.append("=" * 64)

    summary_text = "\n".join(lines)
    print(summary_text)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in enriched:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nSaved enriched records to '{OUTPUT_FILE}'.")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Saved summary to '{SUMMARY_FILE}'.")


if __name__ == "__main__":
    main()
