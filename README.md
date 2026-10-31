# Submerged and Forgotten

Scraper, AI classifier, and analysis pipeline for resolved missing-persons
cases on charleyproject.org. The project asks: when a missing person's vehicle
disappears with them, how often is it eventually found crashed or submerged?

**Status:** Closed. See [CLOSURE.md](CLOSURE.md).

Findings were shared with [Adventures With Purpose](https://www.adventureswithpurpose.com/),
a dive team that locates submerged vehicles in missing-persons cases.

---

## Key Findings

All numbers from the analysis output files in [`results/`](results/):

- **1,995** resolved Charley Project cases scraped and classified
- **148** (7.4%) involved a vehicle found crashed
  ([`results/summary.txt`](results/summary.txt))
- Of those 148, **127 (85.8%)** were found in water — rivers, lakes, ponds,
  canals ([`results/crashed_vehicle_summary.txt`](results/crashed_vehicle_summary.txt))
- **249** cases had a vehicle associated with the disappearance; of those,
  **190** involved the person's own vehicle
  ([`results/own_vehicle_hypothesis.txt`](results/own_vehicle_hypothesis.txt))
- When a person disappears with their own vehicle:
  **P(vehicle found crashed or submerged) = 69.5%** (132/190)
- Median time to discovery for water cases: **2,424 days** (~6.6 years)
  ([`results/location_summary.txt`](results/location_summary.txt))
- In 57.4% of crashed-vehicle cases, the person's remains were found inside
  the car when it was recovered
  ([`results/discovery_order_summary.txt`](results/discovery_order_summary.txt))

## Pipeline

1. **[`charley_scraper.py`](charley_scraper.py)** — Scrapes the "resolved"
   section of charleyproject.org alphabetically. Extracts name, dates,
   location, classification, and narrative. Sends each narrative to
   claude-haiku for vehicle-crash classification (vehicle_found_crashed,
   confidence, evidence).

2. **[`analysis/`](analysis/)** — Nine analysis scripts that classify vehicle
   types, crash locations, discovery order, and test the own-vehicle
   hypothesis. Each uses Claude for narrative classification via
   `ANTHROPIC_API_KEY` environment variable.

3. **[`results/`](results/)** — Aggregate summary statistics (individual
   names and narratives removed).

## Data

Case-level data is not published. The source (charleyproject.org) is publicly
accessible; the scraper and classification pipeline are provided for
reproducibility.

## License

Code: [MIT](LICENSE).
