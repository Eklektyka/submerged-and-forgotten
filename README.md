# Submerged and Forgotten

Scraper and AI classifier for resolved missing-persons cases on
charleyproject.org. Classifies whether a vehicle was found crashed or
submerged in each resolved case using Claude Haiku.

**Status:** Closed — 1,995 cases scraped and classified. No aggregation or
analysis performed. No research question was defined. See [CLOSURE.md](CLOSURE.md).

## What it does

1. Scrapes the "resolved" section of charleyproject.org alphabetically
2. Extracts: name, missing_since, missing_from, classification, narrative
3. Sends each narrative to claude-haiku for vehicle-crash classification
4. Outputs JSONL with: vehicle_found_crashed (bool), confidence, evidence

## Data

Case-level data is not published (personal information about missing persons).
The scraper code is provided for reproducibility.

## License

Code: [MIT](LICENSE).
