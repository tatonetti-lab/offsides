# Concept ID Migration Notes

## Summary
- `build_confounding_matrices.py` now rewrites its output CSVs after each run so that every drug and reaction is keyed by a stable identifier.
  - Drugs use `ingredient.ingredient_rxcui` (`drug_id` column) while preserving the readable `drug_name`.
  - Reactions currently use a deterministic SHA-1 hash of the MedDRA PT text (`reaction_id` column) and keep the original term in `reaction_name` until a true MedDRA code table is integrated.
- Legacy columns (`drug`, `conf_drug`, `reaction`) remain for backwards compatibility but now mirror the ID values.

## Downstream Changes
- PSM scripts (`hdpsm.py`, `two_hdpsm.py`) ingest the new columns and track `*_id`/`*_name` pairs when caching matches.
- Association estimators (`est_assoc_stats.py`, `two_est_assoc_stats.py`) join and emit results using the IDs while keeping the human-readable labels alongside.
- Existing notebooks or analysis code that previously grouped by the text columns should migrate to the `_id` columns to avoid duplicates caused by spelling/whitespace.

## Upgrading Existing Results
- Re-run `python3 src/build_confounding_matrices.py --start_year <y> --end_year <y>` for any year range you plan to analyze. The script’s new `augment_results_with_ids` helper will update the CSVs in-place with ID and name columns.
- For older PSM/association outputs stored under `results/`, re-executing `hdpsm.py` / `two_hdpsm.py` and `est_assoc_stats.py` / `two_est_assoc_stats.py` is recommended so cached files inherit the new schema.

## Next Steps
- Replace the temporary hashed `reaction_id` with the actual MedDRA concept id once the reference table is available.
- Audit notebooks (`notebooks/`) and update joins to use the ID columns to prevent accidental duplication.

## OpenFDA vs. FAERS 'All Versions'
- The totals you’ll see after running `load_openfda.py` match OpenFDA’s published counts (≈19.3 M drug-event reports through the current year). OpenFDA keeps only the most recent submission for each `safetyreportid`, so earlier follow-ups are removed during ingest.
- The FAERS public dashboard advertises ~31 M cases because it retains every follow-up and superseded version. To reproduce that view you’d need to ingest the raw FAERS `public` tables (keyed on `primaryid`/`safetyreportversion`) or extend our schema so `openfda.reports` stores multiple versions per case.
