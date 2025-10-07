# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the high-dimensional propensity score pipeline; entry points include `hdpsm.py`, `est_assoc_stats.py`, and helpers for confounder matrix builds.
- `data/` stores intermediate FAERS extracts; keep large raw dumps outside git.
- `results/` stores association tables; mirror year ranges in filenames (e.g., `offsides_2004_2004.parquet`).
- `notebooks/` contains evaluation notebooks (e.g., `evaluate_performance_confounding.ipynb`).
- Shell wrappers (`run_single_year.sh`, `run_hdpsm_parts.sh`, `run_two_hdpsm_parts.sh`) orchestrate common run modes defined in `config.json`.

## Build, Test, and Development Commands
- `python3 src/build_confounding_matrices.py --start_year 2004 --end_year 2004` materializes confounder features for the selected window.
- `python3 src/hdpsm.py --start_year 2004 --end_year 2004` performs propensity score matching and writes matched cohorts.
- `python3 src/est_assoc_stats.py --start_year 2004 --end_year 2004` derives disproportionality statistics and saves OffSIDES candidates.
- `./run_single_year.sh 2004` runs the end-to-end pipeline for a single FAERS year using the config defaults; use `run_two_hdpsm_parts.sh` for long ranges to stay within memory bounds.
- `python3 src/load_openfda.py --schema openfda --drop-schema` rebuilds the schema, fills `openfda.drug2rxcui`; reruns should start from a fresh schema so the database can enforce de-duplication.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indents, snake_case for functions and variables, UPPER_SNAKE_CASE for module constants (e.g., `MAX_SAMPLES`).
- Keep scripts importable; favor functions over inline logic and guard CLI entry points with `if __name__ == "__main__":`.
- Reuse existing argparse patterns for flags (`--start_year`, `--end_year`), and document new parameters in `README.md` if they alter the workflow.

## Testing & Validation
- There is no automated test suite; validate changes by running a narrow-year pipeline (`--start_year 2004 --end_year 2004`) and confirming outputs in `results/`.
- Use evaluation notebooks to compare with prior runs; include key deltas in PRs.
- When adjusting SQL access layers, dry-run with a small subset or staging database before touching production dumps.

## Commit & Pull Request Guidelines
- Match existing history: short, imperative, lower-case commit subjects (e.g., `refine hdpsm matching loop`).
- Group related changes per commit and reference the affected year range or dataset when relevant.
- Pull requests should describe the motivation, execution command(s), resulting artifacts (paths and counts), and any validation evidence or screenshots.
- Link to tracked issues when available and note any follow-up runs required.

## Data & Configuration Notes
- Connection credentials live outside the repo; point `pgpass.tmp` or your own `.pgpass` via `PGPASSFILE` when running locally.
- Update `config.json` cautiously - retain default keys so orchestration scripts continue to resolve expected paths and resource limits.
