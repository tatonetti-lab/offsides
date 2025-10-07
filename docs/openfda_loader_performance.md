# OpenFDA Loader Performance Notes

## Current Behaviour (March 2025)

- Loader executes sequentially per archive: unzip → JSON parse in CPython → buffer → stage via `COPY` → bulk insert with `ON CONFLICT DO NOTHING`.
- Buffer thresholds: `reports` drive flush cadence (~100 k cap in current runs); high-volume child tables flush alongside the parent but do not trigger their own commits unless they exceed large guards.
- Runtime characteristics measured on the lab workstation (Postgres local, `batch-size` 10 000):

  | `--limit-files` | Reports ingested | User CPU | System | Real time |
  |-----------------|------------------|----------|--------|-----------|
  | 10              | ~103 k           | 72 s     | 12 s   | 4 m 21 s  |
  | 20              | ~206 k           | 151 s    | 23 s   | 10 m 05 s |
  | 100             | ~1.03 M          | 747 s    | 96 s   | 48 m 21 s |

- Scaling is linear (≈30 s per archive, ≈350 reports/s wall clock). CPU utilisation hovers around 30 %, so ~70 % of wall time is I/O wait (zip reads, JSON decode/encode, Postgres fsync).
- Extrapolation: 31 M reports ≈ 24–25 h wall-clock with current single-worker path.

## Known Limitations / Next Targets

1. **JSON parsing overhead** – `json.loads` + object re-serialization dominates CPU. Options: replace with `orjson`, adopt streaming (`ijson`) to avoid building entire dicts, or pre-convert archives to NDJSON to feed `COPY` directly.
2. **Decompression cost** – processing zipped archives serially forces extra I/O. Future ideas: pre-stage uncompressed NDJSON, pipe zcat → loader, or leverage `zipfile.Path` with buffered streaming.
3. **Single-worker design** – loader currently runs one process; consider multiprocessing (one worker per archive) writing to per-worker staging tables or using separate schemas before merging.
4. **Database fsync latency** – staging commits are much larger now, but sustained throughput still depends on Postgres disk speed. Investigate running on fast local SSDs or temporary tablespaces.
5. **Resume support** – intentionally removed during refactor. If we reintroduce a manifest, keep dedupe DB-driven to avoid rehydrating massive Python sets.

## Validation Checklist

- Small-file smoke: `python3 src/load_openfda.py --schema openfda --drop-schema --batch-size 10000 --limit-files 1` (expect ~70 s, one flush).
- Larger sample: `--limit-files 20` (expect ~10 min real time, two flushes).
- Full run: ensure plenty of disk, monitor Postgres `COPY` throughput, expect ~1 day runtime for 31 M reports.

## Open Questions

- Would precomputing the RxCUI explode offline (e.g., Spark job) meaningfully reduce DB time?
- Is there a viable streaming approach that emits rows straight to `COPY` without intermediate CSV buffers?
- When parallelising, how do we prevent deadlocks on shared indexes? Need clear plan (e.g., partitioned staging tables per worker).

Document updated: 2025-10-07.
