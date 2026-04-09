#!/usr/bin/env python3
"""OpenFDA FAERS loader.

This script creates a Postgres schema tailored for OpenFDA FAERS event data,
normalizes locally downloaded JSON bundles into relational tables, tracks load
progress, and explodes RxCUI mappings for downstream ingredient joins.

Example usage:
    python3 src/load_openfda.py --schema openfda --drop-schema --batch-size 5000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2 import sql


REPORT_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "safetyreportversion",
    "receivedate",
    "receivedateformat",
    "receiptdate",
    "receiptdateformat",
    "transmissiondate",
    "transmissiondateformat",
    "reporttype",
    "serious",
    "seriousnessdeath",
    "seriousnesslifethreatening",
    "seriousnesshospitalization",
    "seriousnessdisabling",
    "seriousnesscongenitalanomali",
    "seriousnessother",
    "fulfillexpeditecriteria",
    "duplicate",
    "companynumb",
    "authoritynumb",
    "primarysourcecountry",
    "occurcountry",
    "patientonsetage",
    "patientonsetageunit",
    "patientagegroup",
    "patientsex",
    "patientweight",
    "summary",
)

PRIMARY_SOURCE_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "qualification",
    "reportercountry",
    "literaturereference",
)

SENDER_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "sendertype",
    "senderorganization",
)

RECEIVER_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "receivertype",
    "receiverorganization",
)

REPORT_DUPLICATE_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "duplicatesource",
    "duplicatenumb",
)

DRUG_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "sequence",
    "drugcharacterization",
    "medicinalproduct",
    "drugauthorizationnumb",
    "drugdosagetext",
    "drugadministrationroute",
    "drugindication",
    "drugdosageform",
    "drugstructuredosagenumb",
    "drugstructuredosageunit",
    "drugseparatedosagenumb",
    "drugintervaldosagedefinition",
    "drugintervaldosageunitnumb",
    "drugstartdate",
    "drugstartdateformat",
    "drugenddate",
    "drugenddateformat",
    "drugtreatmentduration",
    "drugtreatmentdurationunit",
    "drugcumulativedosagenumb",
    "drugcumulativedosageunit",
    "drugrecurreadministration",
    "drugrecurrence",
    "actiondrug",
    "drugbatchnumb",
    "activesubstance",
    "openfda",
    "drugadditional",
)

REACTION_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "sequence",
    "reactionmeddrapt",
    "reactionmeddraversionpt",
    "reactionoutcome",
)

DRUG_RXCUI_STAGE_COLUMNS: Sequence[str] = (
    "safetyreportid",
    "sequence",
    "rxcui",
)

TABLE_COLUMNS: Dict[str, Sequence[str]] = {
    "reports": REPORT_COLUMNS,
    "primary_sources": PRIMARY_SOURCE_COLUMNS,
    "senders": SENDER_COLUMNS,
    "receivers": RECEIVER_COLUMNS,
    "report_duplicates": REPORT_DUPLICATE_COLUMNS,
    "drugs": DRUG_COLUMNS,
    "drug_rxcui_stage": DRUG_RXCUI_STAGE_COLUMNS,
    "reactions": REACTION_COLUMNS,
}


class OpenFDALoader:
    def __init__(
        self,
        config: Dict[str, str],
        input_dir: Path,
        schema: str,
        batch_size: int = 2000,
    ) -> None:
        self.config = config
        self.input_dir = input_dir
        self.schema = schema
        self.batch_size = batch_size

        self.conn = psycopg2.connect(**self.config)
        self.conn.autocommit = False

        self.buffers: Dict[str, List[Tuple]] = {
            "reports": [],
            "primary_sources": [],
            "senders": [],
            "receivers": [],
            "report_duplicates": [],
            "drugs": [],
            "reactions": [],
            "drug_rxcui_stage": [],
        }
        self.stats = {name: 0 for name in self.buffers}
        self.stats["drug2rxcui"] = 0
        self.skipped_duplicate_reports = 0

        self._flush_order: Sequence[str] = (
            "reports",
            "primary_sources",
            "senders",
            "receivers",
            "report_duplicates",
            "drugs",
            "drug_rxcui_stage",
            "reactions",
        )
        self._stage_tables = {name: f"tmp_{name}_stage" for name in self._flush_order}
        self._new_report_stage = "tmp_new_reports"
        self._staging_prepared = False

        self._column_sql = {
            name: sql.SQL(", ").join(sql.Identifier(col) for col in TABLE_COLUMNS[name])
            for name in self._flush_order
        }
        self._select_column_sql = {
            name: sql.SQL(", ").join(
                sql.SQL("s.{}").format(sql.Identifier(col)) for col in TABLE_COLUMNS[name]
            )
            for name in self._flush_order
        }
        self._table_thresholds = {
            "reports": self.batch_size,
            "primary_sources": self.batch_size,
            "senders": self.batch_size,
            "receivers": self.batch_size,
            "report_duplicates": max(self.batch_size // 2, 1),
            "drugs": max(self.batch_size * 5, self.batch_size),
            "drug_rxcui_stage": max(self.batch_size * 20, self.batch_size),
            "reactions": max(self.batch_size * 5, self.batch_size),
        }
        self._dedupe_keys = {
            "primary_sources": ("safetyreportid",),
            "senders": ("safetyreportid",),
            "receivers": ("safetyreportid",),
            "drugs": ("safetyreportid", "sequence"),
            "drug_rxcui_stage": ("safetyreportid", "sequence", "rxcui"),
            "reactions": ("safetyreportid", "sequence"),
        }

    def _status(self, message: str) -> None:
        print(f"[load_openfda] {message}")

    def create_schema(self, drop_existing: bool = False) -> None:
        with self.conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.schema)
                    )
                )
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(self.schema)
                )
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.reports (
                        safetyreportid TEXT PRIMARY KEY,
                        safetyreportversion INTEGER,
                        receivedate DATE,
                        receivedateformat VARCHAR(10),
                        receiptdate DATE,
                        receiptdateformat VARCHAR(10),
                        transmissiondate DATE,
                        transmissiondateformat VARCHAR(10),
                        reporttype SMALLINT,
                        serious SMALLINT,
                        seriousnessdeath SMALLINT,
                        seriousnesslifethreatening SMALLINT,
                        seriousnesshospitalization SMALLINT,
                        seriousnessdisabling SMALLINT,
                        seriousnesscongenitalanomali SMALLINT,
                        seriousnessother SMALLINT,
                        fulfillexpeditecriteria SMALLINT,
                        duplicate SMALLINT,
                        companynumb TEXT,
                        authoritynumb TEXT,
                        primarysourcecountry TEXT,
                        occurcountry TEXT,
                        patientonsetage NUMERIC,
                        patientonsetageunit TEXT,
                        patientagegroup TEXT,
                        patientsex SMALLINT,
                        patientweight NUMERIC,
                        summary TEXT
                    )
                    """
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.primary_sources (
                        safetyreportid TEXT PRIMARY KEY REFERENCES {}.reports(safetyreportid) ON DELETE CASCADE,
                        qualification SMALLINT,
                        reportercountry TEXT,
                        literaturereference TEXT
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.senders (
                        safetyreportid TEXT PRIMARY KEY REFERENCES {}.reports(safetyreportid) ON DELETE CASCADE,
                        sendertype SMALLINT,
                        senderorganization TEXT
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.receivers (
                        safetyreportid TEXT PRIMARY KEY REFERENCES {}.reports(safetyreportid) ON DELETE CASCADE,
                        receivertype SMALLINT,
                        receiverorganization TEXT
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.report_duplicates (
                        id BIGSERIAL PRIMARY KEY,
                        safetyreportid TEXT REFERENCES {}.reports(safetyreportid) ON DELETE CASCADE,
                        duplicatesource TEXT,
                        duplicatenumb TEXT
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.drugs (
                        id BIGSERIAL PRIMARY KEY,
                        safetyreportid TEXT REFERENCES {}.reports(safetyreportid) ON DELETE CASCADE,
                        sequence INTEGER,
                        drugcharacterization SMALLINT,
                        medicinalproduct TEXT,
                        drugauthorizationnumb TEXT,
                        drugdosagetext TEXT,
                        drugadministrationroute TEXT,
                        drugindication TEXT,
                        drugdosageform TEXT,
                        drugstructuredosagenumb NUMERIC,
                        drugstructuredosageunit TEXT,
                        drugseparatedosagenumb NUMERIC,
                        drugintervaldosagedefinition TEXT,
                        drugintervaldosageunitnumb NUMERIC,
                        drugstartdate DATE,
                        drugstartdateformat VARCHAR(10),
                        drugenddate DATE,
                        drugenddateformat VARCHAR(10),
                        drugtreatmentduration NUMERIC,
                        drugtreatmentdurationunit TEXT,
                        drugcumulativedosagenumb NUMERIC,
                        drugcumulativedosageunit TEXT,
                        drugrecurreadministration SMALLINT,
                        drugrecurrence SMALLINT,
                        actiondrug SMALLINT,
                        drugbatchnumb TEXT,
                        activesubstance JSONB,
                        openfda JSONB,
                        drugadditional JSONB
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.reactions (
                        id BIGSERIAL PRIMARY KEY,
                        safetyreportid TEXT REFERENCES {}.reports(safetyreportid) ON DELETE CASCADE,
                        sequence INTEGER,
                        reactionmeddrapt TEXT,
                        reactionmeddraversionpt TEXT,
                        reactionoutcome SMALLINT
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.drug_rxcui_stage (
                        safetyreportid TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        rxcui TEXT NOT NULL
                    )
                    """
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.drug2rxcui (
                        drug_id BIGINT REFERENCES {}.drugs(id) ON DELETE CASCADE,
                        rxcui TEXT NOT NULL
                    )
                    """
                ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reports_receivedate_idx ON {}.reports(receivedate)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reports_patientsex_idx ON {}.reports(patientsex)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drugs_report_idx ON {}.drugs(safetyreportid)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drugs_medicinalproduct_idx ON {}.drugs(medicinalproduct)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reactions_report_idx ON {}.reactions(safetyreportid)"
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL("TRUNCATE TABLE {}.drug_rxcui_stage").format(
                    sql.Identifier(self.schema)
                )
            )
            # =====================================================
            # OFFSIDES / SCRUB PERFORMANCE INDEXES
            # =====================================================

            # ---------- DRUG TABLE ----------
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drugs_id_idx "
                    "ON {}.drugs(id)"
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drugs_lower_name_idx "
                    "ON {}.drugs (LOWER(medicinalproduct))"
                ).format(sql.Identifier(self.schema))
            )

            # ---------- REACTIONS TABLE ----------
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reactions_meddra_idx "
                    "ON {}.reactions(reactionmeddrapt)"
                ).format(sql.Identifier(self.schema))
            )

            # MOST IMPORTANT INDEX FOR OFFSIDES
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reactions_pair_idx "
                    "ON {}.reactions(reactionmeddrapt, safetyreportid)"
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reactions_report_meddra_idx "
                    "ON {}.reactions(safetyreportid, reactionmeddrapt)"
                ).format(sql.Identifier(self.schema))
            )

            # ---------- DRUG2RXCUI ----------
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drug2rxcui_drug_idx "
                    "ON {}.drug2rxcui(drug_id)"
                ).format(sql.Identifier(self.schema))
            )

            # ---------- REPORT FILTERING ----------
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reports_country_idx "
                    "ON {}.reports(primarysourcecountry)"
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reports_serious_idx "
                    "ON {}.reports(serious)"
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS reports_date_serious_idx "
                    "ON {}.reports(receivedate, serious)"
                ).format(sql.Identifier(self.schema))
            )
        self.conn.commit()

    def _prepare_staging_tables(self) -> None:
        if self._staging_prepared:
            return
        with self.conn.cursor() as cur:
            for name in self._flush_order:
                cur.execute(
                    sql.SQL(
                        "CREATE TEMP TABLE IF NOT EXISTS {} (LIKE {}.{} INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
                    ).format(
                        sql.Identifier(self._stage_tables[name]),
                        sql.Identifier(self.schema),
                        sql.Identifier(name),
                    )
                )
                if name in {"drugs", "reactions", "report_duplicates"}:
                    cur.execute(
                        sql.SQL("ALTER TABLE {} DROP COLUMN IF EXISTS id")
                        .format(sql.Identifier(self._stage_tables[name]))
                    )
            cur.execute(
                sql.SQL(
                    "CREATE TEMP TABLE IF NOT EXISTS {} (safetyreportid TEXT PRIMARY KEY) ON COMMIT DELETE ROWS"
                ).format(sql.Identifier(self._new_report_stage))
            )
        self._staging_prepared = True
        self._status("prepared staging tables")

    def _copy_buffer_to_stage(self, cur, name: str, rows: Sequence[Tuple]) -> None:
        if not rows:
            return
        stage_table = self._stage_tables[name]
        column_identifiers = sql.SQL(", ").join(
            sql.Identifier(col) for col in TABLE_COLUMNS[name]
        )

        cur.execute(sql.SQL("TRUNCATE {}" ).format(sql.Identifier(stage_table)))

        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow([serialize_value(value) for value in row])
        buf.seek(0)

        copy_sql = sql.SQL(
            "COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '', HEADER false)"
        ).format(sql.Identifier(stage_table), column_identifiers)
        cur.copy_expert(copy_sql, buf)

    def _insert_reports_from_stage(self, cur) -> int:
        stage_table = self._stage_tables["reports"]
        columns_sql = self._column_sql["reports"]
        select_sql = self._select_column_sql["reports"]

        cur.execute(sql.SQL("TRUNCATE {}" ).format(sql.Identifier(self._new_report_stage)))

        insert_sql = sql.SQL(
            """
            WITH inserted AS (
                INSERT INTO {schema}.reports ({columns})
                SELECT {select_columns}
                FROM {stage} AS s
                ON CONFLICT (safetyreportid) DO NOTHING
                RETURNING safetyreportid
            )
            INSERT INTO {temp_new} (safetyreportid)
            SELECT safetyreportid FROM inserted
            """
        ).format(
            schema=sql.Identifier(self.schema),
            columns=columns_sql,
            select_columns=select_sql,
            stage=sql.Identifier(stage_table),
            temp_new=sql.Identifier(self._new_report_stage),
        )
        cur.execute(insert_sql)
        return max(cur.rowcount, 0)

    def _insert_child_from_stage(self, cur, name: str) -> int:
        stage_table = self._stage_tables[name]
        columns_sql = self._column_sql[name]
        select_sql = self._select_column_sql[name]
        dedupe = self._dedupe_keys.get(name)
        if dedupe:
            distinct_sql = sql.SQL("DISTINCT ON ({}) ").format(
                sql.SQL(", ").join(
                    sql.SQL("s.{}").format(sql.Identifier(col)) for col in dedupe
                )
            )
            order_sql = sql.SQL(", ").join(
                sql.SQL("s.{}").format(sql.Identifier(col)) for col in dedupe
            )
            order_clause = sql.SQL(" ORDER BY {}" ).format(order_sql)
        else:
            distinct_sql = sql.SQL("")
            order_clause = sql.SQL("")

        insert_sql = sql.SQL(
            """
            INSERT INTO {schema}.{table} ({columns})
            SELECT {distinct}{select_columns}
            FROM {stage} AS s
            JOIN {temp_new} AS n ON s.safetyreportid = n.safetyreportid
            {order_clause}
            """
        ).format(
            schema=sql.Identifier(self.schema),
            table=sql.Identifier(name),
            columns=columns_sql,
            select_columns=select_sql,
            distinct=distinct_sql,
            stage=sql.Identifier(stage_table),
            temp_new=sql.Identifier(self._new_report_stage),
            order_clause=order_clause,
        )
        cur.execute(insert_sql)
        return max(cur.rowcount, 0)

    def _flush_buffers(self, names_with_rows: Sequence[str]) -> Dict[str, int]:
        self._prepare_staging_tables()
        counts: Dict[str, int] = {}
        try:
            with self.conn.cursor() as cur:
                for name in names_with_rows:
                    self._copy_buffer_to_stage(cur, name, self.buffers[name])

                if "reports" in names_with_rows:
                    counts["reports"] = self._insert_reports_from_stage(cur)
                else:
                    cur.execute(
                        sql.SQL("TRUNCATE {}" ).format(sql.Identifier(self._new_report_stage))
                    )

                for name in self._flush_order:
                    if name == "reports" or name not in names_with_rows:
                        continue
                    counts[name] = self._insert_child_from_stage(cur, name)

            self.conn.commit()
            formatted = ", ".join(f"{key}={counts.get(key, 0)}" for key in names_with_rows)
            self._status(f"committed batch ({formatted})")
        except Exception:
            self.conn.rollback()
            self._staging_prepared = False
            raise
        return counts

    def flush(self, force: bool = False) -> None:
        if not force:
            report_count = len(self.buffers["reports"])
            should_flush = report_count >= self._table_thresholds["reports"]
            if not should_flush:
                for name in self._flush_order:
                    rows = self.buffers[name]
                    if not rows:
                        continue
                    limit = self._table_thresholds.get(name, self.batch_size)
                    if len(rows) >= limit:
                        should_flush = True
                        break
            if not should_flush:
                return

        names_with_rows = [name for name in self._flush_order if self.buffers[name]]
        if not names_with_rows:
            return

        self._status(
            "flushing buffers: "
            + ", ".join(f"{name}={len(self.buffers[name])}" for name in names_with_rows)
        )
        reports_buffered = len(self.buffers["reports"])
        counts = self._flush_buffers(names_with_rows)

        for name in names_with_rows:
            self.buffers[name] = []
            self.stats[name] += counts.get(name, 0)

        inserted_reports = counts.get("reports", 0)
        if reports_buffered:
            self.skipped_duplicate_reports += max(reports_buffered - inserted_reports, 0)

    def process_directory(self, limit_files: Optional[int] = None) -> None:
        files = sorted(
            [p for p in self.input_dir.rglob("*") if p.suffix.lower() in {".zip", ".json", ".ndjson"}]
        )
        if not files:
            raise FileNotFoundError(f"No JSON bundles found under {self.input_dir}")

        if limit_files is not None:
            files = files[:limit_files]

        self.skipped_duplicate_reports = 0
        self._prepare_staging_tables()
        self._status(f"processing {len(files)} file(s) from {self.input_dir}")

        total = len(files)
        if total == 0:
            print("No files selected for processing.")
            return

        for idx, path in enumerate(files, start=1):
            rel_key = self._relative_path(path)
            print(f"Processing {rel_key} ({idx}/{total})")
            self._process_path(path)

        self.flush(force=True)

    def _process_path(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        if not name.lower().endswith(".json"):
                            continue
                        payload = archive.read(name)
                        self._process_payload(payload, source=name)
            except zipfile.BadZipFile as exc:
                print(f"Skipping corrupt zip {path}: {exc}")
        elif suffix in {".json", ".ndjson"}:
            self._process_payload(path.read_bytes(), source=path.name)

    def _process_payload(self, raw: bytes, source: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Skipping {source}: JSON decode error {exc}")
            return

        results = data.get("results")
        if isinstance(results, list):
            for report in results:
                if isinstance(report, dict):
                    self._ingest_report(report)
        elif isinstance(results, dict):
            self._ingest_report(results)

    def _ingest_report(self, report: Dict) -> None:
        safetyreportid = sanitize_text(report.get("safetyreportid"))
        if safetyreportid is None:
            return

        patient = report.get("patient") or {}
        report_row = (
            safetyreportid,
            to_int(report.get("safetyreportversion")),
            parse_date(report.get("receivedate")),
            sanitize_text(report.get("receivedateformat")),
            parse_date(report.get("receiptdate")),
            sanitize_text(report.get("receiptdateformat")),
            parse_date(report.get("transmissiondate")),
            sanitize_text(report.get("transmissiondateformat")),
            to_int(report.get("reporttype")),
            to_int(report.get("serious")),
            to_int(report.get("seriousnessdeath")),
            to_int(report.get("seriousnesslifethreatening")),
            to_int(report.get("seriousnesshospitalization")),
            to_int(report.get("seriousnessdisabling")),
            to_int(report.get("seriousnesscongenitalanomali")),
            to_int(report.get("seriousnessother")),
            to_int(report.get("fulfillexpeditecriteria")),
            to_int(report.get("duplicate")),
            sanitize_text(report.get("companynumb")),
            sanitize_text(report.get("authoritynumb")),
            sanitize_text(report.get("primarysourcecountry")),
            sanitize_text(report.get("occurcountry")),
            to_decimal(patient.get("patientonsetage")),
            sanitize_text(patient.get("patientonsetageunit")),
            sanitize_text(patient.get("patientagegroup")),
            to_int(patient.get("patientsex")),
            to_decimal(patient.get("patientweight")),
            sanitize_text(patient.get("summary")),
        )
        self.buffers["reports"].append(report_row)

        primary_source = report.get("primarysource")
        if isinstance(primary_source, dict) and primary_source:
            self.buffers["primary_sources"].append(
                (
                    safetyreportid,
                    to_int(primary_source.get("qualification")),
                    sanitize_text(primary_source.get("reportercountry")),
                    sanitize_text(primary_source.get("literaturereference")),
                )
            )

        sender = report.get("sender")
        if isinstance(sender, dict) and sender:
            self.buffers["senders"].append(
                (
                    safetyreportid,
                    to_int(sender.get("sendertype")),
                    sanitize_text(sender.get("senderorganization")),
                )
            )

        receiver = report.get("receiver")
        if isinstance(receiver, dict) and receiver:
            self.buffers["receivers"].append(
                (
                    safetyreportid,
                    to_int(receiver.get("receivertype")),
                    sanitize_text(receiver.get("receiverorganization")),
                )
            )

        for duplicate in ensure_iterable(report.get("reportduplicate")):
            if isinstance(duplicate, dict):
                self.buffers["report_duplicates"].append(
                    (
                        safetyreportid,
                        sanitize_text(duplicate.get("duplicatesource")),
                        sanitize_text(duplicate.get("duplicatenumb")),
                    )
                )

        for seq, drug in enumerate(ensure_iterable(patient.get("drug")), start=1):
            if not isinstance(drug, dict):
                continue
            self.buffers["drugs"].append(
                (
                    safetyreportid,
                    seq,
                    to_int(drug.get("drugcharacterization")),
                    sanitize_text(drug.get("medicinalproduct")),
                    sanitize_text(drug.get("drugauthorizationnumb")),
                    sanitize_text(drug.get("drugdosagetext")),
                    sanitize_text(drug.get("drugadministrationroute")),
                    sanitize_text(drug.get("drugindication")),
                    sanitize_text(drug.get("drugdosageform")),
                    to_decimal(drug.get("drugstructuredosagenumb")),
                    sanitize_text(drug.get("drugstructuredosageunit")),
                    to_decimal(drug.get("drugseparatedosagenumb")),
                    sanitize_text(drug.get("drugintervaldosagedefinition")),
                    to_decimal(drug.get("drugintervaldosageunitnumb")),
                    parse_date(drug.get("drugstartdate")),
                    sanitize_text(drug.get("drugstartdateformat")),
                    parse_date(drug.get("drugenddate")),
                    sanitize_text(drug.get("drugenddateformat")),
                    to_decimal(drug.get("drugtreatmentduration")),
                    sanitize_text(drug.get("drugtreatmentdurationunit")),
                    to_decimal(drug.get("drugcumulativedosagenumb")),
                    sanitize_text(drug.get("drugcumulativedosageunit")),
                    to_int(drug.get("drugrecurreadministration")),
                    to_int(drug.get("drugrecurrence")),
                    to_int(drug.get("actiondrug")),
                    sanitize_text(drug.get("drugbatchnumb")),
                    dumps_json_or_none(drug.get("activesubstance")),
                    dumps_json_or_none(drug.get("openfda")),
                    dumps_json_or_none(drug.get("drugadditional")),
                )
            )
            rxcui_values = extract_rxcui_values(drug)
            if rxcui_values:
                seen_local = set()
                for value in rxcui_values:
                    if value in seen_local:
                        continue
                    seen_local.add(value)
                    self.buffers["drug_rxcui_stage"].append((safetyreportid, seq, value))

        for seq, reaction in enumerate(ensure_iterable(patient.get("reaction")), start=1):
            if not isinstance(reaction, dict):
                continue
            self.buffers["reactions"].append(
                (
                    safetyreportid,
                    seq,
                    sanitize_text(reaction.get("reactionmeddrapt")),
                    sanitize_text(reaction.get("reactionmeddraversionpt")),
                    to_int(reaction.get("reactionoutcome")),
                )
            )

        self.flush()

    def populate_drug_rxcui(self) -> None:
        """Populate drug-to-RxCUI mapping table from staged RxCUI values."""
        self.flush(force=True)

        truncate_sql = sql.SQL("TRUNCATE TABLE {}.drug2rxcui").format(sql.Identifier(self.schema))
        insert_sql = sql.SQL(
            """
            INSERT INTO {}.drug2rxcui (drug_id, rxcui)
            SELECT DISTINCT d.id, s.rxcui
            FROM {}.drug_rxcui_stage AS s
            JOIN {}.drugs AS d
              ON d.safetyreportid = s.safetyreportid AND d.sequence = s.sequence
            """
        ).format(
            sql.Identifier(self.schema),
            sql.Identifier(self.schema),
            sql.Identifier(self.schema),
        )

        with self.conn.cursor() as cur:
            cur.execute(truncate_sql)
            cur.execute(insert_sql)
        self.conn.commit()

        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS drug2rxcui_unique_idx ON {}.drug2rxcui (drug_id, rxcui)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drug2rxcui_rxcui_idx ON {}.drug2rxcui (rxcui)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.drug2rxcui").format(
                    sql.Identifier(self.schema)
                )
            )
            total = cur.fetchone()[0]
            cur.execute(
                sql.SQL("TRUNCATE TABLE {}.drug_rxcui_stage").format(
                    sql.Identifier(self.schema)
                )
            )
        self.conn.commit()

        self.stats["drug2rxcui"] = total

    def create_drug_ingredient(self) -> None:
        """Create drug_ingredient table linking reports to ingredient RxCUIs."""
        self._status("creating drug_ingredient table")

        drop_sql = sql.SQL(
            "DROP TABLE IF EXISTS {}.drug_ingredient"
        ).format(sql.Identifier(self.schema))

        create_sql = sql.SQL(
            """
            CREATE TABLE {}.drug_ingredient AS
            SELECT DISTINCT
                d.safetyreportid,
                d2r.rxcui AS ingredient_rxcui
            FROM {}.drugs d
            JOIN {}.drug2rxcui d2r
              ON d.id = d2r.drug_id
            WHERE d2r.rxcui IS NOT NULL
            """
        ).format(
            sql.Identifier(self.schema),
            sql.Identifier(self.schema),
            sql.Identifier(self.schema),
        )

        with self.conn.cursor() as cur:
            cur.execute(drop_sql)
            cur.execute(create_sql)

            # Helpful indexes for OFFSIDES joins
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drug_ingredient_report_idx "
                    "ON {}.drug_ingredient (safetyreportid)"
                ).format(sql.Identifier(self.schema))
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drug_ingredient_rxcui_idx "
                    "ON {}.drug_ingredient (ingredient_rxcui)"
                ).format(sql.Identifier(self.schema))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS drug_ingredient_pair_idx "
                    "ON {}.drug_ingredient (ingredient_rxcui, safetyreportid)"
                ).format(sql.Identifier(self.schema))
            )
        self.conn.commit()

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.input_dir))
        except ValueError:
            return str(path)

    def close(self) -> None:
        try:
            self.flush(force=True)
        finally:
            self.conn.close()


def load_config(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_date(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def to_int(value):
    if value in (None, "", "NA"):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


def to_decimal(value):
    if value in (None, "", "NA"):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def sanitize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def dumps_json_or_none(value):
    if value in (None, ""):
        return None
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def extract_rxcui_values(drug: Dict) -> List[str]:
    values: List[str] = []
    openfda = drug.get("openfda") if isinstance(drug, dict) else None
    if isinstance(openfda, dict):
        rxcui_field = openfda.get("rxcui")
        if isinstance(rxcui_field, list):
            for item in rxcui_field:
                text = sanitize_text(item)
                if text:
                    values.append(text)
        elif isinstance(rxcui_field, (str, int, float)):
            text = sanitize_text(rxcui_field)
            if text:
                values.append(text)
    return values


def ensure_iterable(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def serialize_value(value):
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Decimal):
        normalized = value.normalize()
        text = format(normalized, "f").rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load OpenFDA FAERS JSON bundles into Postgres")
    parser.add_argument("--config", default=Path("config.json"), type=Path, help="Database config JSON path")
    parser.add_argument(
        "--input-dir",
        default=Path("/data/home/nguyent/event"),
        type=Path,
        help="Root directory of downloaded OpenFDA files",
    )
    parser.add_argument("--schema", default="openfda", help="Target Postgres schema name")
    parser.add_argument("--batch-size", type=int, default=2000, help="Rows buffered before COPY")
    parser.add_argument("--limit-files", type=int, help="Optional limit on number of files to process")
    parser.add_argument("--drop-schema", action="store_true", help="Drop the schema before creating tables")
    parser.add_argument("--skip-create", action="store_true", help="Skip schema creation step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    loader = OpenFDALoader(config, args.input_dir, args.schema, batch_size=args.batch_size)
    try:
        if not args.skip_create:
            loader.create_schema(drop_existing=args.drop_schema)
        loader.process_directory(limit_files=args.limit_files)
        loader.populate_drug_rxcui()
        loader.create_drug_ingredient()
        print("Insertion stats:")
        for name, count in sorted(loader.stats.items()):
            if name == "drug_rxcui_stage":
                continue
            print(f"  {name}: {count}")
        if loader.skipped_duplicate_reports:
            print(f"  reports_skipped_duplicate: {loader.skipped_duplicate_reports}")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
