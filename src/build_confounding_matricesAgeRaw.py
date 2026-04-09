import os
import csv
import json
import tqdm
import time
import argparse
import hashlib
import psycopg2
import numpy as np
import pandas as pd
import math
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

_DRUG_NAME_TO_ID: Dict[str, str] = {}
_DRUG_ID_TO_NAME: Dict[str, str] = {}


def _normalize(value: str) -> str:
    if value is None:
        return None
    return value.strip()


def _load_drug_lookup(db) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns mappings between normalized drug names and RxCUI IDs.

    Uses 'drugs' and 'drug2rxcui' tables in the openfda schema.
    """
    if _DRUG_NAME_TO_ID:
        return _DRUG_NAME_TO_ID, _DRUG_ID_TO_NAME

    query = """
        SELECT d.medicinalproduct, dr.rxcui
        FROM openfda.drugs AS d
        JOIN openfda.drug2rxcui AS dr ON d.id = dr.drug_id
        WHERE dr.rxcui IS NOT NULL
    """
    
    results = db.execute_query(query, verbose=False)
    
    for name, rxcui in results:
        normalized = _normalize(name)
        if not normalized or not rxcui:
            continue
        _DRUG_NAME_TO_ID[normalized] = rxcui
        _DRUG_ID_TO_NAME.setdefault(rxcui, normalized)

    return _DRUG_NAME_TO_ID, _DRUG_ID_TO_NAME


def _reaction_id_for(name: str) -> str:
    normalized = _normalize(name)
    if normalized is None:
        return None
    digest = hashlib.sha1(normalized.upper().encode("utf-8")).hexdigest()[:16]
    return digest


def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        or_mask = (result['b'] > 0) & (result['c'] > 0) & (result['d'] > 0)
        result['OR'] = np.where(or_mask, (result['a'] / result['b']) / (result['c'] / result['d']), np.nan)

        prr_mask = result['c'] > 0
        result['PRR'] = np.where(
            prr_mask,
            (result['a'] / (result['a'] + result['b'])) / (result['c'] / (result['c'] + result['d'])),
            np.nan,
        )

        denom = (result['a'] + result['b']) * (result['c'] + result['d']) * (result['b'] + result['d']) * (result['a'] + result['c'])
        denom = np.where(denom > 0, np.sqrt(denom.astype(float)), np.nan)
        result['PHI'] = np.where(
            np.isfinite(denom) & (denom > 0),
            (result['a'] * result['d'] - result['b'] * result['c']) / denom,
            np.nan,
        )
    return result


def _aggregate_counts(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    grouped = df.groupby(group_cols, as_index=False)[['a', 'b', 'c', 'd']].sum()
    return _compute_metrics(grouped)

class PostgresDB:
    """Class to handle PostgreSQL connection and queries."""
    
    def __init__(self, config_file="config.json", verbose=True):
        self.config = self.load_config(config_file)
        self.connection = None
        self.cursor = None
        self.verbose = verbose

    def load_config(self, config_file):
        """Load database configuration from JSON file."""
        with open(config_file, "r") as file:
            return json.load(file)

    def connect(self):
        """Establish connection to the PostgreSQL database."""
        if not self.connection:
            self.connection = psycopg2.connect(**self.config)
            self.cursor = self.connection.cursor()

    def execute_query(self, query, fetch_all=True, verbose=None):
        self.connect()
        if verbose is None:
            verbose = self.verbose

        start_time = time.time()
        self.cursor.execute(query)

        q = query.strip().lower()
        first_token = q.split()[0]

        if first_token in {"select", "with", "explain"}:
            result = self.cursor.fetchall() if fetch_all else self.cursor.fetchone()
        else:
            self.connection.commit()
            result = None

        exec_time = time.time() - start_time
        if verbose:
            print(f" Query completed in {exec_time}s")

        return result

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            self.connection = None

def age_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=None):
    """
    Computes contingency tables and association metrics (OR, PRR, PHI)
    using raw patient ages and RXCUI drugs.
    """
    print(f"Querying age–drug counts (RXCUI)...")
    age_drug_counts = defaultdict(int)
    ages = set()
    drugs = set()

    # --- Joint counts: raw age x drug ---
    query = f"""
    SELECT
        r.patientonsetage AS age,
        di.ingredient_rxcui AS drug_rxcui,
        COUNT(DISTINCT r.safetyreportid) AS count_reports
    FROM openfda.reports r
    JOIN openfda.drug_ingredient di
      ON r.safetyreportid = di.safetyreportid
    WHERE di.ingredient_rxcui IS NOT NULL
      AND EXTRACT(YEAR FROM r.receivedate)::int BETWEEN {start_year} AND {end_year}
      AND r.patientonsetageunit = '801'
      AND r.patientonsetage < 130
    GROUP BY age, drug_rxcui
    HAVING COUNT(DISTINCT r.safetyreportid) >= {min_reports};
    """
    results = db.execute_query(query)
    print(f"  Joint pairs: {len(results)}")
    for age, drug, count in results:
        age_drug_counts[(age, drug)] = count
        ages.add(age)
        drugs.add(drug)

    # --- Age marginals ---
    print("Querying age marginals...")
    age_count = defaultdict(int)
    query = f"""
    SELECT r.patientonsetage AS age, COUNT(DISTINCT r.safetyreportid)
    FROM openfda.reports r
    JOIN openfda.drug_ingredient di
      ON r.safetyreportid = di.safetyreportid
    WHERE di.ingredient_rxcui IS NOT NULL
      AND EXTRACT(YEAR FROM r.receivedate)::int BETWEEN {start_year} AND {end_year}
      AND r.patientonsetageunit = '801'
      AND r.patientonsetage < 130
    GROUP BY age;
    """
    results = db.execute_query(query)
    for age, count in results:
        age_count[age] = count

    # --- Drug marginals ---
    print("Querying drug marginals...")
    drug_count = defaultdict(int)
    query = f"""
    SELECT di.ingredient_rxcui, COUNT(DISTINCT r.safetyreportid)
    FROM openfda.drug_ingredient di
    JOIN openfda.reports r
      ON di.safetyreportid = r.safetyreportid
    WHERE di.ingredient_rxcui IS NOT NULL
      AND EXTRACT(YEAR FROM r.receivedate)::int BETWEEN {start_year} AND {end_year}
      AND r.patientonsetageunit = '801'
      AND r.patientonsetage < 130
    GROUP BY di.ingredient_rxcui
    HAVING COUNT(DISTINCT r.safetyreportid) >= {min_reports};
    """
    results = db.execute_query(query)
    for drug, count in results:
        drug_count[drug] = count

    # --- Total reports ---
    print("Querying total report count...")
    query = f"""
    SELECT COUNT(DISTINCT r.safetyreportid)
    FROM openfda.reports r
    JOIN openfda.drug_ingredient di
      ON r.safetyreportid = di.safetyreportid
    WHERE di.ingredient_rxcui IS NOT NULL
      AND EXTRACT(YEAR FROM r.receivedate)::int BETWEEN {start_year} AND {end_year}
      AND r.patientonsetageunit = '801'
      AND r.patientonsetage < 130;
    """
    total_reports = db.execute_query(query)[0][0]

    # --- Print diagnostics ---
    print("---- Global diagnostics ----")
    print(f"N_ages = {len(ages)}")
    print(f"N_drugs (RXCUI) = {len(drug_count)}")
    print(f"Age–drug non-zero pairs = {len(age_drug_counts)}")
    print(f"Total reports = {total_reports}")

    # --- Compute OR, PRR, PHI ---
    data = []
    invalid_tables = 0
    undefined_metrics = 0

    for age in tqdm.tqdm(sorted(ages)):
        for drug in sorted(drug_count.keys()):
            a = age_drug_counts.get((age, drug), 0)
            if a == 0:
                continue
            b = age_count[age] - a
            c = drug_count[drug] - a
            d = total_reports - (a + b + c)

            if b < 0 or c < 0 or d < 0:
                invalid_tables += 1
                continue

            OR = (a / b) / (c / d) if b > 0 and c > 0 and d > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 and (c + d) > 0 else None
            denom = float((a + b) * (c + d) * (a + c) * (b + d))
            PHI = (a * d - b * c) / np.sqrt(denom) if denom > 0 else None

            if OR is None and PRR is None and PHI is None:
                undefined_metrics += 1
                continue

            data.append([age, drug, a, b, c, d, OR, PRR, PHI])

    df = pd.DataFrame(data, columns=['age', 'drug', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # --- Summary diagnostics ---
    print("---- Summary diagnostics ----")
    print(f"Invalid tables: {invalid_tables}")
    print(f"Undefined metrics: {undefined_metrics}")
    if len(df) > 0:
        print(f"Fraction undefined: {undefined_metrics / len(df):.3f}")

    # --- Save ---
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/age_drug_associations.csv'
        print(f"Saving to {ofn}")
        df.to_csv(ofn, index=False)

    return df

def age_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file):
    print("Querying age–reaction contingency tables...")

    # ------------------------------------------------------------------
    # Shared universe CTE
    # ------------------------------------------------------------------
    cohort_cte = f"""
    WITH report_age_reaction AS (
        SELECT DISTINCT
            r.safetyreportid,
            CASE
                WHEN r.patientonsetage < 20 THEN 'Younger'
                WHEN r.patientonsetage > 60 THEN 'Older'
                ELSE 'Middle'
            END AS age_group,
            re.reactionmeddrapt AS reaction
        FROM reports r
        JOIN reactions re ON r.safetyreportid = re.safetyreportid
        WHERE r.patientonsetageunit = '801'
          AND r.patientonsetage < 130
          AND re.reactionmeddrapt IS NOT NULL
          AND r.receivedate >= DATE '{start_year}-01-01'
          AND r.receivedate < DATE '{end_year + 1}-01-01'
    )
    """

    # ------------------------------------------------------------------
    # Pair counts (a)
    # ------------------------------------------------------------------
    df_a = pd.DataFrame(
        db.execute_query(
            cohort_cte
            + f"""
            SELECT age_group, reaction, COUNT(DISTINCT safetyreportid) AS a
            FROM report_age_reaction
            GROUP BY age_group, reaction
            HAVING COUNT(DISTINCT safetyreportid) >= {min_reports};
            """
        ),
        columns=["age", "reaction", "a"]
    )

    if df_a.empty:
        warnings.warn("No age–reaction pairs passed min_reports threshold")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Total reports
    # ------------------------------------------------------------------
    total_reports = db.execute_query(
        cohort_cte + "SELECT COUNT(DISTINCT safetyreportid) FROM report_age_reaction;"
    )[0][0]

    # ------------------------------------------------------------------
    # Marginals
    # ------------------------------------------------------------------
    age_count = dict(
        db.execute_query(
            cohort_cte
            + "SELECT age_group, COUNT(DISTINCT safetyreportid) FROM report_age_reaction GROUP BY age_group;"
        )
    )

    reaction_count = dict(
        db.execute_query(
            cohort_cte
            + "SELECT reaction, COUNT(DISTINCT safetyreportid) FROM report_age_reaction GROUP BY reaction;"
        )
    )

    # ------------------------------------------------------------------
    # Compute contingency tables + metrics
    # ------------------------------------------------------------------
    rows = []
    invalid = 0
    undefined = 0

    for _, row in tqdm.tqdm(df_a.iterrows(), total=len(df_a)):
        age, reaction, a = row["age"], row["reaction"], row["a"]

        b = age_count[age] - a
        c = reaction_count[reaction] - a
        d = total_reports - (a + b + c)

        if any(x < 0 for x in [a, b, c, d]) or (a + b + c + d) != total_reports:
            warnings.warn(f"Invalid contingency: age={age}, reaction={reaction}, a={a}, b={b}, c={c}, d={d}")
            invalid += 1
            OR = PRR = PHI = None
        else:
            OR = PRR = PHI = None
            if b > 0 and c > 0 and d > 0:
                OR = (a * d) / (b * c)
                PRR = (a / (a + b)) / (c / (c + d))
                denom = float((a + b) * (c + d) * (a + c) * (b + d))
                PHI = (a * d - b * c) / np.sqrt(denom) if denom > 0 else None
            else:
                undefined += 1

        rows.append([age, reaction, a, b, c, d, OR, PRR, PHI])

    df_final = pd.DataFrame(rows, columns=["age", "reaction", "a", "b", "c", "d", "OR", "PRR", "PHI"])

    print("---- Summary ----")
    print(f"Invalid tables: {invalid}, undefined metrics: {undefined}, fraction undefined: {undefined / len(df_final):.3f}")

    if save_to_file:
        os.makedirs(f"./results/{start_year}-{end_year}", exist_ok=True)
        df_final.to_csv(f"./results/{start_year}-{end_year}/age_reaction_associations.csv", index=False)

    return df_final

def sex_by_reaction_matrix(
    db,
    start_year: int,
    end_year: int,
    min_reports: int,
    save_to_file: bool,
):
    print("Querying sex–reaction contingency tables")

    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    # ------------------------------------------------------------------
    # Base CTE: shared universe (CRITICAL)
    # ------------------------------------------------------------------
    report_sex_reaction_cte = f"""
    WITH report_sex_reaction AS (
        SELECT DISTINCT
            r.safetyreportid,
            r.patientsex AS sex,
            re.reactionmeddrapt AS reaction
        FROM openfda.reports r
        JOIN openfda.reactions re
          ON r.safetyreportid = re.safetyreportid
        WHERE r.patientsex IS NOT NULL
          AND r.patientsex != '0'
          AND re.reactionmeddrapt IS NOT NULL
          AND r.receivedate >= DATE '{start_year}-01-01'
          AND r.receivedate <  DATE '{end_year + 1}-01-01'
    )
    """

    # ------------------------------------------------------------------
    # Joint counts (a)
    # ------------------------------------------------------------------
    print("Computing joint counts (a)...")

    df_a = pd.DataFrame(
        db.execute_query(
            report_sex_reaction_cte
            + f"""
            SELECT
                sex,
                reaction,
                COUNT(DISTINCT safetyreportid) AS a
            FROM report_sex_reaction
            GROUP BY sex, reaction
            HAVING COUNT(DISTINCT safetyreportid) >= {min_reports};
            """
        ),
        columns=["sex", "reaction", "a"],
    )

    print(f"  Retained pairs (a ≥ {min_reports}): {len(df_a)}")

    if df_a.empty:
        warnings.warn("No sex–reaction pairs passed min_reports threshold")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Total reports (N)
    # ------------------------------------------------------------------
    total_reports = db.execute_query(
        report_sex_reaction_cte
        + "SELECT COUNT(DISTINCT safetyreportid) FROM report_sex_reaction;"
    )[0][0]

    print(f"Total eligible reports (N): {total_reports}")

    # ------------------------------------------------------------------
    # Marginals
    # ------------------------------------------------------------------
    print("Computing sex marginals...")
    sex_count = dict(
        db.execute_query(
            report_sex_reaction_cte
            + """
            SELECT sex, COUNT(DISTINCT safetyreportid)
            FROM report_sex_reaction
            GROUP BY sex;
            """
        )
    )

    print("Computing reaction marginals...")
    reaction_count = dict(
        db.execute_query(
            report_sex_reaction_cte
            + """
            SELECT reaction, COUNT(DISTINCT safetyreportid)
            FROM report_sex_reaction
            GROUP BY reaction;
            """
        )
    )

    # ------------------------------------------------------------------
    # Contingency tables + metrics
    # ------------------------------------------------------------------
    print("Building contingency tables and computing metrics...")

    rows = []
    invalid = 0

    for _, row in tqdm.tqdm(df_a.iterrows(), total=len(df_a)):
        sex = row["sex"]
        reaction = row["reaction"]
        a = int(row["a"])

        b = sex_count[sex] - a
        c = reaction_count[reaction] - a
        d = total_reports - (a + b + c)

        # ---- Step 1: core validity ----
        if any(x < 0 for x in (a, b, c, d)):
            warnings.warn(
                f"Negative contingency: sex={sex}, reaction={reaction}, "
                f"a={a}, b={b}, c={c}, d={d}"
            )
            invalid += 1
            continue

        if a + b + c + d != total_reports:
            warnings.warn(
                f"Inconsistent contingency: a+b+c+d={a+b+c+d}, N={total_reports}"
            )
            invalid += 1
            continue

        # ---- Step 2: numeric validity ----
        OR = PRR = PHI = None
        if b > 0 and c > 0 and d > 0:
            OR = (a * d) / (b * c)
            PRR = (a / (a + b)) / (c / (c + d))

            denom = float((a + b) * (c + d) * (a + c) * (b + d))
            if denom > 0:
                PHI = (a * d - b * c) / np.sqrt(denom)

        rows.append([sex, reaction, a, b, c, d, OR, PRR, PHI])

    # ------------------------------------------------------------------
    # Final dataframe
    # ------------------------------------------------------------------
    df_final = pd.DataFrame(
        rows,
        columns=["sex", "reaction", "a", "b", "c", "d", "OR", "PRR", "PHI"],
    )

    print("---- Summary diagnostics ----")
    print(f"Invalid contingency tables: {invalid}")
    print(f"Final rows written: {len(df_final)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_to_file:
        outdir = Path(f"./results/{start_year}-{end_year}")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "sex_reaction_associations.csv"
        print(f"Saving to {outpath}")
        df_final.to_csv(outpath, index=False)

    return df_final

# -------------------------------
# Sex by Drug Matrix
# -------------------------------
def sex_by_drug_matrix(
    db,
    start_year: int,
    end_year: int,
    min_reports: int,
    save_to_file: bool = True,
):
    print("Querying sex–drug contingency tables (RXCUI, single-ingredient)")

    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    # ------------------------------------------------------------------
    # Base CTE: shared universe (CRITICAL)
    # ------------------------------------------------------------------
    report_sex_drugs_cte = f"""
    WITH report_sex_drugs AS (
        SELECT DISTINCT
            r.safetyreportid,
            r.patientsex AS sex,
            di.ingredient_rxcui AS drug_id
        FROM openfda.reports r
        JOIN openfda.drug_ingredient di
          ON r.safetyreportid = di.safetyreportid
        WHERE di.ingredient_rxcui IS NOT NULL
          AND r.patientsex IS NOT NULL
          AND r.patientsex != '0'
          AND r.receivedate >= DATE '{start_year}-01-01'
          AND r.receivedate <  DATE '{end_year + 1}-01-01'
    )
    """

    # ------------------------------------------------------------------
    # Joint counts (a)
    # ------------------------------------------------------------------
    print("Computing joint counts (a)...")

    df_a = pd.DataFrame(
        db.execute_query(
            report_sex_drugs_cte
            + f"""
            SELECT
                sex,
                drug_id,
                COUNT(DISTINCT safetyreportid) AS a
            FROM report_sex_drugs
            GROUP BY sex, drug_id
            HAVING COUNT(DISTINCT safetyreportid) >= {min_reports};
            """
        ),
        columns=["sex", "drug_id", "a"],
    )

    print(f"  Retained pairs (a ≥ {min_reports}): {len(df_a)}")

    if df_a.empty:
        warnings.warn("No sex–drug pairs passed min_reports threshold")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Total reports (N)
    # ------------------------------------------------------------------
    total_reports = db.execute_query(
        report_sex_drugs_cte
        + "SELECT COUNT(DISTINCT safetyreportid) FROM report_sex_drugs;"
    )[0][0]

    print(f"Total eligible reports (N): {total_reports}")

    # ------------------------------------------------------------------
    # Marginals
    # ------------------------------------------------------------------
    print("Computing sex marginals...")
    sex_count = dict(
        db.execute_query(
            report_sex_drugs_cte
            + """
            SELECT sex, COUNT(DISTINCT safetyreportid)
            FROM report_sex_drugs
            GROUP BY sex;
            """
        )
    )

    print("Computing drug marginals...")
    drug_count = dict(
        db.execute_query(
            report_sex_drugs_cte
            + """
            SELECT drug_id, COUNT(DISTINCT safetyreportid)
            FROM report_sex_drugs
            GROUP BY drug_id;
            """
        )
    )

    # ------------------------------------------------------------------
    # Contingency tables + metrics
    # ------------------------------------------------------------------
    print("Building contingency tables and computing metrics...")

    rows = []
    invalid = 0

    for _, row in tqdm.tqdm(df_a.iterrows(), total=len(df_a)):
        sex = row["sex"]
        drug_id = row["drug_id"]
        a = int(row["a"])

        b = sex_count[sex] - a
        c = drug_count[drug_id] - a
        d = total_reports - (a + b + c)

        # ---- Step 1: core validity ----
        if any(x < 0 for x in (a, b, c, d)):
            warnings.warn(
                f"Negative contingency: sex={sex}, drug_id={drug_id}, "
                f"a={a}, b={b}, c={c}, d={d}"
            )
            invalid += 1
            continue

        if a + b + c + d != total_reports:
            warnings.warn(
                f"Inconsistent contingency: a+b+c+d={a+b+c+d}, N={total_reports}"
            )
            invalid += 1
            continue

        # ---- Step 2: numeric validity ----
        OR = PRR = PHI = None
        if b > 0 and c > 0 and d > 0:
            OR = (a * d) / (b * c)
            PRR = (a / (a + b)) / (c / (c + d))

            denom = (a + b) * (c + d) * (a + c) * (b + d)
            if denom > 0:
                PHI = (a * d - b * c) / np.sqrt(float(denom))

        rows.append([sex, drug_id, a, b, c, d, OR, PRR, PHI])

    # ------------------------------------------------------------------
    # Final dataframe
    # ------------------------------------------------------------------
    df_final = pd.DataFrame(
        rows,
        columns=["sex", "drug_id", "a", "b", "c", "d", "OR", "PRR", "PHI"],
    )

    print("---- Summary diagnostics ----")
    print(f"Invalid contingency tables: {invalid}")
    print(f"Final rows written: {len(df_final)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_to_file:
        outdir = Path(f"./results/{start_year}-{end_year}")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "sex_drug_associations.csv"
        print(f"Saving to {outpath}")
        df_final.to_csv(outpath, index=False)

    return df_final

# -------------------------------
# Drug by Drug Matrix
# -------------------------------
def drug_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file):
    print("Querying drug–drug counts (excluding duplicate drugs per report)...")

    report_drugs_cte = f"""
    WITH report_drugs AS (
        SELECT DISTINCT
               di.safetyreportid,
               di.ingredient_rxcui AS drug
        FROM openfda.drug_ingredient di
        JOIN openfda.reports r USING (safetyreportid)
        WHERE di.ingredient_rxcui IS NOT NULL
          AND r.receivedate >= DATE '{start_year}-01-01'
          AND r.receivedate <  DATE '{end_year + 1}-01-01'
    )
    """

    total_reports = db.execute_query(report_drugs_cte + "SELECT COUNT(DISTINCT safetyreportid) FROM report_drugs;")[0][0]
    print(f"N_reports = {total_reports}")

    drug_count = dict(db.execute_query(report_drugs_cte + "SELECT drug, COUNT(DISTINCT safetyreportid) FROM report_drugs GROUP BY drug;"))
    print(f"N_drugs = {len(drug_count)}")

    df = pd.DataFrame(db.execute_query(report_drugs_cte + f"""
        SELECT d1.drug AS drug1, d2.drug AS drug2, COUNT(DISTINCT d1.safetyreportid) AS a
        FROM report_drugs d1
        JOIN report_drugs d2
          ON d1.safetyreportid = d2.safetyreportid
         AND d1.drug < d2.drug
        GROUP BY drug1, drug2
        HAVING COUNT(DISTINCT d1.safetyreportid) >= {min_reports};
    """), columns=["conf_drug", "drug", "a"])
    print(f"Non-zero drug–drug pairs = {len(df)}")

    df["b"] = df["conf_drug"].map(drug_count) - df["a"]
    df["c"] = df["drug"].map(drug_count) - df["a"]
    df["d"] = total_reports - (df["a"] + df["b"] + df["c"])

    df["sum"] = df["a"] + df["b"] + df["c"] + df["d"]
    df[["OR", "PRR", "PHI"]] = np.nan
    
    # ---- Step 1 & 2 ----
    for col in ['b','c','d']:
        if (df[col] < 0).any():
            warnings.warn(f"Negative contingency values in column {col}")

    valid_step1 = (
        (df[["a","b","c","d"]] >= 0).all(axis=1) &
        (df["sum"] == total_reports) &
        (df["a"] <= df["conf_drug"].map(drug_count)) &
        (df["a"] <= df["drug"].map(drug_count))
    )

    if (~valid_step1).any():
        warnings.warn("Invalid contingency tables detected (negative values or bad sums)")
    valid_step2 = valid_step1 & (df[["b","c","d"]] > 0).all(axis=1)

    df.loc[valid_step2, "OR"] = (
        df.loc[valid_step2, "a"] * df.loc[valid_step2, "d"]
    ) / (
        df.loc[valid_step2, "b"] * df.loc[valid_step2, "c"]
    )

    df.loc[valid_step2, "PRR"] = (
        (df.loc[valid_step2, "a"] / (df.loc[valid_step2, "a"] + df.loc[valid_step2, "b"])) /
        (df.loc[valid_step2, "c"] / (df.loc[valid_step2, "c"] + df.loc[valid_step2, "d"]))
    )

    df.loc[valid_step2, "PHI"] = (
        df.loc[valid_step2, "a"] * df.loc[valid_step2, "d"] -
        df.loc[valid_step2, "b"] * df.loc[valid_step2, "c"]
    ) / np.sqrt(
        (df.loc[valid_step2, "a"] + df.loc[valid_step2, "b"]) *
        (df.loc[valid_step2, "c"] + df.loc[valid_step2, "d"]) *
        (df.loc[valid_step2, "a"] + df.loc[valid_step2, "c"]) *
        (df.loc[valid_step2, "b"] + df.loc[valid_step2, "d"])
    )

    print("Invalid rows:",
          (~valid_step1).sum(),
          "out of", len(df))

    if save_to_file:
        os.makedirs(f"./results/{start_year}-{end_year}", exist_ok=True)
        df.to_csv(f"./results/{start_year}-{end_year}/drug_drug_associations.csv", index=False)

    return df

# -------------------------------
# Drug by Reaction Matrix
# -------------------------------
def drug_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file):
    print("Querying drug–reaction counts (excluding multi-ingredient products)...")

    if min_reports < 5:
        warnings.warn(
            "min_reports < 5 may produce unstable OR/PRR estimates in sparse FAERS data"
        )

    report_drugs_cte = f"""
    WITH report_drugs AS (
        SELECT DISTINCT
            di.safetyreportid,
            di.ingredient_rxcui AS drug
        FROM openfda.drug_ingredient di
        JOIN openfda.reports r
          ON di.safetyreportid = r.safetyreportid
        WHERE di.ingredient_rxcui IS NOT NULL
          AND r.receivedate >= DATE '{start_year}-01-01'
          AND r.receivedate < DATE '{end_year + 1}-01-01'
    )
    """

    # -----------------------------
    # Universe counts
    # -----------------------------
    total_reports = db.execute_query(
        report_drugs_cte +
        "SELECT COUNT(DISTINCT safetyreportid) FROM report_drugs;"
    )[0][0]

    drug_count = dict(
        db.execute_query(
            report_drugs_cte +
            "SELECT drug, COUNT(DISTINCT safetyreportid) FROM report_drugs GROUP BY drug;"
        )
    )

    reaction_count = dict(
        db.execute_query(
            report_drugs_cte + """
            SELECT re.reactionmeddrapt, COUNT(DISTINCT re.safetyreportid)
            FROM report_drugs rd
            JOIN openfda.reactions re USING (safetyreportid)
            WHERE re.reactionmeddrapt IS NOT NULL
            GROUP BY re.reactionmeddrapt;
            """
        )
    )

    # Optional consistency check
    assert sum(drug_count.values()) >= total_reports, \
        "Drug marginal counts smaller than total reports — universe mismatch?"

    # -----------------------------
    # Joint counts (a)
    # -----------------------------
    df = pd.DataFrame(
        db.execute_query(
            report_drugs_cte + f"""
            SELECT
                rd.drug,
                re.reactionmeddrapt AS reaction,
                COUNT(DISTINCT rd.safetyreportid) AS a
            FROM report_drugs rd
            JOIN openfda.reactions re USING (safetyreportid)
            WHERE re.reactionmeddrapt IS NOT NULL
            GROUP BY rd.drug, reaction
            HAVING COUNT(DISTINCT rd.safetyreportid) >= {min_reports};
            """
        ),
        columns=["drug", "reaction", "a"]
    )

    # -----------------------------
    # Build contingency table
    # -----------------------------
    df["b"] = df["drug"].map(drug_count) - df["a"]
    df["c"] = df["reaction"].map(reaction_count) - df["a"]
    df["d"] = total_reports - (df["a"] + df["b"] + df["c"])

    # -----------------------------
    # Validation: structural invariants
    # -----------------------------
    df["sum"] = df["a"] + df["b"] + df["c"] + df["d"]

    valid_core = (
        (df[["a", "b", "c", "d"]] >= 0).all(axis=1) &
        (df["a"] <= df["drug"].map(drug_count)) &
        (df["a"] <= df["reaction"].map(reaction_count)) &
        (df["sum"] == total_reports)
    )

    if (~valid_core).any():
        warnings.warn(
            f"{(~valid_core).sum()} rows failed contingency table validation"
        )

    # -----------------------------
    # Validation: numeric computability
    # Strict positivity required for stable OR/PRR/PHI
    # (excludes perfect separation cases)
    # -----------------------------
    valid_numeric = valid_core & (df[["b", "c", "d"]] > 0).all(axis=1)

    # -----------------------------
    # Compute metrics safely
    # -----------------------------
    df[["OR", "PRR", "PHI"]] = np.nan

    df.loc[valid_numeric, "OR"] = (
        df.loc[valid_numeric, "a"] * df.loc[valid_numeric, "d"]
    ) / (
        df.loc[valid_numeric, "b"] * df.loc[valid_numeric, "c"]
    )

    df.loc[valid_numeric, "PRR"] = (
        df.loc[valid_numeric, "a"] /
        (df.loc[valid_numeric, "a"] + df.loc[valid_numeric, "b"])
    ) / (
        df.loc[valid_numeric, "c"] /
        (df.loc[valid_numeric, "c"] + df.loc[valid_numeric, "d"])
    )

    denom = (
        (df["a"] + df["b"]) *
        (df["c"] + df["d"]) *
        (df["a"] + df["c"]) *
        (df["b"] + df["d"])
    )

    phi_mask = valid_numeric & (denom > 0)
    df.loc[phi_mask, "PHI"] = (
        df.loc[phi_mask, "a"] * df.loc[phi_mask, "d"] -
        df.loc[phi_mask, "b"] * df.loc[phi_mask, "c"]
    ) / np.sqrt(denom[phi_mask])

    # -----------------------------
    # Diagnostics summary
    # -----------------------------
    undefined = (~valid_numeric).sum()

    print("Validation summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Valid contingency tables: {valid_core.sum()}")
    print(f"  Valid for OR/PRR/PHI: {valid_numeric.sum()}")
    print(f"  Undefined OR/PRR/PHI: {undefined}")
    if len(df) > 0:
        print(f"  Fraction undefined: {undefined / len(df):.3f}")

    # -----------------------------
    # Save results
    # -----------------------------
    if save_to_file:
        outdir = f"./results/{start_year}-{end_year}"
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(
            f"{outdir}/drug_reaction_associations.csv",
            index=False
        )

    return df

# -------------------------------
# Indication by Drug Matrix
# -------------------------------
def indication_by_drug_matrix(
    db,
    start_year: int,
    end_year: int,
    min_reports: int,
    save_to_file: bool,
):
    print("Querying indication–drug contingency tables (RXCUI, single-ingredient)")

    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    # ------------------------------------------------------------------
    # Base CTEs (single, canonical universe)
    # ------------------------------------------------------------------
    report_drugs_cte = f"""
    WITH report_drugs AS (
        SELECT DISTINCT
            di.safetyreportid,
            di.ingredient_rxcui AS drug_id
        FROM openfda.drug_ingredient di
        JOIN openfda.reports r
          ON di.safetyreportid = r.safetyreportid
        WHERE di.ingredient_rxcui IS NOT NULL
          AND r.receivedate >= DATE '{start_year}-01-01'
          AND r.receivedate <  DATE '{end_year + 1}-01-01'
    ),
    report_inds AS (
        SELECT DISTINCT
            safetyreportid,
            drugindication AS indication
        FROM openfda.drugs
        WHERE drugindication IS NOT NULL
          AND drugindication NOT LIKE 'Product used for unknown indication'
    )
    """

    # ------------------------------------------------------------------
    # Joint counts (a)
    # ------------------------------------------------------------------
    print("Computing joint counts (a)...")

    df_a = pd.DataFrame(
        db.execute_query(
            report_drugs_cte
            + f"""
            SELECT
                ri.indication,
                rd.drug_id,
                COUNT(DISTINCT rd.safetyreportid) AS a
            FROM report_drugs rd
            JOIN report_inds ri USING (safetyreportid)
            GROUP BY ri.indication, rd.drug_id
            HAVING COUNT(DISTINCT rd.safetyreportid) >= {min_reports};
            """
        ),
        columns=["indication", "drug", "a"],
    )

    print(f"  Retained pairs (a ≥ {min_reports}): {len(df_a)}")

    if df_a.empty:
        warnings.warn("No indication–drug pairs passed min_reports threshold")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Total reports (N) — SAME UNIVERSE
    # ------------------------------------------------------------------
    total_reports = db.execute_query(
        report_drugs_cte
        + """
        SELECT COUNT(DISTINCT safetyreportid)
        FROM report_drugs;
        """
    )[0][0]

    print(f"Total eligible reports (N): {total_reports}")

    # ------------------------------------------------------------------
    # Marginals — SAME UNIVERSE
    # ------------------------------------------------------------------
    print("Computing indication marginals...")
    ind_count = dict(
        db.execute_query(
            report_drugs_cte
            + """
            SELECT
                ri.indication,
                COUNT(DISTINCT rd.safetyreportid)
            FROM report_drugs rd
            JOIN report_inds ri USING (safetyreportid)
            GROUP BY ri.indication;
            """
        )
    )

    print("Computing drug marginals...")
    drug_count = dict(
        db.execute_query(
            report_drugs_cte
            + """
            SELECT
                drug_id,
                COUNT(DISTINCT safetyreportid)
            FROM report_drugs
            GROUP BY drug_id;
            """
        )
    )

    # ------------------------------------------------------------------
    # Contingency tables + metrics
    # ------------------------------------------------------------------
    print("Building contingency tables and computing metrics...")

    rows = []
    invalid = 0
    undefined = 0

    for _, row in tqdm.tqdm(df_a.iterrows(), total=len(df_a)):
        ind = row["indication"]
        drug_id = row["drug"]
        a = int(row["a"])

        b = ind_count[ind] - a
        c = drug_count[drug_id] - a
        d = total_reports - (a + b + c)

        # Structural validation
        if any(x < 0 for x in (a, b, c, d)):
            warnings.warn(
                f"Negative contingency: {ind}, {drug_id} → "
                f"a={a}, b={b}, c={c}, d={d}"
            )
            invalid += 1
            continue

        if a + b + c + d != total_reports:
            warnings.warn(
                f"Inconsistent contingency: a+b+c+d={a+b+c+d}, N={total_reports}"
            )
            invalid += 1
            continue

        OR = (a * d) / (b * c) if b > 0 and c > 0 else None
        PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        PHI = (a * d - b * c) / np.sqrt(float(denom)) if denom > 0 else None

        if OR is None or PHI is None:
            undefined += 1

        rows.append([ind, drug_id, a, b, c, d, OR, PRR, PHI])

    # ------------------------------------------------------------------
    # Final dataframe
    # ------------------------------------------------------------------
    df_final = pd.DataFrame(
        rows,
        columns=["indication", "drug", "a", "b", "c", "d", "OR", "PRR", "PHI"],
    )

    print(f"Undefined OR/PHI: {undefined}")
    if len(df_final) > 0:
        print(f"Fraction undefined: {undefined / len(df_final):.3f}")

    print("---- Summary diagnostics ----")
    print(f"Invalid contingency tables: {invalid}")
    print(f"Final rows written: {len(df_final)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_to_file:
        from pathlib import Path
        outdir = Path(f"./results/{start_year}-{end_year}")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "indication_drug_associations.csv"
        print(f"Saving raw association table to {outpath}")
        df_final.to_csv(outpath, index=False)

    return df_final

# -------------------------------
# Indication by Reaction Matrix
# -------------------------------
def indication_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file):
    print("Querying indication–reaction counts...")

    indications = set()
    reactions = set()
    indrea_count = defaultdict(int)

    # Step 1: indication-reaction pairs
    query = f"""
    WITH reports_in_year AS (
        SELECT safetyreportid
        FROM openfda.reports
        WHERE EXTRACT(YEAR FROM receivedate) BETWEEN {start_year} AND {end_year}
    )
    SELECT d.drugindication AS indication, re.reactionmeddrapt AS reaction, COUNT(DISTINCT d.safetyreportid) AS a
    FROM openfda.drugs d
    JOIN openfda.reactions re USING (safetyreportid)
    JOIN reports_in_year r USING (safetyreportid)
    WHERE d.drugindication IS NOT NULL
      AND d.drugindication NOT LIKE 'Product used for unknown indication'
      AND re.reactionmeddrapt IS NOT NULL
    GROUP BY indication, reaction
    HAVING COUNT(DISTINCT d.safetyreportid) >= {min_reports};
    """
    df = pd.DataFrame(db.execute_query(query), columns=["indication","reaction","a"])
    print(f"Non-zero pairs (a ≥ {min_reports}): {len(df)}")

    # marginals
    ind_count = dict(db.execute_query(f"""
        SELECT d.drugindication, COUNT(DISTINCT d.safetyreportid)
        FROM openfda.drugs d
        JOIN openfda.reports r USING (safetyreportid)
        WHERE d.drugindication IS NOT NULL
          AND d.drugindication NOT LIKE 'Product used for unknown indication'
          AND EXTRACT(YEAR FROM r.receivedate) BETWEEN {start_year} AND {end_year}
        GROUP BY d.drugindication;
    """))
    reac_count = dict(db.execute_query(f"""
        SELECT re.reactionmeddrapt, COUNT(DISTINCT re.safetyreportid)
        FROM openfda.reactions re
        JOIN openfda.reports r USING (safetyreportid)
        WHERE re.reactionmeddrapt IS NOT NULL
          AND EXTRACT(YEAR FROM r.receivedate) BETWEEN {start_year} AND {end_year}
        GROUP BY re.reactionmeddrapt;
    """))
    total_reports = db.execute_query(f"SELECT COUNT(DISTINCT safetyreportid) FROM openfda.reports WHERE EXTRACT(YEAR FROM receivedate) BETWEEN {start_year} AND {end_year};")[0][0]
    print("---- Global diagnostics ----")
    print(f"N_indications = {len(ind_count)}")
    print(f"N_reactions = {len(reac_count)}")
    print(f"Total reports (N) = {total_reports}")

    print(f"Sum indication marginals = {sum(ind_count.values())}")
    print(f"Sum reaction marginals = {sum(reac_count.values())}")
    
    data = []
    invalid = 0
    undefined = 0
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        ind, reac, a = row["indication"], row["reaction"], row["a"]
        b = ind_count[ind] - a
        c = reac_count[reac] - a
        assert a <= ind_count[ind], (ind, a, ind_count[ind])
        assert a <= reac_count[reac], (reac, a, reac_count[reac])
        d = total_reports - (a + b + c)

        if a + b + c + d != total_reports:
            warnings.warn(
                f"Inconsistent contingency: a+b+c+d={a+b+c+d} != N={total_reports}"
            )
            invalid += 1
            continue

        if any(x < 0 for x in (a, b, c, d)):
            warnings.warn(
                f"Negative contingency: indication={ind}, reaction={reac}, "
                f"a={a}, b={b}, c={c}, d={d}"
            )
            invalid += 1
            continue

        OR = (a/b)/(c/d) if b>0 and c>0 and d>0 else None
        PRR = (a/(a+b))/(c/(c+d)) if c>0 else None
        a, b, c, d = float(a), float(b), float(c), float(d)
        denom = (a+b)*(c+d)*(b+d)*(a+c)
        PHI = (a*d - b*c)/np.sqrt(denom) if denom>0 else None

        if OR is None or PHI is None:
            undefined += 1

        data.append([ind,reac,a,b,c,d,OR,PRR,PHI])

    df_final = pd.DataFrame(data, columns=["indication","reaction","a","b","c","d","OR","PRR","PHI"])
    print("---- Summary diagnostics ----")
    print(f"Total rows: {len(df_final)}")
    print(f"Invalid contingency tables: {invalid}")
    print(f"Undefined metrics: {undefined}")

    if len(df_final) > 0:
        print(f"Fraction invalid: {invalid / len(df_final):.3f}")
        print(f"Fraction undefined: {undefined / len(df_final):.3f}")

    print("---- Metric availability ----")
    print(f"OR defined:  {df_final['OR'].notna().sum()} / {len(df_final)}")
    print(f"PHI defined: {df_final['PHI'].notna().sum()} / {len(df_final)}")

    print("OR summary:")
    print(df_final["OR"].describe())

    print("PHI summary:")
    print(df_final["PHI"].describe())

    if save_to_file:
        os.makedirs(f"./results/{start_year}-{end_year}", exist_ok=True)
        df_final.to_csv(f"./results/{start_year}-{end_year}/indication_reaction_associations.csv", index=False)

    return df_final

def _transform_drug_drug(results_dir: Path, db) -> None:
    path = results_dir / 'drug_drug_associations.csv'
    if not path.exists():
        return

    df = pd.read_csv(path)

    # If already transformed, do nothing
    if {'drug_id', 'conf_drug_id'}.issubset(df.columns):
        return

    # 🔒 HARD GUARD: ensure this is actually a drug–drug table
    required_cols = {
        'drug',
        'conf_drug',
        'a', 'b', 'c', 'd',
        'OR', 'PRR', 'PHI'
    }
    if not required_cols.issubset(df.columns):
        # Not a drug–drug matrix → skip silently
        return

    name_to_id, id_to_name = _load_drug_lookup(db)

    # Case 1: drugs are already numeric IDs
    if pd.api.types.is_numeric_dtype(df['drug']):
        df['drug_id'] = df['drug'].astype(int)
        df['conf_drug_id'] = df['conf_drug'].astype(int)

    # Case 2: drugs are names → normalize + lookup
    else:
        df['drug_norm'] = df['drug'].apply(_normalize)
        df['conf_drug_norm'] = df['conf_drug'].apply(_normalize)

        df['drug_id'] = df['drug_norm'].map(name_to_id)
        df['conf_drug_id'] = df['conf_drug_norm'].map(name_to_id)


    df = df.dropna(subset=['drug_id', 'conf_drug_id'])

    counts = df[['conf_drug_id', 'drug_id', 'a', 'b', 'c', 'd']]
    aggregated = _aggregate_counts(counts, ['conf_drug_id', 'drug_id'])

    # normalize key types
    id_to_name = {int(k): v for k, v in id_to_name.items()}
    aggregated['drug_name'] = aggregated['drug_id'].map(id_to_name)
    aggregated['conf_drug_name'] = aggregated['conf_drug_id'].map(id_to_name)

    aggregated = aggregated[
        [
            'conf_drug_id', 'conf_drug_name',
            'drug_id', 'drug_name',
            'a', 'b', 'c', 'd',
            'OR', 'PRR', 'PHI'
        ]
    ]

    aggregated.to_csv(path, index=False)

def _transform_drug_reaction(results_dir: Path, db) -> None:
    path = results_dir / 'drug_reaction_associations.csv'
    if not path.exists():
        return

    df = pd.read_csv(path)

    # Already transformed
    if 'drug_id' in df.columns and 'reaction_id' in df.columns:
        return

    # drug is RXCUI already — treat it as ID
    df['drug_id'] = df['drug']

    # reaction is MedDRA PT string → internal ID
    df['reaction_norm'] = df['reaction'].apply(_normalize)
    df['reaction_id'] = df['reaction_norm'].apply(_reaction_id_for)

    df = df.dropna(subset=['drug_id', 'reaction_id'])

    counts = df[['drug_id', 'reaction_id', 'a', 'b', 'c', 'd']]
    aggregated = _aggregate_counts(counts, ['drug_id', 'reaction_id'])

    # Optional: add drug names for readability
    _, id_to_name = _load_drug_lookup(db)
    aggregated['drug_name'] = aggregated['drug_id'].map(lambda x: id_to_name.get(str(x)))

    # Reaction name lookup
    reaction_name_map = {
        _reaction_id_for(name): name
        for name in df['reaction_norm'].dropna().unique()
    }
    aggregated['reaction_name'] = aggregated['reaction_id'].map(reaction_name_map)

    aggregated = aggregated[
        [
            'drug_id', 'drug_name',
            'reaction_id', 'reaction_name',
            'a', 'b', 'c', 'd',
            'OR', 'PRR', 'PHI'
        ]
    ]

    aggregated.to_csv(path, index=False)

def _transform_indication_drug(results_dir: Path, db) -> None:
    path = results_dir / 'indication_drug_associations.csv'
    if not path.exists():
        return

    df = pd.read_csv(path)
    print(len(df))

    # Already transformed
    if 'drug_id' in df.columns:
        return

    _, id_to_name = _load_drug_lookup(db)

    # drug column is already RxCUI
    df['drug_id'] = df['drug']
    df = df.dropna(subset=['drug_id'])
    print(len(df))

    counts = df[['indication', 'drug_id', 'a', 'b', 'c', 'd']]
    aggregated = (
        counts
        .groupby(['indication', 'drug_id'], as_index=False)[['a', 'b', 'c', 'd']]
        .sum()
    )

    aggregated = _compute_metrics(aggregated)

    aggregated['drug_name'] = aggregated['drug_id'].map(
        lambda x: id_to_name.get(str(x))
    )

    aggregated = aggregated[
        ['indication', 'drug_id', 'drug_name', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI']
    ]

    aggregated.to_csv(path, index=False)


def augment_results_with_ids(db, start_year: int, end_year: int) -> None:
    results_dir = Path('results') / f"{start_year}-{end_year}"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    _transform_drug_drug(results_dir, db)
    _transform_drug_reaction(results_dir, db)
    _transform_indication_drug(results_dir, db)

def parse_args():
    parser = argparse.ArgumentParser(description="Process a range of years.")
    parser.add_argument('--start_year', type=int, required=True, help='Start year (inclusive)')
    parser.add_argument('--end_year', type=int, required=True, help='End year (inclusive)')
    return parser.parse_args()

if __name__ == "__main__":
    db = PostgresDB()

    args = parse_args()
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")
    start_year = args.start_year
    end_year = args.end_year
    min_reports = 25
    
    #ind_rea_df = indication_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #ind_drug_df = indication_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #drug_rea_df = drug_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #drug_drug_df = drug_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #sex_drug_df = sex_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #sex_rea_df = sex_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #age_rea_df = age_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    age_drug_df = age_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    #augment_results_with_ids(db, start_year, end_year)
    db.close()
