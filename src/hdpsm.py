"""
High-dimensional propensity score matching (HDPSM) script — improved.

Features:
- Sparse matrices for memory efficiency
- Parallel processing of drugs
- Automatic skipping of completed drugs
- Clean logging per part
- Reproducible random sampling
"""

import os
os.environ["JOBLIB_START_METHOD"] = "fork"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
import tqdm

from build_confounding_matrices import PostgresDB

MAX_SAMPLES = 5000
NREPS = 3
MATCH_RATIO = 5


def setup_logger(part=None):
    log_fp = f"hdpsm_part{part}.log" if part else "hdpsm.log"
    logging.basicConfig(
        filename=log_fp,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger()


def _normalize(value):
    return value.strip() if isinstance(value, str) else value


def _ensure_drug_columns(df, id_col, name_col, fallback_col):
    if id_col not in df.columns and fallback_col in df.columns:
        df[id_col] = df[fallback_col].apply(_normalize)
    if name_col not in df.columns:
        source = fallback_col if fallback_col in df.columns else id_col
        df[name_col] = df[source]


def _build_drug_metadata(ind_df, drug_df):
    parts = []
    for id_col, name_col in [('drug_id', 'drug_name'), ('conf_drug_id', 'conf_drug_name')]:
        if id_col in drug_df.columns:
            parts.append(drug_df[[id_col, name_col]].dropna())
    if {'drug_id', 'drug_name'} <= set(ind_df.columns):
        parts.append(ind_df[['drug_id', 'drug_name']].dropna())
    if not parts:
        return {}
    merged = pd.concat(parts, ignore_index=True).drop_duplicates()
    return dict(zip(merged.iloc[:, 0], merged.iloc[:, 1]))


def downsample_pos_neg(y, max_samples, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos_keep = min(len(pos_idx), max_samples // 2)
    n_neg_keep = min(max_samples - n_pos_keep, len(neg_idx))
    keep_idx = np.concatenate([
        rng.choice(pos_idx, n_pos_keep, replace=False),
        rng.choice(neg_idx, n_neg_keep, replace=False)
    ])
    rng.shuffle(keep_idx)
    return keep_idx

from sklearn.neighbors import NearestNeighbors

def match_psm(report_ids, y, propensity, ratio=5):

    treated_idx = np.where(y == 1)[0]
    control_idx = np.where(y == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return None

    # sort controls by propensity
    control_idx = control_idx.copy()
    np.random.shuffle(control_idx)
    control_sorted = control_idx[np.argsort(propensity[control_idx])]

    matched_rows = []
    stratum_id = 0
    used_controls = 0

    for t in treated_idx:
        stratum_id += 1

        # treated
        matched_rows.append({
            "report_id": report_ids[t],
            "treatment": 1,
            "propensity_score": propensity[t],
            "stratum": stratum_id
        })

        # take next K controls (greedy)
        for j in range(ratio):
            c = control_sorted[(used_controls + j) % len(control_sorted)]
            matched_rows.append({
                "report_id": report_ids[c],
                "treatment": 0,
                "propensity_score": propensity[c],
                "stratum": stratum_id
            })

        used_controls += ratio

    return pd.DataFrame(matched_rows)

from itertools import repeat

def build_drug_dataset(drug, drug2report, ind2report,
                       ind_drug_df, drug_drug_df,
                       MAX_SAMPLES, rng_global):

    corr_inds = set(
        ind_drug_df[
            (ind_drug_df["drug_id"] == drug) &
            (ind_drug_df["PHI"] > 0)
        ]["indication"].dropna().unique()
    )

    corr_drugs = set(
        drug_drug_df[
            (drug_drug_df["drug_id"] == drug) &
            (drug_drug_df["PHI"] > 0)
        ]["conf_drug_id"].dropna().unique()
    )

    reports = defaultdict(set)

    for ind in corr_inds:
        if ind in ind2report:
            for rid in ind2report[ind]:
                reports[rid].add(ind)

    for d in corr_drugs:
        if d in drug2report:
            for rid in drug2report[d]:
                reports[rid].add(d)

    features = sorted(corr_drugs | corr_inds)
    if not features:
        return None

    sorted_reportids_full = np.array(sorted(reports.keys()))

    drug_pos_set = set(drug2report.get(drug, []))

    y_full = np.array(
        [1 if rid in drug_pos_set else 0 for rid in sorted_reportids_full],
        dtype=np.uint8
    )

    # downsample once
    if len(sorted_reportids_full) > MAX_SAMPLES:
        pos_idx = np.where(y_full == 1)[0]
        neg_idx = np.where(y_full == 0)[0]

        n_pos = min(len(pos_idx), MAX_SAMPLES // 2)
        n_neg = min(len(neg_idx), MAX_SAMPLES - n_pos)

        keep = np.concatenate([
            rng_global.choice(pos_idx, n_pos, replace=False),
            rng_global.choice(neg_idx, n_neg, replace=False)
        ])

        keep.sort()

        sorted_reportids = sorted_reportids_full[keep]
        y = y_full[keep]
    else:
        sorted_reportids = sorted_reportids_full
        y = y_full

    # feature index
    feat_idx = {f: i for i, f in enumerate(features)}

    # sparse matrix build (once!)
    rows = []
    cols = []
    feature_set = set(features)

    for i, rid in enumerate(sorted_reportids):
        valid_feats = [
            feat_idx[f]
            for f in reports[rid]
            if f in feature_set and f in feat_idx
        ]

        rows.extend(repeat(i, len(valid_feats)))
        cols.extend(valid_feats)

    X = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(sorted_reportids), len(features)),
        dtype=np.uint8
    )

    return {
        "X": X,
        "y": y,
        "report_ids": sorted_reportids,
        "drug_pos_set": drug_pos_set
    }

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
def process_drug(
    drug_idx,
    drug,
    drug_meta,
    ind_drug_df,
    drug_drug_df,
    drug2report,
    ind2report,
    results_dir,
    seed = None
):
    drug2rep = drug2report
    ind2rep = ind2report
    rng_global = np.random.default_rng(seed + drug_idx)
    drug_name_for_log = drug_meta.get(drug, drug)
    per_drug_path = results_dir / "psmWnreps" / f"{drug_idx}_{drug}.csv.gz"

    if per_drug_path.exists():
        logging.info(f"Skipping completed drug: {drug_name_for_log} [{drug}]")
        return None

    data = build_drug_dataset(
        drug,
        drug2rep,
        ind2rep,
        ind_drug_df,
        drug_drug_df,
        MAX_SAMPLES,
        rng_global
    )

    if data is None:
        return None

    X = data["X"]
    y = data["y"]
    sorted_reportids = data["report_ids"]

    # -----------------------------
    # PRECOMPUTE ONCE PER DRUG
    # -----------------------------
    treated_idx = np.flatnonzero(y)
    control_idx = np.flatnonzero(1 - y)

    n_treated = len(treated_idx)
    n_control = len(control_idx)

    n = len(y)

    # -----------------------------
    # NREPS LOOP (NEW)
    # -----------------------------
    cv = StratifiedKFold(n_splits=NREPS, shuffle=True, random_state=42 + drug_idx)

    clf = linear_model.SGDClassifier(loss="log_loss", penalty="l2")

    all_rep_results = []

    X_all = X  # alias (clarity + avoids attribute lookup cost)

    for rep, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # CRITICAL FIX: skip bad splits
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logging.warning(f"{drug} rep {rep}: only one class in split — skipping")
            continue

        clf.fit(X_train, y_train)

        auroc = roc_auc_score(
            y_test,
            clf.predict_proba(X_test)[:, 1]
        )

        propensity = clf.predict_proba(X_all)[:, 1]

        matched_df = match_psm(
            sorted_reportids,
            y,
            propensity,
            ratio=MATCH_RATIO
        )

        if matched_df is None:
            continue

        # IMPORTANT: create a COPY so no shared mutation
        matched_df = matched_df.copy()

        matched_df["replicate"] = rep
        matched_df["drug"] = drug
        matched_df["drug_id"] = drug
        matched_df["drug_name"] = drug_meta.get(drug, None)
        matched_df["auroc"] = auroc

        all_rep_results.append(matched_df)

    # -----------------------------
    # combine replicates
    # -----------------------------
    if not all_rep_results:
        logging.info(f"Skipping {drug_name_for_log} [{drug}]: no matches across replicates")
        return None

    df_out = pd.concat(all_rep_results, ignore_index=True)

    df_out.to_csv(per_drug_path, index=False, compression="gzip")

    logging.info(f"Finished {drug_name_for_log} [{drug}] with {NREPS} reps")

    return df_out

def parse_args():
    parser = argparse.ArgumentParser(description="Run HDPSM for a year range")
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--part', type=int, required=False)
    parser.add_argument('--total_parts', type=int, required=False)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument("--mode", type=str, default="full",
                    choices=["full", "single", "chunk"])

    parser.add_argument("--target_drug", type=str, default=None)

    parser.add_argument("--start", type=int, default=0)

    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args()

def process_drug_star(args):
    return process_drug(*args)

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.part)
    rng_global = np.random.default_rng(seed=42)

    results_dir = Path("results") / f"{args.start_year}-{args.end_year}"
    ind_drug_file = results_dir / "indication_drug_associations.csv"
    drug_drug_file = results_dir / "drug_drug_associations.csv"
    if not ind_drug_file.exists() or not drug_drug_file.exists():
        raise FileNotFoundError("Missing confounding matrices CSVs.")

    ind_drug_df = pd.read_csv(ind_drug_file)
    drug_drug_df = pd.read_csv(drug_drug_file)

    # FIX: PHI NaNs silently break GLP-1 structure
    ind_drug_df["PHI"] = ind_drug_df["PHI"].fillna(0)
    drug_drug_df["PHI"] = drug_drug_df["PHI"].fillna(0)

    # Expand directional drug-drug pairs
    if {"drug1", "drug2"}.issubset(drug_drug_df.columns):
        df1 = drug_drug_df.rename(columns={"drug1": "drug_id", "drug2": "conf_drug_id"})
        df2 = drug_drug_df.rename(columns={"drug2": "drug_id", "drug1": "conf_drug_id"})
        drug_drug_df = pd.concat([df1, df2], ignore_index=True)

    _ensure_drug_columns(ind_drug_df, "drug_id", "drug_name", "drug")
    _ensure_drug_columns(drug_drug_df, "drug_id", "drug_name", "drug")
    _ensure_drug_columns(drug_drug_df, "conf_drug_id", "conf_drug_name", "conf_drug")

    ind_drug_df["drug_id"] = ind_drug_df["drug_id"].astype(str)
    drug_drug_df["drug_id"] = drug_drug_df["drug_id"].astype(str)
    drug_drug_df["conf_drug_id"] = drug_drug_df["conf_drug_id"].astype(str)
    ind_drug_df["drug"] = ind_drug_df["drug_id"]
    drug_drug_df["drug"] = drug_drug_df["drug_id"]
    drug_drug_df["conf_drug"] = drug_drug_df["conf_drug_id"]

    drug_meta = _build_drug_metadata(ind_drug_df, drug_drug_df)

    ind_drugs = set(ind_drug_df["drug_id"].unique())
    drug_drugs = set(drug_drug_df["drug_id"].unique())
    common_drugs = sorted(ind_drugs & drug_drugs)

    if args.target_drug:
        keywords = [k.lower() for k in args.target_drug.split(",")]

        filtered = []
        for d in common_drugs:
            name = str(drug_meta.get(d, "")).lower()
            if any(k in name for k in keywords):
                filtered.append(d)

        print(f"Filtering to {len(filtered)} drugs using target_drug")
        common_drugs = filtered

    if args.part is not None:
        if args.total_parts is None:
            raise ValueError("--total_parts must be set when using --part")

        if args.part < 1 or args.part > args.total_parts:
            raise ValueError("--part must be between 1 and --total_parts")

        part_size = int(np.ceil(len(common_drugs) / args.total_parts))

        start_idx = (args.part - 1) * part_size
        end_idx = min(start_idx + part_size, len(common_drugs))

        common_drugs = common_drugs[start_idx:end_idx]

        print(f"Running part {args.part}/{args.total_parts}: "
              f"{start_idx}-{end_idx} ({len(common_drugs)} drugs)")

    logger.info(f"Processing {len(common_drugs)} drugs (part {args.part}/{args.total_parts if args.total_parts else 1})")
    os.makedirs(results_dir / "psmWnreps", exist_ok=True)

    # Load cached mappings
    cache_fp = results_dir / "_tmp_drug2report_ind2report.pkl"
    if cache_fp.exists():
        logger.info("Loading cached report mappings")
        with open(cache_fp, "rb") as fh:
            drug2report, ind2report = pickle.load(fh)
    else:
        logger.info("Building report mappings from DB")
        db = PostgresDB(verbose=False)
        drug2report = defaultdict(set)
        ind2report = defaultdict(set)
        query = f"""
        SELECT DISTINCT
            d2r.rxcui, d.medicinalproduct, d.drugindication, r.safetyreportid
        FROM openfda.drugs d
        JOIN openfda.drug2rxcui d2r ON d.id = d2r.drug_id
        JOIN openfda.reports r ON d.safetyreportid = r.safetyreportid
        WHERE d.drugindication IS NOT NULL
          AND EXTRACT(YEAR FROM r.receivedate) BETWEEN {args.start_year} AND {args.end_year}
        """
        for drug_id, drug_name, ind, report_id in db.execute_query(query):
            if drug_id is None:
                continue
            drug_id = str(drug_id)
            drug2report[drug_id].add(report_id)
            ind2report[ind].add(report_id)
            drug_meta.setdefault(drug_id, drug_name)
        with open(cache_fp, "wb") as fh:
            pickle.dump((drug2report, ind2report), fh)

    # --- Sequential, memory-safe HDPSM loop ---
    all_results = []

    from multiprocessing import Pool, cpu_count

    drug_args = [
        (
            drug_idx,
            drug,
            drug_meta,
            ind_drug_df,
            drug_drug_df,
            drug2report,
            ind2report,
            results_dir,
            42
        )
        for drug_idx, drug in enumerate(common_drugs)
    ]

    n_jobs = args.n_jobs or max(1, cpu_count() - 1)

    logger.info(f"Running Pool with {n_jobs} workers on {len(common_drugs)} drugs")

    with Pool(processes=n_jobs) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(process_drug_star, drug_args),
                total=len(drug_args),
                desc="Drugs (Pool)"
            )
        )

    all_results = [r for r in results if r is not None]

    # --- Save combined output ---
    if all_results:
        results_dir.mkdir(parents=True, exist_ok=True)

        combined_fp = results_dir / f"hdpsm_nrep{NREPS}_mratio{MATCH_RATIO}_maxsamp{MAX_SAMPLES}wNREPS.csv.gz"

        pd.concat(all_results, ignore_index=True).to_csv(
            combined_fp,
            index=False,
            compression="gzip"
        )

        logger.info(f"Saved combined HDPSM results to {combined_fp}")
    else:
        logger.warning("No results to combine from HDPSM run.")

    logger.info("HDPSM run complete.")
