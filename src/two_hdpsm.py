"""
Run high-dimensional propensity score matching to control for confounding effects. 

"""

import os
import tqdm
import time
import shutil
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict

# import gnuplotlib as gp

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

from build_confounding_matrices import PostgresDB

import pandas as pd
import numpy as np

MAX_SAMPLES = 10_000
MIN_REPORTS = 5


def _normalize(value):
    return value.strip() if isinstance(value, str) else value


def _ensure_drug_columns(df: pd.DataFrame, id_col: str, name_col: str, fallback_col: str) -> None:
    if id_col not in df.columns and fallback_col in df.columns:
        df[id_col] = df[fallback_col].apply(_normalize)
    if name_col not in df.columns:
        source = fallback_col if fallback_col in df.columns else id_col
        df[name_col] = df[source]


def _build_drug_metadata(ind_drug_df: pd.DataFrame, drug_drug_df: pd.DataFrame) -> Dict[str, str]:
    parts = []
    for cols in [('drug_id', 'drug_name'), ('conf_drug_id', 'conf_drug_name')]:
        id_col, name_col = cols
        if id_col in drug_drug_df.columns:
            parts.append(drug_drug_df[[id_col, name_col]].dropna())
    if {'drug_id', 'drug_name'} <= set(ind_drug_df.columns):
        parts.append(ind_drug_df[['drug_id', 'drug_name']].dropna())
    if not parts:
        return {}
    merged = pd.concat(parts, ignore_index=True).drop_duplicates()
    return dict(zip(merged.iloc[:, 0], merged.iloc[:, 1]))

def stratified_1n_matching(df, propensity_col='propensity_score', treatment_col='treatment', 
                           n_bins=5, match_ratio=1, random_state=42):
    df = df.copy()
    np.random.seed(random_state)

    # Assign strata based on propensity score
    df['stratum'] = pd.qcut(df[propensity_col], q=n_bins, duplicates='drop')

    matched = []

    # Process each stratum
    for stratum, group in df.groupby('stratum', observed=False):
        treated = group[group[treatment_col] == 1]
        control = group[group[treatment_col] == 0]

        if treated.empty or control.empty:
            continue  # Drop stratum if one group is missing

        # Shuffle control for randomness
        control = control.sample(frac=1, random_state=random_state)

        for _, treated_row in treated.iterrows():
            # Sample n controls for each treated unit
            sampled_controls = control.sample(n=match_ratio, replace=False) \
                if len(control) >= match_ratio else None

            if sampled_controls is not None:
                matched.append(pd.concat([treated_row.to_frame().T, sampled_controls], axis=0))

    if not matched:
        return pd.DataFrame()  # return empty if nothing matched

    matched_df = pd.concat(matched, axis=0).reset_index(drop=True)
    return matched_df

def run_multiple_matchings(df, num_replicates=10, **match_kwargs):
    all_matches = []

    for rep in range(num_replicates):
        matched = stratified_1n_matching(df, random_state=rep, **match_kwargs)
        if not matched.empty:
            matched['replicate'] = rep
            all_matches.append(matched)

    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    else:
        print("No matches found in any replicate.")
        return pd.DataFrame()

def parse_args():
    parser = argparse.ArgumentParser(description="Process a range of years.")
    parser.add_argument('--start_year', type=int, required=True, help='Start year (inclusive)')
    parser.add_argument('--end_year', type=int, required=True, help='End year (inclusive)')
    parser.add_argument('--part', type=int, required=False, help="Which part of the run to execute.")
    parser.add_argument('--total_parts', type=int, required=False, help="Total parts in this run.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")

    if not args.part is None:
        if args.total_parts is None:
            raise Exception("ERROR: --part was set but --total_parts was not. --total_parts must also be set when using --part.")
        if args.part > args.total_parts:
            raise Exception(f"ERROR: Part provided, {args.part}, is greater than the total parts, {args.total_parts}")
        if args.part < 1:
            raise Exception(f"ERROR: Part provided, {args.part}, was less than 1. Value must be between 1 and --total_parts.")

    nreps = 10
    match_ratio = 5    
    start_year = args.start_year
    end_year = args.end_year
    compute_strategy = 'in_mem' # or could be 'in_db'

    # in_db on 2004-2004: 
    # python3 src/hdpsm.py  7091.22s user 8923.49s system 102% cpu 4:19:13.69 total
    # in_mem on 2004-2004:
    # python3 src/hdpsm.py  7089.12s user 8851.79s system 375% cpu 1:10:49.58 total

    results_dir = os.path.join('results', f"{start_year}-{end_year}")

    if not os.path.exists(results_dir):
        raise Exception(f"Confounding matrices must be built first. No results found at {results_dir}")
        
    ind_drug_file = os.path.join(results_dir, 'indication_drug_associations.csv')
    drug_drug_file = os.path.join(results_dir, 'drug_drug_associations.csv')

    if not os.path.exists(ind_drug_file):
        raise Exception(f"Confounding matrices must be built first. Indicaitond-drug assocations missing: {ind_drug_file}")
    
    if not os.path.exists(drug_drug_file):
        raise Exception(f"Confounding matrices must be built first. Drug-drug assocations missing: {drug_drug_file}")


    ind_drug_df = pd.read_csv(ind_drug_file)
    drug_drug_df = pd.read_csv(drug_drug_file)

    _ensure_drug_columns(ind_drug_df, 'drug_id', 'drug_name', 'drug')
    _ensure_drug_columns(drug_drug_df, 'drug_id', 'drug_name', 'drug')
    _ensure_drug_columns(drug_drug_df, 'conf_drug_id', 'conf_drug_name', 'conf_drug')

    ind_drug_df['drug_id'] = ind_drug_df['drug_id'].astype(str)
    drug_drug_df['drug_id'] = drug_drug_df['drug_id'].astype(str)
    drug_drug_df['conf_drug_id'] = drug_drug_df['conf_drug_id'].astype(str)

    ind_drug_df['drug'] = ind_drug_df['drug_id']
    drug_drug_df['drug'] = drug_drug_df['drug_id']
    drug_drug_df['conf_drug'] = drug_drug_df['conf_drug_id']

    drug_meta = _build_drug_metadata(ind_drug_df, drug_drug_df)

    ind_drugs = set(ind_drug_df['drug_id'].unique())
    drug_drugs = set(drug_drug_df['drug_id'].unique())
    common_drugs = sorted(ind_drugs & drug_drugs)

    print(len(ind_drugs), len(drug_drugs), len(common_drugs))

    #print(common_drugs)

    db = PostgresDB(verbose=False)

    drug2report = None
    ind2report = None 
    if compute_strategy == 'in_mem':
        pkl_fp = os.path.join(results_dir, '_tmp_drug2report_ind2report.pkl')
        if os.path.exists(pkl_fp,):
            print("Found pickle file, will load report data from disk.")
            with open(pkl_fp, 'rb') as fh:
                drug2report, ind2report = pickle.load(fh)
        else:
            # load the drugs and indications data into local memory first
            # may be prohbitively large for the system memory to handle for large year ranges
            drug2report = defaultdict(set)
            ind2report = defaultdict(set)

            query = f"""
            select ingredient.ingredient_rxcui,
                   ingredient.ingredient_concept_name,
                   snomed_term,
                   safetyreport_id
            from drug
            join ingredient on (ingredient.id = drug.id)
            join safetyreport on (safetyreport.id = safetyreport_id)
            join drug_indications using (drugindication)
            where left(receivedate, 4)::int between {start_year} and {end_year}
            """
            results = db.execute_query(query)
            for drug_id_value, drug_name_value, ind, reportid in results:
                if drug_id_value is None:
                    continue
                drug_id_value = str(drug_id_value)
                drug2report[drug_id_value].add(reportid)
                ind2report[ind].add(reportid)
                if drug_id_value not in drug_meta and drug_name_value:
                    drug_meta[drug_id_value] = drug_name_value
            
            with open(pkl_fp, 'wb') as fh:
                pickle.dump((drug2report, ind2report), fh)
            
    elif compute_strategy == 'in_db':
        pass
    else:
        raise Exception(f"Unexpected compute strategy provided: {compute_strategy}. Expected 'in_mem' or 'in_db'")

    matched_df = None
    os.makedirs(os.path.join(results_dir, 'twopsm'), exist_ok=True)
    logfh = open(f"logs/twopsm_{start_year}-{end_year}_{time.time()}.log", 'w')

    if not args.part is None:
        start_pos = int((float(args.part-1)/float(args.total_parts))*len(common_drugs))
        stop_pos = int((float(args.part)/float(args.total_parts))*len(common_drugs))
        print(f"Working on part {args.part} of {args.total_parts}. Will execute drugs in index range: ({start_pos}, {stop_pos}]")

    print("Preprocessing the data to speed up analysis later...")
    drug2corrinds = defaultdict(set)
    drug2corrdrugs = defaultdict(set)
    for _, drug in tqdm.tqdm(enumerate(common_drugs), total=len(common_drugs)):
        drug2corrinds[drug] = set(ind_drug_df[(ind_drug_df['drug'] == drug) & (ind_drug_df['PHI'] > 0)]['indication'].unique())
        drug2corrdrugs[drug] = set(drug_drug_df[(drug_drug_df['drug'] == drug) & (drug_drug_df['PHI'] > 0)]['conf_drug'].unique())

    for drug1idx, drug1 in tqdm.tqdm(enumerate(common_drugs), total=len(common_drugs)):
    # for drug1idx, drug1 in enumerate(common_drugs):
        
        if not (args.part is None) and not (start_pos <= drug1idx < stop_pos):
            continue
        
        # print(f"Working on propensity score matching for pairs including {drug1} ({drug1idx+1} of {len(common_drugs)})")
        
        # for idx, drug2 in tqdm.tqdm(enumerate(common_drugs[(drug1idx+1):]), total=len(common_drugs[(drug1idx+1):])):
        for idx, drug2 in enumerate(common_drugs[(drug1idx+1):]):

            drug2idx = (drug1idx+1)+idx

            if drug2idx < drug1idx:
                continue
            
            # if drug1 != '17-alpha-hydroxyprogesterone' or drug2 != 'bupropion':
            #     continue
            
            drug1_name = drug_meta.get(drug1, drug1)
            drug2_name = drug_meta.get(drug2, drug2)

            cache_path = os.path.join(results_dir, 'twopsm', f"{drug1idx}_{drug1}_{drug2idx}_{drug2}.csv.gz")
            if os.path.exists(cache_path):
                # print(' > Found preexisting run. Will load from there.')
                if not args.part is None:
                    continue
                drug_matched_df = pd.read_csv(cache_path)
                if matched_df is None:
                    matched_df = drug_matched_df
                else:
                    matched_df = pd.concat([matched_df, drug_matched_df])
                continue
            
            # print(ind_drug_df.shape)
            # print(ind_drug_df[(ind_drug_df['drug'] == drug1) & (ind_drug_df['PHI'] > 0)].shape)
            # corr_inds_1 = set(ind_drug_df[(ind_drug_df['drug'] == drug1) & (ind_drug_df['PHI'] > 0)]['indication'].unique())
            # corr_drugs_1 = set(drug_drug_df[(drug_drug_df['drug'] == drug1) & (drug_drug_df['PHI'] > 0)]['conf_drug'].unique())

            # corr_inds_2 = set(ind_drug_df[(ind_drug_df['drug'] == drug2) & (ind_drug_df['PHI'] > 0)]['indication'].unique())
            # corr_drugs_2 = set(drug_drug_df[(drug_drug_df['drug'] == drug2) & (drug_drug_df['PHI'] > 0)]['conf_drug'].unique())
            
            corr_inds_1 = drug2corrinds[drug1]
            corr_inds_2 = drug2corrinds[drug2]

            corr_drugs_1 = drug2corrdrugs[drug1]
            corr_drugs_2 = drug2corrdrugs[drug2]

            corr_inds = corr_inds_1 | corr_inds_2
            corr_drugs = corr_drugs_1 | corr_drugs_2
            # print(corr_inds)
            
            if drug2report is None:
                raise Exception("TwoSIDES requires drug2report to be populated.")
            else:
                drug1_report_ids = drug2report.get(drug1, set())
                drug2_report_ids = drug2report.get(drug2, set())
                pair_report_ids = drug1_report_ids & drug2_report_ids
            
            # print(drug1, drug2)
            # print(f"Number of pair reports: {len(pair_report_ids)}")
            
            if len(pair_report_ids) < MIN_REPORTS:
                continue

            corr_inds_reports = defaultdict(set)
            if ind2report is None:
                raise Exception("TwoSIDES doesn't support in_db version.")
            else:
                for corr_ind in corr_inds:
                    corr_inds_reports[corr_ind] = ind2report[corr_ind]
            
            corr_drugs_reports = defaultdict(set)
            if drug2report is None:
                raise Exception("TwoSIDES doesn't support in_db version.")
            else:
                for corr_drug in corr_drugs:
                    corr_drugs_reports[corr_drug] = drug2report.get(corr_drug, set())
            
            reports = defaultdict(set)
            for indication in corr_inds:
                for reportid in corr_inds_reports[indication]:
                    reports[reportid].add(indication)
            for corr_drug in corr_drugs:
                for reprotid in corr_drugs_reports[corr_drug]:
                    reports[reportid].add(corr_drug)
            
            features = sorted(corr_drugs | corr_inds)
            X = np.zeros(shape=(len(reports), len(features)))
            y = np.zeros(shape=(len(reports),))
            sorted_reportids = sorted(reports.keys())
            for i, reportid in enumerate(sorted_reportids):
                for f in reports[reportid]:
                    j = features.index(f)
                    X[i,j] = 1
                
                if reportid in pair_report_ids:
                    y[i] = 1
            
            # print(X.shape, X.sum(), X.sum()/(X.shape[0]*X.shape[1]))
            # print(y.shape, y.sum())

            n_samples = X.shape[0]
            n_positives = int(y.sum())
            n_negatives = n_samples - n_positives

            if n_samples > MAX_SAMPLES:
                # print(f"Downsampling from {n_samples} to {MAX_SAMPLES}...")

                pos_indices = np.where(y == 1)[0]
                neg_indices = np.where(y == 0)[0]

                # We want to keep as many positives as possible without flipping the class ratio
                max_pos_to_keep = min(len(pos_indices), MAX_SAMPLES // 2)
                n_pos_to_keep = min(len(pos_indices), max_pos_to_keep)
                n_neg_to_keep = MAX_SAMPLES - n_pos_to_keep

                # Make sure we don't request more negatives than exist
                n_neg_to_keep = min(MAX_SAMPLES - n_pos_to_keep, len(neg_indices))

                # Adjust positive count again just in case (rare edge case)
                n_pos_to_keep = min(len(pos_indices), MAX_SAMPLES - n_neg_to_keep)

                # Randomly sample negatives
                rng = np.random.default_rng(seed=42)
                sampled_pos_indices = rng.choice(pos_indices, size=n_pos_to_keep, replace=False)
                sampled_neg_indices = rng.choice(neg_indices, size=n_neg_to_keep, replace=False)

                sampled_indices = np.concatenate([sampled_pos_indices, sampled_neg_indices])
                rng.shuffle(sampled_indices)

                X = X[sampled_indices]
                y = y[sampled_indices]
                sorted_reportids = np.array(sorted_reportids)[sampled_indices].tolist()

                # print(f"After downsampling: X.shape={X.shape}, positives={int(y.sum())}, negatives={X.shape[0] - int(y.sum())}")
            
            if y.sum() < MIN_REPORTS:
                # minimum number of reports for a given drug to run the analysis
                continue

            # print('Training PSM model...')
            clf = linear_model.LogisticRegression(max_iter=1000)
            try:
                auroc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
                # print(f" AUROCs: {auroc}")
                clf.fit(X, y)
                probas = clf.predict_proba(X)[:,1]
            except Exception as e:
                print(f'ERROR: Failed for {drug1_name}, {drug2_name} ({drug1}, {drug2}) at fitting the model with exception: {e}. Skipping.')
                logfh.write(f'ERROR: Failed for {drug1_name}, {drug2_name} ({drug1}, {drug2}) at fitting the model with exception: {e}\n')
                continue

            df = pd.DataFrame({
                'report_id': sorted_reportids,
                'treatment': y,
                'propensity_score': probas
            })
            drug_matched_df = run_multiple_matchings(df, num_replicates=nreps, match_ratio=match_ratio)
            drug_matched_df['drug1'] = drug1
            drug_matched_df['drug2'] = drug2
            drug_matched_df['drug1_id'] = drug1
            drug_matched_df['drug2_id'] = drug2
            drug_matched_df['drug1_name'] = drug1_name
            drug_matched_df['drug2_name'] = drug2_name
            drug_matched_df['auroc'] = np.mean(auroc)

            drug_matched_df.to_csv(cache_path, index=False)
            
            if args.part is None:
                if matched_df is None:
                    matched_df = drug_matched_df
                else:
                    matched_df = pd.concat([matched_df, drug_matched_df], ignore_index=True)

    if args.part is None:
        matched_df.to_csv(os.path.join(results_dir, f'two_hdpsm_nrep{nreps}_mratio{match_ratio}_maxsamp{MAX_SAMPLES}.csv.gz'), index=False)
        # clean up temporary individual files
        # shutil.rmtree(os.path.join(results_dir, 'psm'))
        pass
    
    db.close()
