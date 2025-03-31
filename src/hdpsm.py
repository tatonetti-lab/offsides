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

# import gnuplotlib as gp

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

from build_confounding_matrices import PostgresDB

import pandas as pd
import numpy as np

MAX_SAMPLES = 50_000

def stratified_1n_matching(df, propensity_col='propensity_score', treatment_col='treatment', 
                           n_bins=5, match_ratio=1, random_state=42):
    df = df.copy()
    np.random.seed(random_state)

    # Assign strata based on propensity score
    df['stratum'] = pd.qcut(df[propensity_col], q=n_bins, duplicates='drop')

    matched = []

    # Process each stratum
    for stratum, group in df.groupby('stratum'):
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

    ind_drugs = set(ind_drug_df['drug'].unique())
    drug_drugs = set(drug_drug_df['drug'].unique())
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
            select ingredient_concept_name, drugindication, safetyreport_id
            from drug
            join ingredient on (ingredient.id = drug.id)
            join safetyreport on (safetyreport.id = safetyreport_id)
            where left(receivedate, 4)::int between {start_year} and {end_year}
            """
            results = db.execute_query(query)
            for drug, ind, reportid in results:
                drug2report[drug].add(reportid)
                ind2report[ind].add(reportid)
            
            with open(pkl_fp, 'wb') as fh:
                pickle.dump((drug2report, ind2report), fh)
            
    elif compute_strategy == 'in_db':
        pass
    else:
        raise Exception(f"Unexpected compute strategy provided: {compute_strategy}. Expected 'in_mem' or 'in_db'")

    matched_df = None
    os.makedirs(os.path.join(results_dir, 'psm'), exist_ok=True)
    logfh = open(f"logs/psm_{start_year}-{end_year}_{time.time()}.log", 'w')

    if not args.part is None:
        start_pos = int((float(args.part-1)/float(args.total_parts))*len(common_drugs))
        stop_pos = int((float(args.part)/float(args.total_parts))*len(common_drugs))
        print(f"Working on part {args.part} of {args.total_parts}. Will execute drugs in index range: ({start_pos}, {stop_pos}]")

    for drugidx, drug in tqdm.tqdm(enumerate(common_drugs), total=len(common_drugs)):

        if not (args.part is None) and not (start_pos <= drugidx < stop_pos):
            continue
        
        print(f"Working on propensity score matching for {drug} ({drugidx+1} of {len(common_drugs)})")
        if os.path.exists(os.path.join(results_dir, 'psm', f"{drugidx}_{drug}.csv.gz")):
            print(' > Found preexisting run. Will load from there.')
            if not args.part is None:
                continue
            drug_matched_df = pd.read_csv(os.path.join(results_dir, 'psm', f"{drugidx}_{drug}.csv.gz"))
            if matched_df is None:
                matched_df = drug_matched_df
            else:
                matched_df = pd.concat([matched_df, drug_matched_df])
            continue
        
        # print(ind_drug_df.shape)
        # print(ind_drug_df[(ind_drug_df['drug'] == drug) & (ind_drug_df['PHI'] > 0)].shape)
        corr_inds = set(ind_drug_df[(ind_drug_df['drug'] == drug) & (ind_drug_df['PHI'] > 0)]['indication'].unique())
        corr_drugs = set(drug_drug_df[(drug_drug_df['drug'] == drug) & (drug_drug_df['PHI'] > 0)]['conf_drug'].unique())
        # print(corr_inds)

        if drug2report is None:
            query = f"""
            select safetyreport_id
            from drug
            join ingredient on (ingredient.id = drug.id)
            join safetyreport on (safetyreport.id = safetyreport_id)
            where left(receivedate, 4)::int between {start_year} and {end_year}
            and ingredient_concept_name = '{drug}'
            """
            result = db.execute_query(query)
            drug_report_ids = set([row[0] for row in result])
        else:
            drug_report_ids = drug2report[drug]
        
        # print(drug_report_ids)

        corr_inds_reports = defaultdict(set)
        if ind2report is None:
            escaped_corr_inds = [s.replace("'", "''") for s in corr_inds]
            query = f"""
            select drugindication, safetyreport_id
            from drug
            join ingredient on (ingredient.id = drug.id)
            join safetyreport on (safetyreport.id = safetyreport_id)
            where left(receivedate, 4)::int between {start_year} and {end_year}
            and drugindication in ('{"', '".join(escaped_corr_inds)}');
            """
            result = db.execute_query(query)
            for indication, reportid in result:
                corr_inds_reports[indication].add(reportid)
        else:
            for corr_ind in corr_inds:
                corr_inds_reports[corr_ind] = ind2report[corr_ind]
        
        corr_drugs_reports = defaultdict(set)
        if drug2report is None:
            query = f"""
            select ingredient_concept_name, safetyreport_id
            from drug
            join ingredient on (ingredient.id = drug.id)
            join safetyreport on (safetyreport.id = safetyreport_id)
            where left(receivedate, 4)::int between {start_year} and {end_year}
            and ingredient_concept_name in ('{"', '".join(corr_drugs)}');
            """
            result = db.execute_query(query)
            for corr_drug, reportid in result:
                corr_drugs_reports[corr_drug].add(reportid)
        else:
            for corr_drug in corr_drugs:
                corr_drugs_reports[corr_drug] = drug2report[corr_drug]
        
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
            
            if reportid in drug_report_ids:
                y[i] = 1
        
        print(X.shape, X.sum(), X.sum()/(X.shape[0]*X.shape[1]))
        print(y.shape, y.sum())

        n_samples = X.shape[0]
        n_positives = int(y.sum())
        n_negatives = n_samples - n_positives

        if n_samples > MAX_SAMPLES:
            print(f"Downsampling from {n_samples} to {MAX_SAMPLES}...")

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

            print(f"After downsampling: X.shape={X.shape}, positives={int(y.sum())}, negatives={X.shape[0] - int(y.sum())}")
        
        if y.sum() < 5:
            # minimum number of reports for a given drug to run the analysis
            continue

        print('Training PSM model...')
        clf = linear_model.LogisticRegression(max_iter=1000)
        try:

            auroc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            print(f" AUROCs: {auroc}")
            clf.fit(X, y)
            probas = clf.predict_proba(X)[:,1]
        except Exception as e:
            print(f'ERROR: Failed for {drug} at fitting the model with exception: {e}. Skipping.')
            logfh.write(f'ERROR: Failed for {drug} at fitting the model with exception: {e}')
            continue

        df = pd.DataFrame({
            'report_id': sorted_reportids,
            'treatment': y,
            'propensity_score': probas
        })
        drug_matched_df = run_multiple_matchings(df, num_replicates=nreps, match_ratio=match_ratio)
        drug_matched_df['drug'] = drug
        drug_matched_df['auroc'] = np.mean(auroc)

        drug_matched_df.to_csv(os.path.join(results_dir, 'psm', f"{drugidx}_{drug}.csv.gz"), index=False)
        
        if args.part is None:
            if matched_df is None:
                matched_df = drug_matched_df
            else:
                matched_df = pd.concat([matched_df, drug_matched_df], ignore_index=True)

    if args.part is None:
        matched_df.to_csv(os.path.join(results_dir, f'hdpsm_nrep{nreps}_mratio{match_ratio}_maxsamp{MAX_SAMPLES}.csv.gz'), index=False)
        # clean up temporary individual files
        # shutil.rmtree(os.path.join(results_dir, 'psm'))
    
    db.close()
