"""
Use PSM matched reports to estimate drug-reaction association statistics. 

Requires:
- basic drug-reaction association matrices to be built (run build_confounding_matrices.py)
- propensity score matching to be completed (run hdpsm.py)

"""

import os
import sys
import gzip
import tqdm
import argparse
import hashlib
import numpy as np
import pandas as pd

from collections import defaultdict

from build_confounding_matrices import PostgresDB

MIN_REPORTS = 20


def _normalize(value):
    return value.strip() if isinstance(value, str) else value


def _reaction_id(value: str) -> str:
    normalized = _normalize(value)
    if not normalized:
        return None
    return hashlib.sha1(normalized.upper().encode('utf-8')).hexdigest()[:16]

def parse_args():
    parser = argparse.ArgumentParser(description="Process a range of years.")
    parser.add_argument('--start_year', type=int, required=True, help='Start year (inclusive)')
    parser.add_argument('--end_year', type=int, required=True, help='End year (inclusive)')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")

    start_year = args.start_year
    end_year = args.end_year
    results_dir = os.path.join('results', f"{start_year}-{end_year}")

    psm_files = [f for f in os.listdir(os.path.join(results_dir)) if f.startswith('hdpsm') and f.endswith('csv.gz')]

    if len(psm_files) == 0:
        raise Exception(f"Did not find the propensity score matching files at {results_dir}. Was hdpsm.py run?")
    elif len(psm_files) > 1:
        print(f"Found {len(psm_files)} available. Which would you like to use?")
        for i, psm_file in enumerate(psm_files):
            print(f" [{i+1}] {psm_file}")
        
        choice = input("Please select one of the options: ")
        choice = int(choice)-1
    else:
        choice = 0
    
    psm_file = psm_files[choice]
    print(f"Loading PSM data from file: {psm_file}...", end=' ')
    psm = pd.read_csv(os.path.join(results_dir, psm_file))
    if 'drug_id' not in psm.columns:
        psm['drug_id'] = psm['drug'].apply(_normalize)
    if 'drug_name' not in psm.columns:
        psm['drug_name'] = psm.get('drug', psm['drug_id'])
    drugs = set(psm['drug_id'].unique())
    drug_name_map = (
        psm[['drug_id', 'drug_name']]
        .drop_duplicates()
        .set_index('drug_id')['drug_name']
        .to_dict()
    )
    print("OK.")

    print("Total PSM rows:", len(psm))
    print("Num drugs:", len(drugs))


    drug_rea_fn = os.path.join('results', f'{start_year}-{end_year}', 'drug_reaction_associations.csv')
    print(f"Loading original association estimates from file: {drug_rea_fn}", end=' ')
    uncorrected_df = pd.read_csv(drug_rea_fn)
    if 'drug_id' not in uncorrected_df.columns:
        uncorrected_df['drug_id'] = uncorrected_df['drug'].apply(_normalize)
    if 'drug_name' not in uncorrected_df.columns:
        uncorrected_df['drug_name'] = uncorrected_df.get('drug', uncorrected_df['drug_id'])
    if 'reaction_id' not in uncorrected_df.columns:
        uncorrected_df['reaction_id'] = uncorrected_df['reaction'].apply(_reaction_id)
    if 'reaction_name' not in uncorrected_df.columns:
        uncorrected_df['reaction_name'] = uncorrected_df.get('reaction', uncorrected_df['reaction_id'])
    print("Uncorrected columns:", uncorrected_df.columns.tolist())

    uncorrected_df.rename(columns={
        'a': 'uncorrected_a',
        'b': 'uncorrected_b',
        'c': 'uncorrected_c',
        'd': 'uncorrected_d',
        'PRR': 'uncorrected_PRR',
        'OR': 'uncorrected_OR',
        'PHI': 'uncorrected_PHI'
    }, inplace=True)
    if 'patient_sex' not in uncorrected_df.columns:
        uncorrected_df['patient_sex'] = 'All'
    expected_cols = [
        'drug_id',
        'drug_name',
        'reaction_id',
        'reaction_name',
        'patient_sex',
        'uncorrected_a',
        'uncorrected_b',
        'uncorrected_c',
        'uncorrected_d',
        'uncorrected_PRR',
        'uncorrected_OR',
        'uncorrected_PHI',
    ]
    available_cols = [col for col in expected_cols if col in uncorrected_df.columns]
    uncorrected_df = uncorrected_df[available_cols]
    #print(uncorrected_df.head())
    print('OK.')
    # Prevent duplicate reaction_name columns during merge
    if 'reaction_name' in uncorrected_df.columns:
        uncorrected_df = uncorrected_df.drop(columns=['reaction_name'])

    # load report -> reaction data
    db = PostgresDB()
    print("Loading reaction data...", end=' ')
    query = f"""
    select reactionmeddrapt, re.safetyreportid
    from reactions re
    join reports r on (re.safetyreportid = r.safetyreportid)
    WHERE EXTRACT(YEAR FROM r.receivedate) BETWEEN {start_year} AND {end_year}
    """
    results = db.execute_query(query)
    report2reaction = defaultdict(set)
    reaction2report = defaultdict(set)
    reaction_name_map = {}
    reactions = set()
    for rea, reportid in tqdm.tqdm(results):
        reaction_id = _reaction_id(rea)
        if reaction_id is None:
            continue
        report2reaction[reportid].add(reaction_id)
        reaction2report[reaction_id].add(reportid)
        reactions.add(reaction_id)
        reaction_name_map.setdefault(reaction_id, rea)
    print("OK.")
    sample_db_report = next(iter(next(iter(reaction2report.values()))))
    print("Sample DB safetyreportid:", sample_db_report, type(sample_db_report))

    print("Num reactions:", len(reactions))

    print("Loading reported sex data...")
    query = f"""
    select patientsex, safetyreportid
    from reports
    WHERE EXTRACT(YEAR FROM receivedate) BETWEEN {start_year} AND {end_year}
    and patientsex is not null
    and patientsex != '0'
    """
    results = db.execute_query(query)
    sex2report = defaultdict(set)
    for sex, reportid, in tqdm.tqdm(results):
        sex2report[sex].add(reportid)

    print("Sex report counts:", {k: len(v) for k, v in sex2report.items()})

    assocs = list()

    drug_col = 'drug_id'
    
    for drug in tqdm.tqdm(drugs):
        drug_name = drug_name_map.get(drug, drug)
        this_drug = psm[psm[drug_col] == drug]
        replicates = this_drug['replicate'].unique()
        #print(f"Found {len(replicates)} replicates of the PSM matching for {drug}")
        
        for rep in replicates:
            
            for sex in ('All', '1', '2'):
                #print(f" Estimating associations statistics for {drug} replicate {rep+1} of {len(replicates)}.")
                this_replicate = this_drug[this_drug['replicate']==rep]
                treatment_reports = set(
                    this_replicate[this_replicate['treatment'] == 1]['report_id']
                    .astype(str)
                    .unique()
                )

                control_reports = set(
                    this_replicate[this_replicate['treatment'] == 0]['report_id']
                    .astype(str)
                    .unique()
                )

                if sex != 'All':
                    treatment_reports &= sex2report[sex]
                    control_reports &= sex2report[sex]

                for rea in reactions:
                    a = len(reaction2report[rea] & treatment_reports)
                    if a < MIN_REPORTS:
                        continue
                    
                    b = len(treatment_reports)-a
                    c = len(reaction2report[rea] & control_reports)
                    d = len(control_reports)-c

                    try:
                        OR = (a/b)/(c/d)
                        PRR = (a/(a+b))/(c/(c+d))
                        PHI = (a*d-b*c)/np.sqrt((a+b)*(c+d)*(b+d)*(a+c))
                    except ZeroDivisionError:
                        #print(f"ERROR: ZeroDivisionError for {drug} and {rea}")
                        continue

                    assocs.append([
                        drug,
                        drug_name,
                        rea,
                        reaction_name_map.get(rea, rea),
                        sex,
                        rep,
                        a,
                        b,
                        c,
                        d,
                        OR,
                        PRR,
                        PHI,
                    ])
    print("Total associations computed:", len(assocs))

    df = pd.DataFrame(assocs, columns=['drug_id', 'drug_name', 'reaction_id', 'reaction_name', 'patient_sex', 'replicate', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])
    # ================= DEBUG BEFORE MERGE =================
    print("\n=== DEBUG: MERGE KEYS ===")

    print("PSM patient_sex values:", df['patient_sex'].unique()[:10])
    print("Uncorrected patient_sex values:", uncorrected_df['patient_sex'].unique()[:10])

    print("PSM drug_id dtype:", df['drug_id'].dtype)
    print("Uncorrected drug_id dtype:", uncorrected_df['drug_id'].dtype)

    print("PSM reaction_id sample:", df['reaction_id'].iloc[0], type(df['reaction_id'].iloc[0]))
    print("Uncorrected reaction_id sample:",
          uncorrected_df['reaction_id'].iloc[0],
          type(uncorrected_df['reaction_id'].iloc[0]))

    print("Number of exact key overlaps:",
          len(
              set(
                  zip(
                      df['drug_id'],
                      df['reaction_id'],
                      df['patient_sex']
                  )
              )
              &
              set(
                  zip(
                      uncorrected_df['drug_id'],
                      uncorrected_df['reaction_id'],
                      uncorrected_df['patient_sex']
                  )
              )
          )
    )
    print("========================================\n")
    # =====================================================

    df = pd.merge(df, uncorrected_df, on=['drug_id', 'reaction_id', 'patient_sex'], how='left')
    print(
        "Non-null uncorrected_a:",
        df['uncorrected_a'].notna().sum(),
        "out of",
        len(df)
    )

    os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
    ofn = f'./results/{start_year}-{end_year}/{psm_file.split(".")[0]}_drug_reaction_associations.csv'
    print(f"Saving results to file: {ofn}")
    df.to_csv(ofn, index=False)
