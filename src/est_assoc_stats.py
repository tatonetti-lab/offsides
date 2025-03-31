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
import numpy as np
import pandas as pd

from collections import defaultdict

from build_confounding_matrices import PostgresDB

MIN_REPORTS = 5

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
    drugs = set(psm['drug'].unique())
    print("OK.")

    drug_rea_fn = os.path.join('results', f'{start_year}-{end_year}', 'drug_reaction_associations.csv')
    print(f"Loading original association estimates from file: {drug_rea_fn}", end=' ')
    uncorrected_df = pd.read_csv(drug_rea_fn)
    uncorrected_df.rename(columns={
        'a': 'uncorrected_a',
        'b': 'uncorrected_b',
        'c': 'uncorrected_c',
        'd': 'uncorrected_d',
        'PRR': 'uncorrected_PRR',
        'OR': 'uncorrected_OR',
        'PHI': 'uncorrected_PHI'
    }, inplace=True)
    uncorrected_df['sex'] = 'All'
    uncorrected_df.rename(columns={'sex': 'patient_sex'}, inplace=True)
    #print(uncorrected_df.head())
    print('OK.')

    # load report -> reaction data
    db = PostgresDB()
    print("Loading reaction data...", end=' ')
    query = f"""
    select reactionmeddrapt, safetyreport_id
    from reaction
    join safetyreport on (safetyreport_id = safetyreport.id)
    where LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    """
    results = db.execute_query(query)
    report2reaction = defaultdict(set)
    reaction2report = defaultdict(set)
    reactions = set()
    for rea, reportid in tqdm.tqdm(results):
        report2reaction[reportid].add(rea)
        reaction2report[rea].add(reportid)
        reactions.add(rea)
    print("OK.")

    print("Loading reported sex data...")
    query = f"""
    select patientsex, safetyreport.id
    from safetyreport
    where LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    and patientsex is not null
    and patientsex != '0'
    """
    results = db.execute_query(query)
    sex2report = defaultdict(set)
    for sex, reportid, in tqdm.tqdm(results):
        sex2report[sex].add(reportid)

    assocs = list()
    
    for drug in tqdm.tqdm(drugs):
        this_drug = psm[psm['drug']==drug]
        replicates = this_drug['replicate'].unique()
        #print(f"Found {len(replicates)} replicates of the PSM matching for {drug}")
        
        for rep in replicates:
            
            for sex in ('All', '1', '2'):
                #print(f" Estimating associations statistics for {drug} replicate {rep+1} of {len(replicates)}.")
                this_replicate = this_drug[this_drug['replicate']==rep]
                treatment_reports = set(this_replicate[this_replicate['treatment']==1]['report_id'].unique())
                control_reports = set(this_replicate[this_replicate['treatment']==0]['report_id'].unique())

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

                    assocs.append([drug, rea, sex, rep, a, b, c, d, OR, PRR, PHI])

    df = pd.DataFrame(assocs, columns=['drug', 'reaction', 'patient_sex', 'replicate', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])
    df = pd.merge(df, uncorrected_df, on=['drug', 'reaction', 'patient_sex'], how='left')
    
    os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
    ofn = f'./results/{start_year}-{end_year}/{psm_file.split(".")[0]}_drug_reaction_associations.csv'
    print(f"Saving results to file: {ofn}")
    df.to_csv(ofn, index=False)


            
            
            





