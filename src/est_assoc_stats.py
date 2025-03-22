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
import numpy as np
import pandas as pd

from collections import defaultdict

from build_confounding_matrices import PostgresDB

MIN_REPORTS = 5

if __name__ == "__main__":

    start_year = 2004
    end_year = start_year
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

    assocs = list()
    
    for drug in tqdm.tqdm(drugs):
        this_drug = psm[psm['drug']==drug]
        replicates = this_drug['replicate'].unique()
        #print(f"Found {len(replicates)} replicates of the PSM matching for {drug}")
        
        for rep in replicates:
            
            #print(f" Estimating associations statistics for {drug} replicate {rep+1} of {len(replicates)}.")
            this_replicate = this_drug[this_drug['replicate']==rep]
            treatment_reports = set(this_replicate[this_replicate['treatment']==1]['report_id'].unique())
            control_reports = set(this_replicate[this_replicate['treatment']==0]['report_id'].unique())

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

                assocs.append([drug, rea, rep, a, b, c, d, OR, PRR, PHI])

    df = pd.DataFrame(assocs, columns=['drug', 'reaction', 'replicate', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
    ofn = f'./results/{start_year}-{end_year}/{psm_file.split(".")[0]}_drug_reaction_associations.csv'
    print(f"Saving results to file: {ofn}")
    df.to_csv(ofn, index=False)


            
            
            





