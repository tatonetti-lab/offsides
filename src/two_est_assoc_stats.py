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
# import pandas as pd
import polars as pl

from collections import defaultdict

from build_confounding_matrices import PostgresDB

MIN_REPORTS = 3


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

    psm_files = [f for f in os.listdir(os.path.join(results_dir)) if f.startswith('two_hdpsm') and f.endswith('csv.gz')]

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
    psm = pl.read_csv(os.path.join(results_dir, psm_file))
    if 'drug1_id' not in psm.columns:
        psm = psm.with_columns(pl.col('drug1').map_elements(_normalize).alias('drug1_id'))
    if 'drug2_id' not in psm.columns:
        psm = psm.with_columns(pl.col('drug2').map_elements(_normalize).alias('drug2_id'))
    if 'drug1_name' not in psm.columns:
        base = 'drug1' if 'drug1' in psm.columns else 'drug1_id'
        psm = psm.with_columns(pl.col(base).alias('drug1_name'))
    if 'drug2_name' not in psm.columns:
        base = 'drug2' if 'drug2' in psm.columns else 'drug2_id'
        psm = psm.with_columns(pl.col(base).alias('drug2_name'))
    # pandas:
    # unique_pairs = psm[['drug1_id', 'drug2_id']].drop_duplicates()
    # polars:
    unique_pairs = psm.select(['drug1_id', 'drug2_id']).unique()

    drug_meta = {}
    for id_col, name_col in [('drug1_id', 'drug1_name'), ('drug2_id', 'drug2_name')]:
        subset = (
            psm.select([pl.col(id_col), pl.col(name_col)])
            .unique()
            .drop_nulls()
            .to_dict(as_series=False)
        )
        for key, value in zip(subset[id_col], subset[name_col]):
            if key is not None and value is not None:
                drug_meta[str(key)] = value

    print(f"OK. Found {len(unique_pairs)} unique pairs of drugs.")

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

    # pandas:
    # for drug1, drug2 in tqdm.tqdm(unique_pairs.itertuples(index=False, name=None), total=len(unique_pairs)):
    # polars:
    for drug1, drug2 in tqdm.tqdm(unique_pairs.iter_rows(named=False), total=len(unique_pairs)):
        #print(drug1, drug2)
        # pandas:
        # this_pair = psm[(psm['drug1']==drug1)&(psm['drug2']==drug2)]
        # polars:
        drug1_name = drug_meta.get(drug1, drug1)
        drug2_name = drug_meta.get(drug2, drug2)
        this_pair = psm.filter((pl.col('drug1_id') == drug1) & (pl.col('drug2_id') == drug2))
        
        # pandas:
        # replicates = this_pair['replicate'].unique()
        # polars:
        replicates = this_pair.select('replicate').unique().to_series().to_list()

        for rep in replicates:
            for sex in ('All', '1', '2'):
                # pandas:
                # this_replicate = this_pair[this_pair['replicate']==rep]
                # polars:
                this_replicate = this_pair.filter(pl.col('replicate') == rep)

                # pandas:
                # treatment_reports = set(this_replicate[this_replicate['treatment']==1]['report_id'].unique())
                # polars:
                treatment_reports = set(
                    this_replicate
                    .filter(pl.col('treatment') == 1)
                    .select('report_id')
                    .unique()
                    .to_series()
                    .to_list()
                )

                # pandas:
                # control_reports = set(this_replicate[this_replicate['treatment']==0]['report_id'].unique())
                # polars:
                control_reports = set(
                    this_replicate
                    .filter(pl.col('treatment') == 0)
                    .select('report_id')
                    .unique()
                    .to_series()
                    .to_list()
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
                        # print(f"ERROR: ZeroDivisionError for {drug1}, {drug2} and {rea}")
                        continue

                    assocs.append([
                        drug1,
                        drug1_name,
                        drug2,
                        drug2_name,
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
    # pandas:
    # df = pd.DataFrame(assocs, columns=['drug1', 'drug2', 'reaction', 'patient_sex', 'replicate', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])
    # polars:
    df = pl.DataFrame(
        assocs,
        schema=[
            'drug1_id',
            'drug1_name',
            'drug2_id',
            'drug2_name',
            'reaction_id',
            'reaction_name',
            'patient_sex',
            'replicate',
            'a',
            'b',
            'c',
            'd',
            'OR',
            'PRR',
            'PHI',
        ],
    )
    df = df.with_columns([
        pl.col('drug1_name').alias('drug1'),
        pl.col('drug2_name').alias('drug2'),
        pl.col('reaction_name').alias('reaction'),
    ])

    ofn = f'./results/{start_year}-{end_year}/{psm_file.split(".")[0]}_pair_reaction_associations.csv'
    print(f"Saving results to file: {ofn}")
    
    # pandas:
    # df.to_csv(ofn, index=False)
    # polars:
    df.write_csv(ofn)
