"""
Compile a dataset for analysis from the pre-processed files.


"""

import os
import sys
import csv
import gzip
import psutil
import hashlib
import argparse
import operator

import numpy as np
import scipy as sp
import pandas as pd
from tqdm.auto import tqdm
from functools import reduce
from datetime import datetime

# requires python ≥ 3.5
from pathlib import Path

sys.path.append('./src')
from faers_processor import save_json, load_json, generate_file_md5

DATA_DIR = './data/faers'

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # return memory used in gigabytes
    return mem_info.rss/1024/1024/1024

def build_dataset(proc_status, endpoint, start_year, end_year):
    """
    Build a dataset from the pre-processed FAERS event files for the given
    endpoint and year range. There are three files that will be concatenated
    to produce the dataset, reports, reactions, and drugs. Reports will be
    compiled first so that any duplicates can be identified and skipped if
    encountered.

    """
    all_subpaths = False
    if start_year is None and end_year is None:
        all_subpaths = True

    event_dir = os.path.join(DATA_DIR, endpoint, "event")

    print(f"Checking pre-processing readiness...")

    if not os.path.exists(event_dir):
        raise Exception(f"ERROR: No directory exists at path {event_dir}, was faers_processor.py run?")

    if not endpoint in proc_status["endpoints"]:
        raise Exception(f"ERROR: No processing status available for {endpoint}, was faers_processor.py run and completed?")

    if not "processing" in proc_status["endpoints"][endpoint]:
        raise Exception(f"ERROR: No processing status available for {endpoint}, was faers_processor.py run and completed?")

    if not proc_status["endpoints"][endpoint]["status"] == 'processed':
        raise Exception(f"ERROR: No processing status available for {endpoint}, was faers_processor.py run and completed?")

    processing_info = proc_status["endpoints"][endpoint]["processing"]

    subpaths_to_compile = list()
    if not all_subpaths:
        for year in range(start_year, end_year+1):
            for quarter in ('q1', 'q2', 'q3', 'q4'):
                subpath = f"{year}{quarter}"
                if not subpath in processing_info:
                    raise Exception(f"ERROR: Data for {endpoint}/event/{subpath} was not found in processor_status.json. Check faers_processor.py run was completed and try again.")
                else:
                    for fk in ("reports", "drugs", "reactions"):
                        if not os.path.exists(processing_info[subpath][fk]):
                            raise Exception(f"ERROR: Expected data file at {processing_info[subpath][fk]} does not exist.")
                subpaths_to_compile.append(subpath)
    else:
        for subpath in processing_info.keys():
            if subpath == 'all_other':
                continue

            for fk in ("reports", "drugs", "reactions"):
                if not os.path.exists(processing_info[subpath][fk]):
                    raise Exception(f"ERROR: Expected data file at {processing_info[subpath][fk]} does not exist.")
            subpaths_to_compile.append(subpath)

    subpaths_to_compile = sorted(subpaths_to_compile)
    print(f"Required data are available and ready to compile into a dataset.")

    ########
    # Set up dataset directory
    # - Determine the name of the directory to store the dataset
    # - Confirm if the directory already exists, if not create it.
    ########

    if all_subpaths:
        dataset_prefix = f"{endpoint}_{min(subpaths_to_compile)}-{max(subpaths_to_compile)}"
    else:
        dataset_prefix = f"{endpoint}_{start_year}-{end_year}"

    dataset_path = os.path.join(DATA_DIR, endpoint, 'datasets', dataset_prefix)
    print(f"Data files will be saved to {dataset_path}")
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    dataset_info_path = os.path.join(dataset_path, 'dataset.json')
    if not os.path.exists(dataset_info_path):
        dataset_info = {
            "created_on": datetime.now().strftime("%Y-%m-%d"),
            "endpoint": endpoint,
            "start_year": start_year,
            "end_year": end_year,
            "all_years": all_subpaths,
            # concatenated versions of each subpath file
            "reports": {"file": "", "md5": ""},
            "drugs": {"file": "", "md5": ""},
            "reactions": {"file": "", "md5": ""},
            # list of non-duplicate report ids
            "nonduplicate_reportids": {"file": "", "md5": ""},
            # sparse matrices for
            # - (ingredient:RxCUI, route) x (ingredient:RxCUI OR medicinalproduct:STRING)
            # - (ingredient:RxCUI, route) x (reaction)
            # - (ingredient:RxCUI, route) by (indication:STRING)
            "ingredients_by_drugs": {"file": "", "md5": ""},
            "ingredients_by_reports": {"file": "", "md5": ""},
            "ingredients_by_indications": {"file": "", "md5": ""}
        }
        save_json(dataset_info_path, dataset_info)
    else:
        dataset_info = load_json(dataset_info_path)

    #######
    # Reports
    # - collect from subpaths and remove duplicates
    #######
    reports_path = os.path.join(dataset_path, f"reports.csv.gz")
    nonduplicates_path = os.path.join(dataset_path, f"nonduplicate_reportids.json")
    nodup_reportids = None

    if os.path.exists(reports_path) and generate_file_md5(reports_path) == dataset_info["reports"]["md5"]:
        # already complete
        # if nonduplicate_reportids is also available then we do no tneed to load reports into memory
        print(f"Reports file is already generated. Will check if non-duplicate report ids are avaialble.")
        if os.path.exists(nonduplicates_path) and generate_file_md5(nonduplicates_path) == dataset_info["nonduplicate_reportids"]["md5"]:
            print(f"Duplicates have already been identified. Loading from file.")
            nodup_reportids = load_json(nonduplicates_path)["reportids"]
        else:
            nodups = pd.read_csv(reports_path, dtype=str)
    else:
        print(f"Beginning by loading reports. Current process memory usage is: {process_memory():.2f}GB")
        reports = None
        for subpath in subpaths_to_compile:
            if reports is None:
                reports = pd.read_csv(processing_info[subpath]["reports"], dtype=str)
            else:
                reports = pd.concat([reports, pd.read_csv(processing_info[subpath]["reports"], dtype=str)])

            # print(reports.tail())
            print(f"  Concating {subpath} with resulting dimensions: {reports.shape}. Memory usage is: {process_memory():.2f}GB")

        nodups = reports.groupby(by="report_key", as_index=False).nth(0)
        nduplicates = reports.shape[0]-nodups.shape[0]
        print(f"Removed {nduplicates} ({100*nduplicates/float(reports.shape[0]):.2f}%) duplicates. Memory usage is: {process_memory():.2f}GB")

        print(f"Writing resulting reports file ({nodups.shape}) to path: {reports_path}")
        nodups.to_csv(reports_path, index=False)

        dataset_info["reports"]["md5"] = generate_file_md5(reports_path)
        dataset_info["reports"]["file"] = reports_path
        save_json(dataset_info_path, dataset_info)

    if nodup_reportids is None:

        # we only need the reportids to include, the reports dataframes can be released
        nodup_reportids = list(set(nodups['safetyreportid']))
        save_json(nonduplicates_path, {"reportids": nodup_reportids})

        dataset_info["nonduplicate_reportids"]["md5"] = generate_file_md5(nonduplicates_path)
        dataset_info["nonduplicate_reportids"]["file"] = nonduplicates_path
        save_json(dataset_info_path, dataset_info)

        # clean up memory
        del(reports)
        del(nodups)

    ######
    # Drugs
    # - concatenate drug data from subpaths
    # - build sparse matrices for:
    #   - ingredient, route x report
    #   - report x ingredient OR medicinalproduct (ingredient if available)
    #   - (ingredient, route) x (ingredient OR medicinalproduct)
    # For simplicity, (ingredient, route) will be referred to as "ingredient"
    # and (ingredient OR medicinalproduct) will be referred to as "drug"
    ######

    drugs_path = os.path.join(dataset_path, f"drugs.csv.gz")

    if os.path.exists(drugs_path) and generate_file_md5(drugs_path) == dataset_info["drugs"]["md5"]:
        # already completed concatenating drugs file
        print(f"Loading drug data from existing file... ", end="")
        drugs_nodup = pd.read_csv(drugs_path, dtype=str)
        print("ok.")
    else:
        print(f"Loading drug data for non-duplicate reports. Current process memory usage is: {process_memory():.2f}GB")
        drugs = None
        for subpath in subpaths_to_compile:
            if drugs is None:
                drugs = pd.read_csv(processing_info[subpath]["drugs"], dtype=str)
            else:
                drugs = pd.concat([drugs, pd.read_csv(processing_info[subpath]["drugs"], dtype=str)])

            print(f"  Concating {subpath} with resulting dimensions: {drugs.shape}. Memory usage is: {process_memory():.2f}GB")

        drugs_nodup = drugs[drugs['safetyreportid'].isin(nodup_reportids)]
        nduplicates = drugs.shape[0]-drugs_nodup.shape[0]
        print(f"Removed {nduplicates} ({100*nduplicates/float(drugs.shape[0]):.2f}%) duplicates. Memory usage is: {process_memory():.2f}GB")

        print(f"Writing resulting drugs file ({drugs_nodup.shape}) to path: {drugs_path}")
        drugs_nodup.to_csv(drugs_path, index=False)

        dataset_info["drugs"]["md5"] = generate_file_md5(drugs_path)
        dataset_info["drugs"]["file"] = drugs_path
        save_json(dataset_info_path, dataset_info)

        del(drugs)

    # Build ingredient by report sparse matrix
    # We will use the CSR sparse matrix format
    # print(drugs_nodup.shape)
    ingredients_by_drugs_path = os.path.join(dataset_path, f"ingredients_by_drugs.npz")

    if os.path.exists(ingredients_by_drugs_path) and generate_file_md5(ingredients_by_drugs_path) == dataset_info["ingredients_by_drugs"]["md5"]:
        print("Matrix ingrdients by drugs is already built, loading from file...", end="")
        ingredients_by_drugs = sp.sparse.load_npz(ingredients_by_drugs_path)
        print("ok")
    else:
        print(f"Building ingredients_route by reports matrix. Memory usage is: {process_memory():.2f}GB")

        # if not adminstration route is provided, we assume oral
        drugs_nodup['drugadministrationroute'] = drugs_nodup['drugadministrationroute'].fillna('048')

        # We only use ingredients that can be mapped to RxCUIs
        # NOTE: This may result in a loss of up 34% of the rows in a test of 2010 data
        # TODO: Figure out how to recover these rows and what the impact of the
        # TODO: loss of rows might mean. If it's random then it shouldn't have an
        # TODO: impact. But if it's not random (more likely) then it can bias the results.

        print(f"  Dropping rows that are not mapped to RxCUI ingredients. Memory usage is: {process_memory():.2f}GB")
        ingredients = drugs_nodup[~drugs_nodup['ingredient_rxcui'].isna()]

        # drop rows where the drug mapped to multiple RxCUI ingredients
        # this results in about a 3.4% loss of rows in a test of 2010 data.
        # print(ingredients.shape)

        print(f"  Dropping rows that map to multiple RxCUI ingredients. Memory usage is: {process_memory():.2f}GB")
        ingredients = ingredients[~ingredients['ingredient_rxcui'].str.contains(', ')]
        # print(ingredients.shape)

        print(f"  Creating a combined ingredient_route column. Memory usage is: {process_memory():.2f}GB")
        ingredients['ingredient_route'] = ingredients[['ingredient_rxcui', 'drugadministrationroute']].apply(
            lambda row: '_'.join(row.values.astype(str)), axis=1
        )

        print(f"  Building ingredient_route => safetyreportid map. Memory usage is: {process_memory():.2f}GB")
        ingredroute2reports = dict()
        for _, row in tqdm(ingredients.iterrows(), total=ingredients.shape[0]):
            key = str(row['ingredient_route'])
            if not ingredroute2reports.get(key):
                ingredroute2reports[key] = set()
            ingredroute2reports[key].add(row['safetyreportid'])

        print(f"  Building medicinalproduct|ingredient => safetyreportid map. Memory usage is: {process_memory():.2f}GB")
        drug2reports = dict()
        report2drugs = dict()
        for _, row in tqdm(drugs_nodup.iterrows(), total=drugs_nodup.shape[0]):
            # if there is no RxCUI or the RxCUI maps to muliple ingredients
            # then we use the medicinalproduct
            if pd.isna(row['ingredient_rxcui']) or row['ingredient_rxcui'].find(', ') == -1:
                drug_key = str(row['medicinalproduct'])
            else:
                drug_key = str(row['ingredient_rxcui'])

            if not drug2reports.get(drug_key):
                drug2reports[drug_key] = set()

            drug2reports[drug_key].add(row['safetyreportid'])

            if not report2drugs.get(row['safetyreportid']):
                report2drugs[row['safetyreportid']] = set()

            report2drugs[row['safetyreportid']].add(drug_key)

        print(f"  Intersecting maps to build ingredient_by_drugs matrix. Memory usage is: {process_memory():.2f}GB")

        sorted_ingredroutes = sorted(ingredroute2reports.keys())
        sorted_drugs = sorted(drug2reports.keys())
        sorted_drugs2index = dict(zip(sorted_drugs, range(len(sorted_drugs))))

        row_ind = list()
        col_ind = list()
        data = list()

        for row_idx, ingredroute in tqdm(enumerate(sorted_ingredroutes), total=len(sorted_ingredroutes)):

            for drug in reduce(operator.or_, [report2drugs[r] for r in ingredroute2reports[ingredroute]]):
            #for col_idx, drug in enumerate(sorted_drugs):
                col_idx = sorted_drugs2index[drug]
                nreports = len(ingredroute2reports[ingredroute] & drug2reports[drug])
                if nreports == 0:
                    continue

                row_ind.append(row_idx)
                col_ind.append(col_idx)
                data.append(nreports/float(len(ingredroute2reports[ingredroute])))

        nrows = len(sorted_ingredroutes)
        ncols = len(sorted_drugs)
        print(f"  Loading sparse matrix ({nrows}, {ncols}) with density: {100*len(data)/(nrows*ncols):.2f}%. Memory usage is: {process_memory():.2f}GB")
        ingredients_by_drugs = sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nrows, ncols))

        print(f"  Saving sparse matrix in npz format...")
        sp.sparse.save_npz(ingredients_by_drugs_path, ingredients_by_drugs)

        print(f"  Saving map files.")
        save_json(os.path.join(dataset_path, "ingredient_routes.json"), {"sorted_ingredroutes": sorted_ingredroutes})
        save_json(os.path.join(dataset_path, "drugs.json"), {"sorted_drugs": sorted_drugs})

        dataset_info["ingredients_by_drugs"]["md5"] = generate_file_md5(ingredients_by_drugs_path)
        dataset_info["ingredients_by_drugs"]["file"] = ingredients_by_drugs_path
        save_json(dataset_info_path, dataset_info)

    del(drugs_nodup)

    return


    ######
    # Reactions
    ######

    print(f"Loading reactions data for non-duplicate reports. Current process memory usage is: {process_memory():.2f}GB")
    reactions = None
    for subpath in subpaths_to_compile:
        if reactions is None:
            reactions = pd.read_csv(processing_info[subpath]["reactions"], dtype=str)
        else:
            reactions = pd.concat([reactions, pd.read_csv(processing_info[subpath]["reactions"], dtype=str)])

        print(f"  Concating {subpath} with resulting dimensions: {reactions.shape}. Memory usage is: {process_memory():.2f}GB")

    reactions_nodup = reactions[reactions['safetyreportid'].isin(nodup_reportids)]
    nduplicates = reactions.shape[0]-reactions_nodup.shape[0]
    print(f"Removed {nduplicates} ({100*nduplicates/float(reactions.shape[0]):.2f}%) duplicates. Memory usage is: {process_memory():.2f}GB")

    reactions_path = os.path.join(DATA_DIR, 'datasets', dataset_prefix, f"reactions.csv.gz")
    print(f"Writing resulting reactions file ({reactions_nodup.shape}) to path: {reactions_path}")
    reactions_nodup.to_csv(reactions_path, index=False)

    del(reactions)
    del(reactions_nodup)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='drug', help='Which endpoints to download For a list of available endpoints see the keys in the results section of the download.json file. Defautl is all endpoints.', type=str, required=True)
    parser.add_argument('--years', type=str, default=None, help="Restrict dataset to a given set of years. Must be provided in the following format: YYYY-YYYY or for an individual year: YYYY")

    args = parser.parse_args()

    proc_status_path = os.path.join(DATA_DIR, 'processer_status.json')
    if not os.path.exists(proc_status_path):
        raise Exception(f"ERROR: No processor_status.json file present. Was faers_processor.py run?")
    else:
        proc_status = load_json(proc_status_path)

    if args.years is None:
        start_year = None
        end_year = None
    elif args.years.find('-') == -1:
        start_year = end_year = int(args.years)
    else:
        start_year, end_year  = map(int, args.years.split('-'))

    build_dataset(proc_status, args.endpoint, start_year, end_year)

if __name__ == '__main__':
    main()
