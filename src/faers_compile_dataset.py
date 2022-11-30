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
from faers_processor import save_json, load_json, generate_file_md5, save_object, load_object

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
            # non-duplicate report ids and their position in the matrices
            "report2index": {"file": "", "md5": ""},
            # sparse matrices for
            # reports x (ingredient:RxCUI OR medicinalproduct:STRING)
            # reports x (ingredient:RxCUI, route)
            # reports x indications
            # reports by reactions
            "reports_by_drugs": {"file": "", "md5": ""},
            "reports_by_ingredients": {"file": "", "md5": ""},
            "reports_by_indications": {"file": "", "md5": ""},
            "reports_by_reactions": {"file": "", "md5": ""},
            # map files
            "ingredient2index": {"file": "", "md5": ""},
            "reaction2index": {"file": "", "md5": ""},
            "drug2index": {"file": "", "md5": ""},
            "indication2index": {"file": "", "md5": ""}


            # NOTE:
            # 11/25 - these are all derivative tables that will depend on
            #         the results of the propensity score matching. After
            #         completing them, I realized that they are not actually
            #         what we are after. Leaving the code in, but commented out
            #         because we might want use it later.
            #
            # - (ingredient:RxCUI, route) x (ingredient:RxCUI OR medicinalproduct:STRING)
            # - (ingredient:RxCUI, route) x (reaction)
            # - (ingredient:RxCUI, route) by (indication:STRING)
            # "ingredients_by_drugs": {"file": "", "md5": ""},
            # "ingredients_by_reactions": {"file": "", "md5": ""},
            # "ingredients_by_indications": {"file": "", "md5": ""}

        }
        save_json(dataset_info_path, dataset_info)
    else:
        dataset_info = load_json(dataset_info_path)

    #######
    # Reports
    # - collect from subpaths and remove duplicates
    #######
    reports_path = os.path.join(dataset_path, f"reports.csv.gz")
    report2index_path = os.path.join(dataset_path, f"report2index.json")
    report2index = None
    reports = None

    if os.path.exists(reports_path) and generate_file_md5(reports_path) == dataset_info["reports"]["md5"]:
        # already complete
        # if nonduplicate_reportids is also available then we do no tneed to load reports into memory
        print(f"Reports file is already generated. Will check if non-duplicate report ids are avaialble.")
        if os.path.exists(report2index_path) and generate_file_md5(report2index_path) == dataset_info["report2index"]["md5"]:
            print(f"Duplicates have already been identified. Loading from file.")
            report2index = load_json(report2index_path)
            nodup_reportids = report2index.keys()
        else:
            nodups = pd.read_csv(reports_path, dtype=str)
    else:
        print(f"Beginning by loading reports. Current process memory usage is: {process_memory():.2f}GB")

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

        print(f"Removing outliers from age and weight columns.")
        nodups = nodups.astype({'patientonsetage': 'float64', 'patientweight': 'float64'})
        #print(nodups.dtypes)
        perc99 = np.percentile(nodups['patientonsetage'].dropna(), 99)
        nodups.iloc[np.where(nodups['patientonsetage'] >= perc99)[0],nodups.columns.get_loc('patientonsetage')] = np.nan

        perc99 = np.percentile(nodups['patientweight'].dropna(), 99)
        nodups.iloc[np.where(nodups['patientweight'] >= perc99)[0],nodups.columns.get_loc('patientweight')] = np.nan

        print(f"Writing resulting reports file ({nodups.shape}) to path: {reports_path}")
        nodups.to_csv(reports_path, index=False)

        dataset_info["reports"]["md5"] = generate_file_md5(reports_path)
        dataset_info["reports"]["file"] = os.path.basename(reports_path)
        save_json(dataset_info_path, dataset_info)

    if report2index is None:

        # we only need the reportids to include, the reports dataframes can be released
        nodup_reportids = sorted(set(nodups['safetyreportid']))
        print(f"Building an index for reports. Sorted {len(nodup_reportids)} non-duplicate report ids.")
        report2index = dict(zip(nodup_reportids, range(len(nodup_reportids))))

        save_json(report2index_path, report2index)

        dataset_info["report2index"]["md5"] = generate_file_md5(report2index_path)
        dataset_info["report2index"]["file"] = os.path.basename(report2index_path)
        save_json(dataset_info_path, dataset_info)

        # clean up memory
        if not reports is None:
            del(reports)
        del(nodups)

    ######
    # Drugs
    # - concatenate drug data from subpaths
    # - build sparse matrices for:
    #   - report x drug (ingredient_rxcui OR medicinalproduct if ingredient_rxcui not available)
    #   - report x (ingredient, route)
    # For simplicity, (ingredient, route) will be referred to as "ingredient"
    # and (ingredient OR medicinalprodduct) will be referred to as "drug"
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
        dataset_info["drugs"]["file"] = os.path.basename(drugs_path)
        save_json(dataset_info_path, dataset_info)

        del(drugs)


    #####
    # reports by ingredients (ingredient_rxcui, route)
    # reports by drugs (ingredient_rxuci if available else medicinalproduct)
    # reports by indications
    reports_by_ingredients_path = os.path.join(dataset_path, f"reports_by_ingredients.npz")
    reports_by_drugs_path = os.path.join(dataset_path, f"reports_by_drugs.npz")
    reports_by_indications_path = os.path.join(dataset_path, f"reports_by_indications.npz")

    if os.path.exists(reports_by_ingredients_path) and generate_file_md5(reports_by_ingredients_path) == dataset_info["reports_by_ingredients"]["md5"]\
        and os.path.exists(reports_by_drugs_path) and generate_file_md5(reports_by_drugs_path) == dataset_info["reports_by_drugs"]["md5"]\
        and os.path.exists(reports_by_indications_path) and generate_file_md5(reports_by_indications_path) == dataset_info["reports_by_indications"]["md5"]:
        print("Matrices for (reports by ingredients), (reports by drugs) and (reports by indications) are already built. ", end="")
        # reports_by_ingredients = sp.sparse.load_npz(reports_by_ingredients_path)
        # reports_by_drugs = sp.sparse.load_npz(reports_by_drugs_path)
        # reports_by_indications = sp.sparse.load_npz(reports_by_indications_path)
        print(f"ok. Memory usage is: {process_memory():.2f}GB")
    else:
        print(f"Building reports x ingredients, x drugs, and x indications matrices. Memory usage is: {process_memory():.2f}GB")

        # if no adminstration route is provided, we assume oral
        drugs_nodup['drugadministrationroute'] = drugs_nodup['drugadministrationroute'].fillna('048')

        ingredient2index = dict()
        drug2index = dict()
        indication2index = dict()

        ingredient_indices = set()
        drug_indices = set()
        indication_indices = set()

        print(f"  Compiling information to build matrices. Memory usage is: {process_memory():.2f}GB")

        for _, row in tqdm(drugs_nodup.iterrows(), total=drugs_nodup.shape[0]):

            if not report2index.get(row['safetyreportid']):
                # duplicate id that we are not using
                continue

            # this will be the row for all three matrices
            row_idx = report2index[row['safetyreportid']]

            # ingredients
            if not pd.isna(row['ingredient_rxcui']):
                ingredient = f"{row['ingredient_rxcui']}_{row['drugadministrationroute']}"
                if ingredient2index.get(ingredient, -1) == -1:
                    ingredient2index[ingredient] = len(ingredient2index)

                ingredient_indices.add( (row_idx, ingredient2index[ingredient]) )

            # drugs
            if not pd.isna(row['ingredient_rxcui']) or not pd.isna(row['medicinalproduct']):
                drug = row['ingredient_rxcui'] if not pd.isna(row['ingredient_rxcui']) else row['medicinalproduct']
                if drug2index.get(drug, -1) == -1:
                    drug2index[drug] = len(drug2index)

                drug_indices.add( (row_idx, drug2index[drug]) )

            # indications
            if not pd.isna(row['drugindication']):
                if indication2index.get(row['drugindication'], -1) == -1:
                    indication2index[row['drugindication']] = len(indication2index)

                indication_indices.add( (row_idx, indication2index[row['drugindication']]) )

        print(f"  Building reports x ingredients matrix. Memory usage is: {process_memory():.2f}GB")
        row_ind, col_ind = zip(*ingredient_indices)
        data = np.repeat(1, len(row_ind))
        nrows = len(report2index)
        ncols = len(ingredient2index)
        reports_by_ingredients = sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nrows, ncols))

        sp.sparse.save_npz(reports_by_ingredients_path, reports_by_ingredients)
        dataset_info["reports_by_ingredients"]["md5"] = generate_file_md5(reports_by_ingredients_path)
        dataset_info["reports_by_ingredients"]["file"] = os.path.basename(reports_by_ingredients_path)
        dataset_info["reports_by_ingredients"]["shape"] = (nrows, ncols)
        dataset_info["reports_by_ingredients"]["density"] = 100*len(row_ind)/float(nrows*ncols)

        save_json(os.path.join(dataset_path, "ingredient2index.json"), ingredient2index)
        dataset_info["ingredient2index"]["md5"] = generate_file_md5(os.path.join(dataset_path, "ingredient2index.json"))
        dataset_info["ingredient2index"]["file"] = "ingredient2index.json"

        save_json(dataset_info_path, dataset_info)


        del(reports_by_ingredients)
        del(ingredient_indices)
        del(ingredient2index)
        print(f"    Saved matrix to {reports_by_ingredients_path} with shape: ({nrows, ncols}) and density: {100*len(row_ind)/float(nrows*ncols):.2f}%. Memory usage is: {process_memory():.2f}GB")

        print(f"  Building reports x drugs matrix. Memory usage is: {process_memory():.2f}GB")
        row_ind, col_ind = zip(*drug_indices)
        data = np.repeat(1, len(row_ind))
        ncols = len(drug2index)
        reports_by_drugs = sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nrows, ncols))

        sp.sparse.save_npz(reports_by_drugs_path, reports_by_drugs)
        dataset_info["reports_by_drugs"]["md5"] = generate_file_md5(reports_by_drugs_path)
        dataset_info["reports_by_drugs"]["file"] = os.path.basename(reports_by_drugs_path)
        dataset_info["reports_by_drugs"]["shape"] = (nrows, ncols)
        dataset_info["reports_by_drugs"]["density"] = 100*len(row_ind)/float(nrows*ncols)

        save_json(os.path.join(dataset_path, "drug2index.json"), drug2index)
        dataset_info["drug2index"]["md5"] = generate_file_md5(os.path.join(dataset_path, "drug2index.json"))
        dataset_info["drug2index"]["file"] = "drug2index.json"

        save_json(dataset_info_path, dataset_info)


        del(reports_by_drugs)
        del(drug_indices)
        del(drug2index)
        print(f"    Saved matrix to {reports_by_drugs_path} with shape: ({nrows, ncols}) and density: {100*len(row_ind)/float(nrows*ncols):.2f}%. Memory usage is: {process_memory():.2f}GB")

        print(f"  Building reports x indication matrix. Memory usage is: {process_memory():.2f}GB")
        row_ind, col_ind = zip(*indication_indices)
        data = np.repeat(1, len(row_ind))
        ncols = len(indication2index)
        reports_by_indications = sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nrows, ncols))

        sp.sparse.save_npz(reports_by_indications_path, reports_by_indications)
        dataset_info["reports_by_indications"]["md5"] = generate_file_md5(reports_by_indications_path)
        dataset_info["reports_by_indications"]["file"] = os.path.basename(reports_by_indications_path)
        dataset_info["reports_by_indications"]["shape"] = (nrows, ncols)
        dataset_info["reports_by_indications"]["density"] = 100*len(row_ind)/float(nrows*ncols)

        save_json(os.path.join(dataset_path, "indication2index.json"), indication2index)
        dataset_info["indication2index"]["md5"] = generate_file_md5(os.path.join(dataset_path, "indication2index.json"))
        dataset_info["indication2index"]["file"] = "indication2index.json"

        save_json(dataset_info_path, dataset_info)


        del(reports_by_indications)
        del(indication_indices)
        del(indication2index)
        print(f"    Saved matrix to {reports_by_indications_path} with shape: ({nrows, ncols}) and density: {100*len(row_ind)/float(nrows*ncols):.2f}%. Memory usage is: {process_memory():.2f}GB")

    ######
    # Reactions
    ######
    reactions_path = os.path.join(dataset_path, f"reactions.csv.gz")

    if os.path.exists(reactions_path) and generate_file_md5(reactions_path) == dataset_info["reactions"]["md5"]:
        print(f"Loading reaction data from existing file... ", end="")
        reactions_nodup = pd.read_csv(reactions_path, dtype=str)
        print(f"ok. Memory usage is: {process_memory():.2f}GB")
    else:
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


        print(f"Writing resulting reactions file ({reactions_nodup.shape}) to path: {reactions_path}")
        reactions_nodup.to_csv(reactions_path, index=False)

        dataset_info["reactions"]["md5"] = generate_file_md5(reactions_path)
        dataset_info["reactions"]["file"] = os.path.basename(reactions_path)
        save_json(dataset_info_path, dataset_info)

        del(reactions)

    #####
    # reports by reactions

    reports_by_reactions_path = os.path.join(dataset_path, f"reports_by_reactions.npz")

    if os.path.exists(reports_by_reactions_path) and generate_file_md5(reports_by_reactions_path) == dataset_info["reports_by_reactions"]["md5"]:
        print("Matrix for (reports by reactions) is already built. ", end="")
        # reports_by_reactions = sp.sparse.load_npz(reports_by_reactions_path)
        print(f"ok. Memory usage is: {process_memory():.2f}GB")
    else:
        reaction2index = dict()
        reaction_indices = set()

        print(f"  Compiling information to build (report by reaction) matrix. Memory usage is: {process_memory():.2f}GB")

        for _, row in tqdm(reactions_nodup.iterrows(), total=reactions_nodup.shape[0]):

            if report2index.get(row['safetyreportid'], -1) == -1:
                # duplicate id that we are not using
                continue

            if int(row['error_code']) == 1:
                # error matching the meddra term to an ID
                continue

            row_idx = report2index[row['safetyreportid']]

            if not reaction2index.get(row['pt_meddra_id']):
                reaction2index[row['pt_meddra_id']] = len(reaction2index)

            col_idx = reaction2index[row['pt_meddra_id']]

            reaction_indices.add( (row_idx, col_idx) )

        print(f"  Building reports x reactions matrix. Memory usage is: {process_memory():.2f}GB")
        row_ind, col_ind = zip(*reaction_indices)
        data = np.repeat(1, len(row_ind))
        nrows = len(report2index)
        ncols = len(reaction2index)
        reports_by_reactions = sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nrows, ncols))

        sp.sparse.save_npz(reports_by_reactions_path, reports_by_reactions)
        dataset_info["reports_by_reactions"]["md5"] = generate_file_md5(reports_by_reactions_path)
        dataset_info["reports_by_reactions"]["file"] = os.path.basename(reports_by_reactions_path)
        dataset_info["reports_by_reactions"]["shape"] = (nrows, ncols)
        dataset_info["reports_by_reactions"]["density"] = 100*len(row_ind)/float(nrows*ncols)

        save_json(os.path.join(dataset_path, "reaction2index.json"), reaction2index)
        dataset_info["reaction2index"]["md5"] = generate_file_md5(os.path.join(dataset_path, "reaction2index.json"))
        dataset_info["reaction2index"]["file"] = "reaction2index.json"

        save_json(dataset_info_path, dataset_info)

        del(reports_by_reactions)
        del(reaction_indices)
        del(reaction2index)
        print(f"    Saved matrix to {reports_by_reactions_path} with shape: ({nrows, ncols}) and density: {100*len(row_ind)/float(nrows*ncols):.2f}%. Memory usage is: {process_memory():.2f}GB")

    return

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
