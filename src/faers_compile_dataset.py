"""
Compile a dataset for analysis from the pre-processed files.


"""

import os
import sys
import csv
import gzip
import psutil
import argparse

import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

sys.path.append('./src')
from faers_processor import save_json, load_json

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
            for fk in ("reports", "drugs", "reactions"):
                if not os.path.exists(processing_info[subpath][fk]):
                    raise Exception(f"ERROR: Expected data file at {processing_info[subpath][fk]} does not exist.")
            subpaths_to_compile.append(subpath)

    print(f"Data are available and ready to compile into a dataset.")

    # First, we load the reports and remove duplicates
    print(f"Beginning by loading reports. Current process memory usage is: {process_memory():.2f}GB")
    reports = None
    for subpath in subpaths_to_compile:
        if reports is None:
            reports = pd.read_csv(processing_info[subpath]["reports"])
        else:
            reports = pd.concat([reports, pd.read_csv(processing_info[subpath]["reports"])])

        # print(reports.tail())
        print(f"  Concating {subpath} with resulting dimensions: {reports.shape}. Memory usage is: {process_memory():.2f}GB")

    nodups = reports.groupby(by="report_key", as_index=False).nth(0)
    # print(nodups.head())
    # print(nodups.shape)
    nduplicates = reports.shape[0]-nodups.shape[0]
    print(f"Removed {nduplicates} ({100*nduplicates/float(reports.shape[0]):.2f}%) duplicates. Memory usage is: {process_memory():.2f}GB")

    if not os.path.exists(os.path.join(DATA_DIR, 'datasets')):
        os.mkdir(os.path.join(DATA_DIR, 'datasets'))

    dataset_prefix = f"{endpoint}_{start_year}-{end_year}"
    if not os.path.exists(os.path.join(DATA_DIR, 'datasets', dataset_prefix)):
        os.mkdir(os.path.join(DATA_DIR, 'datasets', dataset_prefix))

    reports_path = os.path.join(DATA_DIR, 'datasets', dataset_prefix, f"{dataset_prefix}_reports.csv.gz")
    print(f"Writing resulting reports file ({nodups.shape}) to path: {reports_path}")
    nodups.to_csv(reports_path, index=False)

    # we only need the reportids to include, the reports dataframes can be released
    nodup_reportids = set(nodups['safetyreportid'])
    del(reports)
    del(nodups)

    print(f"Loading drug data for non-duplicate reports. Current process memory usage is: {process_memory():.2f}GB")
    drugs = None
    for subpath in subpaths_to_compile:
        if drugs is None:
            drugs = pd.read_csv(processing_info[subpath]["drugs"])
        else:
            drugs = pd.concat([drugs, pd.read_csv(processing_info[subpath]["drugs"])])

        print(f"  Concating {subpath} with resulting dimensions: {drugs.shape}. Memory usage is: {process_memory():.2f}GB")

    drugs_nodup = drugs[drugs['safetyreportid'].isin(nodup_reportids)]
    nduplicates = drugs.shape[0]-drugs_nodup.shape[0]
    print(f"Removed {nduplicates} ({100*nduplicates/float(drugs.shape[0]):.2f}%) duplicates. Memory usage is: {process_memory():.2f}GB")

    drugs_path = os.path.join(DATA_DIR, 'datasets', dataset_prefix, f"{dataset_prefix}_drugs.csv.gz")
    print(f"Writing resulting drugs file ({drugs_nodup.shape}) to path: {drugs_path}")
    drugs_nodup.to_csv(drugs_path, index=False)

    del(drugs)
    del(drugs_nodup)

    print(f"Loading reactions data for non-duplicate reports. Current process memory usage is: {process_memory():.2f}GB")
    reactions = None
    for subpath in subpaths_to_compile:
        if reactions is None:
            reactions = pd.read_csv(processing_info[subpath]["reactions"])
        else:
            reactions = pd.concat([reactions, pd.read_csv(processing_info[subpath]["reactions"])])

        print(f"  Concating {subpath} with resulting dimensions: {reactions.shape}. Memory usage is: {process_memory():.2f}GB")

    reactions_nodup = reactions[reactions['safetyreportid'].isin(nodup_reportids)]
    nduplicates = reactions.shape[0]-reactions_nodup.shape[0]
    print(f"Removed {nduplicates} ({100*nduplicates/float(reactions.shape[0]):.2f}%) duplicates. Memory usage is: {process_memory():.2f}GB")

    reactions_path = os.path.join(DATA_DIR, 'datasets', dataset_prefix, f"{dataset_prefix}_reactions.csv.gz")
    print(f"Writing resulting reactions file ({reactions_nodup.shape}) to path: {reactions_path}")
    reactions_nodup.to_csv(reactions_path, index=False)

    del(reactions)
    del(reactions_nodup)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='drug', help='Which endpoints to download For a list of available endpoints see the keys in the results section of the download.json file. Defautl is all endpoints.', type=str, required=True)
    parser.add_argument('--years', type=int, default=None, help="Restrict dataset to a given set of years. Must be provided in the following format: YYYY-YYYY or for an individual year: YYYY")

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
