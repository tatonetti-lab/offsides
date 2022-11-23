"""
Compile a dataset for analysis from the pre-processed files.


"""

import os
import sys
import csv
import gzip
import argparse

from tqdm.auto import tqdm
from datetime import datetime

DATA_DIR = './data/faers'

def build_dataset(proc_status, endpoint, start_year, end_year):
    """
    Build a dataset from the pre-processed FAERS event files for the given
    endpoint and year range. There are three files that will be concatenated
    to produce the dataset, reports, reactions, and drugs. Reports will be
    compiled first so that any duplicates can be identified and skipped if
    encountered.

    """

    event_dir = os.path.join(DATA_DIR, endpoint, "event")

    if not os.path.exists(event_dir):
        raise Exception(f"ERROR: No directory exists at path {event_dir}, was faers_processor.py run?")

    if not endpoint in proc_status["endpoints"]:
        raise Exception(f"ERROR: No processing status available for {endpoint}, was faers_processor.py run and completed?")

    if not "processing" in proc_status["endpoints"][endpoint]:
        raise Exception(f"ERROR: No processing status available for {endpoint}, was faers_processor.py run and completed?")

    if not proc_status["endpoints"][endpoint]["status"] == 'processed':
        raise Exception(f"ERROR: No processing status available for {endpoint}, was faers_processor.py run and completed?")

    processing_info = proc_status["endpoints"][endpoint]["processing"]

    for year in range(start_year, end_year+1):
        for quarter in ('q1', 'q2', 'q3', 'q4'):
            subpath = f"{year}{quarter}"
            if not subpath in processing_info:
                raise Exception(f"ERROR: Data for {endpoint}/event/{subpath} was not found in processor_status.json. Check faers_processor.py run was completed and try again.")

    








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='drug', help='Which endpoints to download For a list of available endpoints see the keys in the results section of the download.json file. Defautl is all endpoints.', type=str, required=True)
    parser.add_argument('--years', default=None, help="Restrict dataset to a given set of years. Must be provided in the following format: YYYY-YYYY or for an individual year: YYYY")

    args = parser.parse_args()

    if args.years is None:
        start_year = 2004 # the first year FAERS makes data available for
        end_year = datetime.now().strftime('%Y')
    elif args.years.find('-') == -1:
        start_year = end_year = int(args.years)
    else:
        start_year, end_year  = map(int, args.years.split('-'))

    proc_status_path = os.path.join(DATA_DIR, 'processer_status.json')
    if not os.path.exists(proc_status_path):
        raise Exception(f"ERROR: No processor_status.json file present. Was faers_processor.py run?")
    else:
        proc_status = load_json(proc_status_path)

    build_dataset(proc_status, args.endpoint, start_year, end_year)



if __name__ == '__main__':
    main()
