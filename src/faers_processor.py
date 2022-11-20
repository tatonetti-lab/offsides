"""
Manage the latest available FDA Adverse Event Reporting System downloads.

See for download instructions:
https://open.fda.gov/apis/downloads/

A list of all downloadable files is available at:
https://api.fda.gov/download.json

Browsable data dictionary is availaable at:
https://open.fda.gov/apis/drug/event/searchable-fields/

Computable data dictionary is available as a yaml file at:
https://open.fda.gov/fields/drugevent.yaml
"""

import os
import sys
import csv
import gzip
import json
import time
import yaml
import shutil
import hashlib
import zipfile
import requests
import argparse

from tqdm.auto import tqdm
from datetime import datetime
from urllib.request import urlopen
from collections import defaultdict

from cfuzzyset import cFuzzySet as FuzzySet

# requires python ≥ 3.5
from pathlib import Path

DATA_DIR = './data/faers'
DOWNLOAD_JSON_URL = 'https://api.fda.gov/download.json'
DRUGEVENT_YAML_URL = 'https://open.fda.gov/fields/drugevent.yaml'
RXNORM_P2I_PATH  = './data/rxnorm_product_to_ingredient.csv.gz'
MEDDRA_PT_LLT_PATH = './data/meddra_llt_pt_map.txt'

def load_json(filename):
    fh = open(filename)
    data = json.loads(fh.read())
    fh.close()
    return data

def save_json(filename, data):
    fh = open(filename, 'w')
    fh.write(json.dumps(data, indent=4))
    fh.close()

def download_file(url, local_path):
    # make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))
        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:
            # save the output to a file
            with open(local_path, 'wb') as output:
                shutil.copyfileobj(raw, output)

def download(proc_status, proc_status_path, args):
    # check for download.json and if not last updated today
    # check for an updated remote version, if available download

    download_json_fp = os.path.join(DATA_DIR, 'download.json')
    if not os.path.exists(download_json_fp):
        response = urlopen(DOWNLOAD_JSON_URL)
        download_info = json.loads(response.read())
        save_json(download_json_fp, download_info)
    else:
        download_info = load_json(download_json_fp)

    # TODO: Need to figure out a way to do updates. From the openFDA documentation
    # TODO: all of the files can change at each export. We need to monitor them over time
    # TODO: and see how true that is. Would be nice if we could just download the most recent
    # TODO: files to save time on download and processing.

    # if download_info["meta"]["last_updated"] < datetime.now().strftime("%Y-%m-%d"):
    #     # if not today, redownload and save
    #     response = urlopen(DOWNLOAD_JSON_URL)
    #     download_info = json.loads(response.read())
    #     save_json(download_json_fp, download_info)

    # this is an event analysis, so we only care about data that have adverse events reported
    # at the time of this coding, it was the drug, device, food, and animalandveterinary endpoints
    endpoints_with_events = [ep for ep, v in download_info["results"].items() if "event" in v]

    # Before we download the files for each part, we need to check that we're
    # that the local version of the download file is in sync with the remote version
    # otherwise we may end up with some weird issues with inconsistencies between the files.
    # We do this by looking at the "export_date" for each "event" entry and make
    # sure they match between the local version of the download.json file and the remote version.

    remote_download_info = json.loads(urlopen(DOWNLOAD_JSON_URL).read())

    for ep in endpoints_with_events:
        if not args.endpoint == 'all' and not args.endpoint == ep:
            continue

        if not ep in proc_status["endpoints"]:
            proc_status["endpoints"][ep] = {
                "export_date": download_info["results"][ep]["event"]["export_date"],
                "num_files": len(download_info["results"][ep]["event"]["partitions"]),
                "downloads": {},
                "status": "in_progress"
            }
            save_json(proc_status_path, proc_status)

        print(f"Downloading available files for {ep}...")
        parts_dir = os.path.join(DATA_DIR, ep, "event")

        # consistency check
        remote_export_date = remote_download_info["results"][ep]["event"]["export_date"]
        local_export_date = download_info["results"][ep]["event"]["export_date"]
        if not remote_export_date == local_export_date:
            raise Exception(f"ERROR in endpoint {ep}: Remote export_date {remote_export_date} does not match local export date {local_export_date}.")

        for part_info in download_info["results"][ep]["event"]["partitions"]:

            base_url, filename = os.path.split(part_info["file"])
            sub_dir = os.path.split(base_url)[1]
            if sub_dir == 'event':
                local_filepath = os.path.join(parts_dir, filename)
            else:
                local_filepath = os.path.join(parts_dir, sub_dir, filename)

            Path(os.path.split(local_filepath)[0]).mkdir(parents=True, exist_ok=True)

            # TODO: Do some validation of the files so that we know the downloads are
            # TODO: successful. Unfortunately OpenFDA is not providing md5 hashes right now
            # TODO: so will have to do it some other way.
            if not local_filepath in proc_status["endpoints"][ep]["downloads"]:
                proc_status["endpoints"][ep]["downloads"][local_filepath] = {
                    "downloaded": "no",
                    "md5": ""
                }

            if proc_status["endpoints"][ep]["downloads"][local_filepath]["downloaded"] == "no":
                print(f"  Downloading {part_info['file']} to {local_filepath}")
                download_file(part_info["file"], local_filepath)
                if os.path.exists(local_filepath):
                    md5 = hashlib.md5(open(local_filepath, 'rb').read()).hexdigest()
                    proc_status["endpoints"][ep]["downloads"][local_filepath]["md5"] = md5
                    proc_status["endpoints"][ep]["downloads"][local_filepath]["downloaded"] = "yes"
                    save_json(proc_status_path, proc_status)

    # download the data dictionary
    data_dict_path = os.path.join(DATA_DIR, 'drugevent.json')
    if not os.path.exists(data_dict_path):
        response = urlopen(DRUGEVENT_YAML_URL)
        yaml_string = response.read().decode('utf-8')
        datadict = yaml.safe_load(yaml_string)
        save_json(data_dict_path, datadict)
    else:
        datadict = load_json(data_dict_path)

def check_downloads(proc_status, proc_status_path, args):

    print(f"Checking download status for files as of {proc_status['access_date']} ")

    download_json_fp = os.path.join(DATA_DIR, 'download.json')
    download_info = load_json(download_json_fp)
    endpoints_with_events = [ep for ep, v in download_info["results"].items() if "event" in v]

    # Perform a status check
    all_downloaded = True
    for ep in endpoints_with_events:
        if not args.endpoint == 'all' and not args.endpoint == ep:
            continue

        status_info = proc_status["endpoints"][ep]
        num_downloaded = len([d for d in status_info["downloads"].values() if d['downloaded'] == 'yes'])
        print(f" {ep}: {num_downloaded} of {status_info['num_files']} total files downloaded successfully.")
        if num_downloaded == status_info['num_files']:
            status_info["status"] = "downloaded"
        else:
            all_downloaded = False

        save_json(proc_status_path, proc_status)

    if all_downloaded:
        print("All downloads completed successfully.")
        proc_status["downloaded"] = "yes"
        save_json(proc_status_path, proc_status)

def process(proc_status, proc_status_path, args, single_ep = None, single_subpath = None):

    if not proc_status["downloaded"] == "yes":
        print("WARNING: Processor status shows the files have not all downloaded. Will continue with processing but there may be missing data in the resulting files.")

    if not os.path.exists(RXNORM_P2I_PATH):
        raise Exception(f"ERROR: Necessary file, {RXNORM_P2I_PATH}, is missing. \n\nThe missing file can be downloaded as part of the latest OnSIDES Data Release (see github.com/tatonetti-lab/onsides/releases). The file is included as part of the onsides_VERSION_DATE.tar.gz archive. For example you can download with:\n\nwget https://github.com/tatonetti-lab/onsides/archive/refs/tags/v2.0.0.tar.gz\n\nDownload the archive and extract the 'rxnorm_product_to_ingredient.csv.gz' file to the local ./data directory.")

    # will be set to True automatically in single_ep, single_subpath mode
    local_processing_info = False

    # Load the RxNorm product to ingredient map
    print("Loading the RxNorm dictionary into memory.")
    product2ingredients = defaultdict(set)
    rxnames2rxcuis = dict()
    fh = gzip.open(RXNORM_P2I_PATH, 'rt')
    reader = csv.reader(fh)
    header = next(reader)
    #print(header)

    drug_fuzzyset = FuzzySet()

    for row in reader:
        rowdict = dict(zip(header, row))
        product2ingredients[rowdict['product_rx_cui']].add(rowdict['ingredient_rx_cui'])
        rxnames2rxcuis[rowdict['product_name'].lower()] = rowdict['ingredient_rx_cui']
        drug_fuzzyset.add(rowdict['product_name'].lower())
        rxnames2rxcuis[rowdict['ingredient_name'].lower()] = rowdict['ingredient_rx_cui']
        drug_fuzzyset.add(rowdict['ingredient_name'].lower())

    fh.close()

    # Load the PT to LLT map
    print("Loading the MedDRA dictionary into memory.")
    if not os.path.exists(MEDDRA_PT_LLT_PATH):
        raise Exception(f"ERROR: Necessary file, {MEDDRA_PT_LLT_PATH}, is missing. \n\nThe missing file requires a licence to use MedDRA. It is provided as part of the 'data.zip' archive provided witch each OnSIDES Release (see github.com/tatonetti-lab/onsides/releases). For example you can download the archive with: \n\nwget https://github.com/tatonetti-lab/onsides/releases/download/v2.0.0/data.zip\n\nMove the `meddra_llt_pt_map.txt` file into the OffSIDES local data directory. Please secure the appropriate licensing before use.")

    llt2pt = dict()
    term2pt = dict()
    fh = open(MEDDRA_PT_LLT_PATH)
    reader = csv.reader(fh, delimiter='|')
    header = next(reader)

    for row in reader:
        rowdict = dict(zip(header, row))
        llt2pt[rowdict['llt_concept_id']] = rowdict['pt_concept_id']
        term2pt[rowdict['llt_concept_name'].lower()] = rowdict['pt_concept_id']

    fh.close()

    # Load the data dictionary
    print("Loading the FAERS Data dictionary into memory.")
    datadict = load_json(os.path.join(DATA_DIR, 'drugevent.json'))

    # event report processing template
    template = load_json('./src/faers_report_template.json')

    endpoints = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    for ep in endpoints:

        if single_ep is not None and not ep == single_ep:
            continue

        if not "processing" in proc_status["endpoints"][ep]:
            proc_status["endpoints"][ep]["processing"] = {}

        if single_ep is not None and single_subpath is not None:
            # running in single iteration mode, likely for multiprocessing
            # in this case we use a local version of the processing json
            # to avoid conflicts
            local_processing_info = True
            processing_status_path = os.path.join(DATA_DIR, single_ep, "event", single_subpath, 'local_processing_status.json')
            print(f"Running in single directory mode, will use a local status dictionary saved at {processing_status_path}.")
            if not os.path.exists(processing_status_path):
                processing_info = {}
                save_json(processing_status_path, processing_info)
            else:
                # loaded
                processing_info = load_json(processing_status_path)

        else:
            # this works as a pointer and any changes made in this function
            # will change the proc_status dictionary
            processing_info = proc_status["endpoints"][ep]["processing"]


        print(f"Processing adverse event reports for '{ep}'")

        event_dir = os.path.join(DATA_DIR, ep, "event")

        subpaths = [p for p in os.listdir(event_dir) if not p.startswith('.')]

        # subpaths either have to be all directories or all files
        is_dir_vector = [int(os.path.isdir(os.path.join(event_dir, p))) for p in subpaths]
        if min(is_dir_vector) != max(is_dir_vector):
            dirs = [p for p in subpaths if os.path.isdir(os.path.join(event_dir, p))]
            files = [p for p in subpaths if not os.path.isdir(os.path.join(event_dir, p))]
            raise Exception(f"ERROR: Subpaths of data directory {event_dir} are expected to be only directories or only files. Found both.\n\n  Num Directories: {len(dirs)}\n  Num Files: {len(files)}")

        if max(is_dir_vector) == 0:
            raise Exception(f"ERROR: Implementation for files only directories is not complete.")

        # correct_scores = list()
        # incorrect_scores = list()

        missing_norxcui = 0
        missing_norxcui_matchtoolow = 0
        missing_norxcui_matchle750gt508 = 0

        report_header = [p for p in template["report"]] + [p for p in template["patient"]] + ["report_key"]
        rxn_header = ["safetyreportid", "pt_meddra_id", "error_code"] + [p for p in template["reaction"]]
        drug_header = ["safetyreportid", "ingredient_rxcui", "error_code"] + [p for p in template["drug"]]


        for subpath in subpaths:

            # Code to allow for a single directory to processed
            if not single_subpath is None and not subpath == single_subpath:
                continue

            print(f"  Processing {subpath}...")

            zipjsons = [f for f in os.listdir(os.path.join(event_dir, subpath)) if f.endswith('.json.zip')]

            if not subpath in processing_info:
                processing_info[subpath] = {
                    "status": "in_progress",
                    "processing_time_min": "",
                    "num_files": len(zipjsons),
                    "reports": os.path.join(event_dir, subpath, 'reports.csv.gz'),
                    "drugs": os.path.join(event_dir, subpath, 'drugs.csv.gz'),
                    "reactions": os.path.join(event_dir, subpath, 'reactions.csv.gz'),
                    "log": os.path.join(event_dir, subpath, 'faers_processor.log')
                }
                if local_processing_info:
                    save_json(processing_status_path, processing_info)
                else:
                    save_json(proc_status_path, proc_status)
            else:
                if local_processing_info:
                    # Running in single ep, single subpath mode. It is unusual
                    # that the subpath is already in the json file, which means
                    # that another job may have already started this run. We will
                    # print an error message and quit.
                    raise Exception("ERROR: Might have encountered a job in progress already. Quitting.")

            if processing_info[subpath]["status"] == "complete":
                print(f"    > Processing for these files is already complete.")
                continue

            print(f"    > Found {len(zipjsons)} archived json files.")

            start_time = time.time()

            report_fh = gzip.open(os.path.join(event_dir, subpath, 'reports.csv.gz'), 'wt')
            report_writer = csv.writer(report_fh)
            report_writer.writerow(report_header)

            rxn_fh = gzip.open(os.path.join(event_dir, subpath, 'reactions.csv.gz'), 'wt')
            rxn_writer = csv.writer(rxn_fh)
            rxn_writer.writerow(rxn_header)

            drug_fh = gzip.open(os.path.join(event_dir, subpath, 'drugs.csv.gz'), 'wt')
            drug_writer = csv.writer(drug_fh)
            drug_writer.writerow(drug_header)

            log_fh = open(os.path.join(event_dir, subpath, 'faers_processor.log'), 'w')

            for zjfn in zipjsons:
                zjfp = os.path.join(event_dir, subpath, zjfn)
                with zipfile.ZipFile(zjfp, "r") as z:
                    if len(z.namelist()) > 1:
                        raise Exception(f"ERROR: Zipped json file at {zjfp} has more than one file.")

                    jfn = z.namelist()[0]
                    with z.open(jfn) as f:
                        data = json.loads(f.read().decode('utf-8'))

                        print(f"Processing {jfn} with {len(data['results'])} event reports:")

                        for report in tqdm(data["results"]):
                            report_key_items = list()

                            save_json('./data/drug-event-report.json', report)

                            # extract report level data
                            report_data = list()
                            safetyreportid = report.get("safetyreportid")

                            for property in template["report"]:
                                report_data.append((property, report.get(property, None)))

                            # extract the patient data
                            patient = report["patient"]
                            patient_data = list()
                            for property in template["patient"]:
                                patient_data.append((property, patient.get(property, None)))
                                report_key_items.append(patient.get(property, None))

                            report_header, report_data = zip(*(report_data + patient_data))
                            # print(report_header)

                            # extract the drug information
                            drugs_data = list()
                            for drug in patient["drug"]:

                                drug_data = list()
                                for property in template["drug"]:
                                    drug_data.append((property, drug.get(property, None)))
                                    report_key_items.append(drug.get(property, None))

                                    # openfda has annotated a lot of the reports
                                    # with mapping data to rxcuis, which makes our
                                    # lives much simpler. We extract that if available.
                                    if "openfda" in drug:
                                        openfda = drug["openfda"]
                                        for property in template["openfda"]:
                                            drug_data.append((property, openfda.get(property, None)))

                                drugs_data.append(dict(drug_data))

                            # normalize the drug data
                            ingredients_data = set()
                            for row in drugs_data:
                                # print(row)
                                ingredient_rxcui = None
                                error_code = 0
                                if "rxcui" in row and not row["rxcui"] is None:
                                    ingredient_rxcuis = list(set([', '.join(product2ingredients[prx]) for prx in row["rxcui"]]))
                                    if len(ingredient_rxcuis) > 1:
                                        log_fh.write(f"WARNING: Found more than one ingredient (or ingredient set) for product {row['rxcui']}\n")
                                        error_code = 1
                                    elif len(ingredient_rxcuis) == 0:
                                        log_fh.write(f"WARNING: Found NO ingredients for product {row['rxcui']}\n")
                                        error_code = 2
                                    else:
                                        ingredient_rxcui = ingredient_rxcuis[0]
                                else:
                                    drugname = row["medicinalproduct"].lower()
                                    if drugname in rxnames2rxcuis:
                                        # exact match
                                        ingredient_rxcui = rxnames2rxcuis[drugname]
                                    else:
                                        # find the closest match
                                        if row["medicinalproduct"] in ("NO CONCURRENT MEDICATION", "HORMONE", "MULTI-VITAMIN", "COMMIT", "ALL OTHER THERAPEUTIC PRODUCTS", "[THERAPY UNSPECIFIED]"):
                                            error_code = 3

                                        match = drug_fuzzyset.get(row["medicinalproduct"].lower())[0]

                                        # Some code to see what the score should be
                                        # See notebook for analysis.
                                        # print(row["medicinalproduct"])
                                        # print(match)
                                        # resp = input('Is this match correct? (y/N) ')
                                        # if resp == 'y':
                                        #     ingredient_rxcui = match[1]
                                        #     correct_scores.append(match[0])
                                        # else:
                                        #     incorrect_scores.append(match[0])

                                        # From the above analysis, we can recover about 12% of the missing
                                        # reports with near perfect accuracy by using a threshold >= 0.75.
                                        # We *could* recover 76% of missing reports with 66% precision if
                                        # we use a threshold of >= 0.508.

                                        # For now we are going with the high preicion approach.
                                        # We will count how many are lost this way and then revisit if we
                                        # need to increase our numbers.
                                        if match[0] >= 0.750:
                                            ingredient_rxcui = match[1]
                                        elif match[0] >= 0.508:
                                            missing_norxcui_matchle750gt508 += 1
                                            missing_norxcui_matchtoolow += 1
                                            error_code = 4
                                        else:
                                            missing_norxcui_matchtoolow += 1
                                            error_code = 5

                                ingredients_data.add( tuple([safetyreportid, ingredient_rxcui, error_code] + [row[p] for p in template["drug"]]) )

                                if ingredient_rxcui is None:
                                    missing_norxcui += 1

                            for row in ingredients_data:
                                # Write out normalized drug data
                                drug_writer.writerow(row)

                            # extract the reactions
                            reactions_data = list()
                            for reaction in patient["reaction"]:
                                reaction_data = list()
                                for property in template["reaction"]:
                                    reaction_data.append((property, reaction.get(property, None)))
                                    report_key_items.append(reaction.get(property, None))

                                reactions_data.append(dict(reaction_data))

                            # normalize reaction data
                            error_code = 0
                            for row in reactions_data:

                                if row['reactionmeddrapt'].lower() in term2pt:
                                    pt_meddra_id = term2pt[row['reactionmeddrapt'].lower()]
                                else:
                                    error_code = 1
                                    log_fh.write(f"WARNING: No meddra term found for {row['reactionmeddrapt']}\n")

                                # Write out normalized reaction data
                                rxn_writer.writerow([safetyreportid, pt_meddra_id, error_code] + [row[p] for p in template["reaction"]])

                            # if (len(correct_scores)+len(incorrect_scores)) > 50:
                            #     print(correct_scores)
                            #     print(incorrect_scores)
                            #     break


                            # completed parsing the report, now we create a key to identify duplicates
                            report_key = hashlib.md5(('-'.join(map(str, report_key_items)).encode())).hexdigest()

                            # Write out report keys for duplicate identification
                            # Write out report data
                            report_writer.writerow(report_data + (report_key,))


                # end "for zjfn in zipjsons"

            processing_info[subpath]["status"] = "complete"
            processing_info[subpath]["processing_time_min"] = (time.time()-start_time)/60.

            if local_processing_info:
                save_json(processing_status_path, processing_info)
            else:
                save_json(proc_status_path, proc_status)

            # end "for subpath in subpaths:"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='all', help='Which endpoints to download For a list of available endpoints see the keys in the results section of the download.json file. Defautl is all endpoints.', type=str, required=False)
    parser.add_argument('--subpath', default=None, help="Used to identify a particular event sub-directory to process. Must be spected with a specific endpoint using the --endpoint flag. Purpose to to enable simple multi-processing.")

    args = parser.parse_args()

    if args.subpath is not None and args.endpoint == 'all':
        raise Exception("ERROR: --subpath cannot be set while --endpoint is set to 'all'")

    if not args.endpoint in ('all', 'device', 'drug', 'food'):
        raise Exception("ERROR: --endpoint must be one of 'all', 'device', 'drug', or 'food'")

    # confirm working data directory is available
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    proc_status_path = os.path.join(DATA_DIR, 'processer_status.json')

    if not os.path.exists(proc_status_path):
        proc_status = {
            "downloaded": "no",
            "processed": "no",
            "endpoints": {},
            "access_date": datetime.now().strftime("%Y-%m-%d")
        }
        save_json(proc_status_path, proc_status)
    else:
        proc_status = load_json(proc_status_path)

    #####
    # Download the openFDA files
    #####

    if proc_status["downloaded"] == "no":
        download(proc_status, proc_status_path, args)

    check_downloads(proc_status, proc_status_path, args)

    #####
    # Process downloaded files
    #  - map products to ingredients
    #  - map adverse event terms to MedDRA identifiers
    #  - map MedDRA LLTs to PTs
    #  - remove duplicates
    #  - build sparse matrix files for:
    #    - report x (ingredient, adminroute)
    #    - report x adverse_reaction
    #  - compile a meta data table for reports
    #####

    process(proc_status, proc_status_path, args, single_ep = args.endpoint, single_subpath=args.subpath)

if __name__ == '__main__':
    main()
