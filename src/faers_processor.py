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
import json
import shutil
import hashlib
import requests
import argparse

from tqdm.auto import tqdm
from datetime import datetime
from urllib.request import urlopen

# requires python ≥ 3.5
from pathlib import Path

DATA_DIR = './data/faers'
DOWNLOAD_JSON_URL = 'https://api.fda.gov/download.json'

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
        if not args.endpoints == 'all' and not args.endpoints == ep:
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

def check_downloads(proc_status, proc_status_path):

    print(f"Checking download status for files as of {proc_status['access_date']} ")

    download_json_fp = os.path.join(DATA_DIR, 'download.json')
    download_info = load_json(download_json_fp)
    endpoints_with_events = [ep for ep, v in download_info["results"].items() if "event" in v]

    # Perform a status check
    all_downloaded = True
    for ep in endpoints_with_events:
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoints', default='all', help='Which endpoints to download For a list of available endpoints see the keys in the results section of the download.json file. Defautl is all endpoints.', type=str, required=False)
    args = parser.parse_args()

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

    check_downloads(proc_status, proc_status_path)

    #####
    # Process downloaded files
    #  - map products to ingredients
    #  - map adverse event terms to meddra identifiers
    #####

    if not proc_status["downloaded"] == "yes":
        print("WARNING: Processor status shows the files have not all downloaded. Will continue with processing but there may be missing data in the resulting files.")








if __name__ == '__main__':
    main()
