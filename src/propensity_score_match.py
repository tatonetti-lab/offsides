"""
The first step of the SCRUB Method (Tatonetti, et al. Sci Trans Med 2012)
is to use propensity score matching to control for potential confounding
biases between exposed reports and unexposed reports.

@author Nicholas Tatonetti

"""

import os
import sys
import csv
import argparse

import numpy as np
import scipy as sp
import pandas as pd

sys.path.append('./src')
from faers_processor import save_json, load_json, generate_file_md5, save_object, load_object

def droprare(dataset_path, dataset_info, min_reports):

    print(f"  Loading feature matrices from disk.")
    reports_by_drugs = sp.sparse.load_npz(os.path.join(dataset_path, dataset_info['reports_by_drugs']['file']))
    reports_by_indications = sp.sparse.load_npz(os.path.join(dataset_path, dataset_info['reports_by_indications']['file']))

    print(f"  Concatenating feature matrices into single matrix.")
    features = sp.sparse.hstack([reports_by_drugs, reports_by_indications])

    print(f"  Dropping columns without enough data.")
    original_ncols = features.shape[1]
    gtrmin_mask = np.squeeze(np.asarray(features.sum(axis=0) > min_reports))
    features = features[:,gtrmin_mask]
    print(f"    Reduced columns from {original_ncols} to {features.shape[1]}, a {100*(original_ncols-features.shape[1])/original_ncols:.2f}% reduction.")

    return features

def get_features(dataset_info, feature_method, dataset_path, droprare_min):

    if feature_method == 'droprare':
        return droprare(dataset_path, dataset_info, droprare_min)
    else:
        raise Exception(f"ERROR: Unexpected feature method type provided: {feature_method}")

def get_metadata(dataset_info, dataset_path):

    print(f"  Loading report metadata from disk.")
    reports = pd.read_csv(os.path.join(dataset_path, dataset_info['reports']['file']))
    print(f"    {reports.shape[0]} rows loaded, will resort to match labels and feature matrices.")

    report2index = load_json(os.path.join(dataset_path, dataset_info['report2index']['file']))
    metadata = reports.groupby(by="safetyreportid", as_index=False).nth(0).sort_values(by=['safetyreportid'])
    sorted_reports = np.asarray(list(zip(*sorted([(index,reportid) for reportid, index in report2index.items()])))[1])

    if not (sorted_reports == metadata['safetyreportid'].values).all():
        raise Exception(f"ERROR: Failed to reorder report data.")

    return metadata

def get_labels(dataset_info, dataset_path, exposed_min):

    print(f"  Loading labels matrix from disk.")
    reports_by_ingredients = sp.sparse.load_npz(os.path.join(dataset_path, dataset_info['reports_by_ingredients']['file']))
    ingredient2index = load_json(os.path.join(dataset_path, dataset_info['ingredient2index']['file']))
    ordered_ingredients = list(zip(*sorted([(idx, ing) for ing, idx in ingredient2index.items()])))[1]

    print(f"  Removing examples with too few reports.")
    original_ncols = reports_by_ingredients.shape[1]
    gtrmin_mask = np.squeeze(np.asarray(reports_by_ingredients.sum(axis=0) > exposed_min))
    labels = reports_by_ingredients[:,gtrmin_mask]
    colnames = [ing for i, ing in enumerate(ordered_ingredients) if gtrmin_mask[i]]

    print(f"    Reduced columns from {original_ncols} to {labels.shape[1]}, a {100*(original_ncols-labels.shape[1])/original_ncols:.2f}% reduction.")

    # NOTE: Based on how we want to access the matrix,
    # the CSR sparse matrix format may not be efficient.
    # Come back to this if there are perfomrance issues
    # and perhaps transpose the matrix or convert to CSC.

    return labels, colnames

def build_propensity_models(features, labels, colnames, args):

    print(features.shape)
    print(labels.shape)
    print(len(colnames))

    idx = 0
    print(colnames[idx])
    print(labels[:,idx].sum())

    print(type(features))
    print(type(features.tocsc()))

    print(features[:,0:4].shape)
    b = labels[:,idx].toarray()
    print(b.shape)

    result = sp.sparse.linalg.lsqr(features, b)
    x = result[0]

    prediction = features @ x
    # rmse = np.sqrt(np.mean((b-prediction)**2))
    # error = np.abs(b-prediction)
    print(type(prediction))
    print(prediction.sum())
    print(type(b), b.ravel().shape)
    error = prediction-b.ravel()
    # mean absolute error
    print(np.abs(error).mean())







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to the spontaneous reporting system dataset files built with faers_compile_dataset.py', type=str, required=True)
    parser.add_argument('--exposed-min', help='The minimum number of reports for an ingredient (ingredient_rxcui, route) to build a PSM for. Default is 10.', type=int, required=False, default=10)
    parser.add_argument('--feature-method', help="Which method to use to pre-process the features for propensity score matching.", type=str, default='droprare', required=False)
    parser.add_argument('--num-samples', help="The number of sets of controls that will be sampled using the propensity scores.", type=int, required=False, default=100)

    # arguments specific to sub-methods and/or sub-steps
    parser.add_argument('--droprare-min', help="Used for 'droprare' feature method. The minimum number of reports for a feature to be included in the PSM. Default is 10.", type=int, required=False, default=10)

    args = parser.parse_args()

    print(f"Running propensity_score_match.py with: ")
    print(f"  dataset: {args.dataset}")
    print(f"  exposed-min: {args.exposed_min}")
    print(f"  num-samples: {args.num_samples}")
    print(f"  feature-method: {args.feature_method}")
    if args.feature_method == 'droprare':
        print(f"    droprare-min: {args.droprare_min}")
    print()

    ###########
    # Propensity Score Matching Steps
    #
    # 1. Pre-process the feature matrix (reports by drugs + reports by indications)
    #    using the pre-processing method.
    #    Methods Available:
    #      droprare - minimal processing, only drop columns that have fewer than args.droprare_min reports
    #      factors - perform sparse matrix factorization to build latent space, use latent space as features
    #      autoencoder - use an autoencoder to build latent space, use latent space as features
    #
    # 2. Fit a PSM for each ingredient=(ingredient_rxcui, route) and calcualte internal quality
    #    statistics (e.g. AUROC) and external evaluation statistics (e.g. correction demographic differences).
    #
    # 3. Apply PSM for each ingredient and choose reports. We choose the reports to match the
    #    distribution of propensity scores for the exposed reports to the unexposed reports through a biased random
    #    sampling of the unexposed reports. We repeat this sampling args.num_samples number of times. These will be used
    #    to generate a distribution of association statistics in downstream analysis.
    ############

    # Check the dataset
    print(f"Checking dataset...", end="")
    if not os.path.exists(args.dataset):
        raise Exception(f"ERROR: No directory found at provided dataset path: {args.dataset}")

    if not os.path.exists(os.path.join(args.dataset, 'dataset.json')):
        raise Exception(f"ERROR: No {args.dataset}/dataset.json found in provided dataset directory. Was faers_compile_dataset.py completed?")

    dataset_info = load_json(os.path.join(args.dataset, 'dataset.json'))

    required_dataset_files = ("reports_by_ingredients", "reports_by_drugs", "reports_by_indications")
    for fn in required_dataset_files:
        if not os.path.exists(dataset_info[fn]['file']):
            raise Exception(f"ERROR: Required dataset file: {fn} was not found in dataset: {args.dataset}")
    print("ok")

    # pre-process the feature matrix
    print(f"Pre-processing and getting the feature matrix.")
    features = get_features(dataset_info, args.feature_method, args.dataset, args.droprare_min)

    # build psms
    print(f"Building propensity score models.")
    labels, colnames = get_labels(dataset_info, args.exposed_min)

    build_propensity_models(features, labels, colnames, args)



if __name__ == '__main__':
    main()
