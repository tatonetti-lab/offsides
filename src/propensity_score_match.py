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

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression

sys.path.append('./src')
from faers_processor import save_json, load_json, generate_file_md5, save_object, load_object

# Methods for the basic linear regression model on all available features

def resample_examples(X, y, min_unexposed = 10_000):

    p = max(y.sum(), min_unexposed)/y.shape[0]
    mask = np.where(np.squeeze(np.asarray((y.T + np.random.binomial(1, p, y.shape[0])) > 0)))[0]

    return X[mask,:], y[mask], mask

def filter_features_by_tanimoto(X, y, min_t = 0, max_t = 0.4):
    intersection = (y.multiply(X)).sum(axis=0)
    union = y.sum() + X.sum(axis=0) - intersection
    tanimoto = intersection / union

    mask = np.where(np.squeeze(np.asarray(tanimoto > 0)) & np.squeeze(np.asarray(tanimoto < 0.4)))[0]

    return X[:,mask], mask

# Fits a linear model given the features
def build_model(features, labels, i, tanimoto_filter):

    i_features, i_labels, example_mask = resample_examples(features, labels[:,i])

    if tanimoto_filter:
        i_features, feature_mask = filter_features_by_tanimoto(i_features, i_labels, 0, 0.4)
        feature_masks.append(feature_mask)

    if i_labels.sum() == 0 or (tanimoto_filter and len(feature_mask) == 0):
        return None

    i_labels = i_labels.toarray()

    lr = LinearRegression()
    lr.fit(i_features, i_labels)

    return lr, i_features, i_labels

def build_models(features, labels, max_iters=50, show_progress=True, tanimoto_filter=False):
    models = list()

    if not show_progress:
        iterator = range(labels.shape[1])[:max_iters]
    else:
        iterator = tqdm(range(labels.shape[1])[:max_iters], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for i in iterator:

        lr, i_features, i_labels = build_model(features, labels, i, tanimoto_filter)
        models.append( lr )

    if tanimoto_filter:
        return models, feature_masks

    return models

# Method to resample the unexposed to match exposed using the fitted propensity score model
def resample_using_model(lr, features, labels, nsamples = 100_000, nbins=100, min_unexposed=3, min_exposed=1, unexposed_exposed_ratio=3, use_lists=False):
    if lr is None:
        return {
            "predictions": np.array([]),
            "exposed_sample": np.array([]),
            "unexposed_sample": np.array([]),
            "exposed_mask": np.squeeze((labels==1).A)
        }

    # get trained PSM model and make predictions for all reports
    predictions = lr.predict(features).squeeze()

    # generate a boolean mask to identify exposed reports vs unexposed
    exposed_mask = np.squeeze((labels==1).A)
    nexposed = exposed_mask.sum()
    #print(nexposed)

    # identify the indexes that correspond to each group
    exposed_indexes = np.where(exposed_mask)[0]
    unexposed_indexes = np.where(~exposed_mask)[0]

    # build a kernel based on the exposed data (we want to match this distr.)
    try:
        kernel = sp.stats.gaussian_kde(predictions[exposed_indexes])
    except numpy.linalg.LinAlgError:
        return {
            "predictions": np.array([]),
            "exposed_sample": np.array([]),
            "unexposed_sample": np.array([]),
            "exposed_mask": np.squeeze((labels==1).A)
        }

    counts, bins = np.histogram(kernel.resample(nsamples), bins=nbins)

    # perform the resampling
    unexposed_sample = np.array([], dtype=int)
    exposed_sample = np.array([], dtype=int)

    for i, lower_bound in enumerate(bins[:-1]):
        upper_bound = bins[i+1]
        in_bin = (lower_bound <= predictions) & (predictions < upper_bound)
        exposed_in_bin = in_bin & exposed_mask
        unexposed_in_bin = in_bin & ~exposed_mask

        if unexposed_in_bin.sum() < min_unexposed or exposed_in_bin.sum() < min_exposed:
            # too few samples for this bin, skipping
            continue
        else:
            #ntosample = int(np.round(unexposed_exposed_ratio*nexposed*counts[i]/nsamples))
            ntosample = int(unexposed_exposed_ratio*exposed_in_bin.sum())
            sampled = np.random.choice(np.where(unexposed_in_bin)[0], ntosample, replace=True)
            unexposed_sample = np.hstack([unexposed_sample, sampled])
            exposed_sample = np.hstack([exposed_sample, np.where(exposed_in_bin)[0]])

    if use_lists:
        predictions = predictions.tolist()
        exposed_sample = exposed_sample.tolist()
        unexposed_sample = unexposed_sample.tolist()
        exposed_mask = exposed_mask.tolist()

    return {
        "predictions": predictions,
        "exposed_sample": exposed_sample,
        "unexposed_sample": unexposed_sample,
        "exposed_mask": exposed_mask
    }

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

def build_propensity_models(dataset_info, dataset_path, features, labels, colnames, n=10, k=100):

    # Decompose the binary matrix using SVD/PCA and take the top k singular values
    print(f"Building propensity score models.")

    print(f"Using PCA to perform feature reduction on co-medications and indication matrix.")
    U, S, VT = sp.sparse.linalg.svds(features.astype(float), k=k)
    # Use the singular values to produce the pinciple components
    PCs = (U @ sp.sparse.diags(S))
    # Train linear propensity score models using this new feature space
    print(f"Building propensity score models.")
    models = build_models(PCs, labels, max_iters=labels.shape[1])

    # resample the data to match between exposed and unexposed
    print(f"Performing matching using the fitted models.")
    match_data = dict()
    too_few_samples = set()
    too_few_controls = set()

    for i in tqdm(range(labels.shape[1]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        samples = list()
        for _ in range(n):
            match = resample_using_model(models[i], PCs, labels[:,i], nbins=100, use_lists=True)

            if len(match['exposed_sample']) < 10:
                # too few samples
                too_few_samples.add(i)
                continue
            elif len(set(match['unexposed_sample'])) <= 2*len(match['exposed_sample']):
                # too few unique controls
                too_few_controls.add(i)
                continue

            samples.append( match )

        if len(samples) < n:
            # didn't get enough samples
            continue

        match_data[i] = samples

    psm_match_data_path = os.path.join(dataset_path, 'psm_match_data.pkl')
    print(f"Saving matching data to file: {psm_match_data_path}")
    save_object(psm_match_data_path, match_data)
    dataset_info['match_data'] = {
        'file': psm_match_data_path,
        'md5': generate_file_md5(psm_match_data_path),
        'n': n,
        'k': k,
        'ningredients_pass_qc': len(match_data),
        'num_failed_too_few_samples': len(too_few_samples),
        'num_failed_too_few_controls': len(too_few_controls)
    }
    print(f"Updating dataset.json file to include match data file.")
    save_json(os.path.join(dataset_path, 'dataset.json'), dataset_info)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to the spontaneous reporting system dataset files built with faers_compile_dataset.py', type=str, required=True)
    parser.add_argument('--exposed-min', help='The minimum number of reports for an ingredient (ingredient_rxcui, route) to build a PSM for. Default is 10.', type=int, required=False, default=10)
    parser.add_argument('--feature-method', help="Which method to use to pre-process the features for propensity score matching.", type=str, default='droprare', required=False)
    parser.add_argument('--num-samples', help="The number of sets of controls that will be sampled using the propensity scores.", type=int, required=False, default=100)
    parser.add_argument('-n', help="The number of times to match between cases and controls. Default is 10.", type=int, default=10, required=False)
    parser.add_argument('-k', help="The number of singular values to use to generate the principle components. Default is 100.", type=int, default=100, required=False)
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
        if not os.path.exists(os.path.join(args.dataset, dataset_info[fn]['file'])):
            raise Exception(f"ERROR: Required dataset file: {dataset_info[fn]['file']} was not found in dataset: {args.dataset}")
    print("ok")

    # pre-process the feature matrix
    print(f"Pre-processing and getting the feature matrix.")
    features = get_features(dataset_info, args.feature_method, args.dataset, args.droprare_min)

    # build psms
    print(f"Building propensity score models.")
    labels, colnames = get_labels(dataset_info, args.dataset, args.exposed_min)

    build_propensity_models(dataset_info, args.dataset, features, labels, colnames, n=args.n, k=args.k)



if __name__ == '__main__':
    main()
