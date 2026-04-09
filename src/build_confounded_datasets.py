import os
import tqdm
import argparse
import numpy as np
# import pandas as pd
import polars as pd
import matplotlib.pyplot as plt

def load_csv_files(start_year, end_year):
    """
    Loads the indication-reaction, indication-drug, and drug-reaction association CSV files into Pandas DataFrames.

    Parameters:
        start_year (int): The start year of the dataset.
        end_year (int): The end year of the dataset.

    Returns:
        tuple: (ind_rea_df, ind_drug_df, drug_rea_df)
    """
    # Define directory path
    results_dir = f'results/{start_year}-{end_year}'
    
    confounders = {
        'ind': 'indication',
        'drug': 'drug',
        'sex': 'sex',
        'age': 'age'
    }
    file_paths = dict()
    for ccode, cname in confounders.items():
        file_paths[f"{ccode}_rea_df"] = f"{cname}_reaction_associations.csv"
        file_paths[f"{ccode}_drug_df"] = f"{cname}_drug_associations.csv"
    
    dfs = dict()
    for dfkey, filename in file_paths.items():
        file_path = os.path.join(results_dir, filename)
        # Load CSVs into DataFrames
        try:
            df = pd.read_csv(file_path)

            # --- CANONICALIZE drug_id type ---
            if 'drug_id' in df.columns:
                df = df.with_columns(
                    pd.col('drug_id').cast(pd.Utf8)
                )

            dfs[dfkey] = df
            print(f"Loaded {len(df)} rows from {file_path}. Stored at key: {dfkey}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dfs[dfkey] = None

    return dfs

def polars_join_with_suffixes(
    df1,
    df2,
    left_on,
    right_on,
    how="inner",
    suffixes=("_x", "_y"),
):
    # rename overlapping non-key columns
    overlap = set(df1.columns) & set(df2.columns)
    overlap -= {left_on, right_on}

    df1_renamed = df1.rename({c: f"{c}{suffixes[0]}" for c in overlap if c in df1.columns})
    df2_renamed = df2.rename({c: f"{c}{suffixes[1]}" for c in overlap if c in df2.columns})

    return df1_renamed.join(
        df2_renamed,
        left_on=left_on,
        right_on=right_on,
        how=how,
    )

def confounder_drug_reaction_correlation(
    ind_rea_df, ind_drug_df, drug_rea_df, 
    conf_rea_threshold, min_threshold_for_drug, type,conf_col_name, title, num_bins=30,
    conf_rea_col = 'PRR', conf_drug_col = 'PHI', drug_rea_col = 'PRR',
    newfig=True, color='k', label=None
):
    """
    Plots:
    - X-axis: Average PHI (conf_drug_col) in equally sized bins of drugs.
    - Y-axis: Mean (or Proportion of drugs in each bin with statistic (drug_rea_col) > min_threshold_for_drug) for the indication's reactions.

    Parameters:
        ind_rea_df (pd.DataFrame): DataFrame with confounder-reaction associations. Initially implemented with indications, but can be other things.
        ind_drug_df (pd.DataFrame): DataFrame with confounder-drug associations. Initiatlly implemented with indications, but can be other things. 
        drug_rea_df (pd.DataFrame): DataFrame with drug-reaction associations.
        ind_rea_threshold (float): Threshold for identifying strong confounder-reaction links. Note used if type == 'mean'
        min_threshold_for_drug (float): PRR threshold for drugs.
        num_bins (int): Number of bins to group drugs based on PHI.
        type (str): 'proportion' or 'mean' to indicate to plot proportion above threshold or mean.
        conf_col (str): Which reaction association statistic to use in the confounder matrix. 
        drug_col (str): which reaction association statistic to use in the drug matrix.
        title (str): override the title

    Returns:
        data (pd.DataFrame): DataFrame with possibly confounded drug-reaction associations.
    """
    if conf_col_name == "conf_drug":
        df_conf_col = "conf_drug_name"  # or "conf_drug_id" if you prefer
    else:
        df_conf_col = conf_col_name
    # Convert to Python list of strings
    indication_names = ind_drug_df.select(df_conf_col).unique().to_series().cast(str).to_list()
    indication_reactions = ind_rea_df.filter(pd.col(conf_rea_col) > conf_rea_threshold)
    indication_reactions = indication_reactions.filter(
        ~pd.col("reaction_name").is_in(indication_names)
    )
    
    data = None
    import csv
    skipped_indications = []
    ind_drug_df = ind_drug_df.with_columns(pd.col("drug_id").cast(pd.Utf8))
    drug_rea_df = drug_rea_df.with_columns(pd.col("drug_id").cast(pd.Utf8))

    for ind in tqdm.tqdm(indication_reactions[df_conf_col].unique()):
        relevant_reactions = (
            indication_reactions
            .filter(pd.col(df_conf_col) == ind)
            .select("reaction_name")
            .unique()
            .to_series()
            .to_list()
        )
        id = ind_drug_df.filter(
            (ind_drug_df[df_conf_col] == ind) &
            (~ind_drug_df[conf_drug_col].is_null())
        )
        dr = drug_rea_df.filter(drug_rea_df['reaction_name'].is_in(relevant_reactions))

        if id.height == 0 or dr.height==0:
            skipped_indications.append({
                'indication': ind,
                'num_drug_rows': id.height,
                'num_reaction_rows': dr.height,
                'num_relevant_reactions': len(relevant_reactions)
            })
            continue  # skip the join

        # DEBUG: print shapes
        print(f"\nProcessing confounder: {ind}")
        print(f"id shape: {id.shape}, dr shape: {dr.shape}")
        
        # Skip if either df is empty
        if id.height == 0:
            print(f"Skipping {ind}: no confounder-drug rows")
            continue
        if dr.height == 0:
            print(f"Skipping {ind}: no drug-reaction rows")
            continue

        # Proceed with join
        re = polars_join_with_suffixes(
            id,
            dr,
            left_on="drug_id",
            right_on="drug_id",
            suffixes=("_conf_drug", "_drug_rea")
        )[[
            df_conf_col,
            "reaction_name",
            "drug_id",
            f"{conf_drug_col}_conf_drug",
            f"{drug_rea_col}_drug_rea",
        ]]
        if data is None:
            data = re
        else:
            #data = pd.concat([data, re], ignore_index=True)
            data = pd.concat([data, re])

    import csv

    # Save skipped indications to CSV in current folder
    outpath = "skipped_indications.csv"

    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['indication', 'num_drug_rows', 'num_reaction_rows', 'num_relevant_reactions']
        )
        writer.writeheader()
        writer.writerows(skipped_indications)

    print(f"Skipped indications saved to: {outpath}")

    if data is None or len(data) == 0:
        print("No valid drug-reaction pairs found. Exiting function.")
        breakpoint()
        print("Check skipped_indications.csv for details.")
        return None
    #sorted_data = data.sort_values(by=f'{conf_drug_col}_conf_drug')
    sorted_data = data.sort(f'{conf_drug_col}_conf_drug')
    
    bin_size = sorted_data.height // num_bins

    plot_data = list()
    #for b, df in tqdm.tqdm(enumerate(np.array_split(sorted_data, num_bins))):
    for b in range(num_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < num_bins - 1 else sorted_data.height
        df = sorted_data.slice(start, end - start)

        if df.height == 0:
            continue
        colname = f"{drug_rea_col}_drug_rea"
        yprop = (
            df.filter(df[colname] > min_threshold_for_drug).height / df.height
        )
        ymu = df[f'{drug_rea_col}_drug_rea'].mean()
        xmu = df[f'{conf_drug_col}_conf_drug'].mean()

        plot_data.append((xmu, ymu, yprop))

    x, ymu, yprop = zip(*plot_data)
    if type == 'proportion':
        y = yprop
        ylabel = f"Proportion of Drugs with {drug_rea_col} > {min_threshold_for_drug}"
    elif type == 'mean':
        y = ymu
        ylabel = f"Average {drug_rea_col}"
    else:
        raise Exception(f"Unknown type provided: {type}")
    
    if newfig:
        plt.figure(figsize=(5, 4))
    plt.scatter(x,y,color=color, label=label)

    plt.xlabel(f"Average {conf_drug_col} in Bin")
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Drug-{conf_col_name.title()} Correlation vs. Drug-Reaction Proportion (Binned)")

    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()

    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Process a range of years.")
    parser.add_argument('--start_year', type=int, required=True, help='Start year (inclusive)')
    parser.add_argument('--end_year', type=int, required=True, help='End year (inclusive)')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")

    # Load DataFrames
    start_year = args.start_year
    end_year = args.end_year
    min_threshold_for_drug = 200
    dfs = load_csv_files(start_year, end_year)

    results_dir = os.path.join('results', f'{start_year}-{end_year}')
    
    print("Processing confounded by indication...")
    indications = confounder_drug_reaction_correlation(
        dfs['ind_rea_df'],  # DataFrame for confounder-reaction links
        dfs['ind_drug_df'], # DataFrame for confounder-drug links
        dfs['drug_rea_df'], # DataFrame for drug-reaction links
        conf_rea_threshold=10,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=30, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='indication', # name of confounder
        title = f'Confounding by Indication Years {start_year}-{end_year}',
        label='Uncorrected'
    )

    psm_10_5_df = pd.read_csv(os.path.join(results_dir, 'hdpsm_nrep5_mratio5_maxsamp25000_drug_reaction_associations.csv'))
    psm_10_5_df.head()

    rep0 = psm_10_5_df.filter((psm_10_5_df['replicate']==0) & (psm_10_5_df['patient_sex']=='All'))
    rep0 = rep0.with_columns(
        pd.col("drug_id").cast(pd.Utf8)
    )

    _ = confounder_drug_reaction_correlation(
        dfs['ind_rea_df'],  # DataFrame for confounder-reaction links
        dfs['ind_drug_df'], # DataFrame for confounder-drug links
        rep0, # DataFrame for drug-reaction links
        conf_rea_threshold=10,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=30, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='indication', # name of confounder
        title = f'Confounding by Indication Years {start_year}-{end_year}',
        newfig=False, color='red', label='SCRUB'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'indication_confounding.pdf'))

    # indications.sort_values(by='PRR_drug_rea', ascending=False).head(20)
    # indications[indications['PRR_drug_rea'] > min_threshold_for_drug].to_csv(f'../results/{start_year}-{end_year}/indication_confounded_examples.csv')
    # Map drug_id to drug_name using drug_rea_df
    drug_id_to_name = dict(zip(dfs['drug_rea_df']['drug_id'], dfs['drug_rea_df']['drug_name']))
    drug_id_to_name = {str(k): v for k, v in drug_id_to_name.items()}
    indications = indications.with_columns([pd.col("drug_id").str.strip_chars(' "\'').alias("drug_id")])
    indications = indications.with_columns([pd.col("drug_id").map_elements(lambda x: drug_id_to_name.get(x, None), return_dtype=pd.Utf8).alias("drug_name")])


    # Save CSV with drug_name included
    indications.filter(indications['PRR_drug_rea'] > min_threshold_for_drug).write_csv(
        os.path.join(results_dir, 'indication_confounded_examples.csv')
    )

    # conf_drug_rea_df = dfs['drug_rea_df'].copy()
    #conf_drug_rea_df = dfs['drug_rea_df'].clone()
    # conf_drug_rea_df.rename(columns={'drug': 'conf_drug'}, inplace=True)
    #conf_drug_rea_df = conf_drug_rea_df.rename({'drug_id': 'conf_drug'})
    conf_drug_rea_df = dfs['drug_rea_df'].clone().with_columns(
        pd.col("drug_name").alias("conf_drug_name")
    )

    print("Processing confounded by drug...")
    drugs = confounder_drug_reaction_correlation(
        conf_drug_rea_df,  # DataFrame for confounder-reaction links
        dfs['drug_drug_df'], # DataFrame for confounder-drug links
        dfs['drug_rea_df'], # DataFrame for drug-reaction links
        conf_rea_threshold=10,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=30, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='conf_drug', # name of confounder
        title = f'Confounding by Drug Years {start_year}-{end_year}',
        label='Uncorrected'
    )

    _ = confounder_drug_reaction_correlation(
        conf_drug_rea_df,  # DataFrame for confounder-reaction links
        dfs['drug_drug_df'], # DataFrame for confounder-drug links
        rep0, # DataFrame for drug-reaction links
        conf_rea_threshold=10,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=30, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='conf_drug', # name of confounder
        title = f'Confounding by Drug Years {start_year}-{end_year}',
        newfig=False, color='red', label='SCRUB'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'drug_confounding.pdf'))

    # drugs.sort_values(by='PRR_drug_rea', ascending=False).head(20)
    # drugs[drugs['PRR_drug_rea'] > min_threshold_for_drug].to_csv(f'../results/{start_year}-{end_year}/drug_confounded_examples.csv')
    drugs = drugs.with_columns([pd.col("drug_id").str.strip_chars(' "\'').alias("drug_id")])
    drugs = drugs.with_columns([pd.col("drug_id").map_elements(lambda x: drug_id_to_name.get(x, None), return_dtype=pd.Utf8).alias("drug_name")])
    drugs.filter(drugs['PRR_drug_rea'] > min_threshold_for_drug).write_csv(
        os.path.join(results_dir, 'drug_confounded_examples.csv')
    )

    print("Processing confounded by sex...")
    sexes = confounder_drug_reaction_correlation(
        dfs['sex_rea_df'],  # DataFrame for confounder-reaction links
        dfs['sex_drug_df'], # DataFrame for confounder-drug links
        dfs['drug_rea_df'], # DataFrame for drug-reaction links
        conf_rea_threshold=30,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=10, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='sex', # name of confounder
        title = f'Confounding by Sex Years {start_year}-{end_year}',
        label='Uncorrected'
    )

    _ = confounder_drug_reaction_correlation(
        dfs['sex_rea_df'],  # DataFrame for confounder-reaction links
        dfs['sex_drug_df'], # DataFrame for confounder-drug links
        rep0, # DataFrame for drug-reaction links
        conf_rea_threshold=30,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=10, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='sex', # name of confounder
        title = f'Confounding by Sex Years {start_year}-{end_year}',
        newfig=False, color='red', label='SCRUB'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sex_confounding.pdf'))

    # sexes.sort_values(by='PRR_drug_rea', ascending=False).head(20)
    # sexes[sexes['PRR_drug_rea'] > min_threshold_for_drug].to_csv(f'../results/{start_year}-{end_year}/sex_confounded_examples.csv')
    sexes = sexes.with_columns([pd.col("drug_id").str.strip_chars(' "\'').alias("drug_id")])
    sexes = sexes.with_columns([pd.col("drug_id").map_elements(lambda x: drug_id_to_name.get(x, None), return_dtype=pd.Utf8).alias("drug_name")])
    sexes.filter(sexes['PRR_drug_rea'] > min_threshold_for_drug).write_csv(
        os.path.join(results_dir, 'sex_confounded_examples.csv')
    )

    print("Processing confounded by age...")
    ages = confounder_drug_reaction_correlation(
        dfs['age_rea_df'],  # DataFrame for confounder-reaction links
        dfs['age_drug_df'], # DataFrame for confounder-drug links
        dfs['drug_rea_df'], # DataFrame for drug-reaction links
        conf_rea_threshold=10,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=30, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='age', # name of confounder
        title = f'Confounding by Age Years {start_year}-{end_year}',
        label='Uncorrected'
    )

    _ = confounder_drug_reaction_correlation(
        dfs['age_rea_df'],  # DataFrame for confounder-reaction links
        dfs['age_drug_df'], # DataFrame for confounder-drug links
        rep0, # DataFrame for drug-reaction links
        conf_rea_threshold=10,  # Identify strong indication-reaction links
        min_threshold_for_drug=min_threshold_for_drug,  # Threshold for PRR in drugs
        num_bins=30, # Number of bins to group data in plot
        type='mean', # calculate proportion above threshold or mean
        conf_rea_col = 'PRR', # statistic to use for confounder-reaction relationship
        conf_drug_col = 'PHI', # statistic to use for confounder-drug relationship
        drug_rea_col = 'PRR', # statistic to use for drug-reaction relationship
        conf_col_name='age', # name of confounder
        title = f'Confounding by Age Years {start_year}-{end_year}',
        label='SCRUB',
        newfig=False, color='red'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'age_confounding.pdf'))

    # ages.sort_values(by='PRR_drug_rea', ascending=False).head(20)
    # ages[ages['PRR_drug_rea'] > min_threshold_for_drug].to_csv(f'../results/{start_year}-{end_year}/age_confounded_examples.csv')
    ages = ages.with_columns([pd.col("drug_id").str.strip_chars(' "\'').alias("drug_id")])
    ages = ages.with_columns([pd.col("drug_id").map_elements(lambda x: drug_id_to_name.get(x, None), return_dtype=pd.Utf8).alias("drug_name")])
    ages.filter(ages['PRR_drug_rea'] > min_threshold_for_drug).write_csv(
        os.path.join(results_dir, 'age_confounded_examples.csv')
    )
