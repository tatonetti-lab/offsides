import pandas as pd
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

# --- Load age per report ---
age_df = pd.read_csv('./results/2024-2024/age_drug_long.csv')  # columns: 'report_id','age','drug_id'
age_df['report_id'] = age_df['report_id'].astype(str)
report2age = dict(zip(age_df['report_id'], age_df['age']))

# Mapping: report_id -> age
report2age = dict(zip(age_df['report_id'].astype(str), age_df['age']))

# Load PSM matched reports
psm = pd.read_csv('./results/2024-2024/hdpsm_nrep5_mratio5_maxsamp25000.csv')
psm['report_id'] = psm['report_id'].astype(str)
if 'drug_name_x' in psm.columns:
    psm = psm.rename(columns={'drug_name_x': 'drug_name'})

drugs = psm['drug_id'].unique()

# Prepare output
age_bias_results = []

# --- Loop over drugs and replicates ---
for drug in tqdm(drugs, desc='Drugs'):
    this_drug = psm[psm['drug_id'] == drug]
    drug_name = this_drug['drug_name'].iloc[0]

    replicates = this_drug['replicate'].unique()
    for rep in replicates:
        this_rep = this_drug[this_drug['replicate'] == rep]

        # Exposed reports
        treated_reports = set(this_rep[this_rep['treatment'] == 1]['report_id'])
        treated_ages = [report2age[r] for r in treated_reports if r in report2age]

        # Corrected control reports
        control_reports = set(this_rep[this_rep['treatment'] == 0]['report_id'])
        control_ages = [report2age[r] for r in control_reports if r in report2age]

        # Uncorrected: all non-exposed reports from age_drug_long
        all_reports_with_age = set(age_df['report_id'])
        all_non_exposed_reports = all_reports_with_age - treated_reports
        all_non_exposed_ages = [report2age[r] for r in all_non_exposed_reports if r in report2age]

        if len(treated_ages) == 0:
            continue

        mean_diff_uncorrected = np.mean(treated_ages) - np.mean(all_non_exposed_ages) if all_non_exposed_ages else np.nan
        mean_diff_corrected = np.mean(treated_ages) - np.mean(control_ages) if control_ages else np.nan

        age_bias_results.append({
            'drug_id': drug,
            'drug_name': drug_name,
            'replicate': rep,
            'mean_diff_uncorrected': mean_diff_uncorrected,
            'mean_diff_corrected': mean_diff_corrected
        })
        print(drug_name, mean_diff_uncorrected, mean_diff_corrected)

# --- Save CSV for plotting ---
out_df = pd.DataFrame(age_bias_results)
os.makedirs('./results/2024-2024', exist_ok=True)
out_file = './results/2024-2024/age_bias_by_drug.csv'
out_df.to_csv(out_file, index=False)
print(f"Saved age bias CSV: {out_file}")
