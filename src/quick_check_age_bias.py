import pandas as pd
import numpy as np

# Paths
PSM_FILE = '~/offsides/results/2024-2024/hdpsm_nrep5_mratio5_maxsamp25000.csv'
AGE_FILE = '~/offsides/results/2024-2024/age_drug_long.csv'

# Load files
psm = pd.read_csv(PSM_FILE)
age_df = pd.read_csv(AGE_FILE)

# Filter for one drug
drug_id = 1000126
psm_drug = psm[psm['drug_id'] == drug_id]
age_drug = age_df[age_df['drug_id'] == drug_id]

# Sets of report_ids
psm_reports = set(psm_drug['report_id'])
age_reports = set(age_drug['report_id'])

# Exposed reports
treated_reports = set(psm_drug[psm_drug['treatment'] == 1]['report_id'])
treated_ages = age_drug[age_drug['report_id'].isin(treated_reports)]['age'].values

# Uncorrected: all non-exposed age reports
all_non_exposed_reports = age_reports - treated_reports
all_non_exposed_ages = age_drug[age_drug['report_id'].isin(all_non_exposed_reports)]['age'].values

# Compute mean difference
mean_diff_uncorrected = np.mean(treated_ages) - np.mean(all_non_exposed_ages)
print(f"Drug {drug_id} mean_diff_uncorrected: {mean_diff_uncorrected}")
