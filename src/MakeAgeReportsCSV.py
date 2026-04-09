import pandas as pd
from collections import defaultdict
from build_confounding_matrices import PostgresDB
import os

# --- CONFIG ---
start_year = 2024
end_year = 2024
min_reports = 1
psm_file = '~/offsides/results/2024-2024/hdpsm_nrep5_mratio5_maxsamp25000.csv'
out_csv = f'~/offsides/results/{start_year}-{end_year}/age_drug_long.csv'

# --- Connect to DB ---
db = PostgresDB(verbose=False)

# --- Load PSM matched reports ---
psm = pd.read_csv(psm_file)
psm['report_id'] = psm['report_id'].astype(str)
psm['drug_id'] = psm['drug_id'].astype(str)

# --- Get unique report_ids we need ages for ---
report_ids = psm['report_id'].unique().tolist()
print(f"Fetching ages for {len(report_ids)} reports from DB...")

query = f"""
SELECT DISTINCT safetyreportid, patientonsetage
FROM openfda.reports
WHERE patientonsetageunit = '801'
  AND patientonsetage < 130
  AND EXTRACT(YEAR FROM receivedate) BETWEEN {start_year} AND {end_year};
"""

results = db.execute_query(query)

# --- Build mapping report_id -> age ---
report2age = {str(rid): age for rid, age in results}

# --- Add age to PSM dataframe ---
psm['age'] = psm['report_id'].map(report2age)

# Drop rows where age is missing
psm = psm.dropna(subset=['age'])

# --- Save CSV for est_assoc_statsAgeBias.py ---
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
psm[['report_id', 'age', 'drug_id']].to_csv(out_csv, index=False)
print(f"Saved age-drug long CSV: {out_csv}")
