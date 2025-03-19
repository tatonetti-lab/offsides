import os
import csv
import json
import tqdm
import time
import psycopg2
import numpy as np
import pandas as pd
from collections import defaultdict

class PostgresDB:
    """Class to handle PostgreSQL connection and queries."""
    
    def __init__(self, config_file="config.json"):
        self.config = self.load_config(config_file)
        self.connection = None
        self.cursor = None

    def load_config(self, config_file):
        """Load database configuration from JSON file."""
        with open(config_file, "r") as file:
            return json.load(file)

    def connect(self):
        """Establish connection to the PostgreSQL database."""
        if not self.connection:
            self.connection = psycopg2.connect(**self.config)
            self.cursor = self.connection.cursor()

    def execute_query(self, query, fetch_all=True):
        """Execute a SQL query and return results."""
        self.connect()  # Ensure connection is open
        start_time = time.time()
        self.cursor.execute(query)

        if query.strip().lower().startswith("select"):
            result = self.cursor.fetchall() if fetch_all else self.cursor.fetchone()
        else:
            self.connection.commit()
            result = None  # For INSERT, UPDATE, DELETE
        
        exec_time = time.time()-start_time
        print(f" Query completed in {exec_time}s")
        
        return result

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            self.connection = None

def drug_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    drugs = set()
    reactions = set()

    # (drug, reaction) counts
    print('Querying for drug, reaction counts...')
    drug_rea_count = defaultdict(int)
    query = f"""
    SELECT generic_name, reactionmeddrapt, COUNT(DISTINCT safetyreport_id)
    FROM drug
    JOIN safetyreport ON (safetyreport_id = safetyreport.id)
    JOIN reaction USING (safetyreport_id)
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    AND generic_name IS NOT NULL
    GROUP BY generic_name, reactionmeddrapt
    HAVING COUNT(DISTINCT safetyreport_id) >= {min_reports};
    """
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for drug, rea, count in results:
            drugs.add(drug)
            reactions.add(rea)
            drug_rea_count[(drug, rea)] = count
    except Exception as e:
        print(f"Database error: {e}")

    # Drug counts
    print('Querying for drug counts...')
    drug_count = defaultdict(int)
    query = f"""
    SELECT generic_name, COUNT(DISTINCT safetyreport_id)
    FROM drug
    JOIN safetyreport ON (safetyreport_id = safetyreport.id)
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    AND generic_name IS NOT NULL
    GROUP BY generic_name
    HAVING COUNT(DISTINCT safetyreport_id) >= {min_reports};
    """
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for drug, count in results:
            drugs.add(drug)
            drug_count[drug] = count
    except Exception as e:
        print(f"Database error: {e}")

    # Reaction counts
    print('Querying for reaction counts...')
    rea_count = defaultdict(int)
    query = f"""
    SELECT reactionmeddrapt, COUNT(DISTINCT safetyreport_id)
    FROM reaction
    JOIN safetyreport ON (safetyreport_id = safetyreport.id)
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    GROUP BY reactionmeddrapt
    HAVING COUNT(DISTINCT safetyreport_id) >= {min_reports};
    """
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for rea, count in results:
            reactions.add(rea)
            rea_count[rea] = count
    except Exception as e:
        print(f"Database error: {e}")

    # Total number of reports
    print('Querying total number of reports...')
    query = f"""
    SELECT COUNT(DISTINCT id)
    FROM safetyreport
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year};
    """
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        total_reports = results[0][0]
    except Exception as e:
        print(f"Database error: {e}")

    print(f"N_drugs = {len(drugs)}")
    print(f"N_reactions = {len(reactions)}")
    print(f"Drug, reaction non-zero pairs = {len(drug_rea_count)}")
    print(f"Drug non-zero counts = {len(drug_count)}")
    print(f"Reaction non-zero counts = {len(rea_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for drug in tqdm.tqdm(sorted(drugs)):
        for rea in sorted(reactions):
            a = drug_rea_count.get((drug, rea), 0)
            if a == 0:
                continue
            b = drug_count[drug] - a
            c = rea_count[rea] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            PHI = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (b + d) * (a + c))

            if OR is None and PRR is None:
                continue

            data.append([drug, rea, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['drug', 'reaction', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/drug_reaction_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)

    return df

def indication_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    indications = set()
    drugs = set()

    print('Querying for indication, drug counts...')
    inddrug_count = defaultdict(int)
    query = f"""
select drugindication, generic_name, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
and generic_name is not NULL
group by drugindication, generic_name
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for ind, drug, count in results:
            indications.add(ind)
            drugs.add(drug)
            inddrug_count[(ind, drug)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for indication counts...')
    ind_count = defaultdict(int)
    query = f"""
select drugindication, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
and generic_name is not NULL
group by drugindication
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for ind, count in results:
            indications.add(ind)
            ind_count[ind] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for drug counts...')
    drug_count = defaultdict(int)
    query = f"""
select generic_name, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
and generic_name is not NULL
group by generic_name
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for drug, count in results:
            drugs.add(drug)
            drug_count[drug] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    # total reports
    print('Querying total number of reports...')
    query = f"""
select count(distinct id)
from safetyreport
where left(receivedate, 4)::int between {start_year} and {end_year};
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        total_reports = results[0][0]
    except Exception as e:
        print(f"Database error: {e}")
    
    print(f"N_indications = {len(indications)}")
    print(f"N_drugs = {len(drugs)}")
    print(f"ind, rea non zero pairs = {len(inddrug_count)}")
    print(f"ind non-zero counts = {len(ind_count)}")
    print(f"drug non-zero counts = {len(drug_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for ind in tqdm.tqdm(sorted(indications)):
        for drug in sorted(drugs):
            a = inddrug_count.get((ind, drug), 0)
            if a == 0:
                continue
            b = ind_count[ind] - a
            c = drug_count[drug] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            PHI = (a*d-b*c)/np.sqrt((a+b)*(c+d)*(b+d)*(a+c))

            if OR is None and PRR is None:
                continue

            data.append([ind, drug, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['indication', 'drug', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/indication_drug_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)
    
    return df


def indication_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    indications = set()
    reactions = set()
    
    # (indication, reactions) counts
    print('Querying for indication, reaction counts...')
    indrea_count = defaultdict(int)
    query = f"""
select drugindication, reactionmeddrapt, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
join reaction using (safetyreport_id)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
group by drugindication, reactionmeddrapt
having count(distinct safetyreport_id) >= {min_reports};
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for ind, rea, count in results:
            #print((ind, rea, count))
            indications.add(ind)
            reactions.add(rea)
            indrea_count[(ind, rea)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    # indication counts
    print('Querying for indication counts...')
    ind_count = defaultdict(int)
    query = f"""
select drugindication, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
group by drugindication
having count(distinct safetyreport_id) >= {min_reports};
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for ind, count in results:
            indications.add(ind)
            ind_count[ind] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    # reaction counts
    print('Querying for reaction counts...')
    rea_count = defaultdict(int)
    query = f"""
select reactionmeddrapt, count(distinct safetyreport_id)
from reaction
join safetyreport on (safetyreport_id = safetyreport.id)
where left(receivedate, 4)::int between {start_year} and {end_year}
group by reactionmeddrapt
having count(distinct safetyreport_id) >= {min_reports};
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for rea, count in results:
            reactions.add(rea)
            rea_count[rea] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    # total reports
    print('Querying total number of reports...')
    query = f"""
select count(distinct id)
from safetyreport
where left(receivedate, 4)::int between {start_year} and {end_year};
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        total_reports = results[0][0]
    except Exception as e:
        print(f"Database error: {e}")
    
    print(f"N_indications = {len(indications)}")
    print(f"N_reactions = {len(reactions)}")
    print(f"ind, rea non zero pairs = {len(indrea_count)}")
    print(f"ind non-zero counts = {len(ind_count)}")
    print(f"rea non-zero counts = {len(rea_count)}")
    print(f"N_reports = {total_reports}")

    # debugging
    # print(len(set(ind_count.keys()) & set([i for i, r in indrea_count.keys()])))
    # print(len(set(rea_count.keys()) & set([r for i, r in indrea_count.keys()])))

    # print(max(ind_count.values()))
    # print(max(rea_count.values()))
    # print(max(indrea_count.values()))

    data = []

    for ind in tqdm.tqdm(sorted(indications)):
        for rea in sorted(reactions):
            a = indrea_count.get((ind, rea), 0)
            if a == 0:
                continue
            b = ind_count[ind] - a
            c = rea_count[rea] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            PHI = (a*d-b*c)/np.sqrt((a+b)*(c+d)*(b+d)*(a+c))

            if OR is None and PRR is None:
                continue

            data.append([ind, rea, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['indication', 'reaction', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/indication_reaction_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)
    
    return df

# Example usage
if __name__ == "__main__":
    db = PostgresDB()

    year = "2004"
    min_reports = 10

    ind_rea_df = indication_by_reaction_matrix(db, year, year, min_reports, save_to_file=True)
    ind_drug_df = indication_by_drug_matrix(db, year, year, min_reports, save_to_file=True)
    drug_rea_df = drug_by_reaction_matrix(db, year, year, min_reports, save_to_file=True)

    db.close()