import os
import csv
import json
import tqdm
import time
import argparse
import psycopg2
import numpy as np
import pandas as pd
from collections import defaultdict

class PostgresDB:
    """Class to handle PostgreSQL connection and queries."""
    
    def __init__(self, config_file="config.json", verbose=True):
        self.config = self.load_config(config_file)
        self.connection = None
        self.cursor = None
        self.verbose = verbose

    def load_config(self, config_file):
        """Load database configuration from JSON file."""
        with open(config_file, "r") as file:
            return json.load(file)

    def connect(self):
        """Establish connection to the PostgreSQL database."""
        if not self.connection:
            self.connection = psycopg2.connect(**self.config)
            self.cursor = self.connection.cursor()

    def execute_query(self, query, fetch_all=True, verbose=None):
        """Execute a SQL query and return results."""
        self.connect()  # Ensure connection is open
        if verbose is None:
            verbose = self.verbose
        
        start_time = time.time()
        self.cursor.execute(query)

        if query.strip().lower().startswith("select"):
            result = self.cursor.fetchall() if fetch_all else self.cursor.fetchone()
        else:
            self.connection.commit()
            result = None  # For INSERT, UPDATE, DELETE
        
        exec_time = time.time()-start_time
        if verbose:
            print(f" Query completed in {exec_time}s")
        
        return result

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            self.connection = None

def age_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    ages = set()
    drugs = set()

    print('Querying for age, drug counts...')
    age_drug_counts = defaultdict(int)
    query = f"""
select 
    case
        when patientonsetage::int < 20 then 'Younger'
        when patientonsetage::int > 60 then 'Older'
        else 'Middle'
    end as age_group, 
    ingredient_concept_name, count(distinct safetyreport.id)
from safetyreport
join drug on (safetyreport_id = safetyreport.id)
join ingredient on (drug.id = ingredient.id)
WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
and patientonsetageunit = '801'
and patientonsetage::int < 130
group by age_group, ingredient_concept_name
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for age, drug, count in results:
            ages.add(age)
            drugs.add(drug)
            age_drug_counts[(age, drug)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for age counts...')
    age_count = defaultdict(int)
    query = f"""
select 
    case
        when patientonsetage::int < 20 then 'Younger'
        when patientonsetage::int > 60 then 'Older'
        else 'Middle'
    end as age_group, 
    count(distinct safetyreport.id)
from safetyreport
WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
and patientonsetageunit = '801'
and patientonsetage::int < 130
group by age_group 
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for age, count in results:
            ages.add(age)
            age_count[age] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for drug counts...')
    drug_count = defaultdict(int)
    query = f"""
select ingredient_concept_name, count(distinct safetyreport_id)
from drug
join ingredient on (ingredient.id = drug.id)
join safetyreport on (safetyreport_id = safetyreport.id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
and patientsex is not NULL
and patientsex != '0'
group by ingredient_concept_name
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
    
    print(f"N_ages = {len(ages)}")
    print(f"N_drugs = {len(drugs)}")
    print(f"Age, drug non zero pairs = {len(age_drug_counts)}")
    print(f"Age non-zero counts = {len(age_count)}")
    print(f"drug non-zero counts = {len(drug_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for age in tqdm.tqdm(sorted(ages)):
        for drug in sorted(drugs):
            a = age_drug_counts.get((age, drug), 0)
            if a == 0:
                continue
            b = age_count[age] - a
            c = drug_count[drug] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            denom = np.sqrt(float((a+b)*(c+d)*(b+d)*(a+c)))
            PHI = (a*d-b*c)/denom if denom > 0 else None

            if OR is None and PRR is None:
                continue
            
            data.append([age, drug, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['age', 'drug', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/age_drug_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)
    
    return df

def age_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    ages = set()
    reactions = set()

    print('Querying for age, reaction counts...')
    age_rea_count = defaultdict(int)
    query = f"""
select 
    case
        when patientonsetage::int < 20 then 'Younger'
        when patientonsetage::int > 60 then 'Older'
        else 'Middle'
    end as age_group, 
    reactionmeddrapt, count(distinct safetyreport.id)
from safetyreport
join reaction on (safetyreport_id = safetyreport.id)
WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
and patientonsetageunit = '801'
and patientonsetage::int < 130
and reactionmeddrapt is not null
group by age_group, reactionmeddrapt
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for age, rea, count in results:
            ages.add(age)
            reactions.add(rea)
            age_rea_count[(age, rea)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for age counts...')
    age_count = defaultdict(int)
    query = f"""
select 
    case
        when patientonsetage::int < 20 then 'Younger'
        when patientonsetage::int > 60 then 'Older'
        else 'Middle'
    end as age_group, 
    count(distinct safetyreport.id)
from safetyreport
WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
and patientonsetageunit = '801'
and patientonsetage::int < 130
group by age_group 
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for age, count in results:
            ages.add(age)
            age_count[age] = count
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
    and reactionmeddrapt is not null
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

    print(f"N_ages = {len(ages)}")
    print(f"N_reactions = {len(reactions)}")
    print(f"Age, reaction non-zero pairs = {len(age_rea_count)}")
    print(f"Age non-zero counts = {len(age_count)}")
    print(f"Reaction non-zero counts = {len(rea_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for age in tqdm.tqdm(sorted(ages)):
        for rea in sorted(reactions):
            a = age_rea_count.get((age, rea), 0)
            if a == 0:
                continue
            b = age_count[age] - a
            c = rea_count[rea] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            denom = np.sqrt(float((a+b)*(c+d)*(b+d)*(a+c)))
            PHI = (a*d-b*c)/denom if denom > 0 else None

            if OR is None and PRR is None:
                continue

            data.append([age, rea, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['age', 'reaction', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/age_reaction_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)

    return df

def sex_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    sexes = set()
    reactions = set()

    print('Querying for sex, reaction counts...')
    sex_rea_count = defaultdict(int)
    query = f"""
select patientsex, reactionmeddrapt, count(distinct safetyreport.id)
from reaction
join safetyreport on (safetyreport_id = safetyreport.id)
where LEFT(receivedate,4)::int BETWEEN {start_year} and {end_year}
and patientsex is not null
and patientsex != '0'
and reactionmeddrapt is not NULL
group by patientsex, reactionmeddrapt
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for sex, rea, count in results:
            sexes.add(sex)
            reactions.add(rea)
            sex_rea_count[(sex, rea)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for sex counts...')
    sex_count = defaultdict(int)
    query = f"""
select patientsex, count(distinct safetyreport.id)
from safetyreport
where left(receivedate, 4)::int between {start_year} and {end_year}
and patientsex is not NULL
and patientsex != '0'
group by patientsex
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for ind, count in results:
            sexes.add(ind)
            sex_count[ind] = count
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
    AND reactionmeddrapt is not NULL
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

    print(f"N_sexes = {len(sexes)}")
    print(f"N_reactions = {len(reactions)}")
    print(f"Sex, reaction non-zero pairs = {len(sex_rea_count)}")
    print(f"Sex non-zero counts = {len(sex_count)}")
    print(f"Reaction non-zero counts = {len(rea_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for sex in tqdm.tqdm(sorted(sexes)):
        for rea in sorted(reactions):
            a = sex_rea_count.get((sex, rea), 0)
            if a == 0:
                continue
            b = sex_count[sex] - a
            c = rea_count[rea] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            denom = np.sqrt(float((a+b)*(c+d)*(b+d)*(a+c)))
            PHI = (a*d-b*c)/denom if denom > 0 else None

            if OR is None and PRR is None:
                continue

            data.append([sex, rea, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['sex', 'reaction', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/sex_reaction_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)

    return df

def sex_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    drugs = set()
    sexes = set()

    # (sex, drug) counts
    print('Querying for sex, drug counts...')
    sex_drug_count = defaultdict(int)
    query = f"""
select patientsex, ingredient_concept_name, count(distinct safetyreport_id)
from drug
join ingredient on (ingredient.id = drug.id)
join safetyreport on (safetyreport_id = safetyreport.id)
where LEFT(receivedate,4)::int BETWEEN {start_year} and {end_year}
and patientsex is not null
and patientsex != '0'
group by patientsex, ingredient_concept_name
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for sex, drug, count in results:
            drugs.add(drug)
            sexes.add(sex)
            sex_drug_count[(sex, drug)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for sex counts...')
    sex_count = defaultdict(int)
    query = f"""
select patientsex, count(distinct safetyreport.id)
from safetyreport
where left(receivedate, 4)::int between {start_year} and {end_year}
and patientsex is not NULL
and patientsex != '0'
group by patientsex
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for ind, count in results:
            sexes.add(ind)
            sex_count[ind] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    print('Querying for drug counts...')
    drug_count = defaultdict(int)
    query = f"""
select ingredient_concept_name, count(distinct safetyreport_id)
from drug
join ingredient on (ingredient.id = drug.id)
join safetyreport on (safetyreport_id = safetyreport.id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
and patientsex is not NULL
and patientsex != '0'
group by ingredient_concept_name
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
    
    print(f"N_sexes = {len(sexes)}")
    print(f"N_drugs = {len(drugs)}")
    print(f"sex, drug non zero pairs = {len(sex_drug_count)}")
    print(f"sex non-zero counts = {len(sex_count)}")
    print(f"drug non-zero counts = {len(drug_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for sex in tqdm.tqdm(sorted(sexes)):
        for drug in sorted(drugs):
            a = sex_drug_count.get((sex, drug), 0)
            if a == 0:
                continue
            b = sex_count[sex] - a
            c = drug_count[drug] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            denom = np.sqrt(float((a+b)*(c+d)*(b+d)*(a+c)))
            PHI = (a*d-b*c)/denom if denom > 0 else None

            if OR is None and PRR is None:
                continue
            
            data.append([sex, drug, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['sex', 'drug', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/sex_drug_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)
    
    return df
    
    


def drug_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    drugs = set()

    # (drug, drug) counts
    drug_drug_count = defaultdict(int)
    query = f"""
select aa.ingredient_concept_name, bb.ingredient_concept_name, count(distinct safetyreport_id)
from drug a
join ingredient aa on (aa.id = a.id)
join safetyreport ON (safetyreport_id = safetyreport.id)
join drug b using (safetyreport_id)
join ingredient bb on (bb.id = b.id)
where aa.ingredient_concept_name != bb.ingredient_concept_name
and LEFT(receivedate,4)::int BETWEEN {start_year} and {end_year}
group by aa.ingredient_concept_name, bb.ingredient_concept_name
having count(distinct safetyreport_id) >= {min_reports}
"""
    try:
        results = db.execute_query(query)
        print(f" N results: {len(results)}")
        for d1, d2, count in results:
            drugs.add(d1)
            drugs.add(d2)
            drug_drug_count[(d1, d2)] = count
    except Exception as e:
        print(f"Database error: {e}")
    
    # Drug counts
    print('Querying for drug counts...')
    drug_count = defaultdict(int)
    query = f"""
    SELECT ingredient_concept_name, COUNT(DISTINCT safetyreport_id)
    FROM drug
    JOIN ingredient on (ingredient.id = drug.id)
    JOIN safetyreport ON (safetyreport_id = safetyreport.id)
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    GROUP BY ingredient_concept_name
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
    print(f"Drug, drug non-zero pairs = {len(drug_drug_count)}")
    print(f"Drug non-zero counts = {len(drug_count)}")
    print(f"N_reports = {total_reports}")

    data = []

    for d1 in tqdm.tqdm(sorted(drugs)):
        for d2 in sorted(drugs):
            a = drug_drug_count.get((d1, d2), 0)
            if a == 0:
                continue
            b = drug_count[d1] - a
            c = drug_count[d2] - a
            d = total_reports - (a + b + c)

            OR = (a / b) / (c / d) if b > 0 and d > 0 and c > 0 else None
            PRR = (a / (a + b)) / (c / (c + d)) if c > 0 else None
            PHI = (a * d - b * c) / np.sqrt(float((a + b) * (c + d) * (b + d) * (a + c)))

            if OR is None and PRR is None:
                continue
            
            data.append([d1, d2, a, b, c, d, OR, PRR, PHI])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['conf_drug', 'drug', 'a', 'b', 'c', 'd', 'OR', 'PRR', 'PHI'])

    # Optionally save to file
    if save_to_file:
        os.makedirs(f'./results/{start_year}-{end_year}', exist_ok=True)
        ofn = f'./results/{start_year}-{end_year}/drug_drug_associations.csv'
        print(f"Saving results to file: {ofn}")
        df.to_csv(ofn, index=False)

    return df

def drug_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=False):
    drugs = set()
    reactions = set()

    # (drug, reaction) counts
    print('Querying for drug, reaction counts...')
    drug_rea_count = defaultdict(int)
    query = f"""
    SELECT ingredient_concept_name, reactionmeddrapt, COUNT(DISTINCT safetyreport_id)
    FROM drug
    JOIN ingredient on (ingredient.id = drug.id)
    JOIN safetyreport ON (safetyreport_id = safetyreport.id)
    JOIN reaction USING (safetyreport_id)
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    and reactionmeddrapt is not null
    GROUP BY ingredient_concept_name, reactionmeddrapt
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
    SELECT ingredient_concept_name, COUNT(DISTINCT safetyreport_id)
    FROM drug
    JOIN ingredient on (ingredient.id = drug.id)
    JOIN safetyreport ON (safetyreport_id = safetyreport.id)
    WHERE LEFT(receivedate, 4)::int BETWEEN {start_year} AND {end_year}
    GROUP BY ingredient_concept_name
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
    and reactionmeddrapt is not null
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
            PHI = (a * d - b * c) / np.sqrt(float((a + b) * (c + d) * (b + d) * (a + c)))

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
select snomed_term, ingredient_concept_name, count(distinct safetyreport_id)
from drug
join ingredient on (ingredient.id = drug.id)
join safetyreport on (safetyreport_id = safetyreport.id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
group by snomed_term, ingredient_concept_name
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
select snomed_term, count(distinct safetyreport_id)
from drug
join ingredient on (ingredient.id = drug.id)
join safetyreport on (safetyreport_id = safetyreport.id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
group by snomed_term
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
select ingredient_concept_name, count(distinct safetyreport_id)
from drug
join ingredient on (ingredient.id = drug.id)
join safetyreport on (safetyreport_id = safetyreport.id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
group by ingredient_concept_name
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
            PHI = (a*d-b*c)/np.sqrt(float((a+b)*(c+d)*(b+d)*(a+c)))

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
select snomed_term, reactionmeddrapt, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
join reaction using (safetyreport_id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
and reactionmeddrapt is not NULL
group by snomed_term, reactionmeddrapt
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
select snomed_term, count(distinct safetyreport_id)
from drug
join safetyreport on (safetyreport_id = safetyreport.id)
join drug_indications using (drugindication)
where left(receivedate, 4)::int between {start_year} and {end_year}
and drugindication is not NULL
group by snomed_term
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
and reactionmeddrapt is not null
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
            PHI = (a*d-b*c)/np.sqrt(float((a+b)*(c+d)*(b+d)*(a+c)))

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

def parse_args():
    parser = argparse.ArgumentParser(description="Process a range of years.")
    parser.add_argument('--start_year', type=int, required=True, help='Start year (inclusive)')
    parser.add_argument('--end_year', type=int, required=True, help='End year (inclusive)')
    return parser.parse_args()

if __name__ == "__main__":
    db = PostgresDB()

    args = parse_args()
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")
    start_year = args.start_year
    end_year = args.end_year
    min_reports = 5
    
    ind_rea_df = indication_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    ind_drug_df = indication_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    drug_rea_df = drug_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    drug_drug_df = drug_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    sex_drug_df = sex_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    sex_rea_df = sex_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    age_rea_df = age_by_reaction_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    age_drug_df = age_by_drug_matrix(db, start_year, end_year, min_reports, save_to_file=True)
    db.close()