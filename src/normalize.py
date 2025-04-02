import logging
import requests
import pandas as pd
from tqdm import tqdm
from typing import Optional

# Configure logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def normalize_texts(
    input_csv: str,
    text_column: str,
    output_csv: str,
    checkpoint_csv: str,
    ragnorm_endpoint: str,
    model: str = "gpt-4o-mini",
    vocabulary: str = "SNOMED_CT",
    n_results: int = 25,
    user: str = "jacob",
    save_checkpoint_every: int = 5000
) -> None:
    """
    Normalizes text in the given column of input_csv using our RAGNorm API.
    Writes results to output_csv (and checkpoint_csv periodically).

    Args:
        input_csv (str): Path to the input CSV file.
        text_column (str): Column name containing the text to normalize.
        output_csv (str): Path to the final output CSV file.
        checkpoint_csv (str): Path to a checkpoint CSV file to store progress periodically.
        ragnorm_endpoint (str): URL of the RAGNorm API endpoint.
        model (str): Model name to send to RAGNorm (default: "gpt-4o-mini").
        vocabulary (str): Vocabulary to use for normalization (default: "SNOMED_CT").
        n_results (int): Number of potential results to retrieve (default: 25).
        user (str): User identifier for RAGNorm (default: "jacob").
        save_checkpoint_every (int): Frequency of checkpoint saving in rows (default: 5000).
    """
    logger.info(f"Starting normalization: {input_csv} -> {output_csv}")
    df = pd.read_csv(input_csv)
    results_records = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing Text"):
        text_value = str(row[text_column]).strip()

        payload = {
            "words": [text_value],
            "model": model,
            "vocabulary": vocabulary,
            "n_results": n_results,
            "user": user,
            "include_usage_counts": False,
            "return_mode": "best"
        }

        justification = ""
        best_term = "None"
        best_code = -1

        try:
            response = requests.post(ragnorm_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json().get("results", {})
                rag_data = data.get(text_value, {})
                justification = rag_data.get("justification", "")
                terms_list = rag_data.get("terms", [])

                if len(terms_list) > 0:
                    best_term = terms_list[0].get("term", "None")
                    best_code = terms_list[0].get("code", -1)
            else:
                logger.warning(f"[Row {i}] RAGNorm error {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            logger.error(f"[Row {i}] Timeout error calling RAGNorm API.")
        except requests.exceptions.RequestException as e:
            logger.error(f"[Row {i}] Request exception occurred: {str(e)}")

        results_records.append({
            text_column: text_value,
            "justification": justification,
            "term": best_term,
            "code": best_code
        })

        # checkpoint
        if (i + 1) % save_checkpoint_every == 0:
            pd.DataFrame(results_records).to_csv(checkpoint_csv, index=False)
            logger.info(f"Checkpoint saved at row {i + 1}")

    pd.DataFrame(results_records).to_csv(output_csv, index=False)
    logger.info(f"Normalization complete. Final CSV saved to {output_csv}.")

def resolve_none_entries(
    input_csv: str,
    text_column: str,
    output_csv: str,
    retrieval_endpoint: str,
    vocabulary: str = "SNOMED_CT"
) -> None:
    """
    Searches for rows with "None" or -1 code in the previous step in case gpt failed, and tries our fallback
    retrieval endpoint to find a match.

    Args:
        input_csv (str): Path to the CSV containing normalized text (output of normalize_texts).
        text_column (str): Column name containing the text to check.
        output_csv (str): Path to store the updated CSV.
        retrieval_endpoint (str): URL of the retrieval API endpoint.
        vocabulary (str): Vocabulary to use for fallback retrieval (default: "SNOMED_CT").
    """
    logger.info(f"Resolving None entries in {input_csv} -> {output_csv}")
    df = pd.read_csv(input_csv)

    required_cols = {text_column, "term", "code", "justification"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {input_csv}")

    updated_records = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Resolving 'None' terms"):
        text_value = str(row[text_column])
        best_term = str(row["term"])
        best_code = row["code"]
        justification = str(row["justification"])
        source = "llm"  # default

        if best_term == "None" or best_code == -1:
            payload = {
                "words": [text_value],
                "vocabulary": vocabulary,
                "n_results": 1
            }

            try:
                resp = requests.post(retrieval_endpoint, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()["results"].get(text_value, {})
                    terms_list = data.get("terms", [])
                    if len(terms_list) > 0:
                        best_term = terms_list[0].get("term", "None")
                        meta = terms_list[0].get("metadata", {})
                        best_code = meta.get("code", -1)
                        justification += " [fallback to retrieval]"
                    else:
                        best_term = "None"
                        best_code = -1
                else:
                    justification += f" [error calling retrieval: {resp.text}]"
                    logger.warning(f"[Row {i}] Retrieval error {resp.status_code}: {resp.text}")

                source = "retrieval"
            except requests.exceptions.Timeout:
                justification += " [retrieval timeout]"
                logger.error(f"[Row {i}] Timeout calling retrieval endpoint")
                best_term = "None"
                best_code = -1
                source = "retrieval"
            except requests.exceptions.RequestException as e:
                justification += f" [retrieval exception: {str(e)}]"
                logger.error(f"[Row {i}] Retrieval request exception: {str(e)}")
                best_term = "None"
                best_code = -1
                source = "retrieval"

        updated_records.append({
            text_column: text_value,
            "justification": justification,
            "term": best_term,
            "code": best_code,
            "source": source
        })

    out_df = pd.DataFrame(updated_records)
    out_df.to_csv(output_csv, index=False)
    logger.info(f"Fallback retrieval complete. Updated file written to {output_csv}.")

def clean_and_save(
    input_csv: str,
    text_column: str,
    output_csv: str
) -> None:
    """
    Cleans the CSV file by filtering out certain rows and saves the final result.

    Args:
        input_csv (str): CSV file resulting from resolve_none_entries.
        text_column (str): Column name of interest (not specifically used here, but for consistency).
        output_csv (str): Final cleaned CSV file path.
    """
    logger.info(f"Cleaning and saving final data from {input_csv} -> {output_csv}")
    df = pd.read_csv(input_csv)

    # Check if gpt-4 produced an error (we wanna leave these in!)
    df_filtered = df[
        ~(
            (df["source"] == "retrieval") &
            (df["justification"] != "nan [fallback to retrieval]")
        )
    ]

    df_filtered.to_csv(output_csv, index=False)
    logger.info(f"Data cleaning complete. Final CSV saved to {output_csv}.")

def main() -> None:

    input_csv = "faers_public_indication.csv"          # Original data
    text_column = "drugindication"                     # Column to normalize
    checkpoint_csv = "faers_ragnorm_checkpoint.csv"    # Checkpoint
    intermediate_csv_ragnorm = "faers_ragnorm.csv"     # After RAGNorm
    intermediate_csv_retrieval = "faers_retrieval.csv" # After fallback retrieval
    final_csv = "faers_corrected.csv"                  # Final "cleaned" output

    # RAGNorm and retrieval endpoints / configuration
    ragnorm_endpoint = "http://localhost:6676/RAGnorm"
    retrieval_endpoint = "http://localhost:6676/retrieval"
    model = "gpt-4o-mini"
    vocabulary = "SNOMED_CT"
    n_results = 25
    user = "jacob"

    # Step 1: Normalize via RAGNorm
    normalize_texts(
        input_csv=input_csv,
        text_column=text_column,
        output_csv=intermediate_csv_ragnorm,
        checkpoint_csv=checkpoint_csv,
        ragnorm_endpoint=ragnorm_endpoint,
        model=model,
        vocabulary=vocabulary,
        n_results=n_results,
        user=user,
        save_checkpoint_every=5000
    )

    # Step 2: Resolve 'None' entries with retrieval fallback
    resolve_none_entries(
        input_csv=intermediate_csv_ragnorm,
        text_column=text_column,
        output_csv=intermediate_csv_retrieval,
        retrieval_endpoint=retrieval_endpoint,
        vocabulary=vocabulary
    )

    # Step 3: Clean & save final
    clean_and_save(
        input_csv=intermediate_csv_retrieval,
        text_column=text_column,
        output_csv=final_csv
    )

if __name__ == "__main__":
    main()