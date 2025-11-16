import dspy
from dotenv import load_dotenv
import os
import pandas as pd
import json

load_dotenv()
base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
api_key = os.getenv("GEMINI_API_KEY")
lm = dspy.LM('gemini/gemini-2.0-flash', api_key=api_key)
dspy.configure(lm=lm)


# Fields to extract from raw JSON data
RELEVANT_FIELDS = [
    # Identifiers & Linking
    "publication_number",
    "application_number",
    "patent_number",
    # Dates (as epoch ints)
    "date_published",
    "filing_date",
    "patent_issue_date",
    "abandon_date",
    # Status & Classes
    "decision",
    "main_cpc_label",
    "main_ipcr_label",
    # Retrievable Text
    "title",
    "abstract",
    "claims",  ## The legally enforceable boundaries of the invention — the essence of what’s protected.
    # "summary",
]

def get_IP_data():
    """Load and filter IP data from JSON files, skipping files with decode errors."""
    ip_files = []
    for file in os.listdir("./Patent_data"):
        file_path = os.path.join("Patent_data", file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                filtered = {key: value for key, value in data.items() if key in RELEVANT_FIELDS}
                ip_files.append(filtered)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Skipping {file}: {e}")
    return ip_files


class ExtractInfo(dspy.Signature):
    """you are gonna be given an abtract of a patent and you need to generate 3 queries that
    can be used to search for prior art patents related to the given patent abstract. 
    The queries should be concise and relevant to the key aspects of the patent abstract.
    only return the queries as a json array of strings with no other text."""

    patent_abstract: str = dspy.InputField()
    number_of_queries: int = dspy.InputField(default=3, desc="Number of search queries to generate.")
    queries: list[str] = dspy.OutputField(desc="List of search queries for prior art patents.")

module = dspy.Predict(ExtractInfo)

def get_queries_from_abstract(patent_abstract: str, number_of_queries=3) -> list[str]:
    """Generate search queries for prior art patents based on the given patent abstract."""
    try:
        return module(patent_abstract=patent_abstract, 
                      number_of_queries=number_of_queries).queries
    except Exception as e:
        raise print(f"Error generating queries: {e}")
        
    
def create_search_queries(patent_data: list[str], number_of_queries=3)-> pd.DataFrame:
    """Generate search queries for a list of patent abstracts."""
    all_queries = []
    for patent in patent_data:
        queries = get_queries_from_abstract(patent['abstract'], number_of_queries)
        for query in queries:
            all_queries.append({'publication_number': patent['publication_number'], 'query': query, 'abstract': patent['abstract']})
    out = pd.DataFrame(all_queries)
    out.to_csv("Evals/patent_search_queries.csv", index=False)
    return out


if __name__ == "__main__":
    patent_data = get_IP_data()
    create_search_queries(patent_data, number_of_queries=3)
