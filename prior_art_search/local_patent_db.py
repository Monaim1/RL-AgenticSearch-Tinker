import os
import json
import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
import asyncio
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Required for Weights & Biases
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

CHROMA_DB_DIR = ".chroma_db"
_chroma_semaphore: asyncio.Semaphore | None = None

def get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(20)
    return _chroma_semaphore



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
    for file in os.listdir("Patent_data"):
        file_path = os.path.join("Patent_data", file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                filtered = {key: value for key, value in data.items() if key in RELEVANT_FIELDS}
                ip_files.append(filtered)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Skipping {file}: {e}")
    return ip_files


def init_chroma_collection(force_recreate: bool = False) :
    """Connect to an existing ChromaDB collection for patent data, or create if not exists."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # get the collection if it exists
    try:
        if force_recreate:
            chroma_client.delete_collection(name="patent_collection")
        collection = chroma_client.get_collection(
            name="patent_collection",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        )
        return collection
    except Exception:
        # If not found, create and ingest
        collection = chroma_client.get_or_create_collection(
            name="patent_collection",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200,
                    "ef_search": 150
                }
            }
        )
        collection.add(
            documents=[patent["abstract"] for patent in patent_data],  # Using abstract as the main text for embeddings
            ids=[patent["publication_number"] for patent in patent_data],
            metadatas=[{k: v for k, v in patent.items() if k != 'abstract'} for patent in patent_data],
        )
        return collection


if __name__ == "__main__":
    patent_data = get_IP_data()
    collection = init_chroma_collection(force_recreate=True)
    print(f"ChromaDB collection 'patent_collection' initialized with {len(collection.get()['ids'])} patents.")
