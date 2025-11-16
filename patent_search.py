import os
import json
import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
import asyncio


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
    # "claims",  ## The legally enforceable boundaries of the invention — the essence of what’s protected.
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



CHROMA_DB_DIR = ".chroma_db"
_chroma_semaphore: asyncio.Semaphore | None = None

def _get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(20)
    return _chroma_semaphore


def create_chroma_collection():
    """Create or get a ChromaDB collection for patent data."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name="patent_collection",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200,
            "ef_search":150
        }
    }
    )
    collection.add(
        documents=[patent["abstract"] for patent in patent_data],
        ids=[patent["publication_number"] for patent in patent_data],
        metadatas=[{ k:v for k, v in patent.items() if k !='abstract'} 
                   for patent in patent_data],
        )
    
    return collection

## search tool
async def search_patents(query: str, n_results: int = 10) -> list[dict]:
    """Search for top 10 relevant patents using title embedding similarity."""
    async with _get_chroma_semaphore():
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    if not results:
        raise ValueError(f"No results found for query: {query}")
    if not results["metadatas"]:
        raise ValueError(f"No results metadata found for query: {query}")
    
    output = []
    for i in range(len(results["ids"][0])):
        patent_title = results["metadatas"][0][i]["title"]
        publication_number = results["ids"][0][i]
        similarity_score = results["distances"][0][i]
        output.append({"patent_title": patent_title, 
                       "publication_number": publication_number,
                       "similarity_score": similarity_score})
    
    return output

# Patent lookup tool
async def lookup_patent(publication_number: str) -> dict:
    """Lookup patent details by publication number."""
    sem = _get_chroma_semaphore()
    async with sem:
        results = await asyncio.to_thread(
            collection.get,
            ids=[publication_number],
        )

    if not results or not results.get("metadatas"):
        raise ValueError(f"No patent found with publication number: {publication_number}")

    patent_content = results["documents"][0]
    patent_metadata = results["metadatas"][0]
    return {**patent_metadata, "abstract": patent_content}
    




if __name__ == "__main__":
    patent_data = get_IP_data()
    collection = create_chroma_collection()