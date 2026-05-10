import argparse
import csv
import json
import tarfile
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import hf_hub_download


DATASET = "HUPD/hupd"
DEFAULT_FILE = "data/sample-jan-2016.tar.gz"
DEFAULT_CHROMA_DIR = ".chroma_db"
DEFAULT_COLLECTION = "patent_collection"
DEFAULT_CSV = "Evals/patent_search_queries.csv"

FIELDS = [
    "publication_number",
    "application_number",
    "patent_number",
    "date_published",
    "filing_date",
    "patent_issue_date",
    "abandon_date",
    "decision",
    "main_cpc_label",
    "cpc_labels",
    "main_ipcr_label",
    "ipcr_labels",
    "uspc_class",
    "uspc_subclass",
    "title",
    "abstract",
    "claims",
]


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def patent_document(patent: dict) -> str:
    parts = [
        ("Title", patent["title"]),
        ("Abstract", patent["abstract"]),
        ("Claims", patent["claims"]),
        ("CPC", patent["cpc_labels"] or patent["main_cpc_label"]),
        ("IPCR", patent["ipcr_labels"] or patent["main_ipcr_label"]),
        ("USPC", " ".join(x for x in [patent["uspc_class"], patent["uspc_subclass"]] if x)),
        ("Decision", patent["decision"]),
        ("Published", patent["date_published"]),
        ("Filed", patent["filing_date"]),
    ]
    return "\n\n".join(f"{label}: {value}" for label, value in parts if value)


def load_hupd(limit: int, hf_file: str) -> list[dict]:
    archive = hf_hub_download(DATASET, filename=hf_file, repo_type="dataset")
    patents = []

    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            raw = json.load(f)
            patent = {key: clean(raw.get(key, "")) for key in FIELDS}
            if not patent["publication_number"] or not patent["abstract"]:
                continue

            patents.append(patent)
            if len(patents) >= limit:
                break

    return patents


def write_eval_csv(patents: list[dict], csv_path: str) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["publication_number", "query", "abstract"])
        writer.writeheader()
        for patent in patents:
            writer.writerow(
                {
                    "publication_number": patent["publication_number"],
                    "query": patent["title"],
                    "abstract": patent["abstract"],
                }
            )


def build_chroma(patents: list[dict], chroma_dir: str, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": 200,
                "ef_search": 150,
            }
        },
    )

    collection.add(
        ids=[p["publication_number"] for p in patents],
        documents=[patent_document(p) for p in patents],
        metadatas=patents,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--hf-file", default=DEFAULT_FILE)
    parser.add_argument("--csv-path", default=DEFAULT_CSV)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    patents = load_hupd(limit=args.limit, hf_file=args.hf_file)
    if len(patents) < 10:
        raise ValueError(f"Need at least 10 usable patents, got {len(patents)}")

    write_eval_csv(patents, args.csv_path)
    build_chroma(patents, args.chroma_dir, args.collection_name)

    print(f"Prepared {len(patents)} patents")
    print(f"Wrote {args.csv_path}")
    print(f"Built Chroma collection '{args.collection_name}' in {args.chroma_dir}")


if __name__ == "__main__":
    main()
