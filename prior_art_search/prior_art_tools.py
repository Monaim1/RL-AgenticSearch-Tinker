import asyncio
from typing import List, Dict

from prior_art_search.local_patent_db import init_chroma_collection, get_chroma_semaphore


collection = init_chroma_collection(force_recreate=False)

## search tool
async def search_patents(query: str, n_results: int = 10) -> list[dict]:
    """Search for top 10 relevant patents using title embedding similarity."""
    async with get_chroma_semaphore():
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
    sem = get_chroma_semaphore()
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

async def main():
    res = await search_patents("battery management", n_results=3)
    print("Search Results:")
    for patent in res:
        print(patent)


if __name__ == "__main__":
    asyncio.run(main())
