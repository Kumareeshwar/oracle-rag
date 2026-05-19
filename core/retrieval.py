"""Retrieval module for Oracle RAG system.

Connects to ChromaDB and retrieves relevant schema context
for a given natural language query.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List


def load_collection():
    """Connect to ChromaDB at ingestion/chroma_db/ and return the oracle_ap_schema collection.

    Returns:
        chromadb.Collection: The oracle_ap_schema collection object.
    """
    client = chromadb.PersistentClient(path="ingestion/chroma_db")
    collection = client.get_or_create_collection("oracle_ap_schema")
    return collection


def retrieve_context(query: str, collection, top_k: int = 3) -> List[str]:
    """Retrieve the top_k most relevant schema chunks for the given query.

    Embeds the query using sentence-transformers (all-MiniLM-L6-v2),
    then searches the ChromaDB collection for the closest matches.

    Args:
        query: Natural language query string.
        collection: ChromaDB collection to search.
        top_k: Number of top results to return. Defaults to 3.

    Returns:
        List of matching schema chunk text strings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]
