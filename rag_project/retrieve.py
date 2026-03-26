"""
retrieve.py — Load the FAISS index and retrieve top-k chunks for a query.

Usage:
    python retrieve.py "your query here"
"""

import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(SCRIPT_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(SCRIPT_DIR, "chunks.json")
TOP_K = 3


def load_resources():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}\nRun ingest.py first.")
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks not found: {CHUNKS_PATH}\nRun ingest.py first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query, index, chunks, model, top_k=TOP_K):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        chunk = chunks[idx]
        results.append({
            "rank": rank,
            "distance": float(dist),
            "source": chunk["source"],
            "timestamp": chunk["timestamp"],
            "text": chunk["text"],
        })
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve.py \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_resources()
    results = retrieve(query, index, chunks, model)

    for r in results:
        print(f"Rank {r['rank']} (distance={r['distance']:.4f})")
        print(f"  Source:    {r['source']}")
        print(f"  Timestamp: {r['timestamp']}")
        print(f"  Text:      {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
        print()


if __name__ == "__main__":
    main()
