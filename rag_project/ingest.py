"""
ingest.py — Parse transcripts, embed chunks, save FAISS index and metadata.

Usage:
    python ingest.py

Reads all 4 transcript .txt files from the parent directory (../),
embeds each timestamp block using sentence-transformers all-MiniLM-L6-v2,
and saves index.faiss + chunks.json inside rag_project/.
"""

import os
import re
import json
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.dirname(__file__)

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

# The original CampusX video is in Hindi. Skip it and use the English
# translation file instead so embeddings are meaningful.
SKIP_FILES = {
    "NoteGPT_TRANSCRIPT_What is Deep Learning Deep Learning Vs Machine Learning  Complete Deep Learning Course.txt",
}


def parse_transcript(filepath):
    """Return list of (timestamp, text) tuples from a transcript file."""
    chunks = []
    current_ts = None
    current_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if TIMESTAMP_RE.match(line.strip()):
                # Save previous chunk
                if current_ts is not None and current_lines:
                    text = " ".join(current_lines).strip()
                    if text:
                        chunks.append((current_ts, text))
                current_ts = line.strip()
                current_lines = []
            else:
                if current_ts is not None and line.strip():
                    current_lines.append(line.strip())

    # Don't forget the last chunk
    if current_ts is not None and current_lines:
        text = " ".join(current_lines).strip()
        if text:
            chunks.append((current_ts, text))

    return chunks


def main():
    txt_files = sorted(glob.glob(os.path.join(PARENT_DIR, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {PARENT_DIR}")

    print(f"Found {len(txt_files)} transcript file(s).")

    all_chunks = []  # list of {"source": str, "timestamp": str, "text": str}

    for filepath in txt_files:
        source = os.path.basename(filepath)
        if source in SKIP_FILES:
            print(f"  {source}: skipped (replaced by English translation)")
            continue
        pairs = parse_transcript(filepath)
        print(f"  {source}: {len(pairs)} chunks")
        for ts, text in pairs:
            all_chunks.append({
                "source": source,
                "timestamp": ts,
                "text": text,
            })

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]

    print("Encoding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_path = os.path.join(OUTPUT_DIR, "index.faiss")
    chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")

    faiss.write_index(index, index_path)
    print(f"Saved FAISS index → {index_path}")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk metadata → {chunks_path}")
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
