# RAG Evaluation Project

Retrieval-Augmented Generation (RAG) pipeline built on 4 YouTube transcripts about neural networks and deep learning. Includes ingestion, retrieval, and golden-dataset evaluation.

## Setup

```bash
cd rag_project
pip install -r requirements.txt
```

## Step 1 — Ingest transcripts

Run this once to parse all 4 transcript `.txt` files from the parent folder, embed them, and save the FAISS index:

```bash
python ingest.py
```

This creates:
- `index.faiss` — FAISS vector index of all transcript chunks
- `chunks.json` — chunk metadata (source filename, timestamp, text)

## Step 2 — Run the evaluation

Runs all 5 golden-dataset questions through the retriever and prints a pass/fail table:

```bash
python eval.py
```

## Custom queries with retrieve.py

Retrieve the top 3 most relevant chunks for any question:

```bash
python retrieve.py "How does the attention mechanism allow tokens to communicate?"
```

Example output:
```
Rank 1 (distance=0.3821)
  Source:    NoteGPT_TRANSCRIPT_Transformers, the tech behind LLMs  Deep Learning Chapter 5.txt
  Timestamp: 00:03:55
  Text:      This sequence of vectors then passes through an operation that's known as an attention block...
```
