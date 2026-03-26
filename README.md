# simple-RAG

A retrieval-augmented generation (RAG) system evaluation project built for an internship assignment. Tests a simple vector-based retriever against a curated golden dataset of 5 question-answer pairs sourced from 4 YouTube videos on neural networks, transformers, and deep learning.

## Overview

**What it does:**
- Chunks 4 video transcripts by timestamp
- Embeds chunks using `sentence-transformers` (all-MiniLM-L6-v2)
- Stores embeddings in a FAISS vector index
- Tests retrieval accuracy against 5 golden QA pairs
- Achieves 5/5 (100%) retrieval accuracy

**Key files:**
- `ingest.py` — Loads transcripts, chunks, embeds, saves FAISS index
- `retrieve.py` — Loads index, takes a query, returns top-3 results
- `eval.py` — Runs 5 golden questions, prints pass/fail table with scores
- `requirements.txt` — Python dependencies

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the index (one time)
```bash
python ingest.py
```

This creates:
- `index.faiss` — Vector index
- `chunks.json` — Chunk metadata (source, timestamp)

### 3. Run evaluation
```bash
python eval.py
```

Outputs a table showing all 5 questions, their retrieved sources, and whether retrieval was correct.

### 4. Try custom queries
```bash
python retrieve.py "your question here"
```

Returns top-3 most relevant chunks with source filename and timestamp.

## Project Structure
```
rag_project/
├── ingest.py                 # Ingestion pipeline
├── retrieve.py               # Retrieval interface
├── eval.py                   # Golden dataset evaluator
├── requirements.txt          # Dependencies
├── index.faiss              # Vector index (generated)
└── chunks.json              # Chunk metadata (generated)
```

## Data Sources

The project indexes 4 video transcripts:

1. **3Blue1Brown — But what is a Neural Network?** (36 chunks)  
   Focus: Weights, bias, sigmoid activation, network structure

2. **3Blue1Brown — Transformers, the tech behind LLMs** (55 chunks)  
   Focus: Softmax, temperature, word embeddings, attention mechanism

3. **CampusX — What is Deep Learning?** (6 chunks, manually translated from Hindi)  
   Focus: ML vs DL, neural network types, why deep learning succeeded

4. **CodeWithHarry — All About ML & Deep Learning** (30 chunks)  
   Focus: Supervised/unsupervised/reinforcement learning types, ML pipeline

**Total:** 127 chunks

## Golden Dataset

The evaluation set contains 5 QA pairs, each testing a specific retrieval scenario:

- **Q1:** Weights and bias in neural networks → Neural Network video
- **Q2:** Softmax and temperature → Transformers video
- **Q3:** ML types (supervised/unsupervised/RL) → CodeWithHarry video
- **Q4:** ML vs DL conceptual difference → CampusX video (English translation)
- **Q5:** Word embeddings and vector arithmetic → Transformers video

Each pair includes a "wrong retrieval" description — the plausible-but-incorrect chunk a weak retriever might return.

## Results
```
Score: 5/5 (100%)

Q1 weights/bias      → Neural Network (00:11:52) ✓
Q2 softmax/temp      → Transformers (00:22:25)  ✓
Q3 ML types          → CodeWithHarry (00:07:54) ✓
Q4 ML vs DL          → CampusX English (00:03:32) ✓
Q5 embeddings        → Transformers (00:14:26)  ✓
```

## Notes

- The CampusX video was originally in Hindi. Key sections were manually translated to English for indexing.
- `index.faiss` and `chunks.json` are generated on first run and can be deleted to rebuild.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (22M params, ~100MB download on first use)
- Context: This is a demonstration of RAG evaluation methodology, not a production system.

## Interview Context

This project demonstrates:
1. How to build a golden dataset (evaluation set) for RAG systems
2. Why wrong retrieval descriptions matter (they catch subtle failures)
3. How to test retriever quality systematically
4. How to handle multilingual content (manual translation + English indexing)

---
