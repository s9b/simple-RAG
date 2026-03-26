"""
eval.py — Evaluate the RAG retriever against the golden dataset.

Hardcodes 5 QA pairs and their expected source videos,
runs each query through the retriever, and prints a results table.

Usage:
    python eval.py
"""

import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from retrieve import load_resources, retrieve

# Golden dataset: (question, expected_source_keyword)
# The keyword is matched as a substring of the retrieved chunk's source filename.
GOLDEN_DATASET = [
    {
        "question": "What role do weights play in a neural network and how does bias modify neuron activation?",
        "expected_source_keyword": "neural network",
        "expected_label": "3Blue1Brown — Neural Network (Video 1)",
    },
    {
        "question": "What is the purpose of softmax and how does the temperature parameter change the distribution?",
        "expected_source_keyword": "Transformers",
        "expected_label": "3Blue1Brown — Transformers (Video 2)",
    },
    {
        "question": "What is the difference between supervised, unsupervised, and reinforcement learning?",
        "expected_source_keyword": "Machine Learning",
        "expected_label": "CodeWithHarry — All About ML & Deep Learning (Video 4)",
    },
    {
        "question": "What is the conceptual difference between machine learning and deep learning?",
        "expected_source_keyword": "ENGLISH_TRANSLATION",
        "expected_label": "CampusX — What is Deep Learning, English Translation (Video 3)",
    },
    {
        "question": "How do word embeddings encode semantic meaning and what does vector arithmetic on them imply?",
        "expected_source_keyword": "Transformers",
        "expected_label": "3Blue1Brown — Transformers (Video 2)",
    },
]

COL_Q = 60
COL_SRC = 45
COL_TS = 10
COL_MATCH = 7


def truncate(s, n):
    return s if len(s) <= n else s[:n - 3] + "..."


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_resources()

    header = (
        f"{'Question':<{COL_Q}} | "
        f"{'Retrieved Source':<{COL_SRC}} | "
        f"{'TS':<{COL_TS}} | "
        f"{'Match?':<{COL_MATCH}}"
    )
    divider = "-" * len(header)

    print("\nRAG Evaluation Results")
    print("=" * len(header))
    print(header)
    print(divider)

    correct = 0
    for i, qa in enumerate(GOLDEN_DATASET, start=1):
        results = retrieve(qa["question"], index, chunks, model, top_k=1)
        top = results[0]

        matched = qa["expected_source_keyword"].lower() in top["source"].lower()
        if matched:
            correct += 1

        q_str = truncate(f"Q{i}: {qa['question']}", COL_Q)
        src_str = truncate(top["source"], COL_SRC)
        ts_str = top["timestamp"]
        match_str = "YES ✓" if matched else "NO  ✗"

        print(f"{q_str:<{COL_Q}} | {src_str:<{COL_SRC}} | {ts_str:<{COL_TS}} | {match_str:<{COL_MATCH}}")

    print(divider)
    total = len(GOLDEN_DATASET)
    print(f"\nScore: {correct}/{total} correct ({100*correct//total}%)")

    print("\nDetailed breakdown:")
    for i, qa in enumerate(GOLDEN_DATASET, start=1):
        results = retrieve(qa["question"], index, chunks, model, top_k=1)
        top = results[0]
        matched = qa["expected_source_keyword"].lower() in top["source"].lower()
        status = "PASS" if matched else "FAIL"
        print(f"\n  [{status}] Q{i}: {qa['question']}")
        print(f"    Expected:  {qa['expected_label']}")
        print(f"    Retrieved: {top['source']}  @  {top['timestamp']}")
        print(f"    Chunk:     {top['text'][:200]}...")


if __name__ == "__main__":
    main()
