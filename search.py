"""
Search the index by meaning. Prints the most relevant papers for a question.

Usage:
  python search.py "how is housing instability detected in clinical notes"
  python search.py "large language models for social needs" --k 10
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

# bge models are trained to expect this prefix on search queries (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class PaperIndex:
    def __init__(self, index_dir="index"):
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "papers.jsonl"), encoding="utf-8") as f:
            self.papers = [json.loads(line) for line in f if line.strip()]
        self.model = SentenceTransformer(self.meta["model"])

    def search(self, query, k=5):
        if "bge" in self.meta["model"].lower():
            query = BGE_QUERY_PREFIX + query
        q = self.model.encode([query], normalize_embeddings=True)[0]
        # With a few hundred papers, comparing against every vector is instant;
        # a vector database (FAISS, Chroma) only pays off at much larger scale.
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.papers[i]) for i in top]


def main():
    p = argparse.ArgumentParser(description="Semantic search over indexed abstracts")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index", default="index")
    args = p.parse_args()

    index = PaperIndex(args.index)
    for rank, (score, paper) in enumerate(index.search(args.query, args.k), 1):
        print(f"\n{rank}. [{score:.3f}] {paper['title']}")
        print(f"   {paper['year']} | {paper['journal']}")
        print(f"   {paper['url']}")


if __name__ == "__main__":
    main()
