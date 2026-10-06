"""
Step 2: Turn each abstract into an embedding (a vector that captures its meaning)
and save everything needed for semantic search.

Usage:
  python build_index.py
  python build_index.py --model BAAI/bge-small-en-v1.5 --input data/abstracts.jsonl --out index

Output (in index/):
  embeddings.npy  - one normalized vector per paper
  papers.jsonl    - paper metadata, in the same order as the vectors
  meta.json       - which model built the index, and when
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

# bge-small reads up to 512 tokens, so most full abstracts fit without being cut off.
# (all-MiniLM-L6-v2, a common default, stops at 256 tokens and would truncate many abstracts.)
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_papers(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def paper_text(paper):
    """What gets embedded: the title carries a lot of signal, so include it."""
    return f"{paper['title']}\n{paper['abstract']}"


def main():
    p = argparse.ArgumentParser(description="Build a semantic search index from abstracts")
    p.add_argument("--input", default="data/abstracts.jsonl")
    p.add_argument("--out", default="index")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    papers = load_papers(args.input)
    print(f"Loaded {len(papers)} papers from {args.input}")

    print(f"Loading model {args.model} (downloads once, then cached)...")
    model = SentenceTransformer(args.model)

    texts = [paper_text(pp) for pp in papers]

    # Check how many papers are longer than the model can read, so we know
    # whether truncation is affecting results (worth reporting in the README).
    max_len = model.max_seq_length
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    truncated = sum(1 for n in lengths if n > max_len)
    print(
        f"Token lengths: median {int(np.median(lengths))}, max {max(lengths)}. "
        f"{truncated}/{len(texts)} papers exceed the model's {max_len}-token limit and will be truncated."
    )

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,  # unit length, so a dot product = cosine similarity
        show_progress_bar=True,
    )

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "embeddings.npy"), embeddings.astype(np.float32))
    with open(os.path.join(args.out, "papers.jsonl"), "w", encoding="utf-8") as f:
        for pp in papers:
            f.write(json.dumps(pp, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "num_papers": len(papers),
                "dim": int(embeddings.shape[1]),
                "truncated": truncated,
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    print(f"Saved index for {len(papers)} papers ({embeddings.shape[1]}-dim vectors) to {args.out}/")


if __name__ == "__main__":
    main()
