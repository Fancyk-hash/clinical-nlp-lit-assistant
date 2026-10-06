"""
Step 3: Ask a question, get a written answer grounded in the indexed papers,
with citations that are checked against the papers actually retrieved.

Setup: put your OpenAI key in a file named .env in the project folder:
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini        (optional; change if this model is retired)

Usage:
  python ask.py "How is housing instability detected in clinical notes?"
  python ask.py "How have methods changed over time?" --k 8
"""

import argparse
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from search import PaperIndex

SYSTEM_PROMPT = """You are a research assistant summarizing biomedical informatics literature.
Answer the question using ONLY the numbered sources provided.
Rules:
- Cite every claim with the source's PMID in square brackets, e.g. [PMID 12345678].
- Do not use outside knowledge. If the sources do not answer the question, say so plainly.
- When describing methods or results, mention the publication year, since methods in this field changed quickly.
- Be concise: 1-3 short paragraphs."""


def build_context(results):
    """Format retrieved papers as numbered sources for the prompt."""
    blocks = []
    for i, (score, p) in enumerate(results, 1):
        blocks.append(
            f"Source {i} [PMID {p['pmid']}] ({p['year']}, {p['journal']})\n"
            f"Title: {p['title']}\nAbstract: {p['abstract']}"
        )
    return "\n\n".join(blocks)


def check_citations(answer, results):
    """Return (valid, invalid) cited PMIDs. Invalid ones were not in the retrieved sources,
    which means the model invented or misremembered a citation."""
    cited = set(re.findall(r"PMID[:\s]*(\d+)", answer))
    retrieved = {p["pmid"] for _, p in results}
    return sorted(cited & retrieved), sorted(cited - retrieved)


def answer_question(index, client, model, question, k=6):
    results = index.search(question, k)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # low randomness: we want faithful summaries, not creativity
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"},
        ],
    )
    answer = response.choices[0].message.content
    return answer, results


def main():
    p = argparse.ArgumentParser(description="Ask a question answered from the indexed papers")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=6, help="How many papers to retrieve")
    p.add_argument("--index", default="index")
    args = p.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("No OPENAI_API_KEY found. Add it to a .env file in the project folder.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    index = PaperIndex(args.index)
    client = OpenAI()
    answer, results = answer_question(index, client, model, args.question, args.k)

    print("\n" + answer + "\n")
    print("Sources retrieved:")
    for score, paper in results:
        print(f"  [PMID {paper['pmid']}] ({paper['year']}) {paper['title']}  (similarity {score:.3f})")

    valid, invalid = check_citations(answer, results)
    print(f"\nCitation check: {len(valid)} citations verified against retrieved sources.")
    if invalid:
        print(f"WARNING: cited PMIDs not among retrieved sources: {', '.join(invalid)}")


if __name__ == "__main__":
    main()
