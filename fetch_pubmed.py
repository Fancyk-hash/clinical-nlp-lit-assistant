"""
Step 1: Fetch PubMed abstracts for a search query and save them as JSONL.

Uses NCBI's free E-utilities API:
  - esearch: find PMIDs matching a query
  - efetch:  download full records (title, abstract, metadata) as XML

Usage:
  python fetch_pubmed.py --query "social determinants of health clinical notes natural language processing" --max 500
  python fetch_pubmed.py --query "..." --max 1000 --email you@example.com --api-key YOUR_KEY

Output: data/abstracts.jsonl (one paper per line)
"""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 200  # efetch handles up to ~200 IDs per request comfortably


def esearch(query, max_results, email=None, api_key=None, min_year=None):
    """Return a list of PMIDs matching the query."""
    term = query
    if min_year:
        term = f"({query}) AND {min_year}:3000[dp]"
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    result = r.json()["esearchresult"]
    print(f"PubMed found {result['count']} matches; fetching up to {max_results}.")
    return result["idlist"]


def efetch(pmids, email=None, api_key=None, retries=3):
    """Fetch PubMed XML for a batch of PMIDs, with simple retry on failure."""
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    for attempt in range(retries):
        try:
            r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  efetch failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("efetch failed after retries")


def _text(el):
    """All text inside an element, including nested tags like <i> or <sup>."""
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_articles(xml_text):
    """Turn PubMed XML into a list of clean dicts. Skips papers with no abstract."""
    root = ET.fromstring(xml_text)
    papers = []
    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//PMID"))
        title = _text(art.find(".//ArticleTitle"))

        # Structured abstracts have labeled sections (BACKGROUND, METHODS, ...)
        sections = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            body = _text(ab)
            if body:
                sections.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(sections)
        if not abstract:
            continue

        year = _text(art.find(".//JournalIssue/PubDate/Year")) or _text(
            art.find(".//JournalIssue/PubDate/MedlineDate")
        )[:4]
        journal = _text(art.find(".//Journal/Title"))

        authors = []
        for a in art.findall(".//AuthorList/Author"):
            last, fore = _text(a.find("LastName")), _text(a.find("ForeName"))
            if last:
                authors.append(f"{fore} {last}".strip())

        keywords = [_text(k) for k in art.findall(".//KeywordList/Keyword") if _text(k)]

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "authors": authors,
                "keywords": keywords,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return papers


def main():
    p = argparse.ArgumentParser(description="Fetch PubMed abstracts to JSONL")
    p.add_argument("--query", required=True, help="PubMed search query")
    p.add_argument("--max", type=int, default=500, help="Max papers to fetch")
    p.add_argument("--min-year", type=int, default=None, help="Only papers from this year on")
    p.add_argument("--email", default=os.getenv("NCBI_EMAIL"), help="Your email (NCBI asks for it)")
    p.add_argument("--api-key", default=os.getenv("NCBI_API_KEY"), help="Optional NCBI API key")
    p.add_argument("--out", default="data/abstracts.jsonl")
    args = p.parse_args()

    # NCBI limit: 3 requests/sec without a key, 10/sec with one
    delay = 0.11 if args.api_key else 0.34

    pmids = esearch(args.query, args.max, args.email, args.api_key, args.min_year)
    time.sleep(delay)

    all_papers = []
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i : i + BATCH_SIZE]
        xml_text = efetch(batch, args.email, args.api_key)
        papers = parse_articles(xml_text)
        all_papers.extend(papers)
        print(f"  batch {i // BATCH_SIZE + 1}: {len(papers)}/{len(batch)} had abstracts")
        time.sleep(delay)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_papers)} papers to {args.out}")


if __name__ == "__main__":
    main()
