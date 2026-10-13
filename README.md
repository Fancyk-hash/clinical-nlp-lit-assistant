# Clinical NLP Literature Assistant

A retrieval-augmented question-answering tool over the research literature on **extracting social determinants of health (SDOH) from clinical notes**. Ask a question in plain English; the tool retrieves the most relevant PubMed abstracts and generates a short answer in which every claim cites a PMID, and every cited PMID is checked against the papers actually retrieved.

> Example: *"How is housing instability detected in clinical notes?"* returns a written answer citing work such as ReHouSED (PMID 34474188) and an NLP vs. structured-data comparison (PMID 37293237), followed by the retrieved sources and a citation check.

## Why this topic

Social factors such as housing instability, food insecurity, and transportation barriers strongly affect health outcomes, but they are rarely captured in structured EHR fields like diagnosis codes. They usually appear only in free-text clinical notes ("staying with her sister since the eviction"). A growing body of NLP research tries to extract this information automatically. This project makes that literature searchable and summarizable, with an emphasis on citation faithfulness, a central concern for AI used in clinical contexts.

## How it works

```
PubMed  ──fetch──▶  abstracts.jsonl  ──embed──▶  vector index  ──retrieve──▶  top-k abstracts  ──LLM──▶  cited answer  ──verify──▶  citation check
```

| Step | Script | What it does |
|---|---|---|
| 1. Collect | `fetch_pubmed.py` | Searches PubMed via NCBI E-utilities and saves title, abstract, authors, journal, year, keywords, and URL for each paper |
| 2. Index | `build_index.py` | Embeds each title + abstract with `BAAI/bge-small-en-v1.5` (384-dim, normalized) and reports how many abstracts exceed the model's token limit |
| 3. Search | `search.py` | Semantic search: ranks papers by cosine similarity to the question |
| 4. Answer | `ask.py` | Sends the top-k abstracts to an LLM with instructions to answer only from the sources and cite PMIDs, then verifies every cited PMID was among the retrieved papers |

## Corpus

Built from the PubMed query in [`queries/sdoh_nlp.txt`](queries/sdoh_nlp.txt), which combines three concept groups with OR-ed synonyms:

- **Social determinants:** SDOH, social needs, social risk, housing instability, homelessness, food insecurity, financial strain, social isolation, transportation
- **Methods:** natural language processing, NLP, text mining, information extraction, (large) language models, BERT, transformers
- **Data source:** clinical notes, clinical text, EHR/EMR, medical records, clinical documentation

| | |
|---|---|
| Date range | 2015 onward, to span the rule-based, transformer, and LLM eras |
| PubMed matches | 197 (fetched October 2026) |
| Papers with abstracts | 193 |
| Papers from 2015–2017 | 8 |

Almost all of the literature dates from 2018 onward. The small corpus size reflects how young and specialized this research area is.

## Design decisions

**Embedding model chosen for abstract length.** A common default, `all-MiniLM-L6-v2`, truncates input at 256 tokens. The median abstract in this corpus is **390 tokens** (max 823), so that model would have cut off more than half of the papers, often losing the results and conclusions. `bge-small-en-v1.5` reads up to 512 tokens; with it, **27 of 193 papers (14%)** are truncated, and only at the end. `build_index.py` measures and reports this rather than assuming it.

**No vector database.** With fewer than 200 documents, a single matrix-vector product against all embeddings takes well under a millisecond. FAISS or Chroma would add complexity without benefit at this scale.

**Grounded generation.** The LLM is instructed to use only the provided sources, cite every claim, say so when the sources don't answer the question, and mention publication years (methods in this field changed quickly). Temperature is set low (0.2) to favor faithful summaries.

**Citation verification.** LLMs can invent or misattribute citations. After generation, `ask.py` extracts every cited PMID and checks it against the retrieved set, flagging any that weren't retrieved.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project folder (it is git-ignored):

```
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini      # optional
```

## Usage

```bash
# 1. Fetch papers (NCBI asks for a contact email)
python fetch_pubmed.py --query-file queries/sdoh_nlp.txt --max 1000 --min-year 2015 --email you@example.com

# 2. Build the embedding index
python build_index.py

# 3. Semantic search only
python search.py "how is housing instability detected in clinical notes"

# 4. Ask a question with a cited answer
python ask.py "What are the main challenges in extracting social needs from clinical notes?"
```

The query is read from a file rather than typed on the command line because some shells (notably Windows PowerShell) strip inner quotes from arguments, which breaks PubMed phrase searches. Keeping the query in the repository also documents exactly how the corpus was built.

## Preliminary observations

From three test questions (housing instability detection, how methods changed over time, and main challenges):

- **Citation validity:** 13 of 13 cited PMIDs were among the retrieved sources; no invented citations.
- **Retrieval relevance:** for the housing question, all top-5 results were directly on topic, including a paper using the phrase "residential instability," showing that semantic search matches meaning beyond exact wording.

These are spot checks, not a systematic evaluation.

## Limitations

1. **Citation check ≠ claim support.** Verification confirms that a cited paper was retrieved, not that it supports the specific claim. Some generated sentences contained general statements not clearly tied to any source. Checking claim-level support is a harder, open problem.
2. **Duplicate papers.** The corpus contains at least one scoping review indexed twice (PMIDs 38370703 and 39478779), which used two of the six retrieval slots for the same content.
3. **Similarity ranking is not temporal coverage.** For "how have methods changed over time," the six most similar papers skipped the LLM-era work present in the corpus, so the answer's timeline had gaps.
4. **Incomplete use of sources.** In 2 of 3 answers, the top-ranked retrieved paper was not cited.
5. **Inconsistent instruction-following.** The prompt asks for publication years; one of three answers mostly omitted them.
6. **Abstracts only.** Full texts are not indexed, and 14% of abstracts are truncated at the end.

## Next steps

- Remove near-duplicate papers by title similarity during fetching
- Year-stratified retrieval for questions about change over time
- A small labeled evaluation set (questions with known relevant papers) to measure retrieval precision/recall, and to test whether chunking long abstracts improves results
- Claim-level faithfulness checks
- A simple web interface that loads the model once, so each question returns in seconds

## Project structure

```
clinical-nlp-lit-assistant/
├── fetch_pubmed.py      # Step 1: PubMed → data/abstracts.jsonl
├── build_index.py       # Step 2: abstracts → index/ (embeddings + metadata)
├── search.py            # Semantic search over the index
├── ask.py               # Retrieval + LLM answer + citation check
├── queries/
│   └── sdoh_nlp.txt     # The PubMed query used to build the corpus
├── requirements.txt
└── .gitignore           # data/, index/, .env, .venv/ are not committed
```

`data/` and `index/` are not committed; anyone can regenerate them with steps 1 and 2.

## Author

Fancy Kiruok, MS Data Analytics, The Catholic University of America