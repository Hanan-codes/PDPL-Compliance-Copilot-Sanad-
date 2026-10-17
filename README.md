# PDPL Corpus Preparation Scripts

Scripts derived from the team's four existing Jupyter notebooks. The extraction and preprocessing are **not legal verification**; review markers are retained.

## Setup (from repository root)

```powershell
python -m pip install -r requirements.txt
```

## Rebuild from official source webpages

Run these commands from the project root in order:

```powershell
python data/download_pdpl.py
python data/download_sdaia_policies.py
python src/preprocess_pdpl.py
python src/build_rag_corpus.py
```

The last command produces `rag_output/pdpl_9_documents_merged.json`, `rag_output/pdpl_9_documents_rag_chunks.json`, and `rag_output/pdpl_9_sources_report.csv`. Original core JSON is read from `pdpl_corpus/data/extracted/`, supplementary JSON from `sdaia_six_corpus/data/extracted/`. The validation script outputs additional audit data in `PDPL_quality_outputs/`; the merge deliberately uses the originally extracted core JSON as in the notebook used for the 223-chunk run, to avoid silently changing the dataset.

**Source access caveat:** SDAIA pages may block automated requests or change their HTML. The original extraction scripts support local HTML/TXT fallbacks under each corpus's `data/manual_html/` directory. Rebuilding from a new download may give different counts from your previous 223 chunks. The scripts do not embed or redistribute the source texts.

**Validation caveat:** Records marked `needs_review` have not been legally authenticated. Inspect audit reports and document/article boundaries before using the corpus as authoritative. Check publisher licence terms before redistributing generated JSON.

## Already have extracted data?

Place the three `*_ar_articles_UNVERIFIED.json` files in `pdpl_corpus/data/extracted/` and the `sdaia_additional_ar_processed_UNVERIFIED.json` file in `sdaia_six_corpus/data/extracted/`. Then run only:

```powershell
python src/build_rag_corpus.py
```

For team members, `rag_output/pdpl_9_documents_rag_chunks.json` is the dataset consumed by the RAG pipeline: `clean_text` is retrieval text, and `document_id`, `article_number`, `source_url` are citation metadata. This package contains scripts, not the redistributed corpus.
