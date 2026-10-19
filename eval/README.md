# Evaluation

This folder contains the evaluation datasets, scripts, and results for the **PDPL Compliance Copilot (Track B)**.

## Evaluation Components

### 1. Gold Q&A Dataset
- A gold evaluation dataset containing **60 Arabic questions**.
- **50 answerable questions** and **10 unanswerable questions** to evaluate the system's ability to abstain when sufficient evidence is unavailable.
- Reference answers, source documents, supporting evidence, and gold article or section references.
- Article-level mappings to the PDPL corpus where applicable.
- Manual review of question quality, evidence alignment, and legal references.
- Conversion of the reviewed Excel dataset into UTF-8 JSON for evaluation.

### 2. Retrieval Evaluation
- Evaluate the ability of the retrieval system to identify relevant legal articles and passages.
- Compare lexical retrieval (BM25), dense retrieval, and hybrid retrieval.
- Evaluate retrieval using metrics such as **Recall@K, Precision@K, MRR, and Hit@K**.
- Assess article-level citation accuracy and supporting evidence relevance.

### 3. Answer Generation Evaluation
- Evaluate the correctness and completeness of generated answers against reference answers.
- Assess whether answers are grounded in retrieved legal evidence.
- Evaluate citation accuracy and unsupported claims.
- Test abstention behavior for unanswerable or out-of-corpus questions.

### 4. Privacy Policy Gap Analysis
- Evaluate the system's ability to identify potential PDPL compliance gaps in privacy policies.
- Use team-annotated privacy policies from applications or websites operating in Saudi Arabia.
- Compare predicted compliance findings against manually annotated references.
- Evaluate gap detection precision, recall, and F1-score.

## Dataset and Scripts

| File | Description |
|---|---|
| `PDPL_Gold_QA_60_Annotated.xlsx` | Working gold Q&A dataset with evidence mappings and review information. |
| `convert_gold_qa_excel_to_json.py` | Converts the Excel dataset into UTF-8 JSON. |
| `gold_qa.json` | Machine-readable Q&A dataset used for evaluation after conversion. |



