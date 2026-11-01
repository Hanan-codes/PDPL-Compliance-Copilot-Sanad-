# SANAD (سَنَد): PDPL Compliance Copilot

**SANAD** is a bilingual (Arabic/English) research assistant for the Saudi Personal Data Protection Law (PDPL) and its Implementing Regulation. It answers questions **only from the official texts**, cites the article behind each answer, and abstains when the texts do not cover a question. It also analyses a privacy policy against 23 PDPL requirements and reports each one as *covered*, *partial*, or *missing*, with the supporting evidence.

> **Disclaimer:** SANAD is a decision-support tool. It is not legal advice. Consult a qualified legal professional for binding compliance decisions.

Built as an AIEP graduation project (Track B: PDPL Compliance Copilot).

---

## Features

- **Grounded Q&A.** Answers in Arabic or English, with inline article citations such as `[Source: pdpl | Article: 4]`. If no retrieved passage supports an answer, the system says the question is not covered.
- **Citation verification.** Every citation tag is checked against the retrieved passages. Invalid citations trigger a repair attempt, and the system abstains if repair fails.
- **Privacy policy gap analysis.** Splits a policy into paragraphs, classifies them with an OPP-115 clause classifier, maps each of the 23 requirements to candidate paragraphs, and asks an LLM for a verdict. Every quoted sentence must appear verbatim in the policy; otherwise it is removed and the verdict is downgraded.
- **Two LLM providers.** Google Gemini (cloud) or Ollama (local, e.g. `qwen2.5:7b`).
- **Exports.** JSON and PDF reports, including Arabic text rendered right-to-left.

## Architecture

    Question ─► Hybrid retrieval (E5 dense + FAISS, BM25 fusion)
              ─► Confidence gate (cross-encoder bge-reranker-v2-m3, τ = 0.5)
              ─► LLM answer with citation tags
              ─► Citation verifier ─► repair (≤ 2 retries) ─► answer or abstain

    Policy ─► Paragraph split ─► OPP-115 classifier (TF-IDF + Logistic Regression)
           ─► Requirement → category mapping (23 PDPL requirements)
           ─► LLM verdict (covered / partial / missing) with evidence
           ─► Verbatim quote check ─► JSON / PDF report

## Quick start

**Requirements:** Python 3.10+, about 8 GB RAM, internet access on first run (retrieval models are downloaded from Hugging Face), and one LLM provider (a Gemini API key, or Ollama).

    git clone https://github.com/<your-user>/sanad.git
    cd sanad
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

**Configure a provider.**

- *Gemini:* copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GEMINI_API_KEY`. Never commit this file.
- *Ollama:* install Ollama, then run `ollama pull qwen2.5:7b` and `ollama serve`.

**Run.**

    streamlit run app.py

In the sidebar, choose a provider and click **Connect / Initialise**. The first start loads the models and can take several minutes.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface (Q&A and policy analysis) |
| `rag_inference.py` | Retrieval, cited answers, citation verification |
| `gap_analyser.py` | Gap analysis: classification, mapping, verdicts, quote checks |
| `gap_analyser_inference.py` | Adapter between the UI and the gap analyser |
| `clause_classifier_inference.py` | OPP-115 paragraph classifier |
| `pdf_report.py` | PDF report export |
| `pdpl_requirements_checklist.json` | The 23 requirements and their category mapping |
| `e4_tfidf_lr_model.joblib` | Trained classifier (requires scikit-learn 1.8.0) |
| `data/stage1_outputs/` | Article-level and fixed-size legal chunks (AR and EN) |
| `data/stage2_outputs/` | FAISS indexes built from those chunks |
| `assets/` | Logo and bundled font |



## Responsible use

- Do not use SANAD to make legal decisions or to certify compliance.
- Do not paste personal data into the policy analyser beyond the policy text itself.
- Keep API keys in `.streamlit/secrets.toml` or the hosting platform's secrets store, never in code.

## Troubleshooting

- **`OSError ... 1455` on Windows:** the paging file is too small. Increase its size in System settings.
- **Model fails to load with a version error:** install `scikit-learn==1.8.0` exactly.
- **Every question returns "not covered":** check that `data/` contains all eight files and that the FAISS indexes match the chunks.

## Acknowledgements

- OPP-115 corpus (Usable Privacy Policy Project).
- Official PDPL and Implementing Regulation texts from SDAIA.
- Open-source models: multilingual E5, bge-reranker-v2-m3, Qwen2.5.
