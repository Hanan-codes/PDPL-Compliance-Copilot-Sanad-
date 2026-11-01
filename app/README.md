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
