
# -*- coding: utf-8 -*-
"""
gap_analyser.py — PDPL Compliance Gap Analyzer.

Components:
1. PDPL requirements checklist.
2. Explicit mapping between classifier categories and requirements.
3. Structured gap report with policy evidence and legal references.
"""

import os
import json
import re

import clause_classifier_inference as clause_clf


_CHECKLIST = None
_USE_RAG = False
_RAG = None
_BASIS_CACHE = {}


# ---------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------
def init(
    checklist="pdpl_requirements_checklist.json",
    classifier_model=None,
    use_rag=True,
    rag_provider="ollama",
):
    """Load the checklist, classifier, and optional RAG pipeline."""
    global _CHECKLIST, _USE_RAG, _RAG, _BASIS_CACHE

    with open(checklist, encoding="utf-8") as f:
        _CHECKLIST = json.load(f)

    clf_info = (
        clause_clf.load_classifier(classifier_model)
        if classifier_model
        else clause_clf.load_classifier()
    )

    _USE_RAG = use_rag
    _RAG = None
    _BASIS_CACHE = {}

    rag_info = None

    if use_rag:
        import rag_inference as rag

        _RAG = rag
        rag_info = rag.load_pipeline(rag_provider)

    return {
        "requirements": len(_CHECKLIST["requirements"]),
        "classifier_labels": clf_info["labels"],
        "rag": rag_info,
    }


# ---------------------------------------------------------------
# Official PDPL requirement basis
# ---------------------------------------------------------------
def _pdpl_basis(req):
    """
    Retrieve the legal basis for a requirement.
    Falls back to the checklist if RAG is unavailable.
    """
    fallback = {
        "text": req["requirement_ar"],
        "citation": None,
        "articles": {
            "pdpl": req.get("pdpl_articles", []),
            "impl": req.get("impl_articles", []),
        },
        "source": "checklist",
    }

    if not (_USE_RAG and _RAG):
        return fallback

    req_id = req["id"]

    if req_id in _BASIS_CACHE:
        return _BASIS_CACHE[req_id]

    try:
        result = _RAG.query_copilot(req["check_question"])

        basis = {
            "text": result.get("answer", ""),
            "citation": result.get("citations", []),
            "abstained": result.get("abstained"),
            "articles": {
                "pdpl": req.get("pdpl_articles", []),
                "impl": req.get("impl_articles", []),
            },
            "source": "rag",
        }

    except Exception as exc:
        basis = {
            **fallback,
            "source": "checklist_fallback",
            "error": f"{type(exc).__name__}: {str(exc)[:120]}",
        }

    _BASIS_CACHE[req_id] = basis
    return basis


# ---------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------
def _norm(text):
    """Normalize whitespace for quote verification."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _quote_in(quote, paragraphs):
    """
    Check whether the evidence quote appears in the actual policy.

    Exact matching is intentionally used to avoid accepting invented
    or paraphrased evidence as a direct quotation.
    """
    q = _norm(quote).strip('"“”')

    if len(q) < 15:
        return False

    return any(q in _norm(paragraph) for paragraph in paragraphs)


_NON_LATIN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uac00-\ud7af\u0400-\u04ff]"
)


def _clean_rationale(text):
    """Replace unsupported non-Arabic/non-English rationale text."""
    if _NON_LATIN_RE.search(text or ""):
        return (
            "[The rationale contains unsupported characters; "
            "manual review is recommended.]"
        )

    return text


# ---------------------------------------------------------------
# LLM compliance judgment
# ---------------------------------------------------------------
_JUDGE_SYS = """
You are a careful PDPL privacy-policy compliance auditor.

Your task is to evaluate whether the ENTITY'S PRIVACY POLICY satisfies
the stated compliance requirement.

Return exactly one valid JSON object with these keys:
{
  "status": "covered|partial|missing",
  "evidence": "an exact quotation from the supplied policy or an empty string",
  "rationale": "one concise sentence in English"
}

Classification rules:

1. covered:
   The policy explicitly addresses all material elements of the
   requirement. The supplied policy text must contain sufficient
   evidence to support this judgment.

2. partial:
   The policy explicitly addresses at least one material element,
   but other material elements are missing, incomplete, or unclear.

3. missing:
   None of the supplied policy paragraphs meaningfully addresses
   the requirement.

Important restrictions:
- The legal basis is not evidence that the entity complies.
- Do not mark a requirement covered just because a PDPL article exists.
- Do not infer compliance from generic or unrelated statements.
- Do not invent policy statements.
- Evidence must be copied verbatim from the supplied policy paragraphs.
- If no relevant evidence exists, return an empty evidence string.
- Judge the policy against the stated requirement, not against a
  generic description of PDPL.
- Return JSON only, without Markdown fences.
- Write the rationale in English only.
"""


def _judge(req_text, candidates):
    """
    Evaluate policy coverage using the LLM.

    Without an available LLM, return needs_review rather than claiming
    compliance based solely on the presence of a candidate paragraph.
    """
    if not candidates:
        return {
            "status": "missing",
            "evidence": "",
            "rationale": "No candidate policy paragraphs were available.",
        }

    if not (
        _USE_RAG
        and _RAG
        and getattr(_RAG, "LLM", None)
    ):
        return {
            "status": "needs_review",
            "evidence": "",
            "rationale": (
                "Candidate paragraphs exist, but an LLM judgment is "
                "unavailable. Manual review is required."
            ),
        }

    user_prompt = (
        "### REQUIREMENT TO EVALUATE\n"
        f"{req_text}\n\n"
        "### ENTITY PRIVACY POLICY PARAGRAPHS\n"
        + "\n---\n".join(str(c)[:1500] for c in candidates[:60])
        + "\n\nReturn the required JSON object only."
    )

    try:
        from rag_inference import generate_with_retry

        raw = generate_with_retry(
            _RAG.LLM,
            _JUDGE_SYS,
            user_prompt,
        )

        # Extract a JSON object, including multiline JSON responses.
        match = re.search(r"\{.*\}", raw, re.S)
        result = json.loads(match.group(0)) if match else {}

        status = result.get("status", "needs_review")

        if status not in ("covered", "partial", "missing"):
            status = "needs_review"

        evidence = str(result.get("evidence") or "").strip()[:500]
        rationale = str(result.get("rationale") or "").strip()[:300]
        rationale = _clean_rationale(rationale)

        return {
            "status": status,
            "evidence": evidence,
            "rationale": rationale,
        }

    except Exception as exc:
        return {
            "status": "needs_review",
            "evidence": "",
            "rationale": (
                f"LLM judgment failed ({type(exc).__name__}); "
                "manual review is required."
            ),
        }


# ---------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------
def analyse_policy(clauses, entity_name=""):
    """
    Analyze an entity privacy policy against all checklist requirements.

    Each requirement is evaluated using its checklist description.
    The retrieved legal basis is retained separately for reporting.
    """
    if _CHECKLIST is None:
        raise RuntimeError("Call init() before analyse_policy().")

    clauses = [
        str(clause).strip()
        for clause in (clauses or [])
        if str(clause).strip()
    ]

    # 1. Classify policy paragraphs.
    clf_results = clause_clf.classify_clauses(clauses)

    by_cat = {}

    for classified in clf_results:
        for label in classified.get("labels", []):
            by_cat.setdefault(label, []).append(
                classified.get("text", "")
            )

    # 2. Evaluate each requirement.
    items = []
    counts = {
        "covered": 0,
        "partial": 0,
        "missing": 0,
        "needs_review": 0,
    }

    llm_available = bool(
        _USE_RAG
        and _RAG
        and getattr(_RAG, "LLM", None)
    )

    judge_mode = (
        "llm"
        if llm_available
        else "manual_review (LLM unavailable)"
    )

    for req in _CHECKLIST["requirements"]:
        category = req.get("classifier_category")

        if category:
            candidates = by_cat.get(category, [])

            # If the classifier did not identify any paragraph for
            # this category, examine the full policy to reduce misses.
            if not candidates:
                candidates = clauses
                mapped_by = (
                    f"classifier:{category} "
                    "(no matching label; examined all paragraphs)"
                )
            else:
                mapped_by = f"classifier:{category}"

        else:
            # Requirements without a classifier category require
            # examining the complete policy.
            candidates = clauses
            mapped_by = "direct_review (no classifier category)"

        # Retrieve legal references separately from the compliance
        # criterion used by the judgment.
        basis = _pdpl_basis(req)

        # IMPORTANT: Judge against the checklist requirement,
        # not the potentially broad answer returned by RAG.
        verdict = _judge(
            req.get("requirement_ar", ""),
            candidates,
        )

        evidence = verdict.get("evidence", "")
        verified = _quote_in(evidence, clauses) if evidence else False

        # A covered result must include a verifiable policy quotation.
        if verdict["status"] == "covered" and not verified:
            verdict["status"] = "needs_review"
            verdict["rationale"] = (
                "The model did not provide a verifiable exact policy "
                "quotation; manual review is required."
            )

        # An unverified quotation must never be presented as evidence.
        if evidence and not verified:
            evidence = ""
            verdict["evidence"] = ""
            if verdict["status"] == "covered":
                verdict["status"] = "needs_review"

        verdict["evidence_verified"] = verified

        status = verdict["status"]
        counts[status] += 1

        items.append({
            "requirement_id": req.get("id"),
            "requirement": req.get("requirement_ar", ""),
            "requirement_en": req.get("requirement_en", ""),
            "mapped_by": mapped_by,

            "pdpl_basis": {
                "articles": basis.get("articles", {}),
                "citation": basis.get("citation"),
                "text": str(basis.get("text") or "")[:400],
                "source": basis.get("source", "checklist"),
            },

            "entity_evidence": evidence,

            # Preserve the full policy for missing requirements.
            "entity_policy_full": (
                "\n\n".join(clauses)
                if status == "missing"
                else ""
            ),

            "evidence_verified": verified,
            "status": status,
            "rationale": verdict.get("rationale", ""),
        })

    return {
        "entity": entity_name,
        "judge_mode": judge_mode,
        "n_clauses": len(clauses),
        "n_requirements": len(items),
        "summary": counts,

        # Strict coverage: only verified covered requirements count.
        "coverage_pct": round(
            100 * counts["covered"] / max(len(items), 1),
            1,
        ),

        "clause_classification": clf_results,
        "gap_report": items,
    }


# ---------------------------------------------------------------
# Save report
# ---------------------------------------------------------------
def save_report(report, path="gap_report.json"):
    """Save the gap report as UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return path


# ---------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------
if __name__ == "__main__":
    info = init(use_rag=False)

    print(
        "Initialized:",
        info["requirements"],
        "requirements |",
        len(info["classifier_labels"]),
        "classifier labels",
    )

    demo_policy = [
        "We collect your name, email, and device information "
        "when you use our app.",
        "We may share aggregated data with third-party analytics providers.",
        "You can request access to or deletion of your personal data "
        "at any time.",
        "We retain personal data for as long as your account is active.",
    ]

    report = analyse_policy(
        demo_policy,
        entity_name="DemoCo",
    )

    print("Summary:", report["summary"])
    print("Coverage:", report["coverage_pct"], "%")
