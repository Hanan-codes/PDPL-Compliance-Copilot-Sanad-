# -*- coding: utf-8 -*-
"""
gap_analyser_inference.py — نقطة دخول إنفرنس واحدة للـ Gap Analyser (للتطبيق).
يكمّل التناظر مع rag_inference.query_copilot و clause_classifier_inference.classify_clause.

ياخذ سياسة جهة (نص خام / قائمة فقرات / مسار ملف) ويعيد تقرير فجوات منظّم بنداء واحد:

    import gap_analyser_inference as gi
    report = gi.analyse_policy_text(open("policy.txt").read(), entity_name="شركة س", provider="ollama")
    print(report["summary"], report["coverage_pct"])

يهيّئ الأنبوب تلقائيًا عند أول نداء (قائمة + مُصنّف + استرجاع RAG).
"""
import os, re, json
import gap_analyser as ga

_INITED = False


def _ensure_init(provider: str = "ollama", use_rag: bool = True,
                 checklist: str = "pdpl_requirements_checklist.json",
                 classifier_model: str = None):
    """تهيئة كسولة مرة واحدة."""
    global _INITED
    if not _INITED:
        cm = classifier_model or os.getenv("CLAUSE_MODEL_PATH", "e4_tfidf_lr_model.joblib")
        ga.init(checklist=checklist, classifier_model=cm, use_rag=use_rag, rag_provider=provider)
        _INITED = True


def split_clauses(text: str) -> list:
    """يقسّم نص سياسة خام إلى فقرات/بنود قابلة للتصنيف.
    يفصل على الأسطر الفارغة، والبنود المرقّمة، والعناوين — ثم يستبعد الشظايا القصيرة جدًا."""
    text = (text or "").replace("\r", "\n")
    # افصل على سطر فارغ أو بداية بند مرقّم (1. / 1) / •) أو عنوان قصير متبوع بنقطتين
    parts = re.split(r"\n\s*\n+|\n(?=\s*(?:\d{1,2}[\.\)]|[-•*])\s)", text)
    clauses = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip(" \t-•*·")
        if len(p) >= 25:                         # استبعاد العناوين/الشظايا
            clauses.append(p)
    return clauses


def analyse_policy_text(policy, entity_name: str = "", provider: str = "ollama",
                        use_rag: bool = True) -> dict:
    """
    نقطة الدخول الموحّدة: سياسة جهة ⟵ تقرير فجوات (dict).
    policy يقبل: نصًّا خامًا (str) → يُقسّم تلقائيًا، أو قائمة فقرات (list) تُستخدم كما هي.
    """
    _ensure_init(provider=provider, use_rag=use_rag)
    clauses = policy if isinstance(policy, list) else split_clauses(str(policy))
    return ga.analyse_policy(clauses, entity_name=entity_name)


def analyse_policy_file(path: str, entity_name: str = "", provider: str = "ollama",
                        use_rag: bool = True) -> dict:
    """يحلّل سياسة من ملف: .json (بصيغة {clauses, meta}) أو .txt/.md (نص خام)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if path.lower().endswith(".json"):
        obj = json.loads(raw)
        clauses = obj.get("clauses") or []
        entity_name = entity_name or obj.get("meta", {}).get("entity", "")
        return analyse_policy_text(clauses, entity_name=entity_name, provider=provider, use_rag=use_rag)
    return analyse_policy_text(raw, entity_name=entity_name or os.path.basename(path),
                               provider=provider, use_rag=use_rag)


def analyse_policy_json(policy, entity_name: str = "", provider: str = "ollama",
                        use_rag: bool = True) -> str:
    """نفس analyse_policy_text لكن يعيد JSON string (ملائم للتطبيق/الـAPI)."""
    return json.dumps(analyse_policy_text(policy, entity_name, provider, use_rag),
                      ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # عرض سريع بلا LLM على نص سياسة خام
    _ensure_init(use_rag=False)
    demo = ("We collect your name and email when you register.\n\n"
            "We may share data with third-party providers.\n\n"
            "You can delete your account at any time.\n\n"
            "We retain data for 12 months after closure.")
    rep = analyse_policy_text(demo, entity_name="DemoText", use_rag=False)
    print("clauses:", rep["n_clauses"], "| summary:", rep["summary"])
