# -*- coding: utf-8 -*-
"""
clause_classifier_inference.py — واجهة إنفرنس لمُصنّف بنود السياسات (E4).
يحمّل نموذج TF-IDF + One-vs-Rest LogisticRegression المُدرَّب (OPP-115، متعدد التصنيفات)
ويطبّق عتبة لكل فئة (per-class thresholds) المضبوطة على التحقق.

    import clause_classifier_inference as clf
    clf.load_classifier("e4_output/e4_tfidf_lr_model.joblib")
    clf.classify_clause("We retain your data for 90 days.")
    # -> {"text":..., "labels":["data_retention"], "scores":{...}}
"""
import os
import numpy as np
import joblib

_MODEL = None
_DEFAULT_PATH = os.getenv("CLAUSE_MODEL_PATH", "e4_output/e4_tfidf_lr_model.joblib")


def load_classifier(path: str = None) -> dict:
    """يحمّل النموذج مرة واحدة. يعيد قائمة الفئات."""
    global _MODEL
    _MODEL = joblib.load(path or _DEFAULT_PATH)
    assert all(k in _MODEL for k in ("vectorizer", "model", "label_cols", "thresholds")), \
        "ملف النموذج يجب أن يحوي: vectorizer, model, label_cols, thresholds"
    return {"labels": list(_MODEL["label_cols"]), "n_labels": len(_MODEL["label_cols"])}


def labels() -> list:
    if _MODEL is None:
        load_classifier()
    return list(_MODEL["label_cols"])


def classify_clause(text: str) -> dict:
    """يصنّف فقرة واحدة (متعدد التصنيفات) بعتبة كل فئة. يعيد الفئات المتوقّعة + كل الاحتمالات."""
    if _MODEL is None:
        load_classifier()
    vec, model, cols = _MODEL["vectorizer"], _MODEL["model"], list(_MODEL["label_cols"])
    thr = np.asarray(_MODEL["thresholds"], dtype=float)
    proba = model.predict_proba(vec.transform([str(text or "")]))[0]
    predicted = [cols[j] for j in range(len(cols)) if proba[j] >= thr[j]]
    scores = {cols[j]: round(float(proba[j]), 4) for j in range(len(cols))}
    return {"text": text, "labels": predicted, "scores": scores}


def classify_clauses(texts: list) -> list:
    """دفعة من الفقرات."""
    if _MODEL is None:
        load_classifier()
    vec, model, cols = _MODEL["vectorizer"], _MODEL["model"], list(_MODEL["label_cols"])
    thr = np.asarray(_MODEL["thresholds"], dtype=float)
    probas = model.predict_proba(vec.transform([str(t or "") for t in texts]))
    out = []
    for i, t in enumerate(texts):
        p = probas[i]
        out.append({"text": t,
                    "labels": [cols[j] for j in range(len(cols)) if p[j] >= thr[j]],
                    "scores": {cols[j]: round(float(p[j]), 4) for j in range(len(cols))}})
    return out


if __name__ == "__main__":
    info = load_classifier()
    print("labels:", info["labels"])
    for s in ["We may share your data with advertising partners.",
              "You can delete your account and personal data anytime."]:
        print(classify_clause(s))
