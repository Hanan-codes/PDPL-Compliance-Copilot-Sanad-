"""Original conservative PDPL validation pipeline; does not imply legal verification."""
from pathlib import Path
import json, re, csv, hashlib, unicodedata, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

# Optional: uncomment and change to the exact folder on your computer.
# INPUT_DIR_OVERRIDE = r"C:\Users\YourName\Desktop\PDPL_Project\pdpl_corpus\data\extracted"
INPUT_DIR_OVERRIDE = None

EXPECTED_FILES = {
    "pdpl": "pdpl_ar_articles_UNVERIFIED.json",
    "implementing_regulation": "implementing_regulation_ar_articles_UNVERIFIED.json",
    "transfer_regulation": "transfer_regulation_ar_articles_UNVERIFIED.json",
}
EXPECTED = {"pdpl": 43, "implementing_regulation": 38, "transfer_regulation": 9}

SEARCH_DIRS = ([Path(INPUT_DIR_OVERRIDE).expanduser()] if INPUT_DIR_OVERRIDE else [
    ROOT / "pdpl_corpus" / "data" / "extracted",
    ROOT / "extracted",
    ROOT / "pdpl_input",
    ROOT,
])

# Choose the location with the greatest number of matching files; zero matches is an error.
counts = {p: sum((p / name).is_file() for name in EXPECTED_FILES.values())
          for p in SEARCH_DIRS}
INPUT_DIR = max(counts, key=counts.get)
OUTPUT_DIR = ROOT / "PDPL_quality_outputs"
PROCESSED = OUTPUT_DIR / "processed"
REPORTS = OUTPUT_DIR / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

print("Notebook working directory:", ROOT)
print("Paths checked:")
for p, count in counts.items():
    print(f"  {count}/3 found | {p.resolve()}")
print("Selected input:", INPUT_DIR.resolve())
print("Outputs:", OUTPUT_DIR.resolve())
if counts[INPUT_DIR] == 0:
    raise FileNotFoundError(
        "No matching *_articles_UNVERIFIED.json files found. "
        "Set INPUT_DIR_OVERRIDE to the full folder path and rerun."
    )


# No upload step needed locally.
# Show exactly which source files are present / missing.
for doc_id, filename in EXPECTED_FILES.items():
    path = INPUT_DIR / filename
    print(("FOUND  " if path.is_file() else "MISSING"), doc_id, "->", path)


LOADED, REJECTED = [], []
for doc_id, filename in EXPECTED_FILES.items():
    path = INPUT_DIR / filename
    if not path.is_file():
        REJECTED.append((filename, "missing_file"))
        continue
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict) and isinstance(obj.get("articles"), list):
            obj = obj["articles"]
        if not isinstance(obj, list) or not obj or not all(isinstance(x, dict) for x in obj):
            raise ValueError("Expected a nonempty list of article objects")
        if not all("article_number" in x and "text" in x for x in obj):
            raise ValueError("Missing article_number or text fields")
        # Stop on mismatched documents rather than accidentally mixing regulations.
        unexpected = {str(r.get("document_id", "")).strip() for r in obj}
        if unexpected != {doc_id}:
            raise ValueError(f"Unexpected document_id values: {unexpected}; expected {doc_id}")
        LOADED.append((path, obj))
        print(f"Loaded {filename}: {len(obj)} articles")
    except Exception as e:
        REJECTED.append((filename, str(e)))

print("Rejected or missing files:", REJECTED)
if not LOADED:
    raise RuntimeError("No usable input documents. Check INPUT_DIR_OVERRIDE and filenames.")


def conservative_clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def make_search_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي")
    text = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()

assert conservative_clean("أهلاً  بالعالم\n\n\nنص") == "أهلاً بالعالم\n\nنص"
assert "البيانات" in make_search_text("البيانات")
print("Normalization tests passed")

CHROME_PHRASES = ["موقع حكومي رسمي", "كيف تتحقق", "روابط المواقع الالكترونية الرسمية", "chevron_left", "Complementary Content", "${title}", "تقييم الأثر على الخصوصية إشعار تسرب"]
SUSPICIOUS_BROKEN = [r"المباش\s+ر", r"\bالبيانا\s+ت\b"]
URL_PATTERN = re.compile(r"https?://[^\s<>]+", re.I)

def normalized_document_id(row):
    did = str(row.get("document_id", "unknown")).strip()
    aliases = {"pdpl_ar": "pdpl", "implementing_regulation_ar": "implementing_regulation", "transfer_regulation_ar": "transfer_regulation"}
    return aliases.get(did, did)

def check_article(row):
    issues = []
    text = row.get("text")
    if not isinstance(text, str):
        issues.append("text_not_string")
        text = ""
    if not text.strip(): issues.append("empty_text")
    if 0 < len(text.strip()) < 45: issues.append("very_short_text_review")
    if "<div" in text.lower() or "</p>" in text.lower(): issues.append("html_tags_in_text")
    if any(x.casefold() in text.casefold() for x in CHROME_PHRASES): issues.append("possible_site_chrome")
    if URL_PATTERN.search(text): issues.append("url_inside_legal_text_review")
    if any(re.search(p, text) for p in SUSPICIOUS_BROKEN): issues.append("possible_broken_arabic_word")
    if "\ufffd" in text: issues.append("unicode_replacement_char")
    if re.search(r"[\u200b-\u200d\ufeff]", text): issues.append("invisible_unicode")
    if not row.get("source_url"): issues.append("missing_source_url")
    elif "dgp.sdaia.gov.sa" not in str(row["source_url"]): issues.append("unexpected_source_domain_review")
    if not row.get("id"): issues.append("missing_id")
    if not row.get("html_sha256"): issues.append("missing_html_hash")
    if str(row.get("language", "")).strip() != "ar": issues.append("unexpected_language_review")
    return issues

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

all_reports, processed_docs, seen_ids, loaded_source_files = [], defaultdict(list), defaultdict(list), []
for path, rows in LOADED:
    loaded_source_files.append(path.name)
    for idx, original in enumerate(rows):
        row = dict(original)
        did = normalized_document_id(row)
        num_raw = row.get("article_number")
        try:
            num = int(num_raw)
            if str(num_raw).strip() not in {str(num), str(float(num))}:
                raise ValueError("non-integer number")
        except (ValueError, TypeError):
            num = None
        text = row.get("text")
        issues = check_article(row)
        if num is None or num <= 0: issues.append("invalid_article_number")
        if num is not None and did in EXPECTED and num > EXPECTED[did]: issues.append("article_number_over_expected_range")
        row["clean_text"] = conservative_clean(text) if isinstance(text,str) else ""
        row["search_text"] = make_search_text(row["clean_text"])
        row["extraction_status"] = "needs_review"  # لا نعتمد أي مادة تلقائيًا
        row["quality_flags"] = issues
        row["source_input_file"] = path.name
        row["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
        row["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text,str) else ""
        processed_docs[did].append(row)
        seen_ids[did].append(str(row.get("id", "")))
        all_reports.append({
            "document_id":did, "input_file":path.name, "id":row.get("id",""),
            "article_number":num if num is not None else "", "text_length":len(text) if isinstance(text,str) else 0,
            "issues":issues, "source_url":row.get("source_url", ""), "text_sha256":row["text_sha256"]
        })

summary, checks = [], {}
for did, rows in sorted(processed_docs.items()):
    nums=[x.get("article_number") for x in rows]
    parsed_nums=[]
    for num in nums:
        try: parsed_nums.append(int(num))
        except (ValueError,TypeError): pass
    counts=Counter(parsed_nums)
    duplicates=sorted(k for k,v in counts.items() if v>1)
    repeated_ids=sorted(k for k,v in Counter(seen_ids[did]).items() if v>1 and k)
    expected=EXPECTED.get(did)
    missing=sorted(set(range(1,expected+1))-set(parsed_nums)) if expected is not None else []
    out_of_order = parsed_nums != sorted(parsed_nums)
    duplicate_text_hashes=[h for h,v in Counter(x.get("text_sha256") for x in rows).items() if v>1 and h]
    flags=[]
    if duplicates: flags.append("duplicate_article_numbers")
    if repeated_ids: flags.append("duplicate_ids")
    if missing: flags.append("missing_article_numbers")
    if out_of_order: flags.append("articles_out_of_order")
    if duplicate_text_hashes: flags.append("duplicate_exact_article_text")
    checks[did] = {"article_count":len(rows),"expected_count_reference_only":expected,"missing_numbers":missing,"duplicate_numbers":duplicates,"repeated_ids":repeated_ids,"out_of_order":out_of_order,"duplicate_text_hash_count":len(duplicate_text_hashes),"document_issues":flags,"status":"needs_review"}
    summary.append({"document_id":did,"count":len(rows),"expected_count":expected if expected is not None else "", "missing":str(missing),"duplicate_numbers":str(duplicates),"document_flags":"; ".join(flags),"articles_flagged":sum(bool(r["quality_flags"]) for r in rows),"verification_status":"needs_review"})
    out = PROCESSED / f"{did}_ar_processed.json"
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print(did, ":", len(rows), "articles | missing:", missing, "| duplicates:", duplicates)

for doc_name in EXPECTED:
    if doc_name not in processed_docs:
        summary.append({"document_id":doc_name,"count":0,"expected_count":EXPECTED[doc_name],"missing":"not uploaded","duplicate_numbers":"","document_flags":"document_not_uploaded","articles_flagged":"","verification_status":"not_processed"})

report_rows=[{**r,"issues":"; ".join(r["issues"])} for r in all_reports]
write_csv(REPORTS/"article_quality_report.csv",report_rows,["document_id","input_file","id","article_number","text_length","issues","source_url","text_sha256"])
write_csv(REPORTS/"manual_review_queue.csv",[r for r in report_rows if r["issues"]],["document_id","input_file","id","article_number","text_length","issues","source_url","text_sha256"])
write_csv(REPORTS/"document_summary.csv",summary,["document_id","count","expected_count","missing","duplicate_numbers","document_flags","articles_flagged","verification_status"])
(REPORTS/"checks.json").write_text(json.dumps({"documents":checks,"rejected_files":REJECTED,"input_files":loaded_source_files},ensure_ascii=False,indent=2),encoding="utf-8")
print("Created reports:", REPORTS)
