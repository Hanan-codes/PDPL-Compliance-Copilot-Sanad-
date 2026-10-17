"""Combine 3 core PDPL and 6 supplementary SDAIA sources; chunk and audit."""
from pathlib import Path
import json, re, hashlib, unicodedata
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "pdpl_corpus" / "data" / "extracted"
ADDITIONAL_FILE = ROOT / "sdaia_six_corpus" / "data" / "extracted" / "sdaia_additional_ar_processed_UNVERIFIED.json"
OUTPUT_DIR = ROOT / "rag_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_FILES = {
    "pdpl": "pdpl_ar_articles_UNVERIFIED.json",
    "implementing_regulation": "implementing_regulation_ar_articles_UNVERIFIED.json",
    "transfer_regulation": "transfer_regulation_ar_articles_UNVERIFIED.json",
}
EXPECTED_EXTRA = {
    "data_sharing_policy", "data_monetization_policy", "appointing_dpo",
    "accreditation_certificates", "accreditation_licensing", "secondary_data_use"
}

print("CORE_DIR exists:", CORE_DIR.exists(), CORE_DIR)
print("Additional file exists:", ADDITIONAL_FILE.exists(), ADDITIONAL_FILE)
print("Output directory:", OUTPUT_DIR.resolve())

def load_array(path):
    if not path.exists():
        raise FileNotFoundError(f"ملف غير موجود: {path}")
    with path.open('r', encoding='utf-8-sig') as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected JSON array in {path}")
    return obj

core_records = []
for doc_id, filename in CORE_FILES.items():
    source = CORE_DIR / filename
    records = load_array(source)
    if not records:
        raise ValueError(f"Empty file: {source}")
    for r in records:
        item = dict(r)
        if item.get('document_id') not in (None, doc_id):
            raise ValueError(f"Unexpected document_id in {filename}: {item.get('document_id')}")
        item['document_id'] = doc_id
        item['clean_text'] = item.get('clean_text') or item.get('text') or ''
        item.setdefault('extraction_status', 'needs_review')
        core_records.append(item)
    print(doc_id, len(records))

additional_records = load_array(ADDITIONAL_FILE)
print('Additional records:', len(additional_records))

actual_extra = {x.get('document_id') for x in additional_records}
if actual_extra != EXPECTED_EXTRA:
    raise ValueError(f"Unexpected extra document set. Missing: {EXPECTED_EXTRA-actual_extra}; Extra: {actual_extra-EXPECTED_EXTRA}")

# Merge defensively: duplicate IDs with different content are an error.
merged_records, by_id = [], {}
for original in core_records + additional_records:
    record = dict(original)
    rid = record.get('id')
    if not isinstance(rid, str) or not rid.strip():
        raise ValueError(f"Record without valid id: {record.get('document_id')}")
    record['clean_text'] = record.get('clean_text') or record.get('text') or ''
    if not record['clean_text'].strip():
        raise ValueError(f"Empty legal text: {rid}")
    record.setdefault('extraction_status', 'needs_review')
    if rid in by_id:
        if by_id[rid] != record:
            raise ValueError(f"Conflicting duplicate id: {rid}")
        continue
    by_id[rid] = record
    merged_records.append(record)

counts = Counter(r['document_id'] for r in merged_records)
EXPECTED_DOCS = set(CORE_FILES) | EXPECTED_EXTRA
if set(counts) != EXPECTED_DOCS:
    raise AssertionError(f"Expected 9 docs. Missing: {EXPECTED_DOCS-set(counts)}")

report = pd.DataFrame([{'document_id':k,'records':v} for k,v in sorted(counts.items())])
print(report.to_string(index=False))
print('Documents:',len(counts),'| Records:',len(merged_records))

merged_path = OUTPUT_DIR/'pdpl_9_documents_merged.json'
with merged_path.open('w',encoding='utf-8') as f:
    json.dump(merged_records,f,ensure_ascii=False,indent=2)
print('Saved:',merged_path.resolve())

MAX_CHARS = 1200
OVERLAP = 150

def split_preserving_words(s, max_chars=MAX_CHARS, overlap=OVERLAP):
    s = re.sub(r'\s+', ' ', s).strip()
    if not s: return []
    pieces, start = [], 0
    while start < len(s):
        end = min(len(s),start+max_chars)
        if end < len(s):
            space = s.rfind(' ', start + max_chars // 2, end)
            if space > start:
                end = space
        part = s[start:end].strip()
        if part: pieces.append(part)
        if end >= len(s): break
        start = max(start+1,end-overlap)
    return pieces

def normalize_ar_search(txt):
    txt = unicodedata.normalize('NFKC',txt)
    txt = re.sub(r'[\u064b-\u065f\u0670]', '', txt)
    txt = txt.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ـ':''}))
    return re.sub(r'\s+', ' ', txt).strip()

rag_chunks=[]
for r in merged_records:
    pieces=split_preserving_words(r['clean_text'])
    for j,piece in enumerate(pieces,1):
        rag_chunks.append({
            'chunk_id': f"{r['id']}_chunk_{j:02d}",
            'parent_id': r['id'],
            'document_id': r['document_id'],
            'language': r.get('language','ar'),
            'article_number': r.get('article_number'),
            'article_title': r.get('article_title'),
            'source_url': r.get('source_url'),
            'extraction_status': r.get('extraction_status','needs_review'),
            'clean_text': piece,
            'search_text': normalize_ar_search(piece)
        })

assert len({r['chunk_id'] for r in rag_chunks}) == len(rag_chunks)
chunk_path = OUTPUT_DIR/'pdpl_9_documents_rag_chunks.json'
with chunk_path.open('w',encoding='utf-8') as f:
    json.dump(rag_chunks,f,ensure_ascii=False,indent=2)

chunk_stats = (pd.DataFrame(rag_chunks)
    .assign(length=lambda df:df.clean_text.str.len())
    .groupby('document_id',as_index=False)
    .agg(chunks=('chunk_id','size'), min_length=('length','min'), max_length=('length','max')))
print(chunk_stats.to_string(index=False))
print('Total chunks:',len(rag_chunks),'| Unique documents:',len(chunk_stats))
print('Saved:',chunk_path.resolve())

# Final report and safety gates
final = report.merge(chunk_stats, on='document_id', how='left', validate='one_to_one')
final_path=OUTPUT_DIR/'pdpl_9_sources_report.csv'
final.to_csv(final_path,index=False,encoding='utf-8-sig')

assert len(final)==9, 'Incomplete merge!'
assert sum(final['chunks']) == len(rag_chunks)
assert all('source_url' in r for r in merged_records), 'Some source URLs missing as keys'

print('✅ Merge completed with nine sources')
print('Merged:',merged_path.resolve())
print('Chunks:',chunk_path.resolve())
print('Report:',final_path.resolve())
print('Review markers retained; do not represent records as source-verified.')
