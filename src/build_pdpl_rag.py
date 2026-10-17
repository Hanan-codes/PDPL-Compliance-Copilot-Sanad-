"""Create traceable RAG chunks from six SDAIA sources and three core PDPL sources.
Usage: python build_pdpl_rag.py --input-dir . --output-dir pdpl_rag_output
Never changes the input texts or upgrades needs_review to verified.
"""
import argparse, collections, hashlib, json, re
from pathlib import Path

SIX_FILE='sdaia_additional_ar_processed_UNVERIFIED.json'
CORE_CANDIDATES={
'pdpl':['pdpl_ar_processed.json','pdpl_ar_processed_UNVERIFIED.json'],
'implementing_regulation':['implementing_regulation_ar_processed.json','implementing_regulation_ar_processed_UNVERIFIED.json','pdpl_implementing_regulation_ar_processed.json'],
'transfer_regulation':['transfer_regulation_ar_processed.json','transfer_regulation_ar_processed_UNVERIFIED.json','personal_data_transfer_regulation_ar_processed.json']}
NAV_RE=re.compile(r'موقع حكومي رسمي تابع لحكومة المملكة|جميع الحقوق محفوظة للهيئة|Complementary Content')

def load_records(p):
    data=json.loads(p.read_text(encoding='utf-8-sig'))
    if not isinstance(data,list): raise ValueError(f'{p}: expected JSON list')
    return data

def split_long(text,limit=1250,overlap=140):
    # Keep paragraphs intact when possible; long paragraphs split at sentence/space boundaries.
    paragraphs=[x.strip() for x in re.split(r'\n\s*\n+',text) if x.strip()]
    if not paragraphs: paragraphs=[text.strip()]
    units=[]
    for para in paragraphs:
        if len(para)<=limit:
            units.append(para);continue
        parts=re.split(r'(?<=[.!؟؛])\s+',para)
        for part in parts:
            if len(part)<=limit: units.append(part);continue
            words=part.split();piece=''
            for word in words:
                if piece and len(piece)+len(word)+1>limit:
                    units.append(piece);piece=''
                piece=(piece+' '+word).strip()
            if piece: units.append(piece)
    chunks=[];buf=''
    for unit in units:
        if len(buf)+len(unit)+1 <= limit:
            buf=(buf+'\n'+unit).strip()
        else:
            if buf: chunks.append(buf)
            prefix=buf[-overlap:].strip() if buf else ''
            # Avoid overlap crossing entire short unit; prefix is clearly duplicated for retrieval only.
            buf=((prefix+'\n') if prefix else '')+unit
            if len(buf)>limit:
                buf=unit
    if buf: chunks.append(buf)
    return chunks

def build(input_dir, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    paths={}
    six=input_dir/SIX_FILE
    if not six.exists(): raise FileNotFoundError(f'Put {SIX_FILE} in {input_dir}')
    paths['supplemental_six']=six
    missing=[]
    for doc,choices in CORE_CANDIDATES.items():
        p=next((input_dir/x for x in choices if (input_dir/x).exists()),None)
        if p: paths[doc]=p
        else: missing.append(doc)
    records=[]; report=[]; ids=set()
    for label,path in paths.items():
        source=load_records(path)
        accepted=0
        for record in source:
            item=dict(record)
            key=item.get('id'); doc=item.get('document_id');body=item.get('clean_text') or item.get('text') or ''
            if not key or not doc or not isinstance(body,str) or not body.strip():
                raise ValueError(f'Invalid record in {path}: {key}')
            if key in ids: raise ValueError(f'Duplicate id: {key}')
            if NAV_RE.search(body):raise ValueError(f'Navigation text inside {key}')
            ids.add(key)
            item.setdefault('clean_text',body)
            item.setdefault('language','ar')
            item.setdefault('extraction_status','needs_review')
            item['corpus_group']='core_pdpl' if label!='supplemental_six' else 'supplementary_sdaia'
            records.append(item);accepted+=1
        report.append({'input_file':path.name,'records':accepted,'group':label})
    chunks=[]
    for r in records:
        bodies=split_long(r['clean_text'])
        for n,chunk_text in enumerate(bodies,1):
            chunks.append({'chunk_id':f"{r['id']}__chunk_{n:02d}",
                           'parent_record_id':r['id'],'document_id':r['document_id'],
                           'corpus_group':r['corpus_group'],'language':r['language'],
                           'article_number':r.get('article_number'),
                           'article_title':r.get('article_title'),
                           'source_url':r.get('source_url'),
                           'extraction_status':r.get('extraction_status','needs_review'),
                           'clean_text':chunk_text,
                           'search_text':re.sub(r'\s+',' ',chunk_text).strip(),
                           'chunk_index':n,'chunk_count':len(bodies)})
    def dump(name,data):
        dest=output_dir/name;dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');return dest
    files=[dump('pdpl_corpus_merged.json',records),dump('pdpl_rag_chunks.json',chunks)]
    summary={'documents':dict(collections.Counter(x['document_id'] for x in records)),
             'total_records':len(records),'total_chunks':len(chunks),
             'missing_core_documents':missing,'input_files':report,
             'review_status':'unverified — original extraction_status preserved',
             'note':'This is structured corpus preprocessing, not legal verification or an embedding index.'}
    files.append(dump('rag_build_report.json',summary))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Files:',*(str(x) for x in files),sep='\n  ')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',default='.');p.add_argument('--output-dir',default='pdpl_rag_output');a=p.parse_args()
    build(Path(a.input_dir),Path(a.output_dir))
