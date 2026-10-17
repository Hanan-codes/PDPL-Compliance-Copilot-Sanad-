"""Extract original SDAIA pages. Results are UNVERIFIED and require legal review."""
ROOT = Path(__file__).resolve().parents[1]

from pathlib import Path
import requests, re, json, hashlib, unicodedata
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import pandas as pd

BASE=ROOT/'sdaia_six_corpus'; RAW=BASE/'data'/'raw_html'; OUT=BASE/'data'/'extracted'; QA=BASE/'data'/'review'; MANUAL=BASE/'data'/'manual_html'
for d in (RAW,OUT,QA,MANUAL): d.mkdir(parents=True,exist_ok=True)
SOURCES={
 'data_sharing_policy':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/DataSharingPolicy1',
 'data_monetization_policy':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/DataMonetizationPolicy',
 'appointing_dpo':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/AppointingPersonalDataProtectionOfficer',
 'accreditation_certificates':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/RulesGoverningtheIssuanceofAccreditationCertificatesforControllersandProcessers',
 'accreditation_licensing':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/RulesGoverningtheLicensingtoIssueAccreditationCertificatestoControllersandProcessors',
 'secondary_data_use':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/GeneralRulesforSecondaryUseofData',
}
print('Documents:',len(SOURCES))

print("Local output:", OUT.resolve())

def load_source(doc_id,url):
    local_html=MANUAL/f'{doc_id}_ar.html'; local_txt=MANUAL/f'{doc_id}_ar.txt'
    if local_html.exists():
        html=local_html.read_text(encoding='utf-8-sig'); method='manual_html'
    elif local_txt.exists():
        # Escape plain text and treat it as a single content block.
        import html as html_lib
        html='<html><body><main>'+html_lib.escape(local_txt.read_text(encoding='utf-8-sig'))+'</main></body></html>';method='manual_text'
    else:
        r=requests.get(url,timeout=40,headers={'User-Agent':'Mozilla/5.0','Accept-Language':'ar,en;q=0.8'})
        r.raise_for_status(); r.encoding=r.apparent_encoding if 'utf' not in (r.encoding or '').lower() else r.encoding
        html=r.text;method='http'
    (RAW/f'{doc_id}_ar.html').write_text(html,encoding='utf-8')
    return html, {'document_id':doc_id,'source_url':url,'retrieval_method':method,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'html_sha256':hashlib.sha256(html.encode('utf-8')).hexdigest()}

HTML={}; PROV={}
for doc_id,url in SOURCES.items():
    try:
        HTML[doc_id], PROV[doc_id]=load_source(doc_id,url)
        print(doc_id, 'HTML characters:',len(HTML[doc_id]))
    except Exception as e:
        print('FETCH FAILED:',doc_id, repr(e))


# NO global deletion of header, footer or nav: on some portals these wrap the legal content.
def full_page_text(html):
    soup=BeautifulSoup(html,'lxml')
    for t in soup(['script','style','noscript','template','svg']): t.decompose()
    # spaces intentionally preserve separation across nested spans.
    text=soup.get_text(separator=' ',strip=True)
    return re.sub(r'\s+',' ',text).strip()

def search_normalize(s):
    s=unicodedata.normalize('NFKC',str(s))
    s=re.sub(r'[\u064b-\u065f\u0670]','',s)
    s=re.sub('[إأآٱ]','ا',s).replace('ى','ي')
    return re.sub(r'\s+',' ',s).strip()

# Arabic legal article titles (1..60), including both forms for 20/30/40 and conjunction forms.
UNITS=['','الأولى','الثانية','الثالثة','الرابعة','الخامسة','السادسة','السابعة','الثامنة','التاسعة']
TEENS={10:'العاشرة',11:'الحادية عشرة',12:'الثانية عشرة',13:'الثالثة عشرة',14:'الرابعة عشرة',15:'الخامسة عشرة',16:'السادسة عشرة',17:'السابعة عشرة',18:'الثامنة عشرة',19:'التاسعة عشرة'}
TENS={20:'العشرون',30:'الثلاثون',40:'الأربعون',50:'الخمسون',60:'الستون'}
UNIT_COMPOUND=['','الحادية','الثانية','الثالثة','الرابعة','الخامسة','السادسة','السابعة','الثامنة','التاسعة']
ORDINAL={}
for i in range(1,61):
    if i<10: forms=[UNITS[i]]
    elif i<20: forms=[TEENS[i]]
    elif i in TENS: forms=[TENS[i]]
    else:
        u=i%10;t=i-u
        forms=[UNIT_COMPOUND[u]+' وال'+TENS[t][2:], UNIT_COMPOUND[u]+' و'+TENS[t]]
    for form in forms: ORDINAL[search_normalize(form)]=i

# Patterns run on an unaccented normalized string while preserving original indices by a length-preserving mask.
def mask_for_matching(s):
    # Avoid altering original offsets: convert only alef variants / alef maqsura and vowels to spaces.
    trans=str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي','ـ':' '})
    s=s.translate(trans)
    return re.sub(r'[\u064b-\u065f\u0670]',' ',s)

names=sorted(ORDINAL,key=len,reverse=True)
expr='|'.join(re.escape(x).replace(r'\ ',r'\s+') for x in names)
# Anchor on heading followed by colon, including a newline between words.
HEADING_RE=re.compile(r'(?<![\w])المادة\s+('+expr+r'|[0-9٠-٩]{1,3})\s*[:：]',re.UNICODE)

def find_headings(text):
    masked=mask_for_matching(text); found=[]
    for m in HEADING_RE.finditer(masked):
        s=m.group(1); number=ORDINAL.get(search_normalize(s))
        if number is None and re.fullmatch('[0-9٠-٩]{1,3}',s):
            number=int(s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789')))
        if number is not None:
            found.append(dict(start=m.start(),end=m.end(),number=number,title=text[m.start():m.end()]))
    return found

for doc_id,html in HTML.items():
    txt=full_page_text(html); (QA/f'{doc_id}_FULL_TEXT_DEBUG.txt').write_text(txt,encoding='utf-8')
    matches=find_headings(txt)
    print(doc_id,'total page chars',len(txt),'candidate headings',len(matches),'first ten numbers:',[x['number'] for x in matches[:10]])
    print(' has article 1:',any(x['number']==1 for x in matches))


FOOTER=[r'chevron_left\s+عودة', r'ملخص عن المنصة\s+تواصل معنا',r'خريطة الموقع\s+سياسة الخصوصية',r'جميع الحقوق محفوظة']

def select_sequence(headings):
    # pick the longest strictly increasing-by-one path beginning with article 1
    options=[]
    for i,h in enumerate(headings):
        if h['number']!=1: continue
        seq=[h]; next_num=2
        for cand in headings[i+1:]:
            if cand['number']==next_num:
                seq.append(cand);next_num+=1
        options.append(seq)
    return max(options,key=len) if options else []

def extract(doc_id,html,prov):
    text=full_page_text(html); heads=select_sequence(find_headings(text))
    if len(heads)<2: raise ValueError('No consecutive article sequence found (1,2,...). Inspect FULL_TEXT_DEBUG.txt')
    result=[]
    for i,h in enumerate(heads):
        stop=heads[i+1]['start'] if i+1<len(heads) else len(text)
        body=text[h['end']:stop].strip()
        if i==len(heads)-1:
            cuts=[m.start() for p in FOOTER if (m:=re.search(p,body))]
            if cuts: body=body[:min(cuts)].strip()
        result.append({
            'id':f'{doc_id}_ar_{h["number"]:03d}', 'document_id':doc_id,
            'language':'ar','article_number':h['number'],'article_title':h['title'],
            'text':body,'search_text':search_normalize(body),
            'source_url':prov['source_url'],'html_sha256':prov['html_sha256'],
            'extraction_status':'needs_review'
        })
    if not result or any(not a['text'].strip() for a in result):
        raise ValueError('Empty article detected; check extraction manually')
    if any('موقع حكومي رسمي تابع لحكومة المملكة العربية السعودية' in a['text'] for a in result):
        raise ValueError('Government website header leaked into legal content')
    if sum(len(a['text']) for a in result)<500:
        raise ValueError('Suspiciously short extraction (<500 chars)')
    return result

EXTRACTED={}
FAILURES={}
for doc_id,html in HTML.items():
    try:
        articles=extract(doc_id,html,PROV[doc_id]); EXTRACTED[doc_id]=articles
        nums=[a['article_number'] for a in articles]
        print('\n',doc_id,'EXTRACTED:',len(articles),'article numbers:',nums)
        print('FIRST:',articles[0]['article_title'],repr(articles[0]['text'][:220]))
        print('LAST:',articles[-1]['article_title'],repr(articles[-1]['text'][-220:]))
        (OUT/f'{doc_id}_ar_articles_UNVERIFIED.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2),encoding='utf-8')
        (QA/f'{doc_id}_LEGAL_ONLY_PREVIEW.txt').write_text('\n\n'.join(f'{a["article_title"]}\n{a["text"]}' for a in articles),encoding='utf-8')
    except Exception as e:
        FAILURES[doc_id]=str(e)
        print('EXTRACTION FAILED:',doc_id,repr(e))


# Policies are divided by sections; article_number stays None.
# Normalize headings for MATCHING ONLY; leave source text unchanged.
ARABIC = re.compile(r'[\u0621-\u064A]')
NOISE = re.compile(r'موقع حكومي رسمي تابع|روابط المواقع الالكترونية الرسمية|جميع الحقوق محفوظة|خريطة الموقع')
END_MARKERS = ('chevron_left', 'Complementary Content', 'جميع الحقوق محفوظة')
TITLE_ALIASES = {
    'data_sharing_policy': ['سياسة مشاركة البيانات'],
    'data_monetization_policy': ['سياسة تحقيق الإيرادات من البيانات', 'سياسة تحقيق الإيرادات من البيانات'],
    'secondary_data_use': ['القواعد العامة للاستخدام الثانوي للبيانات']
}
SECTION_NORM = re.compile(
    r'^(?:اولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s*[:：\-]'
    r'|^المبدا\s+(?:الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)(?:\s*[:：\-]|\s*$)'
)

def policy_sections(doc_id, html, prov):
    soup=BeautifulSoup(html,'lxml')
    for t in soup(['script','style','noscript','svg','template']): t.decompose()
    # preserve line separation, including section labels
    lines=[re.sub(r'\s+',' ',s).strip() for s in soup.get_text('\n').splitlines()]
    lines=[s for s in lines if s]
    titles={search_normalize(x) for x in TITLE_ALIASES[doc_id]}
    starts=[i for i,s in enumerate(lines) if search_normalize(s) in titles]
    if not starts: raise ValueError('Arabic document title not found; inspect FULL_TEXT_DEBUG')
    start=starts[-1]+1
    end=next((j for j in range(start,len(lines)) if lines[j] in END_MARKERS or lines[j].startswith('جميع الحقوق محفوظة')),len(lines))
    body=lines[start:end]
    if not body or len('\n'.join(body))<500: raise ValueError('Insufficient body text')
    records=[]; title='مقدمة'; fragments=[]
    def flush():
        text='\n'.join(fragments).strip()
        if text:
            records.append(dict(id=f'{doc_id}_ar_{len(records)+1:03d}', document_id=doc_id,
                language='ar', article_number=None, article_title=title, text=text,
                clean_text=text, search_text=search_normalize(text),
                source_url=prov['source_url'],html_sha256=prov['html_sha256'],
                extraction_status='needs_review',quality_flags=['section_boundary_needs_review'],
                source_input_file=f'{doc_id}_ar.html'))
    for line in body:
        if SECTION_NORM.match(search_normalize(line)):
            flush(); title=line;fragments=[]
        else: fragments.append(line)
    flush()
    # Fail closed rather than publishing website navigation as law
    for r in records:
        if NOISE.search(r['text']): raise ValueError('Government header or navigation leaked')
        if len(r['text']) < 60: raise ValueError(f'Potentially truncated section {r["id"]}')
    combined=' '.join(r['text'] for r in records)
    letters=len(re.findall(r'[A-Za-z\u0621-\u064A]', combined))
    ratio=len(ARABIC.findall(combined))/max(letters,1)
    if ratio<0.65: raise ValueError(f'Not predominantly Arabic: {ratio:.2f}')
    if len(records)<2: raise ValueError('No section boundaries found')
    return records

POLICY_IDS=['data_sharing_policy','data_monetization_policy','secondary_data_use']
SECTION_RESULTS=[]
for doc_id in POLICY_IDS:
    if doc_id in EXTRACTED:
        continue
    if doc_id not in HTML:
        SECTION_RESULTS.append(dict(document_id=doc_id,status='failed',reason='HTML unavailable'))
        continue
    try:
        records=policy_sections(doc_id,HTML[doc_id],PROV[doc_id]);EXTRACTED[doc_id]=records
        (QA/f'{doc_id}_LEGAL_ONLY_PREVIEW.txt').write_text(
            '\n\n'.join(f'{r["article_title"]}\n{r["text"]}' for r in records),encoding='utf-8')
        SECTION_RESULTS.append(dict(document_id=doc_id,status='needs_review',record_count=len(records)))
        print(doc_id,'sections:',len(records),[r['article_title'] for r in records])
    except Exception as e:
        SECTION_RESULTS.append(dict(document_id=doc_id,status='failed',reason=str(e)))
        print('FAILED:',doc_id,e)
print(pd.DataFrame(SECTION_RESULTS).to_string(index=False))

def enrich(r):
    o=dict(r)
    o.setdefault('clean_text',o['text'])
    o.setdefault('quality_flags',[])
    o.setdefault('source_input_file',f'{o["document_id"]}_ar.html')
    o['processed_at_utc']=datetime.now(timezone.utc).isoformat()
    o['text_sha256']=hashlib.sha256(o['clean_text'].encode('utf-8')).hexdigest()
    return o

merged=[]; report=[]
for doc_id,url in SOURCES.items():
    rows=[enrich(r) for r in EXTRACTED.get(doc_id,[])]
    problems=[]
    for r in rows:
        if not r['text'].strip() or 'موقع حكومي رسمي تابع' in r['text']:
            problems.append(r['id'])
    if problems:
        rows=[]; reason='Invalid records: '+', '.join(problems)
    else: reason='' if rows else next((str(x.get('reason','')) for x in SECTION_RESULTS if x['document_id']==doc_id),'See extraction logs')
    if rows:
        (OUT/f'{doc_id}_ar_processed_UNVERIFIED.json').write_text(
            json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        merged.extend(rows)
    report.append(dict(document_id=doc_id,status='needs_review' if rows else 'failed',
                       record_count=len(rows),article_count=sum(r['article_number'] is not None for r in rows),
                       section_count=sum(r['article_number'] is None for r in rows),reason=reason))

(OUT/'sdaia_additional_ar_processed_UNVERIFIED.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2),encoding='utf-8')
(QA/'extraction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Output:',OUT.resolve())
print('Records:',len(merged),'(unverified)')
print(pd.DataFrame(report).to_string(index=False))

