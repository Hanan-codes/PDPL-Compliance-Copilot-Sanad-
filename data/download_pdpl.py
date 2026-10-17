"""Extract original SDAIA pages. Results are UNVERIFIED and require legal review."""
ROOT = Path(__file__).resolve().parents[1]

from pathlib import Path
import requests, re, json, hashlib, unicodedata
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import pandas as pd

BASE=ROOT/'pdpl_corpus'; RAW=BASE/'data'/'raw_html'; OUT=BASE/'data'/'extracted'; QA=BASE/'data'/'review'; MANUAL=BASE/'data'/'manual_html'
for d in (RAW,OUT,QA,MANUAL): d.mkdir(parents=True,exist_ok=True)
SOURCES={
 'pdpl':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/PDPL',
 'implementing_regulation':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/PDPL2',
 'transfer_regulation':'https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/details/RegulationonPersonalDataTransferOutsidetheKingdom',
}


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
    return result

EXTRACTED={}
for doc_id,html in HTML.items():
    try:
        articles=extract(doc_id,html,PROV[doc_id]); EXTRACTED[doc_id]=articles
        nums=[a['article_number'] for a in articles]
        print('\n',doc_id,'EXTRACTED:',len(articles),'article numbers:',nums)
        print('FIRST:',articles[0]['article_title'],repr(articles[0]['text'][:220]))
        print('LAST:',articles[-1]['article_title'],repr(articles[-1]['text'][-220:]))
        (OUT/f'{doc_id}_ar_articles_UNVERIFIED.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2),encoding='utf-8')
        (QA/f'{doc_id}_LEGAL_ONLY_PREVIEW.txt').write_text('\n\n'.join(f'{a["article_title"]}\n{a["text"]}' for a in articles),encoding='utf-8')
    except Exception as e: print('EXTRACTION FAILED:',doc_id,repr(e))


for doc_id,arts in EXTRACTED.items():
    rows=[];nums=[a['article_number'] for a in arts]
    for a in arts:
        flags=[]
        if len(a['text'])<35 and search_normalize(a['text']).strip(' :،.') not in ('ملغاة','ملغاه'): flags.append('unexpectedly short')
        if 'موقع حكومي رسمي' in a['text'] or 'جميع الحقوق محفوظة' in a['text']: flags.append('web UI residue')
        if re.search(r'https?://|www\.',a['text']): flags.append('URL inside article: review')
        rows.append({'id':a['id'],'article_number':a['article_number'],'chars':len(a['text']),
                     'qa_flags':'; '.join(flags),'review_status':'pending','reviewer':'','notes':''})
    pd.DataFrame(rows).to_csv(QA/f'{doc_id}_manual_review.csv',index=False,encoding='utf-8-sig')
    if doc_id=='pdpl' and nums!=list(range(1,44)):
        print('WARNING: PDPL expected articles 1..43; found',nums)
    else:print(doc_id,'saved',len(rows),'rows for manual review')
print('Output root:',BASE.resolve())

