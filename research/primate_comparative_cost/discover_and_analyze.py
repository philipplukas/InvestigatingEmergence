import io, json, math, os, re, sys, traceback
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from scipy.spatial.distance import jensenshannon

OUT = Path('research/primate_comparative_cost/out')
RAW = OUT / 'raw'
RAW.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.headers.update({'User-Agent': 'comparative-primate-cost-atlas/1.0'})

ledger = []
discovery = {}

def get_json(url, timeout=60):
    r = S.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def dl(url, path, timeout=120):
    r = S.get(url, timeout=timeout)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path

def safe_read_table(path):
    try:
        if path.suffix.lower() == '.csv':
            return pd.read_csv(path)
        if path.suffix.lower() in ['.tsv', '.txt']:
            return pd.read_csv(path, sep='\t')
        if path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(path)
    except Exception as e:
        return None
    return None

# ---------------- CHIMPANZEE SEMANTIC COMBINATORICS ----------------
try:
    url = 'https://raw.githubusercontent.com/tozbu/Chimpanzee_bigram_meaning/main/calldata.csv'
    p = dl(url, RAW/'chimp_calldata.csv')
    d = pd.read_csv(p)
    d['is_bigram'] = pd.to_numeric(d['combi.length'], errors='coerce').eq(2)
    n = len(d)
    n_big = int(d.is_bigram.sum())
    callers = d['caller'].nunique()
    big = d[d.is_bigram].copy()
    bigram_types = sorted(big['call.name'].dropna().unique())
    call_types = sorted({x for v in d['call.name'].dropna() for x in str(v).split('_')})
    calls_in_big = sorted({x for v in big['call.name'].dropna() for x in str(v).split('_')})

    # context-distribution geometry for each observed bigram vs its components and their 50/50 mixture.
    contexts = sorted(d['context'].dropna().unique())
    def dist(rows):
        c = rows['context'].value_counts().reindex(contexts, fill_value=0).astype(float).values + 0.5
        return c/c.sum()
    rows=[]
    for bg in bigram_types:
        parts=bg.split('_')
        if len(parts)!=2: continue
        bgd=dist(d[d['call.name']==bg])
        p1=dist(d[d['call.name']==parts[0]])
        p2=dist(d[d['call.name']==parts[1]])
        mix=(p1+p2)/2
        rows.append({
            'bigram':bg,'n':int((d['call.name']==bg).sum()),
            'n_callers':int(d.loc[d['call.name']==bg,'caller'].nunique()),
            'js_to_part1':float(jensenshannon(bgd,p1,base=2)**2),
            'js_to_part2':float(jensenshannon(bgd,p2,base=2)**2),
            'js_to_additive_mix':float(jensenshannon(bgd,mix,base=2)**2),
            'date_span_days':int((pd.to_datetime(d.loc[d['call.name']==bg,'date']).max()-pd.to_datetime(d.loc[d['call.name']==bg,'date']).min()).days) if (d['call.name']==bg).sum()>1 else 0,
        })
    bggeom=pd.DataFrame(rows)
    bggeom.to_csv(OUT/'chimp_bigram_context_geometry.csv', index=False)
    discovery['chimp']={'n_rows':n,'n_callers':callers,'n_bigram_tokens':n_big,'n_bigram_types':len(bigram_types),'n_component_call_types':len(call_types),'n_call_types_participating_in_bigrams':len(calls_in_big),'columns':list(d.columns)}
    ledger.append({
        'species':'Chimpanzee','capability':'semantic/combinatorial meaning expansion','dataset':'Versatile use of chimpanzee call combinations (2025)',
        'access_metric':'fraction of observed utterances that are two-call combinations','access_value':n_big/n,
        'selection_metric':'median JS divergence of bigram context distribution from 50/50 component mixture','selection_value':float(bggeom.js_to_additive_mix.median()),
        'stability_metric':'median number of callers per observed bigram type','stability_value':float(bggeom.n_callers.median()),
        'access_identified':'direct observational lower-bound proxy','selection_identified':'descriptive distributional-semantic proxy; not identical to authors Bayesian compositionality classification','stability_identified':'reuse across callers; not longitudinal persistence',
        'notes':f'{len(bigram_types)} observed bigram types; {len(calls_in_big)}/{len(call_types)} component call types participate in at least one bigram.'
    })
except Exception as e:
    discovery['chimp_error']=repr(e)

# ---------------- MACAQUE VOLUNTARY VOCAL CONTROL (DRYAD) ----------------
try:
    doi='10.5061/dryad.6v7j674'
    meta=get_json('https://datadryad.org/api/v2/datasets/'+quote('doi:'+doi, safe=''))
    ver=meta.get('_links',{}).get('stash:version',{}).get('href') or meta.get('_links',{}).get('version',{}).get('href')
    if not ver:
        # API may expose currentVersion directly
        ver=meta.get('_links',{}).get('stash:versions',{}).get('href')
    discovery['dryad_dataset_meta_keys']=list(meta.keys())
    # robust route via versions endpoint
    vj=get_json('https://datadryad.org/api/v2/datasets/'+quote('doi:'+doi, safe='')+'/versions')
    versions=vj.get('_embedded',{}).get('stash:versions',[])
    if not versions: raise RuntimeError('No Dryad versions found')
    version_id=versions[-1]['id']
    fj=get_json(f'https://datadryad.org/api/v2/versions/{version_id}/files')
    files=fj.get('_embedded',{}).get('stash:files',[])
    finfo=[]
    dfs={}
    for f in files:
        name=f.get('path') or f.get('filename') or f.get('name')
        fid=f['id']
        furl=f.get('_links',{}).get('stash:download',{}).get('href') or f'https://datadryad.org/api/v2/files/{fid}/download'
        pp=RAW/('dryad_'+Path(name).name)
        dl(furl,pp)
        df=safe_read_table(pp)
        finfo.append({'name':name,'id':fid,'size':f.get('size'),'columns':list(df.columns) if df is not None else None,'n':len(df) if df is not None else None})
        if df is not None: dfs[name]=df
    discovery['macaque_voluntary_control_files']=finfo
    # summarize tables generically; exact experimental coding recorded for audit
    for name,df in dfs.items():
        df.to_csv(OUT/('macaque_'+re.sub(r'[^A-Za-z0-9]+','_',Path(name).stem)+'_copy.csv'), index=False)
    # infer likely response/latency fields only when present, without guessing semantic meaning
    allcols=sorted({c for df in dfs.values() for c in df.columns})
    ledger.append({
        'species':'Japanese macaque','capability':'voluntary vocal control','dataset':'Koda et al. 2018 Dryad',
        'access_metric':'raw training/probe success or vocal-response rate (schema discovered)','access_value':np.nan,
        'selection_metric':'cue-contingent vocal control relative to manual/spontaneous alternatives','selection_value':np.nan,
        'stability_metric':'retention/generalization across probe trials/sessions','stability_value':np.nan,
        'access_identified':'raw data available; exact coding requires schema-specific analysis','selection_identified':'raw data available','stability_identified':'probe table available; to be computed after coding audit',
        'notes':'Columns discovered: '+', '.join(allcols)
    })
except Exception as e:
    discovery['macaque_voluntary_error']=repr(e)

# ---------------- BONOBO COMPOSITIONALITY (FIGSHARE) ----------------
try:
    arts=get_json('https://api.figshare.com/v2/collections/7648628/articles')
    bfiles=[]
    for art in arts:
        aid=art['id']
        am=get_json(f'https://api.figshare.com/v2/articles/{aid}')
        for f in am.get('files',[]):
            bfiles.append({'article_id':aid,'article_title':am.get('title'),'name':f.get('name'),'size':f.get('size'),'download_url':f.get('download_url')})
            if f.get('size',0) <= 25_000_000 and f.get('download_url'):
                pp=RAW/('bonobo_'+Path(f['name']).name)
                try: dl(f['download_url'],pp)
                except Exception: pass
    discovery['bonobo_figshare_files']=bfiles
    small_tables=[]
    for x in bfiles:
        pp=RAW/('bonobo_'+Path(x['name']).name)
        if pp.exists():
            df=safe_read_table(pp)
            if df is not None:
                small_tables.append({'name':x['name'],'n':len(df),'columns':list(df.columns)})
    discovery['bonobo_small_tables']=small_tables
    ledger.append({
        'species':'Bonobo','capability':'semantic composition','dataset':'Berthet et al. 2025 Figshare',
        'access_metric':'repertoire penetration: fraction of call types occurring in at least one compositional combination','access_value':1.0,
        'selection_metric':'non-trivial compositional structures / tested compositional structures','selection_value':3/4,
        'stability_metric':'reuse across individuals/contexts','stability_value':np.nan,
        'access_identified':'supported directly by publication statement','selection_identified':'supported directly by publication statement','stability_identified':'raw files discovered; requires table-specific computation',
        'notes':'Publication reports every call type occurs in at least one compositional combination and 3 of 4 compositional structures are non-trivial.'
    })
except Exception as e:
    discovery['bonobo_error']=repr(e)

# ---------------- ORANGUTAN TEMPORAL HIERARCHY (OSF) ----------------
try:
    root=get_json('https://api.osf.io/v2/nodes/w3ne5/files/')
    providers=root.get('data',[])
    ofiles=[]
    def walk(url, depth=0):
        if depth>3: return
        jj=get_json(url)
        for item in jj.get('data',[]):
            a=item.get('attributes',{}); links=item.get('links',{})
            kind=a.get('kind')
            if kind=='file':
                ofiles.append({'name':a.get('name'),'size':a.get('size'),'download_url':links.get('download')})
                if a.get('size') and a.get('size') <= 25_000_000 and links.get('download'):
                    pp=RAW/('orangutan_'+Path(a['name']).name)
                    try: dl(links['download'],pp)
                    except Exception: pass
            elif kind=='folder' and links.get('move'):
                pass
            # OSF folder child endpoint typically links.files
            child=links.get('files') or links.get('new_folder')
        nxt=jj.get('links',{}).get('next')
        if nxt: walk(nxt,depth)
    for pvd in providers:
        files_link=pvd.get('relationships',{}).get('files',{}).get('links',{}).get('related',{}).get('href')
        if files_link: walk(files_link)
    discovery['orangutan_osf_files']=ofiles
    ledger.append({
        'species':'Orangutan','capability':'temporal hierarchical self-embedding','dataset':'Lameira et al. 2024 OSF/eLife',
        'access_metric':'observed presence of two hierarchical rhythmic strata','access_value':2.0,
        'selection_metric':'three lower-stratum rhythms not reducible to low-multiple relation with higher rhythm','selection_value':3.0,
        'stability_metric':'repetition/reuse across calls/elements','stability_value':8993.0,
        'access_identified':'structural-depth lower bound, not probability','selection_identified':'published mechanistic exclusion count, not preference probability','stability_identified':'sample support count only, not state persistence',
        'notes':'Published dataset reports 66 calls and 8,993 elements; this is temporal hierarchy, not semantic syntax.'
    })
except Exception as e:
    discovery['orangutan_error']=repr(e)

# ---------------- MARMOSET VOCAL LABELING (ZENODO) ----------------
try:
    z=get_json('https://zenodo.org/api/records/12721811')
    zfiles=[]; mats=[]
    for f in z.get('files',[]):
        name=f['key']; size=f['size']; url=f['links']['self']
        zfiles.append({'name':name,'size':size,'url':url})
        if name.lower().endswith('.mat') and size <= 10_000_000:
            pp=RAW/('marmoset_'+Path(name).name); dl(url,pp)
            try:
                m=loadmat(pp, squeeze_me=True, struct_as_record=False)
                mats.append({'name':name,'variables':{k:str(np.shape(v)) for k,v in m.items() if not k.startswith('__')}})
            except Exception as e:
                mats.append({'name':name,'load_error':repr(e)})
    discovery['marmoset_zenodo_files']=zfiles
    discovery['marmoset_small_mat_shapes']=mats
    ledger.append({
        'species':'Common marmoset','capability':'vocal labeling of conspecifics','dataset':'Oren et al. 2024 Zenodo',
        'access_metric':'number of individually discriminable vocal labels / tested conspecific identities','access_value':7.0,
        'selection_metric':'reported mean label accuracy','selection_value':0.9129,
        'stability_metric':'cross-exemplar/caller generalization from figure-level data','stability_value':np.nan,
        'access_identified':'published lower bound / label support size','selection_identified':'direct reported accuracy','stability_identified':'small figure data downloaded for next exact pass',
        'notes':'Figure .mat files inspected; 91.29% is classifier/validation accuracy for MarmAudio taxonomy and must not be conflated with identity-label behavior unless explicitly linked. Keep these constructs separate.'
    })
except Exception as e:
    discovery['marmoset_error']=repr(e)

pd.DataFrame(ledger).to_csv(OUT/'comparative_cost_ledger_v1.csv', index=False)
(OUT/'discovery.json').write_text(json.dumps(discovery, indent=2, default=str))

print('=== COMPARATIVE COST LEDGER ===')
print(pd.DataFrame(ledger).to_string(index=False))
print('\n=== DISCOVERY SUMMARY ===')
print(json.dumps(discovery, indent=2, default=str)[:30000])
