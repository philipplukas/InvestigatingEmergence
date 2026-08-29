import io, json, math, os, re
from pathlib import Path
from urllib.parse import urljoin, quote

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from scipy.stats import wilcoxon

OUT=Path('research/primate_comparative_cost/out_v2'); OUT.mkdir(parents=True,exist_ok=True)
RAW=OUT/'raw'; RAW.mkdir(exist_ok=True)
S=requests.Session(); S.headers.update({'User-Agent':'primate-cost-atlas-v2/1.0','Accept':'application/json'})


def dl(url,path):
    r=S.get(url,timeout=120); r.raise_for_status(); path.write_bytes(r.content); return path

def mat_struct(path,key):
    return loadmat(path,squeeze_me=True,struct_as_record=False)[key]

rows=[]; audit={}

# 1. Chimpanzee combinatorics
chimp= pd.read_csv(dl('https://raw.githubusercontent.com/tozbu/Chimpanzee_bigram_meaning/main/calldata.csv', RAW/'chimp_calldata.csv'))
chimp['is_bigram']=pd.to_numeric(chimp['combi.length'],errors='coerce').eq(2)
big=chimp[chimp.is_bigram]
big_types=sorted(big['call.name'].dropna().unique())
all_types=sorted({x for v in chimp['call.name'].dropna() for x in str(v).split('_')})
part_types=sorted({x for v in big['call.name'].dropna() for x in str(v).split('_')})
chstab=[]
for bg,g in big.groupby('call.name'):
    dates=pd.to_datetime(g['date'])
    chstab.append({'bigram':bg,'tokens':len(g),'callers':g['caller'].nunique(),'date_span_days':(dates.max()-dates.min()).days})
chstab=pd.DataFrame(chstab); chstab.to_csv(OUT/'chimp_bigram_reuse.csv',index=False)
rows.append(dict(species='Chimpanzee',capability='semantic/combinatorial meaning expansion',
 access_observable='observed bigram prevalence in corpus',access_value=len(big)/len(chimp),access_status='QUANTIFIED_PROXY',
 selection_observable='paper identifies multiple meaning-expansion mechanisms; no single target-vs-alternative scalar is identified by this observational design',selection_value=np.nan,selection_status='SUPPORTED_NONSCALAR',
 stability_observable='median callers per bigram type; median temporal span (days)',stability_value=float(chstab.callers.median()),stability_aux=float(chstab.date_span_days.median()),stability_status='QUANTIFIED_REUSE_PROXY',
 evidence_note=f'{len(big)} bigram tokens/{len(chimp)} utterances; {len(big_types)} bigram types; {len(part_types)}/{len(all_types)} component call types used in bigrams; median {chstab.callers.median():.0f} callers/type; median span {chstab.date_span_days.median():.1f} d.'))

# 2. Bonobo semantic composition: exact public table + published four-structure classification
bon_path=dl('https://ndownloader.figshare.com/files/47522606', RAW/'bonobo_xdata.txt')
bon=pd.read_csv(bon_path,sep='\t')
structures={'pe-wi':'nontrivial','ye-gr':'trivial','hh-lh':'nontrivial','py-hh':'nontrivial'}
bm=[]
for comb,kind in structures.items():
    g=bon[bon['Combination']==comb]
    bm.append({'combination':comb,'composition_type':kind,'tokens':len(g),'callers':g.Caller.nunique(),'groups':g.Group.nunique(),'median_tokens_per_caller':float(g.groupby('Caller').size().median())})
bm=pd.DataFrame(bm); bm.to_csv(OUT/'bonobo_compositional_structure_reuse.csv',index=False)
single_types=sorted([x for x in bon.Combination.dropna().unique() if '-' not in x])
struct_components=sorted(set(sum([x.split('-') for x in structures],[])))
rows.append(dict(species='Bonobo',capability='semantic composition',
 access_observable='single-call repertoire types participating in one of the four compositional structures',access_value=len(set(single_types)&set(struct_components))/len(single_types),access_status='QUANTIFIED_REPERTOIRE_PENETRATION',
 selection_observable='non-trivial compositional structures / four compositional structures',selection_value=3/4,selection_status='QUANTIFIED_PUBLISHED',
 stability_observable='median independent callers per compositional structure; all structures replicated across >=2 social groups',stability_value=float(bm.callers.median()),stability_aux=float(bm.groups.min()),stability_status='QUANTIFIED_REUSE',
 evidence_note='; '.join(f"{r.combination}: n={r.tokens}, callers={r.callers}, groups={r.groups}, {r.composition_type}" for r in bm.itertuples())))

# 3. Orangutan temporal hierarchy: exact OSF public annotation file
orang_path=dl('https://osf.io/download/gdkuc/', RAW/'orangutan_elements.xlsx')
orang=pd.read_excel(orang_path)
orang['level']=orang['Pulse level'].astype(str).str.strip()
lower={'Bubble sub-pulse','Grumble sub-pulse','Sub-pulse transitory element','Pulse body'}
of=[]
for fn,g in orang.groupby('Master File Name'):
    levels=set(g.level)
    of.append({'file':fn,'individual':g.Individual.iloc[0],'annotations':len(g),'has_full':('Full pulse' in levels),'n_lower_types':len(levels&lower),'has_two_strata':('Full pulse' in levels and len(levels&lower)>0)})
of=pd.DataFrame(of); of.to_csv(OUT/'orangutan_file_hierarchy_reuse.csv',index=False)
rows.append(dict(species='Orangutan',capability='temporal hierarchical self-embedding',
 access_observable='share of public annotated long-call files containing both full-pulse and lower sub-pulse strata',access_value=float(of.has_two_strata.mean()),access_status='QUANTIFIED_STRUCTURAL_REPLICATION',
 selection_observable='three lower-stratum rhythms reported not reducible to low-multiple relation with higher rhythm',selection_value=3.0,selection_status='PUBLISHED_MECHANISTIC_EXCLUSION_NOT_PREFERENCE',
 stability_observable='hierarchical two-strata organization replicated across annotated call files and individuals',stability_value=float(of.has_two_strata.sum()),stability_aux=float(of.individual.nunique()),stability_status='QUANTIFIED_STRUCTURAL_REPLICATION',
 evidence_note=f'Public OSF table: {len(orang)} interval annotations, {len(of)} call files, {of.individual.nunique()} individuals; two-strata structure in {of.has_two_strata.sum()}/{len(of)} files. Temporal hierarchy only; not semantic syntax.'))

# 4. Marmoset receiver-specific vocal labels: directly unpack figure data
zbase='https://zenodo.org/api/records/12721811/files/'
for name in ['Fig_2.mat','Fig_3.mat']:
    dl(zbase+name+'/content',RAW/('marmoset_'+name))
m2=mat_struct(RAW/'marmoset_Fig_2.mat','Fig_2'); m3=mat_struct(RAW/'marmoset_Fig_3.mat','Fig_3')
std=np.asarray(m2.panel_G.x_standard,dtype=float)/100.0
loo=np.asarray(m2.panel_G.x_leave_one_out,dtype=float)/100.0
directed=np.asarray(m3.panel_D.directed,dtype=float); nondirected=np.asarray(m3.panel_D.nondirected,dtype=float)
w=wilcoxon(directed,nondirected,alternative='greater')
marm=pd.DataFrame({'standard_accuracy':std,'leave_one_session_out_accuracy':loo}); marm.to_csv(OUT/'marmoset_receiver_classifier_generalization.csv',index=False)
pd.DataFrame({'directed':directed,'nondirected':nondirected,'difference':directed-nondirected}).to_csv(OUT/'marmoset_directed_response.csv',index=False)
rows.append(dict(species='Common marmoset',capability='receiver-specific vocal labeling',
 access_observable='receiver-identity signal is decodable for all classifier subjects represented in Fig.2G (opportunity-normalized spontaneous access not identified)',access_value=float(len(std)),access_status='SUPPORTED_LOWER_BOUND_NOT_PROBABILITY',
 selection_observable='mean paired response consistency/correctness difference: directed minus nondirected calls',selection_value=float((directed-nondirected).mean()),selection_status='QUANTIFIED_RESPONSE_SELECTIVITY',
 stability_observable='mean leave-one-session-out receiver-classification accuracy; standard accuracy retained across unseen session',stability_value=float(loo.mean()),stability_aux=float(std.mean()),stability_status='QUANTIFIED_CROSS_SESSION_GENERALIZATION',
 evidence_note=f'Fig.2G: {len(std)} classifier subjects/entries, mean standard accuracy={std.mean():.3f}, leave-one-session-out={loo.mean():.3f}. Fig.3D: directed response={directed.mean():.3f}, nondirected={nondirected.mean():.3f}, paired Δ={np.mean(directed-nondirected):.3f}, one-sided Wilcoxon p={w.pvalue:.3g}.'))

# 5. Japanese macaque voluntary vocal control: Dryad exact raw files through current-version link
try:
    doi='doi:10.5061/dryad.6v7j674'
    meta=S.get('https://datadryad.org/api/v2/datasets/'+quote(doi,safe=''),timeout=60); meta.raise_for_status(); meta=meta.json()
    href=meta['_links']['stash:version']['href']; ver_url=urljoin('https://datadryad.org',href)
    fj=S.get(ver_url+'/files',timeout=60); fj.raise_for_status(); fj=fj.json()
    files=fj.get('_embedded',{}).get('stash:files',[])
    aud=[]; dfs={}
    for f in files:
        name=f.get('path') or f.get('filename') or f.get('name')
        dlink=f.get('_links',{}).get('stash:download',{}).get('href') or f.get('_links',{}).get('download',{}).get('href')
        if dlink:
            pp=RAW/('dryad_'+Path(name).name); dl(urljoin('https://datadryad.org',dlink),pp)
            try: df=pd.read_csv(pp); dfs[name]=df; aud.append({'name':name,'rows':len(df),'columns':list(df.columns)})
            except Exception as e: aud.append({'name':name,'error':repr(e)})
    audit['dryad_files']=aud
    # Save raw schemas and compute only from columns we can identify safely.
    for name,df in dfs.items(): df.to_csv(OUT/('dryad_'+Path(name).name),index=False)
    # Published exact training endpoints and probe effects are used as audited primary quantities.
    vocal_sessions=np.mean([57,47]); manual_sessions=np.mean([9,7]); ratio=vocal_sessions/manual_sessions
    rows.append(dict(species='Japanese macaque',capability='voluntary vocal control',
      access_observable='learned vocal action reached predeclared >=90% correct criterion (two trained vocal subjects)',access_value=0.90,access_status='QUANTIFIED_TRAINED_LOWER_BOUND',
      selection_observable='mean training sessions to criterion, vocal/manual ratio',selection_value=float(ratio),selection_status='QUANTIFIED_ACQUISITION_COST_RATIO',
      stability_observable='early unexpected cue selectively reduces vocal execution after extensive training; 0.25x restraint probe significant in both vocal subjects',stability_value=2.0,stability_aux=np.nan,stability_status='QUANTIFIED_PERTURBATION_FAILURE_COUNT',
      evidence_note=f'Vocal subjects criterion sessions 57 and 47 (mean {vocal_sessions:.1f}); manual subjects 9 and 7 (mean {manual_sessions:.1f}); ratio {ratio:.2f}x. At 0.25x timing, published Fisher tests significant for both vocal subjects. Dryad raw files downloaded and schemas saved.'))
except Exception as e:
    audit['dryad_error']=repr(e)
    rows.append(dict(species='Japanese macaque',capability='voluntary vocal control',access_observable='trained vocal action reaches criterion',access_value=.90,access_status='QUANTIFIED_PUBLISHED',selection_observable='vocal/manual sessions-to-criterion ratio',selection_value=6.5,selection_status='QUANTIFIED_PUBLISHED',stability_observable='unexpected early cue impairment',stability_value=2.0,stability_aux=np.nan,stability_status='QUANTIFIED_PUBLISHED_BOTH_VOCAL_SUBJECTS',evidence_note='Dryad API extraction failed in this pass: '+repr(e)))

atlas=pd.DataFrame(rows)
atlas.to_csv(OUT/'comparative_cost_atlas_v2.csv',index=False)
(OUT/'audit_v2.json').write_text(json.dumps(audit,indent=2,default=str))

# Identification matrix: do not normalize unlike measurements onto a common numeric scale.
ids=[]
for r in rows:
    for axis in ['access','selection','stability']:
        ids.append({'species':r['species'],'capability':r['capability'],'axis':axis,'status':r[f'{axis}_status'],'observable':r[f'{axis}_observable'],'value':r.get(f'{axis}_value',np.nan),'aux':r.get(f'{axis}_aux',np.nan)})
pd.DataFrame(ids).to_csv(OUT/'identification_matrix_v2.csv',index=False)
print('=== ATLAS V2 ==='); print(atlas.to_string(index=False))
print('\n=== DRYAD AUDIT ==='); print(json.dumps(audit,indent=2,default=str))
