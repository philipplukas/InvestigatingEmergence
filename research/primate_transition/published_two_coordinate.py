from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

OUT=Path('research/primate_transition/out_published')
OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(20260829)

# Exact aggregate counts reported in Ferrigno et al. 2020 simple transfer analyses.
# E3 uses the raw/published 30-list generalization aggregate already validated in prior analysis.
DATA={
 'US_Adults':       {'total':240,'CE':224,'Cross':16},
 'Tsimane_Adults':  {'total':251,'CE':157,'Cross':56},
 'US_Children':     {'total':500,'CE':217,'Cross':84},
 'Macaque_E1':      {'total':180,'CE':47, 'Cross':50},
 'Macaque_E2':      {'total':200,'CE':49, 'Cross':19},
 'Macaque_E3':      {'total':450,'CE':135,'Cross':109},
}

def metrics(d):
    ce,cr,total=d['CE'],d['Cross'],d['total']
    structured=ce+cr
    p=ce/structured
    q=(ce-cr)/structured
    cov=structured/total
    gap=np.log((cr+.5)/(ce+.5))
    return dict(total=total,CE=ce,Cross=cr,structured=structured,
                p_CE_given_structured=p,q_rec=q,structural_coverage=cov,
                recursive_energy_gap_log_cross_over_ce=gap,
                unclassified_or_tail=total-structured)

rows=[]
for name,d in DATA.items():
    x=metrics(d); x['cohort']=name; rows.append(x)
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'published_two_coordinate_summary.csv',index=False)

pairs=[
 ('US_Children','Macaque_E2'),
 ('US_Children','Macaque_E1'),
 ('US_Adults','US_Children'),
 ('Tsimane_Adults','US_Children'),
 ('US_Adults','Tsimane_Adults'),
 ('US_Adults','Macaque_E2'),
]
comparisons=[]
for a,b in pairs:
    A=DATA[a]; B=DATA[b]
    # orientation only within structured subspace
    _,p_orient=fisher_exact([[A['CE'],A['Cross']],[B['CE'],B['Cross']]],alternative='two-sided')
    # entry into structured subspace
    As=A['CE']+A['Cross']; Bs=B['CE']+B['Cross']
    _,p_cov=fisher_exact([[As,A['total']-As],[Bs,B['total']-Bs]],alternative='two-sided')
    ma,mb=metrics(A),metrics(B)
    comparisons.append(dict(cohort_a=a,cohort_b=b,
        delta_q_rec=ma['q_rec']-mb['q_rec'],orientation_fisher_p=p_orient,
        delta_structural_coverage=ma['structural_coverage']-mb['structural_coverage'],coverage_fisher_p=p_cov,
        delta_recursive_energy_gap=ma['recursive_energy_gap_log_cross_over_ce']-mb['recursive_energy_gap_log_cross_over_ce']))
cmp=pd.DataFrame(comparisons)
cmp.to_csv(OUT/'published_pairwise_tests.csv',index=False)

# Child developmental robustness on public raw subset: participant is the unit.
URL='https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionKids.csv'
df=pd.read_csv(URL)
df=df[df['Order pressed'].notna()].copy()
def cls(s):
    s=str(s).strip().upper()
    if s in {'ACDB','CABD'}: return 'CE'
    if s in {'ACBD','CADB'}: return 'Cross'
    if s in {'ABCD','CDAB'}: return 'Tail'
    return 'Other'
df['strategy']=df['Order pressed'].map(cls)
df['Age']=pd.to_numeric(df['Age'],errors='coerce')
sub=[]
for sid,g in df.groupby('Sub'):
    age=g['Age'].dropna()
    if age.empty: continue
    ce=(g.strategy=='CE').sum(); cr=(g.strategy=='Cross').sum(); total=len(g); st=ce+cr
    sub.append(dict(Sub=sid,Age=float(age.iloc[0]),n=total,CE=ce,Cross=cr,
                    structural_coverage=st/total,
                    q_rec=(ce-cr)/st if st else np.nan,
                    ce_cond=ce/st if st else np.nan))
sub=pd.DataFrame(sub).sort_values('Age')
sub.to_csv(OUT/'child_subject_two_coordinate.csv',index=False)

def perm_spearman(x,y,nperm=50000):
    ok=np.isfinite(x)&np.isfinite(y); x=np.asarray(x)[ok]; y=np.asarray(y)[ok]
    rho=float(spearmanr(x,y).statistic)
    vals=np.empty(nperm)
    for i in range(nperm): vals[i]=spearmanr(x,RNG.permutation(y)).statistic
    p=(1+np.sum(np.abs(vals)>=abs(rho)))/(nperm+1)
    return rho,float(p),len(x)

age_tests=[]
for out in ['structural_coverage','q_rec','ce_cond']:
    rho,p,n=perm_spearman(sub.Age.to_numpy(),sub[out].to_numpy())
    age_tests.append(dict(outcome=out,n_children=n,spearman_rho=rho,permutation_p=p))
age=pd.DataFrame(age_tests)
age.to_csv(OUT/'child_age_permutation_tests.csv',index=False)

sub['age_bin']=pd.cut(sub.Age,bins=[3,4,5.01],right=False,labels=['3_to_lt4','4_to_5'])
bins=sub.groupby('age_bin',observed=True).agg(n_children=('Sub','size'),mean_age=('Age','mean'),
       mean_coverage=('structural_coverage','mean'),mean_q=('q_rec','mean'),median_q=('q_rec','median')).reset_index()
bins.to_csv(OUT/'child_age_bins.csv',index=False)

C=metrics(DATA['US_Children']); M2=metrics(DATA['Macaque_E2'])
verdict={
 'child_vs_macaque_E2_delta_q':C['q_rec']-M2['q_rec'],
 'child_vs_macaque_E2_delta_CE_cond':C['p_CE_given_structured']-M2['p_CE_given_structured'],
 'child_vs_macaque_E2_delta_coverage':C['structural_coverage']-M2['structural_coverage'],
 'interpretation':'orientation_matched_access_not_matched' if cmp.iloc[0].orientation_fisher_p>0.1 and cmp.iloc[0].coverage_fisher_p<0.01 else 'other',
 'raw_child_subset_n':int(len(sub)),
}
(OUT/'published_two_coordinate_verdict.json').write_text(json.dumps(verdict,indent=2))
print('=== PUBLISHED TWO COORDINATE ==='); print(summary.to_string(index=False))
print('=== PAIRWISE ==='); print(cmp.to_string(index=False))
print('=== CHILD AGE PARTICIPANT-LEVEL ==='); print(age.to_string(index=False))
print('=== AGE BINS ==='); print(bins.to_string(index=False))
print('=== VERDICT ==='); print(json.dumps(verdict,indent=2))
