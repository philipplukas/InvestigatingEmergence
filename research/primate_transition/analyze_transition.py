from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_URL = "https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionMonkeys.csv"
OUT = Path(os.environ.get("OUT_DIR", "research/primate_transition/out"))
OUT.mkdir(parents=True, exist_ok=True)
CATS = ["Center Embedded", "Crossed", "Tail Embedded", "All Other"]
ZMAP = {"Center Embedded": 1.0, "Crossed": -1.0, "Tail Embedded": 0.0, "All Other": 0.0}
EPS = 1e-12


def entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def const_fit(y):
    counts = np.bincount(y, minlength=4).astype(float)
    p = (counts + 0.5) / (counts.sum() + 2.0)
    ll = float(np.log(np.clip(p[y], EPS, 1)).sum())
    bic = -2 * ll + 3 * np.log(len(y))
    return ll, float(bic), p


def trend_fit(y):
    x = np.linspace(-1, 1, len(y)).reshape(-1, 1)
    m = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
    m.fit(x, y)
    P = np.full((len(y), 4), EPS)
    P[:, m.classes_.astype(int)] = m.predict_proba(x)
    ll = float(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).sum())
    bic = -2 * ll + 6 * np.log(len(y))
    return ll, float(bic)


def change_fit(y, min_seg=20):
    best = None
    for cp in range(min_seg, len(y) - min_seg + 1):
        ll1, _, p1 = const_fit(y[:cp])
        ll2, _, p2 = const_fit(y[cp:])
        ll = ll1 + ll2
        bic = -2 * ll + 7 * np.log(len(y))
        rec = (float(bic), cp, float(ll), p1, p2)
        if best is None or rec[0] < best[0]:
            best = rec
    return best


def fb(y, pi, A, B):
    n = len(y)
    a = np.zeros((n, 2)); s = np.zeros(n)
    a[0] = pi * B[:, y[0]]; s[0] = a[0].sum() + EPS; a[0] /= s[0]
    for t in range(1, n):
        a[t] = (a[t-1] @ A) * B[:, y[t]]; s[t] = a[t].sum() + EPS; a[t] /= s[t]
    b = np.ones((n, 2))
    for t in range(n-2, -1, -1):
        b[t] = A @ (B[:, y[t+1]] * b[t+1]); b[t] /= s[t+1]
    g = a * b; g /= g.sum(axis=1, keepdims=True)
    xi = np.zeros((2, 2))
    for t in range(n-1):
        x = a[t][:,None] * A * (B[:, y[t+1]] * b[t+1])[None,:]
        xi += x / (x.sum() + EPS)
    return float(np.log(s + EPS).sum()), g, xi


def hmm_fit(y, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.5]); A = np.array([[0.94,0.06],[0.06,0.94]], float)
    B = rng.dirichlet(np.ones(4), size=2)
    for _ in range(iters):
        ll, g, xi = fb(y, pi, A, B)
        pi = g[0] + 1e-3; pi /= pi.sum()
        A = xi + 1e-2; A /= A.sum(axis=1, keepdims=True)
        BN = np.full((2,4), 1e-2)
        for c in range(4): BN[:,c] += g[y == c].sum(axis=0)
        B = BN / BN.sum(axis=1, keepdims=True)
    ll, g, _ = fb(y, pi, A, B)
    bic = -2 * ll + 9 * np.log(len(y))
    ce_state = int(np.argmax(B[:,0]))
    return {"ll":ll,"bic":float(bic),"pi":pi,"A":A,"B":B,"occ":g[:,ce_state],"ce_state":ce_state}


def rolling(g, w):
    z = g["z"].to_numpy(float); r = g["Response Type"].to_numpy(str); out=[]
    for end in range(w, len(g)+1):
        rs=r[end-w:end]; zs=z[end-w:end]
        c=np.array([(rs==x).sum() for x in CATS],float)
        rho=float(np.corrcoef(zs[:-1],zs[1:])[0,1]) if np.std(zs[:-1])>0 and np.std(zs[1:])>0 else np.nan
        out.append({"end_index":end,"q":float((c[0]-c[1])/w),"entropy":entropy(c),"variance":float(np.var(zs,ddof=1)),"rho1":rho,"switch_rate":float(np.mean(rs[1:]!=rs[:-1])),"p_ce":c[0]/w,"p_cross":c[1]/w,"p_tail":c[2]/w,"p_other":c[3]/w})
    return pd.DataFrame(out)


def main():
    df=pd.read_csv(DATA_URL)
    df=df[df["Response Type"].isin(CATS)].copy()
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df=df.sort_values(["Sub","Exposure","Date","List","Trial","Press"],kind="stable")
    df["z"]=df["Response Type"].map(ZMAP)
    ids={c:i for i,c in enumerate(CATS)}
    summary=[]; rolls=[]; details={}
    for sub,g0 in df.groupby("Sub",sort=True):
        g=g0.reset_index(drop=True); y=g["Response Type"].map(ids).to_numpy(int)
        if len(y)<50: continue
        ll0,b0,p0=const_fit(y); ll1,b1=trend_fit(y); cp=change_fit(y)
        hmm=min([hmm_fit(y,s) for s in range(5)],key=lambda d:d["bic"])
        models={"M0_constant":b0,"M1_smooth":b1,"M2_change_point":cp[0],"M3_hmm":hmm["bic"]}
        best=sorted(models.items(),key=lambda kv:kv[1])
        cross=np.where(hmm["occ"]>=0.5)[0]
        summary.append({"Sub":sub,"n_trials":len(y),"exposures":",".join(map(str,sorted(g["Exposure"].dropna().unique()))),"p_ce":float(np.mean(y==0)),"p_cross":float(np.mean(y==1)),"q_global":float(np.mean(g["z"])),"entropy_global":entropy(np.bincount(y,minlength=4)),"M0_bic":b0,"M1_bic":b1,"M2_bic":cp[0],"M3_bic":hmm["bic"],"best_model":best[0][0],"delta_bic_second":float(best[1][1]-best[0][1]),"change_point_trial":int(cp[1]),"hmm_first_occ_ge_0_5":int(cross[0]+1) if cross.size else None,"hmm_A00":float(hmm["A"][0,0]),"hmm_A11":float(hmm["A"][1,1])})
        details[sub]={"constant_probs":p0.tolist(),"change_point":int(cp[1]),"cp_pre_probs":cp[3].tolist(),"cp_post_probs":cp[4].tolist(),"hmm":{"pi":hmm["pi"].tolist(),"A":hmm["A"].tolist(),"B":hmm["B"].tolist(),"ce_state":int(hmm["ce_state"]),"occupancy_first_20":hmm["occ"][:20].tolist(),"occupancy_last_20":hmm["occ"][-20:].tolist()}}
        for w in (20,40,80):
            if len(g)>=w:
                rr=rolling(g,w); rr.insert(0,"window",w); rr.insert(0,"Sub",sub); rolls.append(rr)
    sdf=pd.DataFrame(summary).sort_values("Sub")
    sdf.to_csv(OUT/"subject_model_summary.csv",index=False)
    pd.concat(rolls,ignore_index=True).to_csv(OUT/"rolling_transition_metrics.csv",index=False)
    (OUT/"model_details.json").write_text(json.dumps(details,indent=2))
    (OUT/"run_summary.json").write_text(json.dumps({"source":DATA_URL,"n_completed_sequences":int(len(df)),"subjects":summary},indent=2))
    print(sdf.to_string(index=False))

if __name__ == "__main__": main()
