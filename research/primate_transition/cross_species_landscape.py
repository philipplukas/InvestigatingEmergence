from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest

OUT = Path("research/primate_transition/out_cross_species")
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "Macaque": "https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionMonkeys.csv",
    "USAdults": "https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionUSAdults.csv",
    "Kids": "https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionKids.csv",
    "Tsimane": "https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionTsimane.csv",
}

CATS = ["Center Embedded", "Crossed", "Tail Embedded", "All Other"]
SEQ_CLASS = {
    "ACDB": "Center Embedded",
    "CABD": "Center Embedded",
    "ACBD": "Crossed",
    "CADB": "Crossed",
    "ABCD": "Tail Embedded",
    "CDAB": "Tail Embedded",
}
ALPHA = 0.5
RNG = np.random.default_rng(20260829)


def classify(seq):
    if pd.isna(seq):
        return None
    s = str(seq).strip().upper().replace(" ", "")
    return SEQ_CLASS.get(s, "All Other") if len(s) == 4 else None


def normalize_author_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().casefold()
    m = {
        "center embedded": "Center Embedded",
        "crossed": "Crossed",
        "tail embedded": "Tail Embedded",
        "all other": "All Other",
    }
    return m.get(s)


def read_cohorts():
    frames = []

    m = pd.read_csv(URLS["Macaque"])
    m = m[m["Order pressed"].notna()].copy()
    m["Subject"] = m["Sub"].astype(str).str.strip().str.casefold().str.title()
    m = m[m["Subject"].isin(["Beyonce", "Coltrane", "Horatio"])].copy()
    m["Exposure"] = pd.to_numeric(m["Exposure"], errors="coerce")
    m = m[m["Exposure"].isin([1, 2, 3])].copy()
    m["Strategy"] = m["Order pressed"].map(classify)
    m["AuthorStrategy"] = m["Response Type"].map(normalize_author_label)
    m["Cohort"] = "Macaque_E" + m["Exposure"].astype(int).astype(str)
    m["OrderIndex"] = pd.to_datetime(m["Date"].astype(str).str.replace(".", "-", regex=False), errors="coerce")
    m["TrialOrder"] = pd.to_numeric(m["Trial"], errors="coerce")
    m["Age"] = np.nan
    frames.append(m[["Cohort","Subject","Strategy","AuthorStrategy","OrderIndex","TrialOrder","Age","Order pressed"]])

    a = pd.read_csv(URLS["USAdults"])
    a = a[a["Order pressed"].notna()].copy()
    if "Session" in a.columns:
        testmask = a["Session"].astype(str).str.contains("test", case=False, na=False)
        if testmask.any():
            a = a[testmask].copy()
    a["Subject"] = a["Sub"].astype(str).str.strip()
    a["Strategy"] = a["Order pressed"].map(classify)
    a["AuthorStrategy"] = a["Response type"].map(normalize_author_label)
    a["Cohort"] = "US_Adults"
    a["OrderIndex"] = pd.to_datetime(a["Date"].astype(str).str.replace(".", "-", regex=False), errors="coerce")
    a["TrialOrder"] = pd.to_numeric(a["Trial"], errors="coerce")
    a["Age"] = np.nan
    frames.append(a[["Cohort","Subject","Strategy","AuthorStrategy","OrderIndex","TrialOrder","Age","Order pressed"]])

    k = pd.read_csv(URLS["Kids"])
    k = k[k["Order pressed"].notna()].copy()
    if "Test or Training" in k.columns:
        testmask = k["Test or Training"].astype(str).str.contains("test", case=False, na=False)
        if testmask.any():
            k = k[testmask].copy()
    k["Subject"] = k["Sub"].astype(str).str.strip()
    k["Strategy"] = k["Order pressed"].map(classify)
    k["AuthorStrategy"] = k["Response type"].map(normalize_author_label)
    k["Cohort"] = "US_Children"
    k["OrderIndex"] = pd.to_datetime(k["Date"].astype(str).str.replace(".", "-", regex=False), errors="coerce")
    k["TrialOrder"] = pd.to_numeric(k["Trial"], errors="coerce")
    k["Age"] = pd.to_numeric(k["Age"], errors="coerce")
    frames.append(k[["Cohort","Subject","Strategy","AuthorStrategy","OrderIndex","TrialOrder","Age","Order pressed"]])

    t = pd.read_csv(URLS["Tsimane"])
    if "Probe?" in t.columns:
        probe = pd.to_numeric(t["Probe?"], errors="coerce")
        if (probe == 1).any():
            t = t[probe == 1].copy()
    t = t[t["Order pressed"].notna()].copy()
    t["Subject"] = t["Sub"].astype(str).str.strip()
    t["Strategy"] = t["Order pressed"].map(classify)
    t["AuthorStrategy"] = None
    t["Cohort"] = "Tsimane_Adults"
    t["OrderIndex"] = pd.NaT
    t["TrialOrder"] = pd.to_numeric(t["Probe Number"], errors="coerce")
    t["Age"] = np.nan
    frames.append(t[["Cohort","Subject","Strategy","AuthorStrategy","OrderIndex","TrialOrder","Age","Order pressed"]])

    df = pd.concat(frames, ignore_index=True)
    df = df[df["Strategy"].isin(CATS)].copy()
    return df


def counts_for(g):
    return np.array([(g["Strategy"] == c).sum() for c in CATS], dtype=float)


def metrics_from_counts(counts):
    n = counts.sum()
    sm = counts + ALPHA
    p = sm / sm.sum()
    ce, cross, tail, other = counts
    denom = ce + cross
    q = (ce - cross) / denom if denom > 0 else np.nan
    coverage = denom / n if n > 0 else np.nan
    ce_cond = ce / denom if denom > 0 else np.nan
    entropy = -float(np.sum(p * np.log(p)))
    energies = -np.log(p)
    energies -= energies.min()
    rec_gap = math.log((cross + ALPHA) / (ce + ALPHA))
    return p, energies, q, coverage, ce_cond, entropy, rec_gap


def participant_bootstrap(g, n_boot=5000):
    subjects = pd.unique(g["Subject"])
    per_sub = {s: counts_for(g[g["Subject"] == s]) for s in subjects}
    qs, covs, ces = [], [], []
    for _ in range(n_boot):
        sampled = RNG.choice(subjects, size=len(subjects), replace=True)
        c = np.sum([per_sub[s] for s in sampled], axis=0)
        _, _, q, cov, ce, _, _ = metrics_from_counts(c)
        qs.append(q); covs.append(cov); ces.append(ce)
    def ci(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return (float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    return ci(qs), ci(covs), ci(ces)


def mismatch_audit(df):
    x = df[df["AuthorStrategy"].notna()].copy()
    x["match"] = x["Strategy"] == x["AuthorStrategy"]
    rows = []
    for cohort, g in x.groupby("Cohort"):
        rows.append({
            "Cohort": cohort,
            "n_author_labeled": len(g),
            "n_matches": int(g["match"].sum()),
            "n_mismatches": int((~g["match"]).sum()),
            "match_rate": float(g["match"].mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def cohort_summaries(df):
    rows, energy_rows = [], []
    order = ["Macaque_E1","Macaque_E2","Macaque_E3","US_Children","US_Adults","Tsimane_Adults"]
    for cohort in order:
        g = df[df["Cohort"] == cohort]
        if g.empty:
            continue
        c = counts_for(g)
        p, e, q, cov, ce_cond, ent, gap = metrics_from_counts(c)
        qci, covci, ceci = participant_bootstrap(g)
        denom = int(c[0] + c[1])
        pbin = float(binomtest(int(c[0]), denom, .5, alternative="two-sided").pvalue) if denom else np.nan
        row = {
            "Cohort": cohort,
            "n_participants": g["Subject"].nunique(),
            "n_trials": len(g),
            "n_CE": int(c[0]), "n_Crossed": int(c[1]), "n_Tail": int(c[2]), "n_Other": int(c[3]),
            "q_rec": q, "q_rec_ci_lo": qci[0], "q_rec_ci_hi": qci[1],
            "structural_coverage": cov, "coverage_ci_lo": covci[0], "coverage_ci_hi": covci[1],
            "p_CE_given_CE_or_Cross": ce_cond, "ce_cond_ci_lo": ceci[0], "ce_cond_ci_hi": ceci[1],
            "CE_vs_Cross_binom_p": pbin,
            "entropy_nats": ent,
            "recursive_energy_gap_log_cross_over_ce": gap,
            "all_four_observed": bool(np.all(c > 0)),
        }
        rows.append(row)
        for cat, count, prob, energy in zip(CATS, c, p, e):
            energy_rows.append({"Cohort": cohort, "Strategy": cat, "count": int(count),
                                "prob_smoothed": prob, "relative_energy": energy})
    return pd.DataFrame(rows), pd.DataFrame(energy_rows)


def barrier_summary(df):
    rows = []
    for cohort, cg in df.groupby("Cohort"):
        total_counts = counts_for(cg)
        base = (total_counts + ALPHA) / (total_counts.sum() + ALPHA * len(CATS))
        stays = np.zeros(len(CATS), float)
        departures = np.zeros(len(CATS), float)
        for subject, sg in cg.groupby("Subject"):
            sg = sg.sort_values(["OrderIndex","TrialOrder"], na_position="last")
            seq = sg["Strategy"].tolist()
            for a, b in zip(seq[:-1], seq[1:]):
                i = CATS.index(a)
                departures[i] += 1
                if a == b:
                    stays[i] += 1
        for i, cat in enumerate(CATS):
            n = departures[i]
            pstay = (stays[i] + .5) / (n + 1.0) if n > 0 else np.nan
            if np.isfinite(pstay):
                lo = math.log(pstay / (1-pstay)) - math.log(base[i] / (1-base[i]))
            else:
                lo = np.nan
            rows.append({"Cohort": cohort, "Strategy": cat, "n_transitions_from_state": int(n),
                         "n_stays": int(stays[i]), "p_stay_smoothed": pstay,
                         "base_occupancy": base[i], "excess_persistence_logodds": lo})
    return pd.DataFrame(rows)


def js_divergence(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = .5 * (p + q)
    return .5 * np.sum(p * np.log(p / m)) + .5 * np.sum(q * np.log(q / m))


def distances(summary, energies):
    cohorts = summary["Cohort"].tolist()
    prob = {c: energies[energies["Cohort"] == c]["prob_smoothed"].to_numpy(float) for c in cohorts}
    rows = []
    for i, a in enumerate(cohorts):
        for b in cohorts[i+1:]:
            rows.append({"cohort_a": a, "cohort_b": b, "js_divergence": float(js_divergence(prob[a], prob[b]))})
    return pd.DataFrame(rows).sort_values("js_divergence")


def bootstrap_difference(df, a, b, n_boot=10000):
    ga, gb = df[df["Cohort"] == a], df[df["Cohort"] == b]
    sa, sb = pd.unique(ga["Subject"]), pd.unique(gb["Subject"])
    ca = {s: counts_for(ga[ga["Subject"] == s]) for s in sa}
    cb = {s: counts_for(gb[gb["Subject"] == s]) for s in sb}
    dq, dcov, dce = [], [], []
    for _ in range(n_boot):
        A = np.sum([ca[s] for s in RNG.choice(sa, size=len(sa), replace=True)], axis=0)
        B = np.sum([cb[s] for s in RNG.choice(sb, size=len(sb), replace=True)], axis=0)
        _, _, qa, cova, cea, _, _ = metrics_from_counts(A)
        _, _, qb, covb, ceb, _, _ = metrics_from_counts(B)
        dq.append(qa-qb); dcov.append(cova-covb); dce.append(cea-ceb)
    def stat(arr):
        arr=np.asarray(arr,float); arr=arr[np.isfinite(arr)]
        return float(np.mean(arr)), float(np.quantile(arr,.025)), float(np.quantile(arr,.975))
    return stat(dq), stat(dcov), stat(dce)


def comparisons(df):
    pairs = [
        ("US_Adults","Macaque_E1"),
        ("US_Children","Macaque_E1"),
        ("Tsimane_Adults","Macaque_E1"),
        ("US_Adults","US_Children"),
        ("US_Adults","Tsimane_Adults"),
        ("US_Adults","Macaque_E2"),
        ("US_Adults","Macaque_E3"),
    ]
    rows=[]
    for a,b in pairs:
        dq, dcov, dce = bootstrap_difference(df,a,b)
        rows.append({"cohort_a":a,"cohort_b":b,
                     "delta_q_rec_mean":dq[0],"delta_q_rec_ci_lo":dq[1],"delta_q_rec_ci_hi":dq[2],
                     "delta_coverage_mean":dcov[0],"delta_coverage_ci_lo":dcov[1],"delta_coverage_ci_hi":dcov[2],
                     "delta_ce_cond_mean":dce[0],"delta_ce_cond_ci_lo":dce[1],"delta_ce_cond_ci_hi":dce[2]})
    return pd.DataFrame(rows)


def fit_binomial_age(kids, success_name):
    rows=[]
    for subject,g in kids.groupby("Subject"):
        age = pd.to_numeric(g["Age"], errors="coerce").dropna()
        if age.empty: continue
        c=counts_for(g)
        if success_name=="CE_cond":
            succ, total = c[0], c[0]+c[1]
        else:
            succ, total = c[0]+c[1], c.sum()
        if total>0:
            rows.append((subject,float(age.iloc[0]),float(succ),float(total)))
    z=pd.DataFrame(rows,columns=["Subject","Age","success","total"])
    if len(z)<3:
        return {}, z
    age0=float(z["Age"].mean())
    X=np.column_stack([np.ones(len(z)), z["Age"].to_numpy()-age0])
    y=z["success"].to_numpy(); n=z["total"].to_numpy()
    def nll(beta):
        eta=X@beta
        p=1/(1+np.exp(-np.clip(eta,-40,40)))
        return float(-np.sum(y*np.log(np.clip(p,1e-12,1))+(n-y)*np.log(np.clip(1-p,1e-12,1))))
    res=minimize(nll,np.zeros(2),method="BFGS")
    beta=res.x
    eta=X@beta; p=1/(1+np.exp(-eta)); W=n*p*(1-p)
    H=X.T@(X*W[:,None])
    try:
        cov=np.linalg.inv(H); se=np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se=np.array([np.nan,np.nan])
    slope=float(beta[1]); slope_se=float(se[1])
    zscore=slope/slope_se if slope_se>0 else np.nan
    pval=math.erfc(abs(zscore)/math.sqrt(2)) if np.isfinite(zscore) else np.nan
    pred3=1/(1+math.exp(-(beta[0]+beta[1]*(3-age0))))
    pred4=1/(1+math.exp(-(beta[0]+beta[1]*(4-age0))))
    pred5=1/(1+math.exp(-(beta[0]+beta[1]*(5-age0))))
    return {"outcome":success_name,"n_children":len(z),"age_mean":age0,
            "log_odds_slope_per_year":slope,"slope_se":slope_se,"wald_p":pval,
            "odds_ratio_per_year":math.exp(slope),"pred_age3":pred3,"pred_age4":pred4,"pred_age5":pred5}, z


def main():
    df=read_cohorts()
    audit=mismatch_audit(df)
    summary, energy=cohort_summaries(df)
    barriers=barrier_summary(df)
    dist=distances(summary, energy)
    comp=comparisons(df)

    kids=df[df["Cohort"]=="US_Children"].copy()
    age1, z1=fit_binomial_age(kids,"CE_cond")
    age2, z2=fit_binomial_age(kids,"structural_coverage")
    age=pd.DataFrame([age1,age2])

    df.to_csv(OUT/"standardized_completed_sequences.csv",index=False)
    audit.to_csv(OUT/"coding_mismatch_audit.csv",index=False)
    summary.to_csv(OUT/"cohort_landscape_summary.csv",index=False)
    energy.to_csv(OUT/"cohort_basin_energies.csv",index=False)
    barriers.to_csv(OUT/"cohort_barrier_proxies.csv",index=False)
    dist.to_csv(OUT/"cohort_js_distances.csv",index=False)
    comp.to_csv(OUT/"cluster_bootstrap_comparisons.csv",index=False)
    age.to_csv(OUT/"child_age_models.csv",index=False)
    z1.to_csv(OUT/"child_subject_age_counts_ce_cond.csv",index=False)
    z2.to_csv(OUT/"child_subject_age_counts_coverage.csv",index=False)

    adult_dist = dist[((dist["cohort_a"]=="US_Adults")|(dist["cohort_b"]=="US_Adults"))].copy()
    adult_dist["other"] = np.where(adult_dist["cohort_a"]=="US_Adults",adult_dist["cohort_b"],adult_dist["cohort_a"])
    closest = adult_dist.sort_values("js_divergence")[["other","js_divergence"]].iloc[0].to_dict()
    verdict={
        "n_completed_sequences":int(len(df)),
        "cohorts":summary["Cohort"].tolist(),
        "coding_mismatch_total":int(audit["n_mismatches"].sum()) if len(audit) else 0,
        "closest_to_US_adults_by_JS":{"other":str(closest["other"]),"js_divergence":float(closest["js_divergence"])},
    }
    (OUT/"cross_species_summary.json").write_text(json.dumps(verdict,indent=2))

    print("=== CODING AUDIT ==="); print(audit.to_string(index=False))
    print("=== COHORT LANDSCAPE SUMMARY ==="); print(summary.to_string(index=False))
    print("=== BASIN ENERGIES ==="); print(energy.to_string(index=False))
    print("=== BARRIER PROXIES ==="); print(barriers.to_string(index=False))
    print("=== CLOSEST DISTRIBUTIONS (JS) ==="); print(dist.head(15).to_string(index=False))
    print("=== CLUSTER BOOTSTRAP COMPARISONS ==="); print(comp.to_string(index=False))
    print("=== CHILD AGE MODELS ==="); print(age.to_string(index=False))
    print("=== SUMMARY ==="); print(json.dumps(verdict,indent=2))


if __name__=="__main__":
    main()
