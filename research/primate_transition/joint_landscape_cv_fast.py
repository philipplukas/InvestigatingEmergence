from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from joint_landscape import DATA_URL, CATS

OUT = Path("research/primate_transition/out_joint")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
ALPHA = 0.5
SUBS = ["Beyonce", "Coltrane", "Horatio"]
EXPS = [1, 2, 3]
MODELS = ["J0_global", "J1_subject", "J2_shared_phase", "J3_subject_phase_interaction"]


def make_counts(df):
    C = np.zeros((3, 3, 4), float)
    for si, s in enumerate(SUBS):
        for ei, e in enumerate(EXPS):
            g = df[(df["Sub"] == s) & (df["Exposure"] == e)]
            C[si, ei] = [(g["Response Type"] == c).sum() for c in CATS]
    return C


def additive_design():
    rows = []
    for si in range(3):
        for ei in range(3):
            rows.append([1.0, float(si == 1), float(si == 2), float(ei == 1), float(ei == 2)])
    return np.asarray(rows)


def softmax3(z):
    z4 = np.column_stack([z, np.zeros(len(z))])
    z4 -= logsumexp(z4, axis=1, keepdims=True)
    return np.exp(z4)


def fit_additive(C):
    X = additive_design()
    Y = (C + ALPHA).reshape(9, 4)
    def obj(theta):
        B = theta.reshape(5, 3)
        P = softmax3(X @ B)
        return float(-(Y * np.log(np.clip(P, EPS, 1))).sum() + 1e-8 * np.dot(theta, theta))
    res = minimize(obj, np.zeros(15), method="L-BFGS-B", options={"maxiter": 1500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(res.message)
    return softmax3(X @ res.x.reshape(5, 3)).reshape(3, 3, 4)


def predictive_probs(C, model):
    Cj = C + ALPHA
    if model == "J0_global":
        p = Cj.sum(axis=(0, 1)); p /= p.sum()
        return np.broadcast_to(p, (3, 3, 4)).copy()
    if model == "J1_subject":
        p = Cj.sum(axis=1); p /= p.sum(axis=1, keepdims=True)
        return np.broadcast_to(p[:, None, :], (3, 3, 4)).copy()
    if model == "J2_shared_phase":
        return fit_additive(C)
    if model == "J3_subject_phase_interaction":
        return Cj / Cj.sum(axis=2, keepdims=True)
    raise ValueError(model)


def main():
    df = pd.read_csv(DATA_URL)
    df = df[df["Response Type"].isin(CATS)].copy()
    df["Sub"] = df["Sub"].astype(str).str.strip().str.casefold().str.title()
    df = df[df["Sub"].isin(SUBS)].copy()
    df["Exposure"] = df["Exposure"].astype(int)
    # Missing list IDs are a real block too; give them an explicit label rather than allowing NaN propagation.
    list_key = df["List"].where(df["List"].notna(), "MISSING").astype(str)
    df["block"] = df["Sub"].astype(str) + "|E" + df["Exposure"].astype(str) + "|L" + list_key
    groups = [g for g in pd.unique(df["block"]) if pd.notna(g)]

    full = make_counts(df)
    per_fold = []
    totals = {m: {"ll": 0.0, "n": 0, "losses": []} for m in MODELS}

    for block in groups:
        test_df = df[df["block"] == block]
        if test_df.empty:
            continue
        si = SUBS.index(test_df["Sub"].iloc[0]); ei = EXPS.index(int(test_df["Exposure"].iloc[0]))
        test_counts = np.array([(test_df["Response Type"] == c).sum() for c in CATS], float)
        train = full.copy(); train[si, ei] -= test_counts
        for model in MODELS:
            P = predictive_probs(train, model)
            p = np.clip(P[si, ei], EPS, 1)
            ll = float((test_counts * np.log(p)).sum())
            n = int(test_counts.sum())
            loss = -ll / max(n, 1)
            totals[model]["ll"] += ll; totals[model]["n"] += n; totals[model]["losses"].append(loss)
            per_fold.append({"model": model, "block": block, "Sub": SUBS[si], "Exposure": EXPS[ei],
                             "n": n, "log_loss": loss})

    rows = []
    for model in MODELS:
        t = totals[model]
        rows.append({"model": model, "n_blocks": len(t["losses"]), "n_test_predictions": t["n"],
                     "loo_log_likelihood": t["ll"], "loo_log_loss_per_trial": -t["ll"] / t["n"],
                     "median_block_log_loss": float(np.median(t["losses"])),
                     "q90_block_log_loss": float(np.quantile(t["losses"], .90))})
    out = pd.DataFrame(rows).sort_values("loo_log_loss_per_trial")
    out["delta_log_loss"] = out["loo_log_loss_per_trial"] - out["loo_log_loss_per_trial"].min()
    out.to_csv(OUT / "blocked_cv_model_comparison.csv", index=False)

    pf = pd.DataFrame(per_fold)
    pf.to_csv(OUT / "blocked_cv_per_list.csv", index=False)
    subj_rows = []
    for (model, sub), g in pf.groupby(["model", "Sub"]):
        subj_rows.append({"model": model, "Sub": sub,
                          "mean_block_log_loss": float(np.average(g["log_loss"], weights=g["n"])),
                          "n_predictions": int(g["n"].sum())})
    subject_summary = pd.DataFrame(subj_rows)
    subject_summary.to_csv(OUT / "blocked_cv_by_subject.csv", index=False)

    print("=== BLOCKED LIST-LEVEL CV ===")
    print(out.to_string(index=False))
    print("=== BY SUBJECT ===")
    print(subject_summary.to_string(index=False))


if __name__ == "__main__":
    main()
