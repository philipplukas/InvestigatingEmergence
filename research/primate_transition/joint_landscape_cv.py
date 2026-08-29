from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from joint_landscape import DATA_URL, CATS, build_design

OUT = Path("research/primate_transition/out_joint")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
SPECS = [("J0_global", "global"), ("J1_subject", "subject"), ("J2_shared_phase", "shared_phase"), ("J3_subject_phase_interaction", "interaction")]


def probs(X, theta):
    B = theta.reshape(X.shape[1], 3)
    z = np.column_stack([X @ B, np.zeros(len(X))])
    z -= logsumexp(z, axis=1, keepdims=True)
    return np.exp(z)


def fit_theta(X, y):
    def obj(theta):
        P = probs(X, theta)
        nll = -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).sum()
        return float(nll + 1e-7 * np.dot(theta, theta))
    res = minimize(obj, np.zeros(X.shape[1] * 3), method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-11})
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def main():
    df = pd.read_csv(DATA_URL)
    df = df[df["Response Type"].isin(CATS)].copy()
    df["Sub"] = df["Sub"].astype(str).str.strip().str.casefold().str.title()
    df = df[df["Sub"].isin(["Beyonce", "Coltrane", "Horatio"])].copy()
    df["Exposure"] = df["Exposure"].astype(int)
    ids = {c: i for i, c in enumerate(CATS)}
    y = df["Response Type"].map(ids).to_numpy(int)

    # A list is the natural experimental block. Include subject and exposure to avoid accidental ID collisions.
    group = df["Sub"].astype(str) + "|E" + df["Exposure"].astype(str) + "|L" + df["List"].astype(str)
    groups = pd.unique(group)
    rows = []
    per_fold = []
    for model, mode in SPECS:
        total_ll = 0.0
        total_n = 0
        fold_losses = []
        for g in groups:
            test = (group.to_numpy() == g)
            train = ~test
            Xtr, _ = build_design(df.loc[train], mode)
            Xte, _ = build_design(df.loc[test], mode)
            theta = fit_theta(Xtr, y[train])
            P = probs(Xte, theta)
            ll = float(np.log(np.clip(P[np.arange(test.sum()), y[test]], EPS, 1)).sum())
            total_ll += ll
            total_n += int(test.sum())
            loss = -ll / max(int(test.sum()), 1)
            fold_losses.append(loss)
            per_fold.append({"model": model, "block": g, "n": int(test.sum()), "log_loss": loss})
        rows.append({"model": model, "n_blocks": len(groups), "n_test_predictions": total_n,
                     "loo_log_likelihood": total_ll, "loo_log_loss_per_trial": -total_ll / total_n,
                     "median_block_log_loss": float(np.median(fold_losses)),
                     "q90_block_log_loss": float(np.quantile(fold_losses, .90))})

    out = pd.DataFrame(rows).sort_values("loo_log_loss_per_trial")
    out["delta_log_loss"] = out["loo_log_loss_per_trial"] - out["loo_log_loss_per_trial"].min()
    out.to_csv(OUT / "blocked_cv_model_comparison.csv", index=False)
    pd.DataFrame(per_fold).to_csv(OUT / "blocked_cv_per_list.csv", index=False)

    # Subject-specific predictive summary to reveal whether one animal drives model differences.
    pf = pd.DataFrame(per_fold)
    pf[["Sub", "Exposure", "List"]] = pf["block"].str.extract(r"^([^|]+)\|E([^|]+)\|L(.+)$")
    subject_summary = pf.groupby(["model", "Sub"], as_index=False).apply(
        lambda g: pd.Series({"mean_block_log_loss": np.average(g["log_loss"], weights=g["n"]),
                             "n_predictions": int(g["n"].sum())}), include_groups=False)
    subject_summary.to_csv(OUT / "blocked_cv_by_subject.csv", index=False)

    print("=== BLOCKED LIST-LEVEL CV ===")
    print(out.to_string(index=False))
    print("=== BY SUBJECT ===")
    print(subject_summary.to_string(index=False))


if __name__ == "__main__":
    main()
