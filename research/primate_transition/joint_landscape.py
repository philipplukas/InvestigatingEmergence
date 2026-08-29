from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

DATA_URL = "https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/RecursionMonkeys.csv"
OUT = Path("research/primate_transition/out_joint")
OUT.mkdir(parents=True, exist_ok=True)
CATS = ["Center Embedded", "Crossed", "Tail Embedded", "All Other"]
EPS = 1e-12


def softmax_logits(z):
    z4 = np.column_stack([z, np.zeros(len(z))])
    z4 -= logsumexp(z4, axis=1, keepdims=True)
    return np.exp(z4)


def fit_multinomial(X, y, name):
    n, d = X.shape
    kfree = 3
    def objective(theta):
        B = theta.reshape(d, kfree)
        P = softmax_logits(X @ B)
        return -float(np.log(np.clip(P[np.arange(n), y], EPS, 1.0)).sum())
    res = minimize(objective, np.zeros(d * kfree), method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"{name} failed: {res.message}")
    B = res.x.reshape(d, kfree)
    P = softmax_logits(X @ B)
    ll = -objective(res.x)
    npar = d * kfree
    bic = -2 * ll + npar * np.log(n)
    aic = -2 * ll + 2 * npar
    return {"name": name, "ll": ll, "bic": bic, "aic": aic, "npar": npar, "coef": B, "P": P}


def build_design(df, mode):
    sub = pd.Categorical(df["Sub"], categories=["Beyonce", "Coltrane", "Horatio"])
    exp = pd.Categorical(df["Exposure"].astype(int), categories=[1, 2, 3])
    S = pd.get_dummies(sub, drop_first=True, dtype=float).to_numpy()
    E = pd.get_dummies(exp, drop_first=True, dtype=float).to_numpy()
    I = np.ones((len(df), 1))
    if mode == "global":
        return I, ["intercept"]
    if mode == "subject":
        return np.column_stack([I, S]), ["intercept", "sub_Coltrane", "sub_Horatio"]
    if mode == "shared_phase":
        return np.column_stack([I, S, E]), ["intercept", "sub_Coltrane", "sub_Horatio", "exp_2", "exp_3"]
    if mode == "interaction":
        inter = np.column_stack([S[:, i] * E[:, j] for i in range(S.shape[1]) for j in range(E.shape[1])])
        return np.column_stack([I, S, E, inter]), ["intercept", "sub_Coltrane", "sub_Horatio", "exp_2", "exp_3",
            "Coltrane_x_exp2", "Coltrane_x_exp3", "Horatio_x_exp2", "Horatio_x_exp3"]
    raise ValueError(mode)


def empirical_landscape(df):
    rows = []
    alpha = 0.5
    for (sub, exp), g in df.groupby(["Sub", "Exposure"], sort=True):
        c = np.array([(g["Response Type"] == cat).sum() for cat in CATS], float)
        p = (c + alpha) / (c.sum() + alpha * len(CATS))
        U = -np.log(p)
        U -= U.min()
        qrec = (c[0] - c[1]) / max(c[0] + c[1], 1)
        coverage = (c[0] + c[1]) / c.sum()
        for j, cat in enumerate(CATS):
            rows.append({"Sub": sub, "Exposure": int(exp), "strategy": cat, "count": int(c[j]), "prob_smoothed": p[j],
                         "relative_energy": U[j], "q_rec": qrec, "structural_coverage": coverage})
    return pd.DataFrame(rows)


def transition_barriers(df):
    rows = []
    alpha = 0.5
    for (sub, exp), g0 in df.groupby(["Sub", "Exposure"], sort=True):
        g = g0.sort_values(["Date", "List", "Trial", "Press"], kind="stable")
        r = g["Response Type"].to_numpy(str)
        # Do not create transitions across list boundaries.
        listv = g["List"].to_numpy()
        counts = np.array([(r == cat).sum() for cat in CATS], float)
        pbase = (counts + alpha) / (counts.sum() + alpha * 4)
        for i, cat in enumerate(CATS):
            valid = (r[:-1] == cat) & (listv[:-1] == listv[1:])
            n = int(valid.sum())
            stay = int(((r[1:] == cat) & valid).sum())
            pstay = (stay + alpha) / (n + 2 * alpha) if n else np.nan
            base = pbase[i]
            if n:
                logit_stay = np.log(pstay / (1 - pstay))
                logit_base = np.log(base / (1 - base))
                excess = logit_stay - logit_base
            else:
                excess = np.nan
            rows.append({"Sub": sub, "Exposure": int(exp), "strategy": cat, "n_departures": n, "n_stays": stay,
                         "p_stay_smoothed": pstay, "base_occupancy": base, "excess_persistence_logodds": excess})
    return pd.DataFrame(rows)


def cell_predictions(df, fits):
    rows = []
    for model_name, fit in fits.items():
        P = fit["P"]
        tmp = df[["Sub", "Exposure"]].copy()
        for j, cat in enumerate(CATS):
            tmp[f"p_{j}"] = P[:, j]
        for (sub, exp), g in tmp.groupby(["Sub", "Exposure"]):
            row = {"model": model_name, "Sub": sub, "Exposure": int(exp)}
            for j, cat in enumerate(CATS):
                row[cat] = float(g[f"p_{j}"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def phase_shift_alignment(land):
    # Compare each subject's energy-shift vector from phase 1 to 2 and 1 to 3.
    piv = land.pivot_table(index=["Sub", "Exposure"], columns="strategy", values="relative_energy")
    rows = []
    subjects = ["Beyonce", "Coltrane", "Horatio"]
    for target in [2, 3]:
        shifts = {}
        for sub in subjects:
            a = piv.loc[(sub, 1), CATS].to_numpy(float)
            b = piv.loc[(sub, target), CATS].to_numpy(float)
            d = b - a
            d -= d.mean()
            shifts[sub] = d
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                a, b = shifts[subjects[i]], shifts[subjects[j]]
                corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) * np.linalg.norm(b) > 0 else np.nan
                rows.append({"phase_shift": f"1->{target}", "subject_a": subjects[i], "subject_b": subjects[j],
                             "energy_shift_corr": corr, "energy_shift_cosine": cos})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_URL)
    df = df[df["Response Type"].isin(CATS)].copy()
    df["Sub"] = df["Sub"].astype(str).str.strip().str.casefold().str.title()
    df = df[df["Sub"].isin(["Beyonce", "Coltrane", "Horatio"])].copy()
    df["Exposure"] = df["Exposure"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    ids = {c: i for i, c in enumerate(CATS)}
    y = df["Response Type"].map(ids).to_numpy(int)

    fits = {}
    specs = [("J0_global", "global"), ("J1_subject", "subject"), ("J2_shared_phase", "shared_phase"), ("J3_subject_phase_interaction", "interaction")]
    for name, mode in specs:
        X, names = build_design(df, mode)
        fit = fit_multinomial(X, y, name)
        fit["feature_names"] = names
        fits[name] = fit

    comparison = pd.DataFrame([{k: fit[k] for k in ["name", "ll", "bic", "aic", "npar"]} for fit in fits.values()]).sort_values("bic")
    comparison["delta_bic"] = comparison["bic"] - comparison["bic"].min()
    comparison.to_csv(OUT / "joint_model_comparison.csv", index=False)

    landscape = empirical_landscape(df)
    landscape.to_csv(OUT / "empirical_basin_energies.csv", index=False)
    barriers = transition_barriers(df)
    barriers.to_csv(OUT / "transition_barrier_proxies.csv", index=False)
    preds = cell_predictions(df, fits)
    preds.to_csv(OUT / "joint_model_cell_predictions.csv", index=False)
    align = phase_shift_alignment(landscape)
    align.to_csv(OUT / "phase_shift_alignment.csv", index=False)

    # Same accessible state-space support test: each subject has observed all four strategy categories.
    support = []
    for sub, g in df.groupby("Sub"):
        present = {c: int((g["Response Type"] == c).sum()) for c in CATS}
        support.append({"Sub": sub, **present, "all_four_observed": bool(all(v > 0 for v in present.values()))})
    support_df = pd.DataFrame(support)
    support_df.to_csv(OUT / "state_space_support.csv", index=False)

    # Save coefficients in long form for the shared-phase model: log-probability biases relative to All Other.
    fit = fits["J2_shared_phase"]
    coef_rows = []
    for i, feat in enumerate(fit["feature_names"]):
        for j, cat in enumerate(CATS[:-1]):
            coef_rows.append({"feature": feat, "strategy_vs_AllOther": cat, "logit_coefficient": float(fit["coef"][i, j])})
    pd.DataFrame(coef_rows).to_csv(OUT / "shared_phase_coefficients.csv", index=False)

    summary = {
        "n_trials": int(len(df)),
        "subjects": sorted(df["Sub"].unique().tolist()),
        "best_model": str(comparison.iloc[0]["name"]),
        "best_bic": float(comparison.iloc[0]["bic"]),
        "delta_bic_shared_vs_interaction": float(fits["J2_shared_phase"]["bic"] - fits["J3_subject_phase_interaction"]["bic"]),
        "all_subjects_observe_all_four_strategies": bool(support_df["all_four_observed"].all()),
    }
    (OUT / "joint_summary.json").write_text(json.dumps(summary, indent=2))

    print("=== JOINT MODEL COMPARISON ===")
    print(comparison.to_string(index=False))
    print("=== STATE SPACE SUPPORT ===")
    print(support_df.to_string(index=False))
    print("=== BASIN ENERGIES ===")
    print(landscape.to_string(index=False))
    print("=== BARRIER PROXIES ===")
    print(barriers.to_string(index=False))
    print("=== PHASE SHIFT ALIGNMENT ===")
    print(align.to_string(index=False))
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
