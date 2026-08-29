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
N_NULL = int(os.environ.get("N_NULL", "100"))
N_BOOT = int(os.environ.get("N_BOOT", "100"))


def entropy(counts):
    p = np.asarray(counts, float)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def const_fit(y):
    counts = np.bincount(y, minlength=4).astype(float)
    p = counts / counts.sum()
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


def exposure_fit(y, exposure):
    ll = 0.0
    probs = {}
    groups = pd.unique(exposure)
    for e in groups:
        idx = np.asarray(exposure == e)
        lle, _, pe = const_fit(y[idx])
        ll += lle
        probs[str(e)] = pe.tolist()
    bic = -2 * ll + 3 * len(groups) * np.log(len(y))
    return float(ll), float(bic), probs


def fb(y, pi, A, B):
    n = len(y)
    a = np.zeros((n, 2)); s = np.zeros(n)
    a[0] = pi * B[:, y[0]]; s[0] = a[0].sum() + EPS; a[0] /= s[0]
    for t in range(1, n):
        a[t] = (a[t - 1] @ A) * B[:, y[t]]
        s[t] = a[t].sum() + EPS; a[t] /= s[t]
    b = np.ones((n, 2))
    for t in range(n - 2, -1, -1):
        b[t] = A @ (B[:, y[t + 1]] * b[t + 1]); b[t] /= s[t + 1]
    g = a * b; g /= g.sum(axis=1, keepdims=True)
    xi = np.zeros((2, 2))
    for t in range(n - 1):
        x = a[t][:, None] * A * (B[:, y[t + 1]] * b[t + 1])[None, :]
        xi += x / (x.sum() + EPS)
    return float(np.log(s + EPS).sum()), g, xi


def hmm_fit(y, seed=0, iters=180):
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.94, 0.06], [0.06, 0.94]], float)
    B = rng.dirichlet(np.ones(4), size=2)
    for _ in range(iters):
        ll, g, xi = fb(y, pi, A, B)
        pi = g[0] + 1e-3; pi /= pi.sum()
        A = xi + 1e-2; A /= A.sum(axis=1, keepdims=True)
        BN = np.full((2, 4), 1e-2)
        for c in range(4):
            BN[:, c] += g[y == c].sum(axis=0)
        B = BN / BN.sum(axis=1, keepdims=True)
    ll, g, _ = fb(y, pi, A, B)
    bic = -2 * ll + 9 * np.log(len(y))
    structure_score = B[:, 0] - B[:, 1]
    high_state = int(np.argmax(structure_score))
    return {"ll": ll, "bic": float(bic), "pi": pi, "A": A, "B": B,
            "gamma": g, "occ": g[:, high_state], "high_state": high_state,
            "structure_score": structure_score}


def best_hmm(y, seeds=range(8), iters=180):
    return min([hmm_fit(y, int(s), iters=iters) for s in seeds], key=lambda d: d["bic"])


def rolling(g, w):
    z = g["z"].to_numpy(float); r = g["Response Type"].to_numpy(str); out = []
    for end in range(w, len(g) + 1):
        rs = r[end - w:end]; zs = z[end - w:end]
        c = np.array([(rs == x).sum() for x in CATS], float)
        rho = float(np.corrcoef(zs[:-1], zs[1:])[0, 1]) if np.std(zs[:-1]) > 0 and np.std(zs[1:]) > 0 else np.nan
        out.append({"end_index": end, "q": float((c[0] - c[1]) / w), "entropy": entropy(c),
                    "variance": float(np.var(zs, ddof=1)), "rho1": rho,
                    "switch_rate": float(np.mean(rs[1:] != rs[:-1])),
                    "p_ce": c[0] / w, "p_cross": c[1] / w, "p_tail": c[2] / w, "p_other": c[3] / w})
    return pd.DataFrame(out)


def region_metrics(g, start, end, label, cp):
    h = g.iloc[max(0, start):min(len(g), end)]
    z = h["z"].to_numpy(float); r = h["Response Type"].to_numpy(str)
    counts = np.array([(r == c).sum() for c in CATS], float)
    rho = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if len(z) > 2 and np.std(z[:-1]) > 0 and np.std(z[1:]) > 0 else np.nan
    return {"region": label, "cp": cp, "start": max(0, start) + 1, "end": min(len(g), end), "n": len(h),
            "q": float((counts[0] - counts[1]) / max(len(h), 1)), "entropy": entropy(counts),
            "variance": float(np.var(z, ddof=1)) if len(z) > 1 else np.nan, "rho1": rho,
            "switch_rate": float(np.mean(r[1:] != r[:-1])) if len(r) > 1 else np.nan}


def shuffle_within_exposure(y, exposure, rng):
    out = y.copy()
    for e in pd.unique(exposure):
        idx = np.flatnonzero(exposure == e)
        out[idx] = rng.permutation(out[idx])
    return out


def circular_block_bootstrap(y, exposure, rng, block=10):
    out = []
    for e in pd.unique(exposure):
        seq = y[np.asarray(exposure == e)]
        n = len(seq); sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend(seq[(start + np.arange(block)) % n].tolist())
        out.extend(sampled[:n])
    return np.asarray(out, int)


def model_bics(y, exposure, hmm_seeds=range(4), hmm_iters=120):
    _, b0, _ = const_fit(y)
    _, b1 = trend_fit(y)
    cp = change_fit(y)
    hmm = best_hmm(y, seeds=hmm_seeds, iters=hmm_iters)
    _, b4, _ = exposure_fit(y, exposure)
    return {"M0_constant": b0, "M1_smooth": b1, "M2_change_point": cp[0], "M3_hmm": hmm["bic"], "M4_exposure": b4}, hmm, cp


def robustness_for_subject(sub, y, exposure, observed_bics, rng):
    obs_delta = min(v for k, v in observed_bics.items() if k != "M3_hmm") - observed_bics["M3_hmm"]
    null_delta = []
    for _ in range(N_NULL):
        yp = shuffle_within_exposure(y, exposure, rng)
        b, _, _ = model_bics(yp, exposure, hmm_seeds=range(3), hmm_iters=90)
        null_delta.append(min(v for k, v in b.items() if k != "M3_hmm") - b["M3_hmm"])
    p = (1 + np.sum(np.asarray(null_delta) >= obs_delta)) / (N_NULL + 1)

    wins = {k: 0 for k in observed_bics}
    deltas = []
    for _ in range(N_BOOT):
        yb = circular_block_bootstrap(y, exposure, rng, block=10)
        b, _, _ = model_bics(yb, exposure, hmm_seeds=range(3), hmm_iters=90)
        best = min(b, key=b.get); wins[best] += 1
        deltas.append(min(v for k, v in b.items() if k != "M3_hmm") - b["M3_hmm"])
    return {"Sub": sub, "observed_hmm_advantage_bic": float(obs_delta),
            "within_exposure_shuffle_p": float(p), "null_delta_mean": float(np.mean(null_delta)),
            "null_delta_q95": float(np.quantile(null_delta, 0.95)),
            "bootstrap_hmm_win_fraction": float(wins["M3_hmm"] / N_BOOT),
            "bootstrap_hmm_advantage_median": float(np.median(deltas)),
            **{f"bootstrap_win_{k}": int(v) for k, v in wins.items()}}


def main():
    df = pd.read_csv(DATA_URL)
    df = df[df["Response Type"].isin(CATS)].copy()
    df["Sub_raw"] = df["Sub"].astype(str)
    df["Sub"] = df["Sub_raw"].str.strip().str.casefold().str.title()  # fixes Beyonce/beyonce
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Sub", "Exposure", "Date", "List", "Trial", "Press"], kind="stable")
    df["z"] = df["Response Type"].map(ZMAP)
    ids = {c: i for i, c in enumerate(CATS)}
    summary = []; rolls = []; details = {}; occ_rows = []; switch_rows = []; crit_rows = []; robust_rows = []
    rng = np.random.default_rng(20260829)

    for sub, g0 in df.groupby("Sub", sort=True):
        g = g0.reset_index(drop=True)
        y = g["Response Type"].map(ids).to_numpy(int)
        exposure = g["Exposure"].to_numpy()
        if len(y) < 50:
            continue

        ll0, b0, p0 = const_fit(y)
        ll1, b1 = trend_fit(y)
        cp = change_fit(y)
        hmm = best_hmm(y)
        ll4, b4, p4 = exposure_fit(y, exposure)
        models = {"M0_constant": b0, "M1_smooth": b1, "M2_change_point": cp[0], "M3_hmm": hmm["bic"], "M4_exposure": b4}
        best = sorted(models.items(), key=lambda kv: kv[1])
        high = hmm["high_state"]
        decoded = np.argmax(hmm["gamma"], axis=1)
        boundaries = np.flatnonzero(exposure[1:] != exposure[:-1]) + 1
        switches = np.flatnonzero(decoded[1:] != decoded[:-1]) + 1
        cross = np.where(hmm["occ"] >= 0.5)[0]
        cp_row = g.iloc[min(cp[1], len(g) - 1)]

        summary.append({"Sub": sub, "raw_ids": "|".join(sorted(g["Sub_raw"].unique())), "n_trials": len(y),
                        "exposures": ",".join(map(str, sorted(pd.unique(exposure)))), "p_ce": float(np.mean(y == 0)),
                        "p_cross": float(np.mean(y == 1)), "q_global": float(np.mean(g["z"])),
                        "entropy_global": entropy(np.bincount(y, minlength=4)), **{f"{k}_bic": float(v) for k, v in models.items()},
                        "best_model": best[0][0], "delta_bic_second": float(best[1][1] - best[0][1]),
                        "change_point_trial": int(cp[1]), "change_point_exposure": cp_row["Exposure"],
                        "change_point_list": cp_row["List"], "hmm_first_occ_ge_0_5": int(cross[0] + 1) if cross.size else None,
                        "hmm_high_state": high, "hmm_A00": float(hmm["A"][0, 0]), "hmm_A11": float(hmm["A"][1, 1]),
                        "hmm_dwell0": float(1 / max(1 - hmm["A"][0, 0], EPS)),
                        "hmm_dwell1": float(1 / max(1 - hmm["A"][1, 1], EPS)), "hmm_n_decoded_switches": int(len(switches)),
                        "n_exposure_boundaries": int(len(boundaries)),
                        "switches_within5_exposure_boundary": int(sum(np.any(np.abs(boundaries - s) <= 5) for s in switches))})

        for e in pd.unique(exposure):
            idx = exposure == e
            occ_rows.append({"Sub": sub, "Exposure": e, "n": int(idx.sum()), "high_state_mean_occupancy": float(hmm["occ"][idx].mean()),
                             "decoded_high_fraction": float(np.mean(decoded[idx] == high)), "q": float(np.mean(g.loc[idx, "z"]))})
        for s in switches:
            row = g.iloc[s]
            switch_rows.append({"Sub": sub, "index": int(s + 1), "Exposure": row["Exposure"], "Date": str(row["Date"].date()),
                                "List": row["List"], "Trial": row["Trial"], "from_state": int(decoded[s - 1]), "to_state": int(decoded[s]),
                                "near_exposure_boundary_5": bool(np.any(np.abs(boundaries - s) <= 5))})

        w = 40
        crit_rows.extend([{**{"Sub": sub}, **region_metrics(g, cp[1] - w, cp[1], "pre", cp[1])},
                          {**{"Sub": sub}, **region_metrics(g, cp[1] - w // 2, cp[1] + w // 2, "center", cp[1])},
                          {**{"Sub": sub}, **region_metrics(g, cp[1], cp[1] + w, "post", cp[1])}])

        for ww in (20, 40, 80):
            if len(g) >= ww:
                rr = rolling(g, ww); rr.insert(0, "window", ww); rr.insert(0, "Sub", sub); rolls.append(rr)

        details[sub] = {"constant_probs": p0.tolist(), "exposure_probs": p4, "change_point": int(cp[1]),
                        "cp_pre_probs": cp[3].tolist(), "cp_post_probs": cp[4].tolist(),
                        "exposure_boundaries": (boundaries + 1).tolist(), "decoded_switches": (switches + 1).tolist(),
                        "hmm": {"pi": hmm["pi"].tolist(), "A": hmm["A"].tolist(), "B": hmm["B"].tolist(),
                                "high_state": int(high), "structure_score": hmm["structure_score"].tolist(),
                                "occupancy_first_20": hmm["occ"][:20].tolist(), "occupancy_last_20": hmm["occ"][-20:].tolist()}}

        if best[0][0] == "M3_hmm":
            robust_rows.append(robustness_for_subject(sub, y, exposure, models, rng))

    sdf = pd.DataFrame(summary).sort_values("Sub")
    sdf.to_csv(OUT / "subject_model_summary.csv", index=False)
    pd.concat(rolls, ignore_index=True).to_csv(OUT / "rolling_transition_metrics.csv", index=False)
    pd.DataFrame(occ_rows).to_csv(OUT / "hmm_exposure_occupancy.csv", index=False)
    pd.DataFrame(switch_rows).to_csv(OUT / "hmm_switches.csv", index=False)
    pd.DataFrame(crit_rows).to_csv(OUT / "criticality_regions.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(OUT / "robustness_summary.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(details, indent=2))
    (OUT / "run_summary.json").write_text(json.dumps({"source": DATA_URL, "n_completed_sequences": int(len(df)),
                                                       "canonical_subjects": sorted(df["Sub"].unique().tolist()),
                                                       "subjects": summary, "robustness": robust_rows}, indent=2))
    print("=== BASE MODEL COMPARISON ===")
    print(sdf.to_string(index=False))
    print("\n=== HMM ROBUSTNESS (only subjects where M3 wins) ===")
    print(pd.DataFrame(robust_rows).to_string(index=False) if robust_rows else "No HMM winner")
    print("\n=== EXPOSURE OCCUPANCY ===")
    print(pd.DataFrame(occ_rows).to_string(index=False))


if __name__ == "__main__":
    main()
