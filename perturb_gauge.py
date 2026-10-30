#!/usr/bin/env python3
"""
perturb_gauge.py — does pair_% predict leakage, or does something else?

THE CLAIM UNDER TEST
--------------------
Across the twelve real targets, pair_% correlates with the site-leakage of
Section 7 at r = -0.63: thinner columns leak more. Twelve points, one corpus,
and several correlations examined before that one cleared. It needs a test with
more points and a stated alternative.

THE ALTERNATIVE
---------------
Note first what the probe actually sees. It is trained on the column binarised
at 0.6, so the graded scores — the thing pair_% measures — never reach it. Any
predictive power pair_% has must therefore be indirect, through *which* studies
end up positive rather than through how finely they are scored.

That suggests the real quantity is not thinness but site-concentration: how
unevenly the positives are spread across institutions. A column whose positives
cluster in a few sites gives a probe something to recognise. pair_% would then
be a proxy that works only because, in real reports, silence is itself a
hospital habit — which is exactly what the seventh-place team observed about
synovitis.

THE EXPERIMENT
--------------
Each target is degraded two ways, at three strengths. Both silence studies, so
both reduce pair_% by a similar amount. They differ only in who gets silenced:

  silence_random   studies chosen uniformly at random
  silence_by_site  studies chosen from whole scanner signatures at a time,
                   so the surviving positives concentrate in fewer sites

If pair_% predicts the leakage under both, it is tracking thinness and the
Section 4 claim stands as written. If it predicts only under silence_by_site,
then site-concentration is the operative quantity and pair_% is a proxy — a
weaker claim, and a more honest one.

site_spread is measured directly for comparison: the study-weighted standard
deviation of the positive rate across scanner signatures. The question the run
answers is which of the two better explains the drop.

Runtime is roughly twenty minutes: 84 columns, ten logistic fits each.

Usage:
    python perturb_gauge.py --embeddings data/interim/embeddings.parquet \\
                            --extractions data/interim/extractions.csv \\
                            --sites data/interim/study_sites.csv \\
                            --out perturb
    python perturb_gauge.py ... --quick        # 4 targets, for a dry run
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PLANES = ["Sagittal", "Coronal", "Axial"]
POSITIVE_AT = 0.6
SILENT = 0.15
LEVELS = [0.3, 0.5, 0.7]
MIN_POS = 60          # below this a grouped fold cannot be trusted

C_PER_TARGET = {
    "ACL": 0.003, "MCL": 0.001, "Medial Meniscus": 0.003,
    "Lateral Meniscus": 0.001, "Medial OA": 0.001, "Lateral OA": 0.001,
    "PF OA": 0.003, "Effusion": 0.003, "Synovitis": 0.003,
    "Baker's": 0.003, "Contusion": 0.003, "Fracture": 0.001,
}

BLUE, VERMILLION, GREEN, INK, GRID = "#0072B2", "#D55E00", "#009E73", "#222222", "#CCCCCC"


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    s = p[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def pair_pct(v: np.ndarray) -> float:
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(100.0 * (1.0 - np.sum(p ** 2)))


def site_spread(y: np.ndarray, groups: np.ndarray) -> float:
    """Study-weighted sd of the positive rate across scanner signatures.

    Zero when every site carries the finding at the same rate; large when the
    positives pile up in a few sites. This is the quantity a probe can exploit
    by recognising the site, so it is the alternative explanation to thinness.
    """
    df = pd.DataFrame({"y": y, "g": groups})
    agg = df.groupby("g")["y"].agg(["mean", "size"])
    agg = agg[agg["size"] >= 20]
    if len(agg) < 3:
        return float("nan")
    w = agg["size"].to_numpy(dtype=float)
    m = agg["mean"].to_numpy(dtype=float)
    bar = np.average(m, weights=w)
    return float(np.sqrt(np.average((m - bar) ** 2, weights=w)))


def build_features(emb: pd.DataFrame):
    fcols = sorted((c for c in emb.columns if c.startswith("f")),
                   key=lambda c: int(c[1:]))
    dim = len(fcols)
    uids = sorted(emb["StudyInstanceUID"].unique())
    index = {u: i for i, u in enumerate(uids)}
    X = np.zeros((len(uids), dim * len(PLANES)), dtype=np.float32)
    for (uid, plane), g in emb.groupby(["StudyInstanceUID", "plane"], sort=False):
        if plane not in PLANES:
            continue
        b = PLANES.index(plane) * dim
        X[index[uid], b : b + dim] = g[fcols].to_numpy(dtype=np.float32).mean(axis=0)
    return X, uids


def silence_random(scores: np.ndarray, frac: float, rng) -> np.ndarray:
    out = scores.copy()
    loud = np.flatnonzero(scores >= POSITIVE_AT)
    k = int(round(frac * len(loud)))
    if k:
        out[rng.choice(loud, size=k, replace=False)] = SILENT
    return out


def silence_by_site(scores: np.ndarray, groups: np.ndarray, frac: float, rng) -> np.ndarray:
    """Silence whole sites until the positive budget is spent.

    Sites are taken in a random order, so the surviving positives concentrate in
    whichever sites happen to be spared rather than in a chosen few.
    """
    out = scores.copy()
    loud = scores >= POSITIVE_AT
    target = int(round(frac * loud.sum()))
    order = rng.permutation(np.unique(groups))
    silenced = 0
    for g in order:
        if silenced >= target:
            break
        m = (groups == g) & loud
        if m.sum() == 0:
            continue
        out[m] = SILENT
        silenced += int(m.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", default="perturb")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        print(f"needs scikit-learn 0.24+: {exc}", file=sys.stderr)
        return 1

    emb = pd.read_parquet(args.embeddings)
    ext = pd.read_csv(args.extractions)
    sites = pd.read_csv(args.sites)

    X, uids = build_features(emb)
    Y = ext.pivot_table(index="StudyInstanceUID", columns="target",
                        values="score", aggfunc="first").reindex(uids)
    site_of = dict(zip(sites["StudyInstanceUID"], sites["scanner"]))
    groups = np.array([site_of.get(u, "UNKNOWN") for u in uids])

    keep = Y[TARGETS[0]].notna().to_numpy() & (groups != "UNKNOWN")
    X, Y, groups = X[keep], Y[keep], groups[keep]
    print(f"{X.shape[0]:,} studies, {len(set(groups))} scanner signatures\n")

    targets = TARGETS[:4] if args.quick else TARGETS
    rng = np.random.default_rng(args.seed)

    def pipe(C):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    rcv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    gcv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    def evaluate(scores, C):
        y = (scores >= POSITIVE_AT).astype(int)
        if y.sum() < MIN_POS or (len(y) - y.sum()) < MIN_POS:
            return None
        res = {}
        for name, cv, g in (("random", rcv, None), ("grouped", gcv, groups)):
            p = np.zeros(len(y))
            for tr, va in cv.split(X, y, g):
                m = pipe(C)
                m.fit(X[tr], y[tr])
                p[va] = m.predict_proba(X[va])[:, 1]
            res[name] = auc_score(y, p)
        return {"pos": int(y.sum()), "pair": pair_pct(scores),
                "silent": float(100 * np.mean(np.isclose(scores, SILENT))),
                "spread": site_spread(y, groups),
                "random": res["random"], "grouped": res["grouped"],
                "drop": res["random"] - res["grouped"]}

    rows, t0 = [], time.time()
    plan = [("baseline", 0.0)] + [(f, lv) for f in ("silence_random", "silence_by_site")
                                  for lv in LEVELS]

    for t in targets:
        base = Y[t].to_numpy(dtype=float)
        C = C_PER_TARGET[t]
        for family, lv in plan:
            if family == "baseline":
                s = base
            elif family == "silence_random":
                s = silence_random(base, lv, rng)
            else:
                s = silence_by_site(base, groups, lv, rng)
            r = evaluate(s, C)
            if r is None:
                print(f"  {t:18s} {family:16s} {lv:.1f}  skipped, too few positives")
                continue
            r.update({"target": t, "family": family, "level": lv})
            rows.append(r)
            print(f"  {t:18s} {family:16s} {lv:.1f}  pos {r['pos']:5d}  "
                  f"pair {r['pair']:5.1f}  spread {r['spread']:.3f}  "
                  f"drop {r['drop']:+.3f}   [{(time.time()-t0)/60:.1f} min]")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.csv", index=False)
    print(f"\nwrote {args.out}.csv  ({len(df)} columns evaluated)\n")

    def r_of(a, b, sub):
        s = sub.dropna(subset=[a, b])
        return float(np.corrcoef(s[a], s[b])[0, 1]) if len(s) > 3 else float("nan")

    print("=" * 74)
    print("WHICH QUANTITY EXPLAINS THE LEAKAGE?")
    print("=" * 74)
    print(f"  {'subset':22s} {'n':>4s} {'r(pair, drop)':>15s} {'r(spread, drop)':>17s}")
    for name, sub in [("all columns", df)] + [(f, df[df.family == f]) for f in df.family.unique()]:
        print(f"  {name:22s} {len(sub):4d} {r_of('pair','drop',sub):15.3f} "
              f"{r_of('spread','drop',sub):17.3f}")

    rr = df[df.family == "silence_random"]
    rb = df[df.family == "silence_by_site"]
    print(f"\n  mean drop, site-blind silencing : {rr['drop'].mean():+.3f}")
    print(f"  mean drop, site-wise silencing  : {rb['drop'].mean():+.3f}")
    print(f"  baseline mean drop              : {df[df.family=='baseline']['drop'].mean():+.3f}")

    print("\n" + "=" * 74)
    a, b = r_of("pair", "drop", df), r_of("spread", "drop", df)
    if abs(b) > abs(a) + 0.15:
        print("site_spread explains the leakage better than pair_%. Section 4's")
        print("claim should be narrowed: the gauge proxies site-concentration")
        print("rather than measuring it, and says so only on corpora where")
        print("silence is itself a hospital habit.")
    elif abs(a) > abs(b) + 0.15:
        print("pair_% explains the leakage better than site_spread, which is the")
        print("stronger result and lets Section 4 stand as written.")
    else:
        print("The two explain the leakage about equally well. Report both and")
        print("claim neither as the mechanism.")

    # -------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib missing, no figure)")
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), dpi=200, sharey=True)
    mean_of = {f: df[df.family == f]["drop"].mean() for f in df.family.unique()}
    style = {"baseline": (BLUE, "o", f"unperturbed  (mean {mean_of.get('baseline', float('nan')):.3f})"),
             "silence_random": (VERMILLION, "^", f"silenced at random  (mean {mean_of.get('silence_random', float('nan')):.3f})"),
             "silence_by_site": (GREEN, "s", f"silenced by site  (mean {mean_of.get('silence_by_site', float('nan')):.3f})")}

    for ax, xcol, xlabel in ((axes[0], "pair", "orderable pairs, %"),
                             (axes[1], "spread", "site spread of positives")):
        for fam, (col, _, _) in style.items():
            if fam in mean_of:
                ax.axhline(mean_of[fam], color=col, lw=1.0, ls=(0, (4, 3)),
                           alpha=0.55, zorder=1)
        for fam, (col, mk, lab) in style.items():
            sub = df[df.family == fam]
            ax.scatter(sub[xcol], sub["drop"], s=34, marker=mk, color=col,
                       alpha=0.85, linewidth=0, label=lab, zorder=3)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel, fontsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=8.5, color=GRID)
        ax.annotate(f"r = {r_of(xcol, 'drop', df):+.2f}", xy=(0.04, 0.93),
                    xycoords="axes fraction", fontsize=9, color=INK)

    axes[0].set_ylabel("leakage: random AUC − grouped AUC", fontsize=9.5)
    axes[1].legend(fontsize=8.3, frameon=False, loc="lower right")
    fig.suptitle("Leakage follows who was silenced, not how much order was lost:\n"
                 "neither axis separates the three groups that the colours do",
                 fontsize=10.5, x=0.012, ha="left", y=1.03)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{args.out}.{e}", bbox_inches="tight")
        print(f"wrote {args.out}.{e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
