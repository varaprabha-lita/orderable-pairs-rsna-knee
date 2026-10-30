#!/usr/bin/env python3
"""
group_folds.py — how much of the out-of-fold score is site leakage?

THE QUESTION
------------
Section 7 reports a macro out-of-fold AUC of 0.775, measured with randomly
stratified folds. Scanner settings identify the hospital, and a hospital has its
own base rates for every finding, so a model given studies from the same site in
both training and validation can learn the site rather than the knee. The
fifth-place team measured 0.019 of inflation from exactly this and switched to
site-grouped folds.

This script runs the twelve probes twice — once with the random folds we used,
once grouped by scanner signature — and reports the gap. That gap is the
leakage. It is a result rather than only a correction: it says how far a
weak-label pipeline validated the obvious way overstates itself.

WHY THE SCANNER STRINGS ARE NOT CLEANED UP
------------------------------------------
The signature list contains what look like duplicates: "SIEMENS | Aera | 1.5"
beside "Siemens Healthineers | MAGNETOM Aera | 1.5", three spellings of Philips
against the same Ingenia. These are not noise to normalise away. The vendor
string is written by the scanner's own software, and it varies by site
convention and installed version rather than by hardware — which is what makes
it a usable stand-in for the hospital. Merging the variants would merge sites
and defeat the grouping.

WHAT A GROUPED FOLD COSTS
-------------------------
Positives are not spread evenly across sites, and some targets are rare:
Fracture has 275 positives in 4,407 studies. StratifiedGroupKFold balances the
positive rate across folds as far as whole groups allow, but it cannot split a
group, so a rare target can still end up lopsided. The per-target positive
counts per fold are printed for this reason — read them before trusting a
target's grouped number.

Usage:
    python group_folds.py --embeddings data/interim/embeddings.parquet \\
                          --extractions data/interim/extractions.csv \\
                          --sites study_sites.csv
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

# The per-target regularisation of head_v3, the submission that scored 0.733
# public and 0.730 private. Reusing it keeps this a comparison of fold schemes
# and nothing else.
C_PER_TARGET = {
    "ACL": 0.003, "MCL": 0.001, "Medial Meniscus": 0.003,
    "Lateral Meniscus": 0.001, "Medial OA": 0.001, "Lateral OA": 0.001,
    "PF OA": 0.003, "Effusion": 0.003, "Synovitis": 0.003,
    "Baker's": 0.003, "Contusion": 0.003, "Fracture": 0.001,
}


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


def build_features(emb: pd.DataFrame):
    """Per-plane mean pooling, exactly as train_head and tune_head do."""
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
    return X, uids, dim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--sites", required=True, help="study_sites.csv from 05_headers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        print(f"needs scikit-learn 0.24 or newer: {exc}", file=sys.stderr)
        return 1

    emb = pd.read_parquet(args.embeddings)
    ext = pd.read_csv(args.extractions)
    sites = pd.read_csv(args.sites)

    X, uids, dim = build_features(emb)
    Y = ext.pivot_table(index="StudyInstanceUID", columns="target",
                        values="score", aggfunc="first").reindex(uids)

    site_of = dict(zip(sites["StudyInstanceUID"], sites["scanner"]))
    groups = np.array([site_of.get(u, "UNKNOWN") for u in uids])

    keep = Y[TARGETS[0]].notna().to_numpy() & (groups != "UNKNOWN")
    missing = int((groups == "UNKNOWN").sum())
    X, Y, groups = X[keep], Y[keep], groups[keep]
    uids = [u for u, k in zip(uids, keep) if k]

    n_sites = len(set(groups))
    print(f"{len(uids):,} studies x {X.shape[1]:,} features")
    print(f"{n_sites} scanner signatures"
          + (f"  ({missing} studies had none and were dropped)" if missing else ""))
    big = pd.Series(groups).value_counts()
    print(f"largest signature holds {big.iloc[0]} studies "
          f"({100 * big.iloc[0] / len(groups):.1f}% of the corpus)\n")

    if big.iloc[0] / len(groups) > 1.0 / args.folds:
        print("  ! one signature is larger than a fold — grouped folds will be "
              "uneven\n", file=sys.stderr)

    def pipe(C):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    random_cv = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.seed)
    grouped_cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                      random_state=args.seed)

    rows, t0 = [], time.time()

    for t in TARGETS:
        y = (Y[t].to_numpy(dtype=float) >= POSITIVE_AT).astype(int)
        if y.sum() < args.folds:
            print(f"  {t}: {int(y.sum())} positives, skipped")
            continue
        C = C_PER_TARGET[t]

        scores = {}
        spread = {}
        for name, cv, g in (("random", random_cv, None),
                            ("grouped", grouped_cv, groups)):
            p = np.zeros(len(y))
            per_fold_pos = []
            for tr, va in cv.split(X, y, g):
                m = pipe(C)
                m.fit(X[tr], y[tr])
                p[va] = m.predict_proba(X[va])[:, 1]
                per_fold_pos.append(int(y[va].sum()))
            scores[name] = auc_score(y, p)
            spread[name] = (min(per_fold_pos), max(per_fold_pos))

        drop = scores["grouped"] - scores["random"]
        rows.append({"target": t, "pos": int(y.sum()),
                     "random": round(scores["random"], 3),
                     "grouped": round(scores["grouped"], 3),
                     "drop": round(drop, 3),
                     "pos_per_fold": f"{spread['grouped'][0]}-{spread['grouped'][1]}"})
        print(f"  {t:18s} random {scores['random']:.3f}  "
              f"grouped {scores['grouped']:.3f}  ({drop:+.3f})   "
              f"positives/fold {spread['grouped'][0]}-{spread['grouped'][1]}"
              f"   [{(time.time()-t0)/60:.1f} min]")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print(df.to_string(index=False))
    print("=" * 86)

    r, g = df["random"].mean(), df["grouped"].mean()
    print(f"\nmacro OOF   random {r:.3f}   grouped {g:.3f}   ({g - r:+.3f})")
    print(f"reference: the fifth-place team measured 0.019 of inflation this way")
    print(f"reference: this head scored 0.733 public and 0.730 private\n")

    leak = r - g
    if leak <= 0.002:
        print("Essentially no leakage. The random folds were not flattering the")
        print("head, and Section 7's numbers stand as reported. Worth stating,")
        print("since it is not the result the fifth-place team got.")
    elif leak < 0.015:
        print(f"Mild leakage of {leak:.3f}. Report the grouped number as the honest")
        print("one and keep the random number beside it, since every offline")
        print("comparison in the paper was made under the random scheme.")
    else:
        print(f"Substantial leakage of {leak:.3f}, at or above what the fifth-place")
        print("team saw. The grouped number becomes the headline figure and the")
        print("Limitations entry about fold grouping becomes a Results entry.")
    print()
    print("Either way the private leaderboard is unaffected — 0.730 was always")
    print("measured on studies from institutions the model never saw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
