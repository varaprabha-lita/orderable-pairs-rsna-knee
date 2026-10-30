#!/usr/bin/env python3
"""
padchest_gauge.py — the two questions this corpus was obtained to answer.

QUESTION ONE: does a graded column clear the binary ceiling here too?
---------------------------------------------------------------------
PadChest publishes presence lists, and a binary column's orderable pairs are
2p(1-p) and nothing more. The knee extractor's grading lifted its columns to
1.79 times their binarised value. This asks whether the same holds on reports
with a median length of 74 characters, where a third of studies are simply
normal. The three numbers per target are the graded column, the same column
thresholded at 0.6, and PadChest's own published binary label.

QUESTION TWO: what does the agreement gauge say when it has enough studies?
---------------------------------------------------------------------------
Section 3 of the paper argues that 58 annotated studies cannot arbitrate
anything: at A = 0.85 with nine positives the 95% interval spans about 0.14 in
each direction, an order of magnitude wider than the differences that separate
leaderboard positions. That argument is sound and it has never been tested
against the alternative, because the alternative did not exist.

Here it does. The held-out physician-labelled studies number roughly 18,000,
and none of them were read when the lexicon was written. The same Hanley-McNeil
interval is printed beside each AUC so the two regimes can be compared directly.

A caveat that belongs in the paper: PadChest's physicians annotated REPORTS,
not images. So this measures extraction accuracy cleanly, isolated from the
report-image disagreement that puts a ceiling on the knee corpus's gauge — a
different question, honestly a cleaner one, and not the same question the
competition scored.

Usage:
    python padchest_gauge.py --extractions padchest_extractions.csv \\
                             --labels padchest_labels.csv \\
                             --heldout vocab_heldout_studies.txt
"""

from __future__ import annotations

import argparse
import ast
import sys

import numpy as np
import pandas as pd

POSITIVE_AT = 0.6


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


def hanley_se(a: float, n_p: int, n_n: int) -> float:
    """Hanley and McNeil's standard error — the same formula Section 3 uses."""
    if n_p < 1 or n_n < 1 or not np.isfinite(a):
        return float("nan")
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n_p - 1) * (q1 - a * a) + (n_n - 1) * (q2 - a * a)) / (n_p * n_n)
    return float(np.sqrt(max(v, 0.0)))


def pair_pct(v: np.ndarray) -> float:
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(100.0 * (1.0 - np.sum(p ** 2)))


def parse_labels(raw) -> list:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        x = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(t).strip() for t in x] if isinstance(x, (list, tuple)) else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--heldout", required=True)
    args = ap.parse_args()

    ext = pd.read_csv(args.extractions)
    S = ext.pivot_table(index="StudyInstanceUID", columns="target",
                        values="score", aggfunc="first")
    targets = list(S.columns)
    print(f"{len(S):,} studies, {len(targets)} targets\n")

    lab = pd.read_csv(args.labels, low_memory=False)
    lab["_l"] = lab["Labels"].map(parse_labels)
    per = lab.groupby("StudyID").agg(
        labels=("_l", lambda ls: {x for l in ls for x in l}),
        method=("MethodLabel", "first"))

    held = {int(x) if x.strip().lstrip("-").isdigit() else x.strip()
            for x in open(args.heldout) if x.strip()}
    idx = [i for i in S.index if i in held]
    print(f"held-out physician studies present in both files: {len(idx):,}\n")
    if len(idx) < 500:
        print("  ! too few to evaluate — check the id types match", file=sys.stderr)

    # ------------------------------------------------- question one: ceiling
    print("=" * 84)
    print("ORDERABLE PAIRS  — does grading clear the binary ceiling on this corpus?")
    print("=" * 84)
    print(f"  {'target':30s} {'graded':>8s} {'binarised':>10s} "
          f"{'published':>10s} {'prev %':>8s} {'lift':>7s}")
    rows = []
    for t in targets:
        s = S[t].to_numpy(dtype=float)
        b = (s >= POSITIVE_AT).astype(int)
        pub = per["labels"].reindex(S.index).map(lambda st: int(t in st) if isinstance(st, set) else 0).to_numpy()
        g, bz, pz = pair_pct(s), pair_pct(b), pair_pct(pub)
        rows.append({"target": t, "graded": g, "binarised": bz,
                     "published": pz, "prev": 100 * pub.mean(), "lift": g - bz})
        print(f"  {t[:30]:30s} {g:8.1f} {bz:10.1f} {pz:10.1f} "
              f"{100*pub.mean():8.2f} {g-bz:7.1f}")

    df = pd.DataFrame(rows)
    mg, mb, mp = df["graded"].mean(), df["binarised"].mean(), df["published"].mean()
    print(f"\n  mean graded {mg:.1f}   binarised {mb:.1f}   published {mp:.1f}")
    print(f"  grading multiplies orderable pairs by {mg/mb:.2f}x "
          f"(knee corpus: 1.79x)")

    # ----------------------------------------------- question two: agreement
    print("\n" + "=" * 84)
    print("AGREEMENT  — the gauge Section 3 says 58 studies cannot support")
    print("=" * 84)
    print("  Held-out physician-labelled studies, none read while writing the lexicon.\n")
    print(f"  {'target':30s} {'pos':>7s} {'AUC':>7s} {'SE':>7s} "
          f"{'95% CI':>16s} {'width':>7s}")

    sub = S.loc[idx]
    labs = per["labels"].reindex(idx)
    aucs, widths = [], []
    for t in targets:
        y = labs.map(lambda st: int(t in st) if isinstance(st, set) else 0).to_numpy()
        p = sub[t].to_numpy(dtype=float)
        n_p, n_n = int(y.sum()), int((1 - y).sum())
        if n_p < 10:
            print(f"  {t[:30]:30s} {n_p:7d}   too few positives")
            continue
        a = auc_score(y, p)
        se = hanley_se(a, n_p, n_n)
        lo, hi = a - 1.96 * se, a + 1.96 * se
        aucs.append(a); widths.append(2 * 1.96 * se)
        print(f"  {t[:30]:30s} {n_p:7d} {a:7.3f} {se:7.4f} "
              f"  [{lo:.3f}, {hi:.3f}] {2*1.96*se:7.3f}")

    if aucs:
        print(f"\n  macro AUC {np.mean(aucs):.3f} over {len(aucs)} targets")
        print(f"  median 95% interval width {np.median(widths):.3f}")
        print(f"  the knee corpus, on 58 studies, gave {0.28:.3f}")
        print(f"  ratio {0.28/np.median(widths):.0f}x narrower")

    print("\n" + "=" * 84)
    print("Both numbers go in the paper. The first says whether the gauge's")
    print("precondition holds on a corpus it was not designed against; the")
    print("second is the first time its own agreement gauge has had the sample")
    print("size to say anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
