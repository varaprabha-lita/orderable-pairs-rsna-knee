#!/usr/bin/env python3
"""
binary_ceiling.py — what the orderable-pairs gauge can and cannot see.

THE CLAIM
---------
For a column with k states holding proportions p_k, the share of orderable
pairs is the Gini impurity 1 - sum p_k^2. When k = 2 that collapses to

    pair_% = 2 p (1 - p)

a function of prevalence alone, maximised at 0.5 when p = 0.5. A binary label
column therefore tells the gauge nothing its prevalence did not already say,
and no binary column anywhere can exceed 50 percent.

This matters because it bounds the gauge's scope. PadChest publishes presence
lists - a study either carries a finding or it does not - so running the gauge
there returns the prevalence back. The RSNA columns clear that ceiling only
because the extractor grades: negated below silent below hedged below asserted.
The gauge and the grading are one design, not two.

This script demonstrates that on both corpora and draws the figure.

WHAT THE FIGURE SHOWS
---------------------
Prevalence on x, orderable pairs on y. The dashed curve is 2p(1-p), the ceiling
for any binary column. PadChest's twelve sit on it, because they are binary.
Each RSNA target appears twice - once binarised at the 0.6 validation cut, which
lands it on the curve, and once graded, which lifts it off - joined by an arrow.
The height of that arrow is what grading buys.

Usage:
    python binary_ceiling.py --padchest ~/padchest/padchest_labels.csv \\
                             --extractions ~/rsna-knee/data/interim/extractions.csv \\
                             --out binary_ceiling
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Spans 12.9% down to 0.31%, with pneumothorax in the rare-target role.
# 'fracture' exists in the taxonomy with 11 studies and is useless; the real
# fracture label in this corpus is 'callus rib fracture'.
PADCHEST_TARGETS = [
    "COPD signs", "cardiomegaly", "aortic elongation", "pleural effusion",
    "interstitial pattern", "pneumonia", "alveolar pattern", "infiltrates",
    "costophrenic angle blunting", "nodule", "atelectasis", "pneumothorax",
]

RSNA_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
POSITIVE_AT = 0.6

# Okabe-Ito. Validated: all six checks pass on a light surface, worst adjacent
# CVD separation dE 11.0 (deutan). Marker shape carries identity too, so the
# figure does not rely on colour alone.
BLUE, VERMILLION, GREEN, INK, GRID = "#0072B2", "#D55E00", "#009E73", "#222222", "#CCCCCC"


def pair_pct(values: np.ndarray) -> float:
    """Gini impurity over the value distribution, as a percentage."""
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(100.0 * (1.0 - np.sum(p ** 2)))


def parse_labels(raw) -> list:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        v = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(x).strip() for x in v] if isinstance(v, (list, tuple)) else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--padchest", required=True)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--out", default="binary_ceiling")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib:  pip install matplotlib", file=sys.stderr)
        return 1

    # ------------------------------------------------------------ PadChest
    print("loading PadChest...")
    pc = pd.read_csv(args.padchest, low_memory=False)
    pc["_l"] = pc["Labels"].map(parse_labels)
    per_study = pc.groupby("StudyID")["_l"].agg(lambda ls: {x for l in ls for x in l})
    n_pc = len(per_study)

    pc_rows = []
    for t in PADCHEST_TARGETS:
        y = per_study.map(lambda s: int(t in s)).to_numpy()
        p = y.mean()
        pc_rows.append({"target": t, "prev": p, "pair": pair_pct(y),
                        "formula": 100 * 2 * p * (1 - p)})
    pc_df = pd.DataFrame(pc_rows)
    pc_df["err"] = (pc_df["pair"] - pc_df["formula"]).abs()

    print(f"\nPadChest: {n_pc:,} studies, labels are presence lists\n")
    print(f"  {'target':30s} {'prev %':>8s} {'pair_%':>8s} {'2p(1-p)':>9s} {'diff':>7s}")
    for r in pc_rows:
        print(f"  {r['target'][:30]:30s} {100*r['prev']:8.2f} {r['pair']:8.2f} "
              f"{r['formula']:9.2f} {abs(r['pair']-r['formula']):7.4f}")
    print(f"\n  largest departure from 2p(1-p): {pc_df['err'].max():.2e}")
    print("  the gauge returns the prevalence and nothing else\n")

    # ---------------------------------------------------------------- RSNA
    print("loading RSNA extractions...")
    ext = pd.read_csv(args.extractions)
    Y = ext.pivot_table(index="StudyInstanceUID", columns="target",
                        values="score", aggfunc="first")

    rs_rows = []
    for t in RSNA_TARGETS:
        s = Y[t].dropna().to_numpy(dtype=float)
        b = (s >= POSITIVE_AT).astype(int)
        p = b.mean()
        rs_rows.append({"target": t, "prev": p, "graded": pair_pct(s),
                        "binarised": pair_pct(b), "states": len(np.unique(s))})
    rs_df = pd.DataFrame(rs_rows)
    rs_df["lift"] = rs_df["graded"] - rs_df["binarised"]

    print(f"\nRSNA: {len(Y):,} studies, extractor emits graded scores\n")
    print(f"  {'target':20s} {'states':>6s} {'prev %':>8s} {'graded':>8s} "
          f"{'binarised':>10s} {'lift':>7s}")
    for r in rs_rows:
        print(f"  {r['target'][:20]:20s} {r['states']:6d} {100*r['prev']:8.2f} "
              f"{r['graded']:8.2f} {r['binarised']:10.2f} "
              f"{r['graded']-r['binarised']:7.2f}")

    g, b = rs_df["graded"].mean(), rs_df["binarised"].mean()
    print(f"\n  mean graded    {g:.1f}%")
    print(f"  mean binarised {b:.1f}%   ({b-g:+.1f})")
    print(f"  grading multiplies orderable pairs by {g/b:.2f}x\n")

    # -------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)

    x = np.linspace(0, 1, 400)
    ax.plot(x, 100 * 2 * x * (1 - x), ls=(0, (5, 3)), lw=1.4, color=INK,
            alpha=0.55, zorder=1)
    ax.annotate("binary ceiling,  2p(1 − p)", xy=(0.555, 35.5),
                fontsize=8.5, color=INK, alpha=0.7)

    for _, r in rs_df.iterrows():
        ax.annotate("", xy=(r["prev"], r["graded"]), xytext=(r["prev"], r["binarised"]),
                    arrowprops=dict(arrowstyle="->", color=INK, alpha=0.35, lw=1.0),
                    zorder=2)

    ax.scatter(pc_df["prev"], pc_df["pair"], s=46, marker="o",
               facecolor="white", edgecolor=BLUE, linewidth=1.8, zorder=4,
               label=f"PadChest, binary labels (n={n_pc:,} studies)")
    ax.scatter(rs_df["prev"], rs_df["binarised"], s=40, marker="s",
               facecolor="white", edgecolor=GREEN, linewidth=1.6, zorder=4,
               label="RSNA, binarised at the 0.6 cut")
    ax.scatter(rs_df["prev"], rs_df["graded"], s=52, marker="^",
               color=VERMILLION, zorder=5,
               label="RSNA, graded scores as extracted")

    # Selective labels only: the two extremes of each corpus.
    for _, r in rs_df.iterrows():
        if r["target"] in ("Effusion", "Fracture"):
            ax.annotate(r["target"], (r["prev"], r["graded"]),
                        textcoords="offset points", xytext=(7, 3),
                        fontsize=8, color=INK)
    for _, r in pc_df.iterrows():
        if r["target"] in ("COPD signs", "pneumothorax"):
            ax.annotate(r["target"], (r["prev"], r["pair"]),
                        textcoords="offset points", xytext=(9, 4),
                        fontsize=8, color=INK)

    ax.set_xlabel("prevalence of the positive class", fontsize=9.5)
    ax.set_ylabel("orderable pairs, %", fontsize=9.5)
    ax.set_title("A binary label column cannot exceed 50% orderable pairs;\n"
                 "grading is what lifts a column off that ceiling",
                 fontsize=10.5, loc="left", pad=10)
    ax.set_xlim(-0.02, 0.78)
    ax.set_ylim(-2.5, 76)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8.5, color=GRID)
    ax.legend(fontsize=8.3, frameon=False, loc="upper right")
    fig.tight_layout()

    for ext_ in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext_}", bbox_inches="tight")
        print(f"wrote {args.out}.{ext_}")

    pd.concat([
        pc_df.assign(corpus="PadChest")[["corpus", "target", "prev", "pair"]],
        rs_df.assign(corpus="RSNA")[["corpus", "target", "prev", "graded", "binarised"]],
    ], ignore_index=True).to_csv(f"{args.out}.csv", index=False)
    print(f"wrote {args.out}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
