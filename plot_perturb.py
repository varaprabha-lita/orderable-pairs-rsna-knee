#!/usr/bin/env python3
"""
plot_perturb.py — redraw the perturbation figure from perturb.csv.

The fits take twenty minutes and the layout takes seconds, so they should not
be coupled. This reads the CSV perturb_gauge.py already wrote and redraws only
the figure, which makes further adjustments free.

Usage:
    python plot_perturb.py --csv perturb.csv --out perturb
    python plot_perturb.py --csv perturb.csv --out perturb --left-only
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

BLUE, VERMILLION, GREEN, INK, GRID = "#0072B2", "#D55E00", "#009E73", "#222222", "#CCCCCC"

STYLE = {
    "baseline":        (BLUE,       "o", "unperturbed"),
    "silence_random":  (VERMILLION, "^", "silenced at random"),
    "silence_by_site": (GREEN,      "s", "silenced by site"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="perturb.csv")
    ap.add_argument("--out", default="perturb")
    ap.add_argument("--left-only", action="store_true",
                    help="drop the site-spread panel, which only restates the left one")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("needs matplotlib", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    mean_of = {f: df[df.family == f]["drop"].mean() for f in df.family.unique()}

    def r_of(xcol):
        s = df.dropna(subset=[xcol, "drop"])
        return float(np.corrcoef(s[xcol], s["drop"])[0, 1])

    panels = [("pair", "orderable pairs, %")]
    if not args.left_only:
        panels.append(("spread", "site spread of positives"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 4.3),
                             dpi=200, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (xcol, xlabel) in zip(axes, panels):
        for fam, (col, _, _) in STYLE.items():
            if fam in mean_of:
                ax.axhline(mean_of[fam], color=col, lw=1.0, ls=(0, (4, 3)),
                           alpha=0.55, zorder=1)
        for fam, (col, mk, _) in STYLE.items():
            sub = df[df.family == fam]
            ax.scatter(sub[xcol], sub["drop"], s=34, marker=mk, color=col,
                       alpha=0.85, linewidth=0, zorder=3,
                       label=f"{STYLE[fam][2]}  (mean {mean_of[fam]:.3f})")
        ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel, fontsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=8.5, color=GRID)
        ax.annotate(f"r = {r_of(xcol):+.2f}", xy=(0.035, 0.94),
                    xycoords="axes fraction", fontsize=9.5, color=INK)

    axes[0].set_ylabel("leakage: random AUC − grouped AUC", fontsize=9.5)

    # The legend rides above the axes, so it cannot sit on the mean lines.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.008, 1.012),
               ncol=3, frameon=False, fontsize=8.6, handletextpad=0.4,
               columnspacing=1.9)
    fig.suptitle("Leakage follows who was silenced, not how much order was lost",
                 fontsize=11, x=0.008, ha="left", y=1.085)
    fig.tight_layout()

    for e in ("png", "pdf"):
        fig.savefig(f"{args.out}.{e}", bbox_inches="tight")
        print(f"wrote {args.out}.{e}")

    print(f"\n  n = {len(df)} columns")
    for f, m in mean_of.items():
        print(f"  {f:18s} mean leakage {m:+.3f}   (n={int((df.family==f).sum())})")
    print(f"\n  r(pair_%, leakage)      {r_of('pair'):+.3f}")
    print(f"  r(site spread, leakage) {r_of('spread'):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
