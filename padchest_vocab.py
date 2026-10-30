#!/usr/bin/env python3
"""
padchest_vocab.py — learn what the corpus says before writing any rule.

WHY THIS COMES FIRST
--------------------
PadChest's reports are not ordinary Spanish. They have been lowercased,
stripped of accents and stopwords, truncated to stems, and compound terms are
split apart:

    siluet cardi mediastin dentr normal      <- silueta cardiomediastinica dentro de lo normal
    no sign enfermed metastas                <- no signos de enfermedad metastasica
    sen costofren libr                       <- senos costofrenicos libres

A lexicon of full Spanish forms would match almost nothing here, and worse, a
miss would be indistinguishable from an absent finding. So the vocabulary is
read off the corpus rather than guessed, exactly as inspect_gaps does for the
knee work.

THE SPLIT, AND WHY IT IS NOT OPTIONAL
-------------------------------------
Tokens are ranked by how strongly they associate with PadChest's own labels.
That is label supervision, so any lexicon built from it would score
optimistically on the studies used to build it.

Only the physician-labelled studies are touched, and only a DEV share of them.
The rest are written to a file of held-out ids and never read again until the
extractor is evaluated. The split is by PatientID rather than by study, since
one patient contributes several studies and reports repeat across them.

WHAT IT PRINTS
--------------
Corpus shape; the commonest tokens, which are the candidate stopwords and
negators; per target, the tokens most over-represented in positive reports by
log-odds with a frequency floor; and a probe of candidate negation and
normality markers. The per-target lists are the raw material for the lexicon,
to be curated by hand rather than adopted wholesale: a token can associate with
a finding because it names it, because it names a cause, or because it names
something a radiologist happens to mention in the same breath.

Usage:
    python padchest_vocab.py --csv padchest_labels.csv --out vocab
    python padchest_vocab.py --csv padchest_labels.csv --out vocab --top 40
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

TARGETS = [
    "COPD signs", "cardiomegaly", "aortic elongation", "pleural effusion",
    "interstitial pattern", "pneumonia", "alveolar pattern", "infiltrates",
    "costophrenic angle blunting", "nodule", "atelectasis", "pneumothorax",
]

# Candidates only. Which of these actually act as negators is a question for
# the output, not an assumption: "sen" is senos (sinuses), not sin (without).
NEG_PROBE = ["no", "sin", "ni", "descart", "ausenc", "aus", "nieg", "negativ",
             "normal", "libr", "conserv", "dentr", "limit", "signific"]
HEDGE_PROBE = ["probabl", "posibl", "sugier", "sugest", "compatibl", "parec",
               "dud", "valor", "versus", "vs", "podri", "aparent", "leve",
               "minim", "sutil", "discret", "ligera"]

TOKEN = re.compile(r"[a-z0-9]+")


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
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="vocab")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--dev-frac", type=float, default=0.30)
    ap.add_argument("--min-docs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading...")
    df = pd.read_csv(args.csv, low_memory=False)
    df["_l"] = df["Labels"].map(parse_labels)

    # One row per study: union the labels, keep the first non-empty report.
    rep = (df.assign(_r=df["Report"].fillna(""))
             .sort_values("_r", key=lambda s: s.str.len(), ascending=False)
             .groupby("StudyID")
             .agg(report=("_r", "first"),
                  patient=("PatientID", "first"),
                  method=("MethodLabel", "first")))
    lab = df.groupby("StudyID")["_l"].agg(lambda ls: {x for l in ls for x in l})
    st = rep.join(lab.rename("labels"))
    st = st[st["report"].str.len() > 0]
    print(f"{len(st):,} studies with a report\n")

    # ------------------------------------------------------------ the split
    phys = st[st["method"].astype(str).str.lower().str.startswith("phys")]
    pats = np.array(sorted(phys["patient"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pats)
    n_dev = int(round(args.dev_frac * len(pats)))
    dev_pats = set(pats[:n_dev].tolist())

    dev = phys[phys["patient"].isin(dev_pats)]
    held = phys[~phys["patient"].isin(dev_pats)]

    print("=" * 76)
    print("SPLIT  — physician-labelled studies only, divided by patient")
    print("=" * 76)
    print(f"  physician-labelled studies {len(phys):>8,}   patients {len(pats):>7,}")
    print(f"  dev   (vocabulary is read from here) {len(dev):>8,}")
    print(f"  held  (never read again until evaluation) {len(held):>8,}")
    print(f"  RNN-labelled studies, unused here {len(st) - len(phys):>8,}")

    with open(f"{args.out}_heldout_studies.txt", "w") as fh:
        fh.write("\n".join(map(str, sorted(held.index))))
    with open(f"{args.out}_split.json", "w") as fh:
        json.dump({"seed": args.seed, "dev_frac": args.dev_frac,
                   "n_dev": len(dev), "n_held": len(held),
                   "dev_patients": sorted(map(str, dev_pats))}, fh)
    print(f"\n  wrote {args.out}_heldout_studies.txt and {args.out}_split.json")

    # ---------------------------------------------------------- tokenisation
    toks = dev["report"].map(lambda s: TOKEN.findall(s.lower()))
    doc_sets = toks.map(set)
    lengths = toks.map(len)
    overall = Counter(t for s in doc_sets for t in s)

    print("\n" + "=" * 76)
    print("CORPUS SHAPE  (dev split)")
    print("=" * 76)
    print(f"  tokens per report: median {int(lengths.median())}, "
          f"95th {int(lengths.quantile(.95))}, max {int(lengths.max())}")
    print(f"  distinct tokens: {len(overall):,}")
    print(f"  appearing in >= {args.min_docs} reports: "
          f"{sum(1 for v in overall.values() if v >= args.min_docs):,}\n")

    print(f"  commonest 50 tokens (candidate stopwords, negators, hedges):")
    for i, (t, n) in enumerate(overall.most_common(50)):
        end = "\n" if i % 5 == 4 else "  "
        print(f"    {t:>14s} {100*n/len(dev):5.1f}%", end=end)
    print()

    # ------------------------------------------------------- per-target rank
    print("\n" + "=" * 76)
    print(f"TOKENS BY ASSOCIATION  — top {args.top} per target, log-odds")
    print("=" * 76)
    print("  Curate these by hand. A token can associate with a finding because")
    print("  it names it, because it names a cause, or because a radiologist")
    print("  mentions it in the same breath. Only the first kind belongs.\n")

    rows = []
    for t in TARGETS:
        y = dev["labels"].map(lambda s: t in s).to_numpy()
        n_p, n_n = int(y.sum()), int((~y).sum())
        if n_p < 20:
            print(f"  {t}: only {n_p} positives in dev, skipped\n")
            continue
        cp = Counter(x for s, k in zip(doc_sets, y) if k for x in s)
        cn = Counter(x for s, k in zip(doc_sets, y) if not k for x in s)
        scored = []
        for tok, c in overall.items():
            if c < args.min_docs:
                continue
            a, b = cp.get(tok, 0), cn.get(tok, 0)
            if a < 5:
                continue
            lo = np.log(((a + 0.5) / (n_p + 1)) / ((b + 0.5) / (n_n + 1)))
            scored.append((lo, tok, a, b, 100 * a / n_p, 100 * b / n_n))
        scored.sort(reverse=True)
        print(f"  {t}  ({n_p:,} positive of {len(dev):,} dev studies)")
        print(f"    {'token':>16s} {'logodds':>8s} {'in pos':>7s} {'in neg':>7s} "
              f"{'pos %':>7s} {'neg %':>7s}")
        for lo, tok, a, b, pp, pn in scored[:args.top]:
            print(f"    {tok:>16s} {lo:8.2f} {a:7d} {b:7d} {pp:7.1f} {pn:7.1f}")
            rows.append({"target": t, "token": tok, "logodds": round(lo, 3),
                         "in_pos": a, "in_neg": b,
                         "pos_pct": round(pp, 2), "neg_pct": round(pn, 2)})
        print()

    pd.DataFrame(rows).to_csv(f"{args.out}_tokens.csv", index=False)
    print(f"wrote {args.out}_tokens.csv  ({len(rows)} rows)\n")

    # ------------------------------------------------------- negation probe
    print("=" * 76)
    print("NEGATION AND HEDGE CANDIDATES  — which of these are real?")
    print("=" * 76)
    print(f"    {'token':>14s} {'reports':>9s} {'share':>7s}   example context")
    for name, probe in (("negation / normality", NEG_PROBE), ("hedging", HEDGE_PROBE)):
        print(f"\n  {name}:")
        for tok in probe:
            n = overall.get(tok, 0)
            if n == 0:
                print(f"    {tok:>14s} {'absent':>9s}")
                continue
            ex = next((r for r, s in zip(dev["report"], doc_sets) if tok in s), "")
            ctx = ex[:72].replace("\n", " ")
            print(f"    {tok:>14s} {n:9,} {100*n/len(dev):6.1f}%   {ctx}")

    print("\n" + "=" * 76)
    print("Send the output back. The next step is the lexicon, written by hand")
    print("from these lists, with grading and negation adapted to stemmed text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
