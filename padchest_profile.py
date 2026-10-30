#!/usr/bin/env python3
"""
padchest_profile.py — what is in PadChest, and what can it replicate?

WHY THIS CORPUS
---------------
PadChest is the replication set this paper needs, and it is better suited than
a second knee dataset would be. Three of its columns matter:

  Report        the free-text radiology report, in Spanish
  MethodLabel   whether a study's labels were assigned by a physician or by
                the authors' recurrent network
  Labels        a list of findings per image, drawn from a 174-term taxonomy

The second of those is the point. RSNA gave 58 expert-labelled studies, which
Section 3 argues cannot arbitrate anything. PadChest's physician-labelled
portion is roughly 27,000 studies. The agreement gauge, underpowered to the
point of uselessness on the first corpus, becomes a real measurement on this
one — which is the obvious objection to Section 4 and the obvious answer to it.

Manufacturer_DICOM supports the site-grouping result of Section 7, and
Rows/Columns/SpatialResolution support the pitch argument of Section 8. All
three findings can be tested here.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It decides nothing. It prints what is in the file so that the mapping decisions
— which findings become targets, what an absent finding means, whether the unit
is the image or the study — are made against counts rather than against
assumptions. Those decisions belong in the paper, so they should be visible.

UNIT OF ANALYSIS
----------------
PadChest has one row per image and several images per study. The RSNA work is
study-level and the competition metric is study-level, so the comparison should
be too. Labels are unioned across a study's images, which is the same choice
PadChest's own authors describe for report-level annotation, since the report
covers the study rather than one projection.

Usage:
    python padchest_profile.py --csv padchest_labels.csv
    python padchest_profile.py --csv padchest_labels.csv --top 60
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Candidate target set: the RSNA twelve have no chest analogue, so these are
# chosen to span the same range of difficulty — common and rare, diffuse and
# focal, and at least one known to be reporting-habit dependent.
CANDIDATES = [
    "cardiomegaly", "pleural effusion", "pneumonia", "atelectasis",
    "infiltrates", "pulmonary edema", "pneumothorax", "nodule",
    "COPD signs", "fracture", "hiatal hernia", "costophrenic angle blunting",
]


def parse_labels(raw) -> list:
    """The Labels column holds a Python list literal, with some rows malformed."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        v = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(v, (list, tuple)):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    print("loading (166k rows, this takes a moment)...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"{len(df):,} image rows, {df.shape[1]} columns\n")

    # ---------------------------------------------------------------- units
    print("=" * 74)
    print("UNITS")
    print("=" * 74)
    for col in ("StudyID", "PatientID", "ReportID"):
        if col in df:
            print(f"  unique {col:12s} {df[col].nunique():,}")
    print(f"  images per study    {len(df) / max(df['StudyID'].nunique(), 1):.2f}")

    # --------------------------------------------------------- label method
    print("\n" + "=" * 74)
    print("LABEL PROVENANCE  — the reason this corpus is worth having")
    print("=" * 74)
    if "MethodLabel" in df:
        vc = df["MethodLabel"].fillna("(missing)").value_counts()
        for k, v in vc.items():
            print(f"  {str(k)[:46]:46s} {v:7,}  ({100*v/len(df):5.1f}%)")
        by_study = df.groupby("StudyID")["MethodLabel"].agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else "(missing)")
        print("\n  per study:")
        for k, v in by_study.value_counts().items():
            print(f"  {str(k)[:46]:46s} {v:7,}  ({100*v/len(by_study):5.1f}%)")

    # --------------------------------------------------------------- labels
    print("\n" + "=" * 74)
    print(f"FINDINGS — top {args.top} of the taxonomy, counted per study")
    print("=" * 74)
    df["_labels"] = df["Labels"].map(parse_labels)
    empty = int((df["_labels"].str.len() == 0).sum())
    print(f"  rows with no parseable label: {empty:,} ({100*empty/len(df):.1f}%)\n")

    per_study = df.groupby("StudyID")["_labels"].agg(
        lambda lists: sorted({x for l in lists for x in l}))
    n_studies = len(per_study)
    counts = Counter(x for l in per_study for x in l)

    for name, n in counts.most_common(args.top):
        print(f"  {name[:44]:44s} {n:7,}  ({100*n/n_studies:5.2f}%)")
    print(f"\n  {len(counts)} distinct findings in all")

    # ------------------------------------------------------ candidate targets
    print("\n" + "=" * 74)
    print("CANDIDATE TARGET SET")
    print("=" * 74)
    print("  Matched against the taxonomy; unmatched names need replacing.\n")
    for c in CANDIDATES:
        n = counts.get(c, 0)
        mark = " " if n else "  <-- NOT FOUND"
        print(f"  {c[:40]:40s} {n:7,}  ({100*n/n_studies:5.2f}%){mark}")

    # ----------------------------------------------------------------- sites
    print("\n" + "=" * 74)
    print("SITE STRUCTURE  — for the Section 7 replication")
    print("=" * 74)
    if "Manufacturer_DICOM" in df:
        man = df.groupby("StudyID")["Manufacturer_DICOM"].agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else "(missing)")
        vc = man.value_counts()
        print(f"  {len(vc)} distinct manufacturers over {len(man):,} studies")
        for k, v in vc.head(12).items():
            print(f"    {str(k)[:40]:40s} {v:7,}  ({100*v/len(man):5.1f}%)")
        print("\n  NOTE: PadChest is one hospital, so manufacturer separates")
        print("  equipment rather than institutions. The Section 7 result may")
        print("  therefore be weaker here, and that is itself worth reporting:")
        print("  it would say the leakage is institutional rather than physical.")

    # ----------------------------------------------------------------- pitch
    print("\n" + "=" * 74)
    print("ACQUISITION GEOMETRY  — for the Section 8 replication")
    print("=" * 74)
    sr = pd.to_numeric(df.get("SpatialResolution_DICOM"), errors="coerce")
    rows = pd.to_numeric(df.get("Rows_DICOM"), errors="coerce")
    cols = pd.to_numeric(df.get("Columns_DICOM"), errors="coerce")
    print(f"  SpatialResolution present on {int(sr.notna().sum()):,} rows")
    if sr.notna().any():
        print(sr.describe(percentiles=[.05, .5, .95]).round(4).to_string())
    print(f"\n  Rows   median {rows.median():.0f}   Columns median {cols.median():.0f}")
    fov = (sr * rows).dropna()
    if len(fov):
        print(f"  implied extent (mm) median {fov.median():.0f}")
        for P in (224, 384):
            p = fov / P
            print(f"    at {P} px: median {p.median():.3f} mm/px")

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 74)
    print("REPORTS")
    print("=" * 74)
    rep = df["Report"].fillna("")
    print(f"  non-empty reports: {int((rep.str.len() > 0).sum()):,}")
    print(f"  median length: {int(rep[rep.str.len() > 0].str.len().median())} characters")
    print("\n  three examples:\n")
    for s in rep[rep.str.len() > 40].head(3):
        print(f"    {s[:200]}\n")

    print("=" * 74)
    print("Send this output back and the next step is the mapping into the")
    print("shape spread.py expects: a target set, a scoring scale, and a rule")
    print("for what an absent finding means.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
