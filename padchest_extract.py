#!/usr/bin/env python3
"""
padchest_extract.py — graded labels from PadChest's stemmed Spanish reports.

WHAT THIS IS FOR
----------------
The orderable-pairs gauge needs a graded column; PadChest publishes binary
presence lists. This produces the graded column, so the gauge has something to
measure on a corpus that is not the one it was designed against, and so the
agreement gauge can run against physician labels at a sample size where it
works.

THE SCALE, UNCHANGED FROM THE KNEE WORK
---------------------------------------
    negated   0.00   the report says the finding is absent
    silent    0.15   the report does not mention it
    hedged    0.60   the report raises it without committing
    asserted  1.00   the report states it

Silent ranks above negated because explicitly clear should rank below
unmentioned on a metric that reads order, and below hedged because a raised
possibility is more than silence.

WHAT THE CORPUS FORCED
----------------------
Four decisions came from the vocabulary profile rather than from Spanish.

1. TERMS MATCH AS PREFIXES, NEGATORS AS WHOLE TOKENS. The corpus is stemmed, so
   `interstici` and `intersticial` both occur and a prefix catches both. But a
   prefix negator is a disaster: `no` would swallow `nodul`. Negators, hedges
   and the normality words are matched whole.

2. PNEUMONIA IS NAMED AS A PROCESS. `proces` with `infecci` appears in 73% of
   pneumonia-positive reports where `neumon` reaches 27%. The house style names
   the process, not the disease, so the rule has to as well.

3. `no pued descart` IS A HEDGE, NOT A DENIAL. "no puede descartarse patologia"
   raises a possibility. Reading the negator alone inverts it — the same trap
   the knee lexicon hit with "without", in a different language.

4. NORMALITY CLEARS AT TWO SCOPES. "siluet cardi dentr normal" clears the heart
   and says nothing about the lungs, so an anchored normality phrase is scoped
   to the structures its clause names.

   But a bare "sin hallazg signific", with no structure named and no finding in
   the clause, is a global clearance in this corpus and not silence. These
   reports run to a median of 74 characters and a third of the studies are
   labelled normal outright; the blanket phrase is the commonest construction
   there is. Treating it as silence discards the largest available source of
   order, and on a rank metric explicitly-clear must sit below unmentioned.
   A later clause naming a finding still overrides it.

WHAT IT DOES NOT DO
-------------------
No laterality, no severity grading beyond hedged/asserted, no localisation.
Those matter for a clinical reader and not for the question being asked, which
is whether a graded column can be built here at all and what the gauge says
about it.

Usage:
    python padchest_extract.py --selftest
    python padchest_extract.py --csv padchest_labels.csv --out padchest_extractions.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

import pandas as pd

NEGATED, SILENT, HEDGED, ASSERTED = 0.0, 0.15, 0.60, 1.00

# Whole-token matches. A prefix here would be a bug: `no` swallows `nodul`.
NEGATORS = {"no", "sin", "ni", "nunca", "descartad"}
NORMALITY = {"normal", "normales", "libr", "libres", "conserv", "despej"}
HEDGES = {"probabl", "posibl", "posibil", "sugier", "sugest", "podri", "dudos",
          "compat", "valor", "vs", "versus", "parec", "aparent", "sospech",
          "indetermin", "descart"}

# "no pued descart X" raises X rather than denying it.
DESCART = re.compile(r"\bno\s+(?:se\s+)?(?:pued\w*|puede\w*)\s+descart")

# Nouns that make a negator into a blanket clearance: "sin hallazgos
# significativos", "no alteraciones valorables". `cambi` is deliberately absent
# — "sin grand cambi respect rx juli" compares with a prior study and asserts
# nothing about normality.
GLOBAL_NOUNS = ["hallazg", "alter", "anomal", "patologi", "signific",
                "relev", "destac", "valorabl", "particular"]

# Per target: `path` names the finding, `anat` names the structure a normality
# phrase can clear, `pair` requires two prefixes in the same clause. Every entry
# was taken from the dev-split token ranking and kept only where it names the
# finding rather than co-occurring with it.
LEXICON: dict[str, dict] = {
    "COPD signs": {
        "path": ["epoc", "enfisem", "bull", "aplan"],
        "pair": [("atrap", "aere")],
        "anat": [],
    },
    "cardiomegaly": {
        "path": ["cardiomegali"],
        "pair": [("ict", "alto"), ("cardiotorac", "alto"), ("cardiotorac", "aument"),
                 ("cardiac", "aument"), ("siluet", "aument")],
        "anat": ["cardiomegali", "cardiac", "cardiotorac", "ict", "siluet"],
    },
    "aortic elongation": {
        "path": ["elongacion", "elong"],
        "pair": [("ensanch", "aort"), ("boton", "aort"), ("desenroll", "aort")],
        "anat": ["aort", "aortic", "supraaort", "boton"],
    },
    "pleural effusion": {
        "path": ["derram"],
        "pair": [],
        "anat": ["pleural", "derram"],
    },
    "interstitial pattern": {
        "path": ["interstici"],
        "pair": [("patron", "interstici"), ("afectacion", "interstici")],
        "anat": ["interstici"],
    },
    "pneumonia": {
        "path": ["neumon", "neumoni", "bronconeumon"],
        "pair": [("proces", "infecci"), ("proces", "inflamatori"),
                 ("foc", "infecci"), ("condensacion", "infecci")],
        "anat": ["neumon", "infecci"],
    },
    "alveolar pattern": {
        "path": ["alveol"],
        "pair": [("ocupacion", "alveol"), ("patron", "alveol"), ("espaci", "aere")],
        "anat": ["alveol"],
    },
    "infiltrates": {
        "path": ["infiltr"],
        "pair": [],
        "anat": ["infiltr"],
    },
    "costophrenic angle blunting": {
        "path": ["pinzamient"],
        "pair": [("borramient", "costofren"), ("borramient", "sen"),
                 ("obliteracion", "costofren")],
        "anat": ["costofren", "sen"],
    },
    "nodule": {
        "path": ["nodul"],
        "pair": [],
        "anat": ["nodul"],
    },
    "atelectasis": {
        "path": ["atelectasi"],
        "pair": [],
        "anat": ["atelectasi"],
    },
    "pneumothorax": {
        "path": ["neumotorax"],
        "pair": [],
        "anat": ["neumotorax"],
    },
}

TARGETS = list(LEXICON)
TOKEN = re.compile(r"[a-z0-9]+")


def clauses(report: str) -> list[list[str]]:
    """Split on the corpus's own ' . ' separators, then tokenise."""
    out = []
    for part in re.split(r"\s*\.\s*", str(report).lower()):
        toks = TOKEN.findall(part)
        if toks:
            out.append(toks)
    return out


def has_prefix(toks: list[str], pref: str) -> int | None:
    for i, t in enumerate(toks):
        if t.startswith(pref):
            return i
    return None


def score_clause(toks: list[str], spec: dict, raw: str) -> float | None:
    """The clause's verdict on one target, or None if it says nothing."""
    hit = None
    for p in spec["path"]:
        i = has_prefix(toks, p)
        if i is not None:
            hit = i if hit is None else min(hit, i)
    if hit is None:
        for a, b in spec["pair"]:
            ia, ib = has_prefix(toks, a), has_prefix(toks, b)
            if ia is not None and ib is not None:
                cand = min(ia, ib)
                hit = cand if hit is None else min(hit, cand)

    neg_before = [i for i, t in enumerate(toks) if t in NEGATORS]
    hedged = any(t in HEDGES for t in toks) or bool(DESCART.search(raw))

    if hit is not None:
        # "no pued descart X" raises X; a plain negator before it denies it.
        if DESCART.search(raw):
            return HEDGED
        if any(i < hit for i in neg_before):
            return NEGATED
        return HEDGED if hedged else ASSERTED

    # No pathology term. A normality phrase can still clear the structure, but
    # only one this clause actually names — "siluet cardi dentr normal" says
    # nothing about the lungs.
    if spec["anat"] and any(t in NORMALITY for t in toks):
        if any(has_prefix(toks, a) is not None for a in spec["anat"]):
            return NEGATED
    # "sin alteraciones en X" / "no se aprecia X" with the structure named.
    if spec["anat"] and neg_before:
        for a in spec["anat"]:
            ia = has_prefix(toks, a)
            if ia is not None and any(i < ia for i in neg_before):
                return NEGATED
    return None


def names_any_finding(toks: list[str]) -> bool:
    for spec in LEXICON.values():
        if any(has_prefix(toks, p) is not None for p in spec["path"]):
            return True
        for a, b in spec["pair"]:
            if has_prefix(toks, a) is not None and has_prefix(toks, b) is not None:
                return True
    return False


def is_global_normal(toks: list[str]) -> bool:
    """A clause that clears the study rather than one structure.

    Requires a normality word, or a negator with one of the blanket nouns, and
    no finding named anywhere in the clause.
    """
    if names_any_finding(toks):
        return False
    if any(t in NORMALITY for t in toks):
        return True
    if any(t in NEGATORS for t in toks):
        return any(has_prefix(toks, n) is not None for n in GLOBAL_NOUNS)
    return False


def extract(report: str) -> dict[str, float]:
    best = {t: None for t in TARGETS}
    cleared = False
    for part in re.split(r"\s*\.\s*", str(report).lower()):
        toks = TOKEN.findall(part)
        if not toks:
            continue
        if is_global_normal(toks):
            cleared = True
        for t, spec in LEXICON.items():
            v = score_clause(toks, spec, part)
            if v is None:
                continue
            # An assertion anywhere outranks a denial elsewhere: a report that
            # clears the lungs in one line and names an infiltrate in the next
            # is reporting the infiltrate.
            best[t] = v if best[t] is None else max(best[t], v)

    floor = NEGATED if cleared else SILENT
    return {t: (floor if best[t] is None else best[t]) for t in TARGETS}


CASES = [
    (". . siluet cardi mediastin dentr normal . cambi pulmonar cronic . . . sen costofren libr . . . no sign enfermed metastas .",
     {"cardiomegaly": NEGATED, "costophrenic angle blunting": NEGATED}),
    ("sin hallazg signific con", {"cardiomegaly": NEGATED, "infiltrates": NEGATED}),
    ("sin hallazg signific . cardiomegali",
     {"cardiomegaly": ASSERTED, "infiltrates": NEGATED}),
    ("sin grand cambi respect rx previ", {"infiltrates": SILENT}),
    ("estudi dentr limit normal", {"nodule": NEGATED}),
    ("no apreci infiltr ni cond", {"infiltrates": NEGATED}),
    ("derram pleural izdo no pued descart patologi parenquimat subyacent",
     {"pleural effusion": HEDGED}),
    ("cardiomegali . sign radiolog epoc . marcapas con catet trav subclavi iz",
     {"cardiomegaly": ASSERTED, "COPD signs": ASSERTED}),
    ("hili prominent probabl orig vascul . sin grand cambi respect rx juli 20",
     {"nodule": SILENT}),
    ("sign radiolog atrap aere", {"COPD signs": ASSERTED}),
    ("pinzamient sen costofren derech", {"costophrenic angle blunting": ASSERTED}),
    ("patron interstici bilateral", {"interstitial pattern": ASSERTED}),
    ("proces infecci lid", {"pneumonia": ASSERTED}),
    ("nodul pulmonar lsd . recomiend tac torac", {"nodule": ASSERTED}),
    ("no neumotorax", {"pneumothorax": NEGATED}),
    ("atelectasi laminar bas izquierd", {"atelectasis": ASSERTED}),
    ("elongacion aortic", {"aortic elongation": ASSERTED}),
    ("sutil infiltr podri correspond proces infecci",
     {"infiltrates": HEDGED, "pneumonia": HEDGED}),
]


def selftest() -> int:
    bad = 0
    for text, want in CASES:
        got = extract(text)
        for t, v in want.items():
            if abs(got[t] - v) > 1e-9:
                bad += 1
                print(f"  FAIL  {t:28s} want {v:.2f} got {got[t]:.2f}")
                print(f"        {text[:96]}")
    print(f"\n{len(CASES)} cases, {bad} failures")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--out", default="padchest_extractions.csv")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.csv:
        print("need --csv or --selftest", file=sys.stderr)
        return 1

    import ast
    df = pd.read_csv(args.csv, low_memory=False)
    rep = (df.assign(_r=df["Report"].fillna(""))
             .sort_values("_r", key=lambda s: s.str.len(), ascending=False)
             .groupby("StudyID")
             .agg(report=("_r", "first"), method=("MethodLabel", "first")))
    rep = rep[rep["report"].str.len() > 0]
    if args.limit:
        rep = rep.head(args.limit)
    print(f"{len(rep):,} studies\n")

    name_of = {NEGATED: "negated", SILENT: "silent",
               HEDGED: "hedged", ASSERTED: "asserted"}
    rows, states = [], {t: Counter() for t in TARGETS}
    for sid, r in zip(rep.index, rep["report"]):
        sc = extract(r)
        for t, v in sc.items():
            rows.append({"StudyInstanceUID": sid, "target": t,
                         "score": v, "state": name_of[v]})
            states[t][v] += 1

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(rows):,} rows)\n")

    n = len(rep)
    print(f"  {'target':30s} {'negated':>9s} {'silent':>8s} {'hedged':>8s} {'asserted':>9s}")
    for t in TARGETS:
        c = states[t]
        print(f"  {t[:30]:30s} {100*c[NEGATED]/n:8.1f}% {100*c[SILENT]/n:7.1f}% "
              f"{100*c[HEDGED]/n:7.1f}% {100*c[ASSERTED]/n:8.1f}%")
    mean_silent = sum(states[t][SILENT] for t in TARGETS) / (n * len(TARGETS))
    print(f"\n  mean silence {100*mean_silent:.1f}%")
    print("\nNext: run spread.py on this file for pair_%, then the agreement")
    print("gauge against the held-out physician studies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
