# Orderable pairs: a label-free gauge for report-derived supervision

Analysis code for the manuscript *Orderable pairs: a label-free gauge for
report-derived supervision, and the frozen-encoder ceiling on knee MRI*.

The paper asks what you can learn about a weak-supervision label column when
almost nothing is annotated. Its three findings are a third label-free gauge
that counts the orderable pairs a column admits, a quantification of how much
randomly stratified folds overstate such a pipeline, and a sampling-bound
argument for why a frozen encoder on a 224-pixel cache stops where it does.

## No data in this repository, by licence

Neither corpus can be redistributed, and that is a condition of use rather
than an oversight:

- **RSNA Knee Abnormality Detection** is released under the
  [RSNA MIRA Dataset Research Use Agreement](https://www.rsna.org/mira-license),
  whose clause 4 forbids distributing, publishing or reproducing any portion
  of it. Access requires individual registration — one per person, not one
  per group.
- **PadChest** is released by BIMCV under its own Research Use Agreement on
  equivalent terms.

Both agreements also forbid any attempt to re-identify a data subject.

So this repository carries no images, no reports, and no derived per-study
records. Tables keyed to `StudyInstanceUID` or `SeriesInstanceUID` are
themselves portions of the datasets, so extraction outputs are excluded too.
`.gitignore` is configured to make committing them by accident difficult;
please do not override it.

Everything the paper reports is an aggregate, and aggregates are reproducible
from these scripts once you have registered for the data yourself.

## What is here

| File | What it does |
| --- | --- |
| `padchest_vocab.py` | Learns the stemmed Spanish vocabulary from a development split, divided by patient rather than by study, and writes the held-out study ids it must never see again |
| `padchest_extract.py` | The Spanish extractor: prefix-matched findings, whole-token negators, two scopes of normality |
| `padchest_gauge.py` | The two PadChest measurements — whether grading clears the binary ceiling, and agreement against ~18,000 held-out physician-labelled studies with Hanley–McNeil intervals |
| `padchest_profile.py` | Corpus shape and label distribution |
| `binary_ceiling.py` | The 2p(1−p) demonstration across both corpora (Figure 3) |
| `perturb_gauge.py` | The falsification experiment: degrades each column at random and by scanner signature, three strengths each |
| `plot_perturb.py` | Redraws Figure 2 from `perturb.csv` without refitting |
| `group_folds.py` | Random against site-grouped folds at the per-target regularisation |
| `05_headers.ipynb` | The DICOM header pass over all 24,371 series, which the sampling-bound argument rests on |

The knee lexicon and its extraction pipeline are maintained separately and are
added here as `lexicon/`.

## Reproducing the figures

`perturb_gauge.py` takes about twenty minutes and writes `perturb.csv`;
`plot_perturb.py` redraws from that file in seconds, which is why they are
separate programs. `binary_ceiling.py` needs extractions from both corpora.

## A note on what did not survive

Two claims in earlier drafts were refuted by these scripts and are reported in
the paper as refuted. The correlation between a column's orderable pairs and
its leakage, which looked strong at r = −0.63 across twelve targets, falls to
−0.03 across the 84 columns `perturb_gauge.py` produces. The twelve-point
result was an artefact of twelve points. The experiment that killed it is in
this repository, and it is worth running before trusting any correlation
computed on twelve targets.

## Citation

Citation details will be added here once the paper is published.

## Licence

Code is released under the MIT Licence. The datasets are not covered by it and
are not included.
