# Making the contrast classifier reproducible on your machine

Four changes, all aimed at one question: **when you point this model at a new dataset in six
months, can you tell whether it is still the model you trained?** Without them the answer is no —
a scikit-learn `Pipeline` is a pickle, and a pickle carries no record of the environment that
made it.

Nothing here depends on this app. Everything runs from a terminal.

## What changed

| # | Change | Where |
|---|---|---|
| 1 | Model bundle now carries a provenance stamp: library versions, source-file hashes, input-file hashes, argv, git commit | `provenance.py`, written by `train_classifier.py` |
| 2 | Exact package pins written at fit time | `requirements-lock.txt`, auto-generated |
| 3 | Machine-readable run record: inputs, hashes, environment, grouped-CV metrics, top features | `training_manifest.json`, auto-generated |
| 4 | Frozen acceptance fixture replayed through the bundle | `selftest_classifier.py` |

`provenance.py` is imported by both `train_classifier.py` and `classify_offline.py`, for the same
reason `contrast_features.py` is: the fields written at training and the fields checked at
inference have to be defined in one place or they drift apart silently.

## Workflow

```bash
# 0. environment
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. train (writes model + report + manifest + pins)
python train_classifier.py \
    --labels contrast_categories.xlsx \
    --comparisons comparison_registry.csv \
    --cache geo_cache \
    --out contrast_classifier.joblib

# 2. freeze the acceptance fixture — ONCE, immediately after training
python selftest_classifier.py --write-fixture \
    --comparisons comparison_registry.csv --cache geo_cache

# 3. before trusting any later run, on any machine
python selftest_classifier.py --cache geo_cache        # exit 0 = pass, 1 = fail

# 4. score new data
python classify_offline.py --comparisons new_registry.csv \
    --model contrast_classifier.joblib --out new_categories.xlsx
#   add --strict-provenance to refuse to run on a mismatched environment (exit 2)
```

Keep these together and treat them as one unit — the model alone is not a deliverable:

```
contrast_classifier.joblib   training_manifest.json   requirements-lock.txt
fixture/                     contrast_features.py     provenance.py
drug_lexicon.csv
```

To rebuild the training environment on another machine, `pip install -r requirements-lock.txt` —
the pins and the interpreter version are recorded there as of fit time. That file is distinct
from the repository's `requirements.txt`, which declares only what the code needs: one records
what a *particular model* was fitted under, the other what the *code* requires.

## The two levels of the self-test, and why both are needed

`selftest_classifier.py` replays 24 frozen contrasts (one per study, so the sample spans series
rather than re-testing the same shared text) and checks two independent things:

- **Level A — prediction reproducibility.** Frozen *feature rows* in, predictions out, compared
  to the frozen predictions. Hermetic: needs only the bundle, no GEO cache, no network. This is
  the check that catches scikit-learn or numpy pickle drift — the failure mode that changes
  answers without raising an exception.
- **Level B — feature reproducibility.** Rebuilds the feature rows from the frozen registry slice
  and the GEO cache, and compares them cell by cell. Catches an edit to `contrast_features.py`,
  a swapped cache, or a GEO record revised upstream. Skipped (not failed) when no cache is
  present.

They fail independently, and that is the point. In the verification run below, editing one
string in `contrast_features.py` left **Level A passing and Level B failing** — the model was
intact but would have been fed different features than it was trained on. That is precisely the
failure that shows up as quietly degraded accuracy on new data and never as an error.

**Regenerate the fixture only when you retrain deliberately — never to make a failing test
pass.** If the model file's hash has changed since the fixture was written, the self-test says so
explicitly rather than letting you misread the failure.

## Severity model

A hash mismatch is not proof the model is wrong; it is proof you no longer know that it is right.
So the check is graded:

- **warn** (run continues) — a numpy/pandas/openpyxl version differs, or the interpreter patch
  version moved.
- **error** (run continues, loudly; exits 2 under `--strict-provenance`) — scikit-learn changed,
  or `contrast_features.py` changed. These are the two that can alter predictions from a pickle
  without raising.
- A bundle with **no stamp at all** (anything trained before this change) warns that its training
  environment is unknown and cannot be verified.

Output workbooks from `classify_offline.py` now carry a `provenance` sheet recording the model
file and hash, when it was trained, the training and scoring commands, and a
trained-vs-scored comparison of every tracked package and module — so a spreadsheet found on disk
later can be traced back to the exact model that produced it.

## Verification

### Continuously, on synthetic data

`pytest tests/` runs the whole cycle — demo data, training, stamping, both self-test levels,
offline scoring, strict-provenance refusal — on every push, across Python 3.10/3.11/3.12. It
needs no network and no API key. The synthetic accuracy is meaningless; what CI asserts is that
the apparatus works, including that a deliberate edit to `contrast_features.py` is caught.

### Once, on real data

The original verification ran on a fixed-seed 20% random sample of the maintainer's
`comparison_registry.csv` (seed `20260920`; 121 of 604 contrasts, 44 studies). Results were
sample-based and described the plumbing, not model quality — grouped-CV accuracy of 0.57 on that
sample is not a model performance claim, and `nutrition` had 4 examples.

| Check | Result |
|---|---|
| Training writes stamped bundle + manifest + pins | pass — 26 manifest fields, 6 packages pinned |
| Clean self-test | pass — 24/24 classes agree, max probability drift 5.6e-17 |
| Self-test after editing `contrast_features.py` | Level A pass, **Level B fail**, exit 1 |
| `--strict-provenance` scoring after that edit | refused, exit 2, mismatch printed with both hashes |
| Clean scoring run | exit 0, `provenance` sheet written |

## What this does not do

- It does not pin transitive dependencies or the BLAS build. If you need bit-identical behaviour
  across machines rather than reproducible-on-this-one, the next step is a lockfile
  (`pip freeze`, or `uv`/`conda-lock`) and ideally a container.
- It does not make the *labels* reproducible. `label_contrasts.py` calls an API whose model
  changes under you; the manifest hashes the label file so you know which labels a model was
  distilled from, but re-running the teacher will not reproduce them exactly.
- It does not validate the GEO cache against live GEO. Series records do get revised; Level B
  detects a changed cache only if you keep the fixture and the cache together.
