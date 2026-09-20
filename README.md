# mtb-contrast-classifier

Assign each differential-expression contrast in a multi-study *Mycobacterium tuberculosis*
transcriptome atlas to one of five perturbation categories — **genetic, drug, environment,
nutrition, infection** — from the contrast's name and its GEO series metadata, reproducibly.

The problem this solves is mundane and unavoidable: an atlas assembled from ~100 independent
public studies arrives as several hundred contrasts whose category has to be decided from
author free text. Doing that by hand is slow and unauditable; doing it with an LLM every time
is slow, costly, and not reproducible, because the model changes under you. So the pipeline
does it **once** with an LLM and then distils those labels into a local scikit-learn model that
runs offline, deterministically, forever after.

Because the whole point is a label you can defend later, every fitted model carries a
provenance stamp and a frozen acceptance fixture — see
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Install

```bash
git clone https://github.com/manjusri1998/mtb-contrast-classifier
cd mtb-contrast-classifier
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. Only stage 1 needs the `anthropic` package and an API key; training, self-testing
and scoring need neither.

## Try it without any data

This repository ships **code only** — no GEO cache, no registry, no labels. To see the full
cycle run on synthetic stand-ins:

```bash
python make_demo_data.py --out demo
python train_classifier.py --labels demo/contrast_categories.xlsx \
    --comparisons demo/comparison_registry.csv --cache demo/geo_cache --out model.joblib
python selftest_classifier.py --write-fixture --model model.joblib \
    --comparisons demo/comparison_registry.csv --cache demo/geo_cache
python selftest_classifier.py --model model.joblib --cache demo/geo_cache
python classify_offline.py --model model.joblib \
    --comparisons demo/comparison_registry.csv --cache demo/geo_cache --no-fetch \
    --out scored.xlsx
```

The demo text is invented and the accessions are fake. Any accuracy figure obtained from it is
meaningless — it exercises the plumbing, nothing else.

For real data, see [docs/INPUTS.md](docs/INPUTS.md) for the three input files and their schemas.

## The pipeline

```
                 ┌─ label_contrasts.py ──┐   LLM, once, needs an API key
your registry ──>│        or             │──> contrast_categories.xlsx ──┐
                 └─ classify_rules.py ───┘   deterministic lexicon rules │
                                                                         v
                                                          train_classifier.py
                                                                         │
                          model.joblib + manifest + lock file + fixture/ │
                                                                         v
                    new registry ────────────────────────────> classify_offline.py
                                                                  no API key, no cost
```

| script | what it does |
|---|---|
| `label_contrasts.py` | Stage 1 teacher. Fetches GEO series metadata, asks an LLM for a category per contrast. One pass, one cost. |
| `classify_rules.py` | Deterministic alternative to stage 1: lexicon rules over the same text, no API. |
| `train_classifier.py` | Stage 2. Distils the labels into TF-IDF + multinomial logistic regression. Writes the model, a report, a manifest and a lock file. |
| `classify_offline.py` | Applies a fitted model to any registry. No API key. Emits probabilities, abstentions, drug class and a review flag. |
| `selftest_classifier.py` | Replays a frozen fixture through the model to prove it still behaves as it did at fit time. |
| `contrast_features.py` | The feature space. Imported by training **and** inference so the two cannot drift. |
| `provenance.py` | The reproducibility stamp. Imported by both, for the same reason. |
| `make_demo_data.py` | Synthetic inputs, so the repo is runnable with no data. |
| `compare_classifiers.py` | Head-to-head: distilled model vs lexicon rules on identical held-out studies, with a selective-prediction table. Decides which classifier to deploy. |

## What is learned, and what deliberately is not

**Learned:** `primary_category`, as TF-IDF (1–2 grams) over the contrast name, series title and
summary, and the series sample-group vocabulary, plus two categorical features — an aeration
signal parsed from the growth protocol, and the platform. A linear model on sparse text is the
right model class at this sample size, and its coefficients are readable: `--top-features`
prints the terms driving each category, so a label can be defended rather than merely asserted.

**Not learned:** `drug_class`. That is a lexicon lookup (`drug_lexicon.csv`) applied at
inference. Thirteen mechanism classes over a few hundred drug contrasts is too thin to fit, and
the agent-to-mechanism mapping is a fact you curate, not a pattern you infer. Contrasts matching
more than one class are reported as ambiguous rather than silently reduced to one.

## Reading the accuracy numbers

`train_classifier.py` reports two estimates and they differ a lot. **Use the grouped one.**

Contrasts from a single GEO series share almost all of their text — the same title, the same
summary, the same sample vocabulary. A random train/test split puts some contrasts from a series
on each side, so the model recognises the series rather than the biology, and the reported
accuracy is one you will never see on a new study. Grouped 5-fold CV holds out whole studies and
is the estimate that matches the intended use. The random split is printed only for comparison,
and labelled as optimistic.

Watch the class counts too: a category with a handful of examples produces a per-class F1 that is
noise, and the script warns when the smallest class is under five.

## Trusting a run

`classify_offline.py` flags rather than hides its weak spots. `needs_review` is set when the top
class probability is below `--abstain-below` (0.55 by default), when a series had no metadata,
when the drug lexicon matched more than one mechanism class, or when more than half the
contrast's tokens are absent from the training vocabulary. That last one, `oov_token_share`, is
the honest domain-shift signal: a high median across a new file means the model is being asked
about vocabulary it never saw, and the predictions deserve suspicion regardless of their
probabilities.

Before trusting any run, `python selftest_classifier.py` — exit 0 means this bundle still
reproduces its frozen predictions on this machine.

## Relationship to prior work

This work began as an extension of **scBaseCount** (Youngblut et al. 2026), which curates a
single-cell repository by having an LLM agent system, SRAgent, annotate SRA records with
ontology-grounded fields. It is no longer an extension of it, and the divergence is deliberate
in some places and a consequence of scale in others. Recording which is which:

**What transfers — the validation discipline.** Every call from the LLM labelling stage carries a
`confidence` level drawn from a fixed vocabulary *and* a free-text `justification`, so a reader
can judge whether a confidence score means what they need it to mean rather than taking it on
faith. Agreement is reported stratified by the model's own confidence, which is what licenses a
high-confidence-only sensitivity analysis. `docs/labelling_spec.md` sets out the protocol.

**What deliberately does not transfer — per-record ontology annotation.** SRAgent annotates
*records* with organism, tissue and disease grounded in Uberon/MONDO. A perturbation category is
not a property of a record; it is a property of the **difference between two arms**. Within one
series, `hypoxia 24 h vs hypoxia 0 h` is *environment* while `dosR-deletion hypoxia vs WT hypoxia`
is *genetic*, and every sample in both contrasts carries "hypoxia" in its own metadata.
Per-sample annotation cannot produce the contrast label, so there is no ontology grounding, no
vector search and no agent hierarchy here.

**Where the architecture inverts.** scBaseCount tunes its prompt against a hand-curated gold
standard and deliberately does not train a model, applying the LLM to every record indefinitely.
This pipeline does the opposite: it calls the LLM once and distils the result into a local
scikit-learn model. That is a scale judgement, not a disagreement — ~600 contrasts against their
~40,000 records, and a requirement that re-scoring be offline, free and deterministic. It is also
why the reproducibility apparatus in `docs/REPRODUCIBILITY.md` exists at all: once a model rather
than a prompt is the artefact, the question becomes whether that artefact still behaves as it did
when fitted.

**A known limitation this creates.** `docs/labelling_spec.md` names "keep multiple values when a
record cannot be disambiguated" as a design choice carried over from scBaseCount, via a
`secondary_category` field for contrasts whose arms differ in two categories at once. The shipped
pipeline does not implement it: `train_classifier.py` fits a single `primary_category` and
`classify_offline.py` emits one label per contrast, so a genuinely two-category contrast is forced
to one. The rule-based path (`classify_rules.py`) retains a partial signal — `n_categories_matched`
and `all_categories_matched` flag the ambiguity for review — but nothing downstream preserves a
second label. Restoring it means a multi-label model, not a configuration change.

## Tests

```bash
pip install pytest && pytest tests/ -v
```

Eight end-to-end tests on synthetic data, no network and no API key. They cover the full CLI
cycle and, in particular, assert that editing the feature module is *caught*: the self-test's
Level A keeps passing (the model is intact) while Level B fails (it is being fed different
features), and strict scoring refuses with exit 2.

## Citation

Anbarasu, M. *mtb-contrast-classifier* (2026). https://github.com/manjusri1998/mtb-contrast-classifier
ORCID [0009-0004-5463-4356](https://orcid.org/0009-0004-5463-4356). Machine-readable metadata in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
