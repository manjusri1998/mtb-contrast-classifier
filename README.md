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
