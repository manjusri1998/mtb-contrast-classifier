# Input files you supply

This repository ships code only. Three inputs are yours to provide; none are committed here.
`make_demo_data.py` generates synthetic stand-ins with the same shape if you just want to see
the pipeline run.

## 1. `comparison_registry.csv` — one row per contrast

| column | required | meaning |
|---|---|---|
| `study_id` | yes | GEO series accession, e.g. `GSE101048`. Used as the CV grouping key. |
| `comparison_id` | yes | contrast name within the series, e.g. `T_24_vs_Control`. |
| `technology` | no | `microarray` / `rnaseq`; defaults to `unknown`. A model feature. |
| `test_condition` | no | free text; appended to the contrast's text if present. |
| `control_condition` | no | free text; appended to the contrast's text if present. |

Any other columns are ignored. `study_id` + `comparison_id` must be unique together.

## 2. `geo_cache/` — one pickle per series

`geo_cache/<GSE>.pkl`, each a dict of the shape GEOparse returns:

```python
{"head":    {"title": ["..."], "summary": ["..."]},
 "samples": {"GSM123": {"title": ["..."],
                        "source_name_ch1": ["..."],
                        "characteristics_ch1": ["genotype: wild-type", "..."],
                        "growth_protocol_ch1": ["..."],
                        "treatment_protocol_ch1": ["..."]}}}
```

`label_contrasts.py` builds this cache. `classify_offline.py` will fetch any missing series
itself (a keyless public GEO request) unless you pass `--no-fetch`. A series absent from the
cache is not an error: the contrast is classified from its name alone and flagged in
`needs_review`.

The cache is also the reason the sample-group vocabulary is a feature at all — it is what makes
an opaque contrast name like `gpl1343_cccp_vs_paired_reference_channel` interpretable.

## 3. `contrast_categories.xlsx` — the teacher labels

A sheet named `full` with at least:

| column | meaning |
|---|---|
| `gse` | series accession, matched to `study_id` in the registry |
| `comparison_id` | contrast name |
| `primary_category` | one of `genetic`, `drug`, `environment`, `nutrition`, `infection` |
| `confidence` | optional; `high` / `medium` / `low`, used by `--min-confidence` |

Produced by `label_contrasts.py` (LLM, one pass, needs an API key) or `classify_rules.py`
(deterministic lexicon rules, no API). See `docs/labelling_spec.md` for the category
definitions and the aeration house rule.

## A note on releasing these

The registry may embed absolute local paths in a `source_result` column. Strip it before
sharing: `pandas.read_csv(...).drop(columns=['source_result'])`.
