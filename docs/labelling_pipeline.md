# Contrast-level perturbation labelling — run it on your terminal

Nothing here has been run against the LLM. The pipeline is built, self-tested with a stubbed
model, and packaged so you run the real job yourself.

## Files

| file | what it is |
|---|---|
| `label_contrasts.py` | the whole pipeline, one file, CLI |
| `selftest_label_contrasts.py` | 16 checks, no network, no API calls — run after any edit |
| `geo_cache.tar.gz` | per-sample GEO metadata for all 86 series, already fetched |

## Setup — run once

```bash
pip install pandas openpyxl anthropic
tar xzf geo_cache.tar.gz
```

`tar xzf` is optional but recommended: it drops in the per-sample GEO metadata for all 86 series
so your run starts from exactly the inputs used here. Without it the script fetches them itself
(~3 minutes, one HTTP request pair per series) and caches them in the same place.

Set the API key. Bash / WSL / macOS:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Put `comparison_registry.csv` and `study_registry.csv` in the working directory, or point at
them with `--comparisons` / `--study`.

---

## Command reference

Every command below has been run as written (the `--dry-run` and inspection ones for real; the
paid ones differ only in that the model is live). Output lines quoted are the actual output.

### 1. Dry run — build all prompts, call nothing, spend nothing

```bash
python label_contrasts.py --dry-run
```

Writes one file per chunk into `prompts/` and a `prompt_manifest.csv` listing chunk, series,
contrast count, aeration evidence, study-level label and token estimate. Prints:

```
registry: 604 contrasts across 86 studies
built 98 prompt chunks (~229226 input tokens)
```

Do this first, every time you change the vocabulary or the prompt.

### 2. Read a prompt before paying for it

```bash
head -80 prompts/000_GSE101048.txt      # the instruction block
sed -n '/=== SAMPLES/,/=== CONTRASTS/p' "$(ls prompts/*GSE1642* | head -1)" | head -40
```

(chunk numbers shift if the registry changes, so glob on the accession rather than hardcoding an
index.)

The second one shows the grouped sample table — the decoding key. If a series' contrast names
can't be matched to anything in that block, the model won't manage it either, and you'll get low
confidence rather than a wrong answer.

### 3. Total the cost before committing

```bash
python -c "import pandas as pd; m=pd.read_csv('prompt_manifest.csv'); print('chunks',len(m),'input_tokens',int(m.est_input_tokens.sum()))"
```

```
chunks 98 input_tokens 229197
```

Output tokens are roughly 80 per contrast, so ~50k for the full 604 plus whatever the model
spends on internal reasoning. Multiply by your per-token rate.

### 4. Dry run on one series

```bash
python label_contrasts.py --dry-run --only GSE1642
```

```
built 6 prompt chunks (~35985 input tokens)
```

`--only` takes any number of accessions:

```bash
python label_contrasts.py --dry-run --only GSE101048 GSE10391 GSE1642
```

```
built 8 prompt chunks (~40334 input tokens)
```

### 5. First real run — one hard series

```bash
python label_contrasts.py --only GSE1642 --out test_GSE1642.xlsx
```

`GSE1642` is the worst case in your registry: the Boshoff compendium, 115 contrasts, 437 samples,
75 agents, contrast names of the form `gpl1343_cccp_vs_paired_reference_channel`. It is tagged
`drug / antibiotics` at study level but actually contains H₂O₂, UV, PZA-at-pH 5.6 and succinate
contrasts. If the labelling holds up here it will hold up anywhere; if it doesn't, fix the
vocabulary before spending on the other 85 studies.

### 6. Full run

```bash
python label_contrasts.py --out contrast_categories.xlsx
```

Progress prints one line per chunk (`[47/98] GSE1642 ok`). On completion:

```
wrote contrast_categories.xlsx
labelled 604/604 contrasts; 0 chunk failures
agreement with study-level label: ...%
confidence: {'high': ..., 'medium': ..., 'low': ...}
drug_class breakdown: ...
```

### 7. Throughput and size controls

```bash
python label_contrasts.py --out contrast_categories.xlsx --workers 8 --chunk 12 --max-tokens 8000
```

| flag | default | when to change it |
|---|---|---|
| `--workers` | 4 | raise to 8 if your rate limit allows; lower to 1–2 if you hit 429s |
| `--chunk` | 20 | **lower it** if responses come back truncated — the symptom is a chunk failing with `no JSON array in response (truncated?)`. `--chunk 10` on GSE1642 gives 12 chunks instead of 6 |
| `--max-tokens` | 6000 | raise with the chunk size; a reasoning model spends most of this budget before it writes any JSON |

### 8. Resume only what failed

Failed chunks land in the `failures` sheet. Extract the accessions and re-run just those:

```bash
python -c "import pandas as pd; sh=pd.read_excel('contrast_categories.xlsx',sheet_name=None); print(' '.join(sorted(sh['failures'].gse.unique())) if 'failures' in sh else '(none)')"
```

```bash
python label_contrasts.py --only GSE1642 GSE21114 --out retry.xlsx
```

Nothing is refetched — the GEO cache makes retries free apart from the API calls themselves.

### 9. Non-default file locations

```bash
python label_contrasts.py --dry-run \
  --comparisons /path/to/comparison_registry.csv \
  --study /path/to/study_registry.csv \
  --cache /path/to/geo_cache
```

### 10. Force a fresh fetch for one series

The cache is one pickle per series; delete the file to refetch it.

```bash
rm geo_cache/GSE10391.pkl
python label_contrasts.py --dry-run --only GSE10391
```

Use this if a series was updated in GEO, or if a fetch failed and cached an error.

### 11. Self-test after any edit

```bash
python selftest_label_contrasts.py
```

16 checks, no network, no API calls: the JSON parser, all three house-rule branches, vocabulary
validation, aeration detection, and an end-to-end pass with a stubbed model that confirms the
workbook comes out with the right sheets and one row per contrast. Exits non-zero on any failure.

### 12. Full help

```bash
python label_contrasts.py --help
```

---

## After the run — what to look at

```bash
# how much is low-confidence, and where
python -c "
import pandas as pd
f=pd.read_excel('contrast_categories.xlsx',sheet_name='full')
print(f.confidence.value_counts())
print(f[f.confidence=='low'][['gse','comparison_id','arm_evidence','justification']].head(20).to_string())"
```

```bash
# drug-class breakdown across the atlas
python -c "
import pandas as pd
f=pd.read_excel('contrast_categories.xlsx',sheet_name='full')
print(f[f.drug_class.notna()].drug_class.value_counts().to_string())"
```

```bash
# where the contrast label departs from the study label -- the actual finding
python -c "
import pandas as pd
f=pd.read_excel('contrast_categories.xlsx',sheet_name='full')
d=f[f.differs_from_study_label]
print(len(d),'contrasts differ from their study label')
print(d[['gse','comparison_id','study_label','primary_category','justification']].to_string())"
```

```bash
# how often the time-only hypoxia rule fired, and how often aeration blocked it
python -c "
import pandas as pd
f=pd.read_excel('contrast_categories.xlsx',sheet_name='full')
print(f.house_rule.value_counts().to_string())"
```

```bash
# anything the model returned off-vocabulary
python -c "
import pandas as pd
f=pd.read_excel('contrast_categories.xlsx',sheet_name='full')
print(f[f.schema_problems.fillna('')!=''][['gse','comparison_id','schema_problems']].to_string())"
```

Check `--model` against your account before any paid run; model ids change and a stale one fails
every call.

## Output workbook

- **`categories`** — exactly what you asked for: `gse | comparison_id | perturbation_category`.
- **`full`** — subcategory, drug_class, agent, dose, duration, control_type,
  `arms_differ_only_in_time`, `house_rule`, confidence, `differs_from_study_label`, the
  study-level label for comparison, `arm_evidence` (which samples the model matched the arm to),
  a ≤25-word justification, and `schema_problems` for anything off-vocabulary.
- **`vs_study`** — crosstab of study-level label against contrast-level label. The off-diagonal
  is the finding: those are contrasts whose study tag was wrong for them.
- **`failures`** — chunks that errored or came back missing a contrast.

## How it works

1. **GEO per-sample metadata** for each series, cached to `geo_cache/` so re-runs are free.
2. **Sample grouping.** Samples collapse into distinct `(source_name, characteristics)` groups.
   This is the decoding key for contrast names: `gpl1343_cccp_vs_paired_reference_channel` is
   only interpretable because a sample group carries `source=CCCP`. Grouping also keeps the
   437-sample series inside a sensible prompt.
3. **One call per ≤20 contrasts**, carrying the series header, the grouped sample table, the
   contrast names (plus the registry's `test_condition`/`control_condition` where present — the
   RNA-seq rows have them, the microarray rows don't), and the study-level label marked
   *reference only*.
4. **Grounding requirement.** Each contrast must report `arm_evidence` naming the sample group
   its treatment arm corresponds to. If the model can't identify the samples it must say so and
   drop to low confidence — this is what stops it from quietly copying the study label.
5. **Deterministic post-pass.** House rule, vocabulary validation and flags are applied in code,
   after the model, so they are auditable and cannot be argued away.

## The house rule and its two guards

Your rule — a contrast whose arms differ only in elapsed time is hypoxia — is applied in code,
not by the model. The model reports the structural fact (`arms_differ_only_in_time`); the script
converts it. Two guards:

- **Aeration blocks it.** A shaken, rolled or explicitly aerobic culture does not go hypoxic; a
  time course there is a growth-phase transition. Those contrasts are flagged and set to low
  confidence rather than relabelled. 13 of the 86 series state aeration, 12 state hypoxia,
  61 say nothing either way.
- **Confidence tracks the evidence.** Protocol states hypoxia → high; silent → medium. A
  convention applied with no protocol statement is a convention, not an observation.

## Drug classes

13 mechanism classes: cell wall synthesis, protein synthesis, RNA polymerase, DNA gyrase, DNA
damaging, energy metabolism, folate pathway, membrane disruptor, redox/oxidative, efflux pump
inhibitor, antimetabolite, other defined mechanism, unknown mechanism.

`drug_class` is assigned whenever a defined chemical or physical agent distinguishes the arms —
**even when the category is not `drug`**. H₂O₂ is `environment / oxidative stress` with
`drug_class = redox_or_oxidative_agent`; UV is `environment / DNA damage` with
`dna_damaging_agent`. That keeps the category column clean while still giving you the mechanism
breakdown across the whole atlas.

## Tuning

Everything under `CONTROLLED VOCABULARY` in `label_contrasts.py` — `CATEGORIES`, `SUBCATS`,
`DRUG_CLASSES`, `PROMPT` — is meant to be edited. After any edit:

```bash
python selftest_label_contrasts.py     # 16 checks, stubbed model, no API calls
python label_contrasts.py --dry-run    # read the prompt back
```

Two vocabulary decisions already baked in from your study registry, worth confirming they're
what you want: metals (copper, iron) and vitamin C are **nutrition**, not environment; and
`pH` is **environment**. Change the `SUBCATS` guidance if either is wrong.

## Calibration — do this before trusting the output

The `vs_study` sheet is not a scoring of the model, because the study labels are not ground truth
at contrast level — disagreement is sometimes the point. To calibrate properly:

1. Take a stratified sample of ~60 contrasts across the five categories and both platforms.
2. Label them by hand, without looking at the script's output.
3. Compare. Report per-category precision/recall and agreement split by the model's own
   confidence level — that last split is what licenses a high-confidence-only sensitivity
   analysis downstream.
4. Systematic disagreements are vocabulary problems, not row problems. Fix the prompt, re-run.

## Known limits

- Series hosted outside GEO are not covered.
- Contrast names that don't encode their arms (`gpl1343_109_vs_paired_reference_channel`) depend
  entirely on the sample table having the matching group. Where it doesn't, expect low confidence
  and an honest `arm_evidence` note — check those rather than assuming.
- The script reads metadata, not papers. Where metadata is genuinely uninformative the correct
  output is low confidence. If low-confidence rows are a large fraction, that's a coverage
  finding about the atlas, not a bug.
