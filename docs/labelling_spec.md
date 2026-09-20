# Contrast-level perturbation labelling — specification

**Problem.** Atlas analysis runs at contrast level; perturbation categories were assigned at study
level. Any series containing more than one kind of contrast is therefore mislabelled for some of
its contrasts, and every category-level result inherits that error.

**Why SRAgent doesn't solve it.** SRAgent annotates *records* (SRX accessions) with organism,
tissue, disease and library fields grounded in Uberon/MONDO. A perturbation category is not a
property of a record — it is a property of the **difference between two arms**. Within one series,
`hypoxia 24 h vs hypoxia 0 h` is *environment* and `ΔdosR hypoxia vs WT hypoxia` is *genetic*, and
every sample in both contrasts carries "hypoxia" in its own metadata. Per-sample annotation cannot
produce the contrast label. What transfers from the paper is its validation discipline, not its code.

---

## 1. Input contract

The comparison registry is one row per contrast:

| column | example |
|---|---|
| `gse` | `GSE101048` |
| `contrast_id` | `T_15_vs_Control` |
| `platform` | `microarray` |

Arm membership is **not** recorded in the registry, so reconstructing it is step 2 of the pipeline
and carries its own review gate. Strain, agent, dose, duration and control type are all recovered
from GEO.

## 2. Pipeline

1. **Fetch per-sample metadata**, one request per series (~100 requests, cached to disk).
   `fetch_series(gse)` returns the series header plus, for every GSM: `title`,
   `source_name_ch1`, `characteristics_ch1`, `growth_protocol_ch1`, `treatment_protocol_ch1`,
   `description`, `platform_id`.
2. **Reconstruct the arms** from the contrast name. `resolve_arms()` splits on `_vs_`, tokenises
   both sides, and matches tokens against each sample's `source_name_ch1`, `title`, `description`
   and characteristic values at **token boundaries** — never substring, because `T_1` substring-matches
   `T 15` and would silently merge two timepoints. The control side additionally matches a fixed
   alias list (`control`, `untreated`, `con`, `mock`, `vehicle`, `wt`, `t0`, …). Returns a status of
   `resolved` / `ambiguous` / `unresolved`; only `resolved` contrasts proceed automatically, and the
   other two are the manual review queue.
3. **Coverage audit — gates everything downstream.** Report the fraction of contrasts that resolve,
   and the fraction whose samples carry structured `characteristics_ch1`, broken down by platform
   and submission year. Older microarray series frequently carry nothing but a sample title; those
   contrasts are label-by-title and must be reported as a separate stratum, not pooled.
4. **Build the contrast record.** Collect the characteristic keys for both arms and split them
   mechanically into `shared_between_arms` (identical across arms) and `DIFFERING_between_arms`
   (the contrast itself). Deterministic — no model involved — and it is what forces the classifier
   to label the difference rather than the study.
5. **Run the deterministic confound flags** (`confound_flags()`, §5) before the model sees anything.
6. **Classify.** One call per contrast against a fixed controlled vocabulary, returning the schema
   in §3. The series summary is supplied but explicitly demoted in the prompt to disambiguating
   abbreviations and compound names only; it may never be the primary basis for a call.
7. **Validate against the existing hand labels** (§4).

## 3. Output schema

| field | notes |
|---|---|
| `primary_category` | one of `genetic`, `drug`, `environment`, `nutrition`, `infection` |
| `secondary_category` | non-null when the arms differ in two categories at once; **not** forced to null |
| `perturbation_detail` | free text |
| `agent_or_condition`, `dose`, `duration`, `strain_or_genotype` | extracted covariates, usable as strata |
| `control_type` | `vehicle`, `untreated`, `wild_type`, `timepoint_zero`, `uninfected`, `in_vitro_reference`, `other` |
| `confidence` | `high` / `medium` / `low`, definitions fixed in the prompt |
| `justification` | ≤40 words, free text — the audit trail |
| `evidence_fields` | which differing fields drove the call |

Two design choices carried over from scBaseCount: **keep multiple values when a record cannot be
disambiguated** (hence `secondary_category`), and **emit a free-text justification with every
confidence score** so a reader can judge whether the score means what they need it to mean.

## 4. Validation protocol

The existing ~600 hand tags are the gold standard — a stronger starting position than the paper's,
which required hand-curating 150 records to calibrate its confidence classifier.

1. Stratify by category and platform; hold out a random 20% as a **tuning set**. Iterate the prompt
   only against this set.
2. Freeze the prompt. Run on the remaining 80% and report, on that untouched set:
   per-category precision / recall / F1; the 5×5 confusion matrix plus a `multi-category` row;
   overall agreement and Cohen's κ; agreement **stratified by the model's own confidence level**
   (this is what licenses a high-confidence-only sensitivity analysis) and **by platform**.
3. Every disagreement is output as a ranked review list with both labels, the differing fields and
   the justification. Disagreements are not automatically the model's error — the point of the
   exercise is that some of them are contrasts whose study-level tag was wrong.

## 4b. House rule — time-only contrasts are hypoxia

Atlas convention, supplied by the PI: a contrast whose arms differ **only in elapsed time**, with
no drug, genotype, medium or host difference, is hypoxia (`environment`). The reasoning is the
standard Mtb one — an undisturbed culture followed over time depletes its own oxygen, which is the
Wayne model.

Encoded in `apply_house_rule()`, applied **deterministically after** the model call so it overrides
model judgement and appears in an auditable `house_rule` column. Three precision points:

- **"Time-only" is tested on the differing-field set, not on the absence of a drug.** A ΔdosR-vs-WT
  contrast at 24 h has no drug and no infection either, but it differs in `genotype`, so the rule
  must not fire. `time_only_contrast()` requires every differing field to be a time field, after
  excluding identifier fields (`title`, `source_name_ch1`, `description`) and technical fields
  (`batch`, `replicate`, `plate`) — the latter are still reported by `confound_flags()`.
- **Aeration blocks the rule.** A shaken or rolled culture does not go hypoxic; a time course there
  is a growth-phase transition. `aeration_evidence()` scans the growth and treatment protocols for
  aeration terms (shaking, roller, aerobic, stirred, sparged) versus hypoxia terms (standing,
  static, sealed, Wayne, hypoxic, anaerobic, oxygen depletion). On `aeration_stated` the rule is
  blocked, confidence set to `low`, and the contrast is flagged for review rather than silently
  relabelled. This matters: of the three series examined here, GSE165673 states "with shaking" and
  GSE101048 "grown aerobically".
- **Confidence tracks the evidence.** `hypoxia_stated` → high; `unstated` → medium. A rule applied
  in the absence of any protocol statement is a convention, not an observation, and the confidence
  column should say so.

## 5. Deterministic confound flags

Computed from the arm comparison, independent of the model, and reported per contrast:

- `asymmetric_field:<k>` — a characteristic present in one arm and entirely absent in the other.
- `arms_not_batch_matched:<k>` — the arms draw on different `batch` values.
- `arms_differ_in_time:<k>` — both arms have a time field and they differ.
- `multiple_substantive_fields_differ:<list>` — more than one non-identifier field distinguishes
  the arms, i.e. the contrast is not clean.

These are cheap and they catch things the model talks itself out of (§6).

## 6. What the output buys beyond the labels

- **A confound audit for free**, from the flags above — invisible at study level, and directly
  relevant to the amplitude question.
- **Control-type strata.** `vehicle` vs `untreated` vs `timepoint_zero` are different baselines;
  pooling them is a real source of heterogeneity inside the "drug" category.
- **A high-confidence subset** for re-running headline results as a sensitivity analysis.

## 7. Worked example — the three GSE101048 rows

Registry input: `GSE101048 / T_15_vs_Control / microarray` and the same for `T_1` and `T_24`.
Output in `contrast_labels_GSE101048.csv`.

All three resolved cleanly at 3 treatment vs 3 control samples. The series is a vitamin C exposure
time course; `source_name_ch1` values are `T 15 I…III`, `T 1 I…III`, `T 24 I…IV` and `T con I…III`,
so the registry names map onto GEO fields exactly.

Three things this surfaced that the registry row cannot:

1. **`T_15` is 15 minutes, not 15 hours.** The characteristics give `time point: 0.25 h`, while
   `T_1` and `T_24` are 1 h and 24 h. The series mixes units in its sample names; any duration
   parsed from the contrast string alone would be wrong for the two sub-hour contrasts.
2. **Every contrast in this series shares one 0 h untreated control.** The control samples carry no
   `time point` characteristic at all, and the summary states profiles were compared against
   untreated culture at 0 h. So `T_24_vs_Control` confounds 24 h of vitamin C with 24 h of culture
   ageing. The flag `asymmetric_field:time point (absent in control arm)` catches this on all three
   rows; the model's own justification for `T_24` claimed a "same timepoint background", which is
   wrong — which is precisely why the flags are computed deterministically rather than asked for.
3. **A category boundary worth deciding explicitly.** The classifier called all three `drug`,
   reasoning that vitamin C is a xenobiotic compound. Vitamin C at 10 mM in Mtb is usually
   discussed as a redox/Fenton stressor, which argues for `environment`. If the atlas has this
   series tagged `environment`, the calibration run will show all three contrasts disagreeing —
   and the correct fix is a vocabulary rule for redox-active chemicals, not three per-row edits.
   This is the single best argument for running §4 before trusting any output.

Four further contrasts, from two other series, are in `contrast_labels_demo.csv`:

| id | series | arms | call | conf. | flags |
|---|---|---|---|---|---|
| C1 | GSE292332 | BDQ+DMSO 2 µM vs DMSO vehicle | drug, control=`vehicle` | high | none |
| C2 | GSE292332 | AMK 4×MIC vs generic Untreated | drug, control=`untreated` | high | not batch-matched |
| C3 | GSE292332 | Untreated 16 h vs Untreated 0 h | environment | **low** | time differs, not batch-matched |
| C4 | GSE165673 | INH both doses vs no drug, 24 h | drug | high | none |

C1 and C2 are the same category against different baselines, and the schema records which. C3 is
the behaviour that matters most: a time-only contrast with no stated perturbation returns **low**
confidence with a justification saying so, rather than inheriting the series' "drug" label.

## 7. Known limits

- GEO `characteristics_ch1` completeness varies by submission era; step 2 exists to measure this
  before anything is interpreted. Contrasts resolvable only from sample titles are a separate stratum.
- Series hosted outside GEO are not covered by this fetcher.
- The classifier reads metadata, not the paper. Where the metadata is genuinely uninformative the
  correct output is `low` confidence, not a guess — check that low-confidence contrasts are a
  plausible fraction rather than a dumping ground.
- Budget ~3,000 output tokens per call; the reasoning model spends most of it on deliberation, and
  truncated responses come back as unparseable JSON rather than as an error.
