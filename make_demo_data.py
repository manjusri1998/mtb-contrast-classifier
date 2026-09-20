#!/usr/bin/env python3
"""Generate a small SYNTHETIC dataset so the pipeline can be run end to end without real data.

    python make_demo_data.py --out demo

This repository ships code only -- no GEO cache, no contrast registry, no labels. That keeps
other people's curated data out of a public repo, but it would otherwise leave nothing here
runnable. So this script fabricates inputs with the same *shape* as the real ones:

    demo/geo_cache/GSE9000xx.pkl      series metadata blobs, the format contrast_features reads
    demo/comparison_registry.csv      one row per contrast
    demo/contrast_categories.xlsx     'full' sheet, the teacher labels train_classifier consumes

WHAT THIS IS NOT
----------------
The text is invented. The GSE accessions are fake and deliberately outside the real GEO range.
Nothing here is Mycobacterium tuberculosis data and no accuracy number obtained from it means
anything biological. Its only job is to prove the plumbing works -- that training, the
provenance stamp, the self-test and offline scoring all run and agree with each other on this
machine. Substitute your own registry, cache and labels for real work.
"""
import argparse
import os
import pickle
import random

import pandas as pd

# Category-distinctive vocabulary, so a text model has something learnable to find. Real
# contrasts are far messier; this is a smoke test, not a benchmark.
SPEC = {
    "genetic": dict(
        series="deletion mutant transcriptional profiling",
        summary="Transcriptional consequences of targeted gene deletion in the reference strain.",
        groups=["genotype: wild-type", "genotype: delta-rvXXXX knockout",
                "genotype: complemented mutant"],
        arms=["delta_mutant_vs_wildtype", "complemented_vs_mutant", "knockout_vs_parental"],
        protocol="cultures grown with shaking to mid-log phase"),
    "drug": dict(
        series="antibiotic exposure time course",
        summary="Expression response to sub-inhibitory antibiotic exposure over several hours.",
        groups=["treatment: isoniazid 0.5 ug/ml", "treatment: rifampicin 1 ug/ml",
                "treatment: untreated control"],
        arms=["isoniazid_vs_untreated", "rifampicin_vs_untreated", "drug_6h_vs_control"],
        protocol="cultures grown with shaking, drug added at mid-log"),
    "environment": dict(
        series="environmental stress response",
        summary="Response to an abrupt environmental shift applied to exponential cultures.",
        groups=["condition: pH 5.0 acid stress", "condition: standing sealed culture",
                "condition: unstressed control"],
        arms=["acid_stress_vs_control", "hypoxia_vs_aerobic", "heat_shock_vs_control"],
        protocol="standing sealed cultures, oxygen depleted gradually"),
    "nutrition": dict(
        series="nutrient limitation",
        summary="Adaptation to withdrawal of a single carbon or nitrogen source.",
        groups=["medium: carbon starved", "medium: nitrogen limited",
                "medium: complete rich medium"],
        arms=["carbon_starvation_vs_rich", "nitrogen_limited_vs_complete",
              "glucose_withdrawal_vs_control"],
        protocol="cultures grown with shaking then washed into limiting medium"),
    "infection": dict(
        series="intracellular transcriptome during macrophage infection",
        summary="Bacterial transcriptome recovered from infected host macrophages.",
        groups=["source: infected macrophage", "source: broth grown inoculum",
                "source: activated macrophage"],
        arms=["macrophage_24h_vs_broth", "intracellular_vs_invitro",
              "activated_macrophage_vs_resting"],
        protocol="bacteria recovered from infected macrophage monolayers"),
}


def make_series(gse, category, tech, rng):
    s = SPEC[category]
    samples = {}
    for i, g in enumerate(s["groups"]):
        for rep in range(1, 3):
            samples["GSM%d%02d%d" % (rng.randint(100, 999), i, rep)] = {
                "title": ["%s replicate %d" % (g.split(": ")[1], rep)],
                "source_name_ch1": [g.split(": ")[1]],
                "characteristics_ch1": [g, "strain: reference", "technology: %s" % tech],
                "growth_protocol_ch1": [s["protocol"]],
                "treatment_protocol_ch1": [s["summary"]],
            }
    head = {"title": ["%s (%s)" % (s["series"], gse)],
            "summary": [s["summary"] + " Synthetic demonstration record, not a real study."],
            "geo_accession": [gse]}
    return {"head": head, "samples": samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--studies-per-category", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cache = os.path.join(args.out, "geo_cache")
    os.makedirs(cache, exist_ok=True)

    rows, labels, n = [], [], 0
    for category, s in SPEC.items():
        for k in range(args.studies_per_category):
            n += 1
            gse = "GSE9%05d" % n                      # fake accession, outside the real range
            tech = "microarray" if n % 2 else "rnaseq"
            with open(os.path.join(cache, gse + ".pkl"), "wb") as fh:
                pickle.dump(make_series(gse, category, tech, rng), fh)
            for arm in s["arms"]:
                rows.append(dict(study_id=gse, comparison_id=arm, technology=tech,
                                 test_condition=arm.split("_vs_")[0].replace("_", " "),
                                 control_condition=arm.split("_vs_")[-1].replace("_", " ")))
                labels.append(dict(gse=gse, comparison_id=arm, primary_category=category,
                                   confidence="high"))

    reg = pd.DataFrame(rows)
    reg.to_csv(os.path.join(args.out, "comparison_registry.csv"), index=False)
    with pd.ExcelWriter(os.path.join(args.out, "contrast_categories.xlsx"),
                        engine="openpyxl") as xl:
        pd.DataFrame(labels).to_excel(xl, sheet_name="full", index=False)

    print("wrote %s/: %d contrasts, %d studies, %d categories"
          % (args.out, len(reg), reg.study_id.nunique(), len(SPEC)))
    print("SYNTHETIC DATA -- fake accessions, invented text. Accuracy on it is meaningless.")


if __name__ == "__main__":
    main()
