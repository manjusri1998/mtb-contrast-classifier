#!/usr/bin/env python3
"""Acceptance test for label_contrasts.py. Makes no network and no API calls.

Run this after editing the vocabulary or prompt:
    python selftest_label_contrasts.py
It stubs the LLM with canned responses, drives the real main() over two cached series, and
checks the deterministic layer and the workbook that comes out the other end.
"""
import json
import os
import sys
import tempfile

import pandas as pd

import label_contrasts as lc

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


# --- 1. parser ------------------------------------------------------------------------------
check(lc.parse_json_list('```json\n[{"comparison_id":"a"}]\n```')[0]["comparison_id"] == "a",
      "parse_json_list strips code fences")
try:
    lc.parse_json_list('{"comparison_id": "a"}')
    check(False, "parse_json_list rejects a bare object")
except ValueError:
    check(True, "parse_json_list rejects a bare object")

# --- 2. house rule --------------------------------------------------------------------------
base = {"primary_category": "drug", "subcategory": "antibiotics", "drug_class": "rna_polymerase_inhibitor",
        "confidence": "high", "arms_differ_only_in_time": False}
check(lc.apply_house_rule(base, "unstated")["primary_category"] == "drug",
      "house rule leaves a non-time contrast untouched")

t = dict(base, arms_differ_only_in_time=True)
r = lc.apply_house_rule(t, "unstated")
check(r["primary_category"] == "environment" and r["subcategory"] == "hypoxia"
      and r["confidence"] == "medium" and r["drug_class"] is None,
      "time-only + unstated aeration -> environment/hypoxia at medium confidence")

r = lc.apply_house_rule(t, "hypoxia_stated")
check(r["primary_category"] == "environment" and r["confidence"] == "high",
      "time-only + stated hypoxia -> high confidence")

r = lc.apply_house_rule(t, "aeration_stated")
check(r["primary_category"] == "drug" and r["confidence"] == "low"
      and r["house_rule"].startswith("blocked"),
      "time-only + stated aeration -> rule blocked, flagged for review, NOT relabelled")

# --- 3. vocabulary validation ---------------------------------------------------------------
check(lc.validate({"primary_category": "drug", "drug_class": "rna_polymerase_inhibitor",
                   "confidence": "high"}) == "",
      "validate accepts an in-vocabulary record")
bad = lc.validate({"primary_category": "antibiotic", "drug_class": "makes_cells_sad",
                   "confidence": "very high"})
check(all(k in bad for k in ("category_off", "drug_class_off", "confidence_off")),
      "validate catches all three off-vocabulary fields")

# --- 4. aeration detection ------------------------------------------------------------------
mk = lambda p: {"samples": {"G": {"growth_protocol_ch1": [p]}}}
check(lc.series_aeration(mk("grown with shaking at 37C")) == "aeration_stated",
      "aeration detected from growth protocol")
check(lc.series_aeration(mk("standing sealed culture, Wayne model")) == "hypoxia_stated",
      "hypoxia detected from growth protocol")
check(lc.series_aeration(mk("7H9 broth with OADC")) == "unstated",
      "silent protocol reports unstated")

# --- 5. end-to-end through main(), LLM stubbed ------------------------------------------------
if not (os.path.exists("comparison_registry.csv") and os.path.isdir("geo_cache")):
    print("SKIP  end-to-end (needs comparison_registry.csv and a populated geo_cache)")
else:
    def fake_llm(prompt, model, max_tokens=6000, retries=3):
        ids = prompt.split("=== CONTRASTS TO LABEL")[1].split("\n", 1)[1].strip().splitlines()
        out = []
        for line in ids:
            cid = line.split("\t")[0]
            out.append({"comparison_id": cid, "primary_category": "drug",
                        "subcategory": "antibiotics", "drug_class": "rna_polymerase_inhibitor",
                        "agent_or_condition": "rifampicin", "dose": "1x", "duration": "24 h",
                        "control_type": "untreated", "arms_differ_only_in_time": False,
                        "arm_evidence": "stub", "confidence": "high",
                        "differs_from_study_label": False, "justification": "stub"})
        return json.dumps(out)

    lc.call_llm = fake_llm
    out = os.path.join(tempfile.mkdtemp(), "selftest.xlsx")
    sys.argv = ["label_contrasts.py", "--only", "GSE101048", "GSE10391", "--out", out]
    lc.main()
    check(os.path.exists(out), "workbook written")
    sheets = pd.read_excel(out, sheet_name=None)
    check(set(sheets) >= {"categories", "full", "vs_study"}, "workbook has the three sheets")
    check(list(sheets["categories"].columns) == ["gse", "comparison_id", "perturbation_category"],
          "'categories' sheet has exactly gse | comparison_id | perturbation_category")
    exp = int((pd.read_csv("comparison_registry.csv").study_id.isin(["GSE101048", "GSE10391"])).sum())
    check(len(sheets["categories"]) == exp,
          "one row per contrast in scope (%d)" % exp)
    check(sheets["full"].schema_problems.fillna("").eq("").all(),
          "no schema problems on a valid stub response")

print("\n%d checks failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
