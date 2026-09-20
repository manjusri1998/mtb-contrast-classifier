#!/usr/bin/env python3
"""Head-to-head: distilled model vs deterministic rules, on studies neither has seen.

    python compare_classifiers.py --comparisons comparison_registry.csv \\
        --gold gold_corrected.csv --cache geo_cache --out comparison

Answers one question: **if I train on the contrasts I have labelled and then apply this to
contrasts from studies that did not exist at training time, how often is it right, and is it
better than the lexicon rules?**

Why grouped by study
--------------------
Contrasts within a GEO series share nearly all their text -- same title, same summary, same
sample vocabulary. A random row split puts the same series on both sides, so the model
recognises the series rather than the biology and reports an accuracy that will not survive
contact with a new study. Every fold here holds out whole studies, which is exactly the
deployment situation: new series arrive, none of their text was in training.

The rules need no training, so they are run once over everything and then scored on each fold's
held-out studies. That is not leakage -- nothing about the held-out studies influenced them.

Selective prediction
--------------------
"Unsupervised" need not mean "accept everything". The model emits a probability, so the real
operating question is: *what fraction can be auto-accepted, at what accuracy, and how many are
left to review?* The coverage/accuracy table is the deliverable -- it converts "can I trust it"
into a threshold you can defend.

Caveat on what this measures
----------------------------
Gold labels are used as the training target here, not the LLM teacher labels used in production.
That makes this the model's **ceiling**: in production it learns from teacher labels that carry
their own error, so the deployed figure will be at or below what this reports.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contrast_features as cfx
from train_classifier import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["study_id", "comparison_id"]


def load_gold(path):
    """gse|study_id + comparison_id + perturbation_category, from csv or xlsx."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(path)
        sheet = "full" if "full" in xl.sheet_names else xl.sheet_names[0]
        g = xl.parse(sheet)
    else:
        g = pd.read_csv(path)
    g = g.rename(columns={"gse": "study_id", "primary_category": "perturbation_category"})
    miss = {"study_id", "comparison_id", "perturbation_category"} - set(g.columns)
    if miss:
        sys.exit("gold file is missing column(s): %s" % sorted(miss))
    # the reviewed workbook contains stray capitalised labels; normalise both sides
    g["perturbation_category"] = g.perturbation_category.astype(str).str.strip().str.lower()
    return g[["study_id", "comparison_id", "perturbation_category"]].drop_duplicates(KEY)


def run_rules(comparisons, cache, workdir, extra_args=()):
    """Run classify_rules.py once over the whole registry; return its per-contrast calls."""
    out = os.path.join(workdir, "_rules_predictions.xlsx")
    cmd = [sys.executable, os.path.join(HERE, "classify_rules.py"),
           "--comparisons", comparisons, "--cache", cache, "--out", out, *extra_args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("classify_rules.py failed:\n%s" % p.stderr[-2000:])
    r = pd.ExcelFile(out).parse("full").rename(columns={"gse": "study_id"})
    r["rules_category"] = r.perturbation_category.astype(str).str.strip().str.lower()
    keep = ["study_id", "comparison_id", "rules_category"]
    if "needs_review" in r.columns:
        keep.append("needs_review")
    return r[keep].drop_duplicates(KEY)


def grouped_model_predictions(X, y, groups, seed=0, n_splits=5):
    """Out-of-fold predictions where every fold holds out whole studies."""
    n = min(n_splits, len(set(groups)))
    if n < 2:
        sys.exit("need at least 2 studies to hold one out")
    pred = np.empty(len(y), dtype=object)
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        pipe = make_pipeline(seed).fit(X.iloc[tr], y[tr])
        pr = pipe.predict_proba(X.iloc[te])
        cls = pipe.named_steps["clf"].classes_
        top = pr.argmax(1)
        pred[te] = cls[top]
        prob[te] = pr[np.arange(len(te)), top]
    return pred, prob, n


def selective_curve(truth, pred, prob, thresholds):
    """Accuracy among contrasts the model is confident about, and how many are left over."""
    rows = []
    for t in thresholds:
        take = prob >= t
        rows.append(dict(
            threshold=round(float(t), 2),
            auto_accepted=int(take.sum()),
            coverage=round(float(take.mean()), 4),
            accuracy_on_accepted=(round(float((pred[take] == truth[take]).mean()), 4)
                                  if take.any() else np.nan),
            left_to_review=int((~take).sum()),
            accuracy_on_reviewed=(round(float((pred[~take] == truth[~take]).mean()), 4)
                                  if (~take).any() else np.nan)))
    return pd.DataFrame(rows)


def mcnemar_exact(truth, a, b):
    """Exact McNemar on the discordant pairs -- the paired test for two classifiers, same data."""
    ca, cb = (a == truth), (b == truth)
    n01 = int((~ca & cb).sum())     # only b right
    n10 = int((ca & ~cb).sum())     # only a right
    if n01 + n10 == 0:
        return dict(only_model_right=0, only_rules_right=0, p_value=1.0)
    p = stats.binomtest(n10, n01 + n10, 0.5).pvalue
    return dict(only_model_right=n10, only_rules_right=n01, p_value=float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparisons", required=True)
    ap.add_argument("--gold", required=True,
                    help="reviewed labels: study_id/gse, comparison_id, perturbation_category")
    ap.add_argument("--cache", default="geo_cache")
    ap.add_argument("--out", default="comparison", help="output prefix")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules-args", default="",
                    help="extra flags passed through to classify_rules.py, space separated")
    args = ap.parse_args()

    reg = pd.read_csv(args.comparisons)
    if "technology" not in reg.columns:
        reg["technology"] = "unknown"
    gold = load_gold(args.gold)
    rules = run_rules(args.comparisons, args.cache, os.path.dirname(os.path.abspath(args.out))
                      or ".", tuple(args.rules_args.split()) if args.rules_args else ())

    X = cfx.build_features(reg, args.cache)
    df = X.merge(gold, on=KEY, how="inner").merge(rules, on=KEY, how="left")
    if df.empty:
        sys.exit("no contrasts matched between the registry and the gold file")
    unscored = int(df.rules_category.isna().sum())
    df = df[df.rules_category.notna()].reset_index(drop=True)

    truth = df.perturbation_category.to_numpy()
    feats = df[[c for c in X.columns]]
    pred, prob, folds = grouped_model_predictions(feats, truth, df.study_id.to_numpy(),
                                                  args.seed, args.folds)
    rpred = df.rules_category.to_numpy()

    acc = lambda p: float((p == truth).mean())
    head = pd.DataFrame([
        dict(classifier="distilled model (grouped CV)", accuracy=round(acc(pred), 4),
             n=len(truth), note="%d folds, whole studies held out" % folds),
        dict(classifier="lexicon rules", accuracy=round(acc(rpred), 4), n=len(truth),
             note="no training; unaffected by the split"),
    ])
    mc = mcnemar_exact(truth, pred, rpred)
    curve = selective_curve(truth, pred, prob, np.arange(0.0, 0.96, 0.05))

    per_cat = pd.DataFrame(classification_report(
        truth, pred, zero_division=0, output_dict=True)).T.add_prefix("model_").join(
        pd.DataFrame(classification_report(
            truth, rpred, zero_division=0, output_dict=True)).T.add_prefix("rules_"))

    per_contrast = df[KEY].assign(truth=truth, model=pred, model_probability=prob.round(4),
                                  rules=rpred, model_correct=(pred == truth),
                                  rules_correct=(rpred == truth))

    with pd.ExcelWriter(args.out + ".xlsx", engine="openpyxl") as xl:
        head.to_excel(xl, sheet_name="headline", index=False)
        # own sheet: stacking a second frame under the first with startrow writes its header
        # into the first frame's columns and makes the sheet unreadable by pd.read_excel
        pd.DataFrame([mc]).to_excel(xl, sheet_name="mcnemar", index=False)
        curve.to_excel(xl, sheet_name="selective_prediction", index=False)
        per_cat.to_excel(xl, sheet_name="per_category")
        per_contrast.to_excel(xl, sheet_name="per_contrast", index=False)
    curve.to_csv(args.out + "_selective_prediction.csv", index=False)

    print("\n%s" % head.to_string(index=False))
    if unscored:
        print("\n%d contrast(s) had no rules call and were dropped" % unscored)
    print("\nexact McNemar on discordant pairs: model-only-right=%d  rules-only-right=%d  p=%.3g"
          % (mc["only_model_right"], mc["only_rules_right"], mc["p_value"]))
    print("\nselective prediction (model):\n%s" % curve.to_string(index=False))
    print("\nwrote %s.xlsx and %s_selective_prediction.csv" % (args.out, args.out))


if __name__ == "__main__":
    main()
