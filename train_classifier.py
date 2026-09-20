#!/usr/bin/env python3
"""Fit a local perturbation-category classifier on the LLM-labelled contrasts, then save weights.

Stage 2 of two. Stage 1 (label_contrasts.py) uses the Anthropic API once to label the contrasts
in your registry; this script distils those labels into a scikit-learn model you can run
offline forever afterwards. No API calls here.

    python train_classifier.py --labels contrast_categories.xlsx --out contrast_classifier.joblib

What is learned and what is not
-------------------------------
LEARNED : primary_category (genetic / drug / environment / nutrition / infection), as a TF-IDF
          + multinomial logistic regression over the contrast text, the series sample-group
          vocabulary, the aeration signal and the platform. Linear-on-sparse-text is the right
          model class at this sample size, and its coefficients are readable -- --top-features
          prints the terms driving each category so the labels can be defended.
NOT LEARNED : drug_class. That is a lexicon lookup (drug_lexicon.csv) applied at inference.
          Thirteen mechanism classes over a few hundred drug contrasts is too thin to fit, and
          the agent-to-mechanism mapping is a fact you can curate rather than a pattern to infer.

The house rule is already baked into the teacher labels, and the aeration evidence it keys on is
an explicit feature, so the student reproduces the rule rather than needing it re-applied.
"""
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GroupKFold, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run from any cwd

import contrast_features as cfx
import provenance as prov

LABEL_COL = "primary_category"
HERE = os.path.dirname(os.path.abspath(__file__))


def make_pipeline(seed=0):
    text = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                           strip_accents="unicode", lowercase=True)
    cats = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([("text", text, cfx.TEXT_COL),
                             ("cat", cats, cfx.CAT_COLS)])
    clf = LogisticRegression(max_iter=4000, C=4.0, class_weight="balanced",
                             random_state=seed)
    return Pipeline([("features", pre), ("clf", clf)])


def top_features(pipe, n=12):
    pre = pipe.named_steps["features"]
    names = np.concatenate([pre.named_transformers_["text"].get_feature_names_out(),
                            pre.named_transformers_["cat"].get_feature_names_out()])
    clf = pipe.named_steps["clf"]
    out = {}
    for i, cls in enumerate(clf.classes_):
        coef = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        out[cls] = [names[j] for j in np.argsort(coef)[::-1][:n]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="contrast_categories.xlsx",
                    help="teacher output from label_contrasts.py (reads the 'full' sheet)")
    ap.add_argument("--comparisons", default="comparison_registry.csv")
    ap.add_argument("--cache", default="geo_cache")
    ap.add_argument("--out", default="contrast_classifier.joblib")
    ap.add_argument("--report", default="training_report.txt")
    ap.add_argument("--manifest", default="training_manifest.json",
                    help="machine-readable record of this run: inputs, hashes, env, metrics")
    ap.add_argument("--requirements", default="requirements-lock.txt",
                    help="exact package pins written at fit time, to recreate this environment. "
                         "Deliberately not requirements.txt: that file declares what the code "
                         "needs, this one records what one particular model was fitted under.")
    ap.add_argument("--lexicon", default="drug_lexicon.csv",
                    help="not used for fitting; hashed so inference can detect lexicon drift")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="held-out fraction for the honest accuracy estimate")
    ap.add_argument("--min-confidence", default=None, choices=[None, "medium", "high"],
                    help="train only on teacher rows at this confidence or better")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-features", type=int, default=12)
    args = ap.parse_args()

    lab = pd.read_excel(args.labels, sheet_name="full")
    lab = lab.rename(columns={"gse": "study_id"})
    if args.min_confidence:
        keep = {"high"} if args.min_confidence == "high" else {"high", "medium"}
        before = len(lab)
        lab = lab[lab.confidence.isin(keep)]
        print("confidence filter: %d -> %d rows" % (before, len(lab)), file=sys.stderr)

    reg = pd.read_csv(args.comparisons)
    merged = lab.merge(reg, on=["study_id", "comparison_id"], how="left", suffixes=("", "_reg"))
    if "technology" not in merged or merged.technology.isna().all():
        merged["technology"] = "unknown"
    missing = merged[LABEL_COL].isna().sum()
    if missing:
        print("dropping %d rows with no label" % missing, file=sys.stderr)
        merged = merged[merged[LABEL_COL].notna()]

    X = cfx.build_features(merged, args.cache)
    y = merged[LABEL_COL].to_numpy()
    print("training set: %d contrasts, %d classes %s"
          % (len(X), len(set(y)), sorted(set(y))), file=sys.stderr)
    counts = pd.Series(y).value_counts()
    print(counts.to_string(), file=sys.stderr)
    if counts.min() < 5:
        print("WARNING: smallest class has %d examples -- the held-out estimate for it will be "
              "noise. Label more contrasts before trusting per-class numbers."
              % counts.min(), file=sys.stderr)

    groups = X.study_id.to_numpy()

    # PRIMARY estimate: grouped by study. Contrasts from one series share almost all of their
    # text, so a random split leaks the series into both sides and reports a number you will
    # not see on new studies. Grouped CV is the estimate that matches the intended use.
    n_groups = len(set(groups))
    gfolds = min(5, n_groups)
    grouped_txt, grouped_conf = "skipped (need >=2 studies)", ""
    grouped_metrics = None
    if gfolds >= 2:
        g_pred = cross_val_predict(make_pipeline(args.seed), X, y,
                                   cv=GroupKFold(n_splits=gfolds), groups=groups)
        grouped_txt = classification_report(y, g_pred, zero_division=0)
        grouped_metrics = classification_report(y, g_pred, zero_division=0, output_dict=True)
        grouped_conf = pd.DataFrame(confusion_matrix(y, g_pred, labels=sorted(set(y))),
                                    index=sorted(set(y)), columns=sorted(set(y))).to_string()

    # SECONDARY, optimistic: random stratified split, reported only for comparison.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          stratify=y, random_state=args.seed)
    held = classification_report(yte, make_pipeline(args.seed).fit(Xtr, ytr).predict(Xte),
                                 zero_division=0)

    folds = min(5, int(pd.Series(y).value_counts().min()))
    cv_txt = "skipped (a class has fewer than 2 examples)"
    if folds >= 2:
        cv_pred = cross_val_predict(make_pipeline(args.seed), X, y,
                                    cv=StratifiedKFold(folds, shuffle=True,
                                                       random_state=args.seed))
        cv_txt = classification_report(y, cv_pred, zero_division=0)

    # final model on everything
    final = make_pipeline(args.seed).fit(X, y)
    feats = top_features(final, args.top_features)

    # Everything needed to establish, on another machine or in a year, whether this bundle is
    # still behaving as it did here: library versions, source hashes, input hashes, the argv.
    stamp = prov.stamp(
        data_files={"labels": args.labels, "comparisons": args.comparisons,
                    "lexicon": cfx.resolve_path(args.lexicon)},
        extra={"seed": args.seed, "label_column": LABEL_COL,
               "n_train": int(len(X)), "n_studies": int(n_groups),
               "class_counts": {k: int(v) for k, v in counts.items()},
               "min_confidence": args.min_confidence,
               "grouped_cv_folds": int(gfolds),
               "grouped_cv_macro_f1": (round(grouped_metrics["macro avg"]["f1-score"], 4)
                                       if grouped_metrics else None),
               "grouped_cv_accuracy": (round(grouped_metrics["accuracy"], 4)
                                       if grouped_metrics else None)},
        module_dir=HERE)

    bundle = {"pipeline": final, "classes": list(final.named_steps["clf"].classes_),
              "trained_on": len(X), "label_source": args.labels,
              "min_confidence": args.min_confidence,
              "feature_columns": [cfx.TEXT_COL] + cfx.CAT_COLS,
              "provenance": stamp}
    joblib.dump(bundle, args.out)

    vers = prov.write_requirements(args.requirements)
    prov.write_manifest(args.manifest, dict(
        stamp, model_file=args.out, model_sha256=prov.file_sha256(args.out),
        report_file=args.report, requirements_file=args.requirements,
        grouped_cv_report=grouped_metrics, top_features=feats))

    with open(args.report, "w") as fh:
        fh.write("trained on %d contrasts from %s\n\n" % (len(X), args.labels))
        fh.write("class counts:\n%s\n\n" % counts.to_string())
        fh.write("PRIMARY -- GROUPED BY STUDY (%s folds), the estimate that matches applying\n"
                 "this model to new studies:\n%s\n\n%s\n\n" % (gfolds, grouped_txt, grouped_conf))
        fh.write("SECONDARY, OPTIMISTIC -- random split leaks series text across the split and\n"
                 "overstates performance on unseen studies. For comparison only.\n")
        fh.write("random held-out (%.0f%%):\n%s\n\nrandom %s-fold CV:\n%s\n\n"
                 % (100 * args.test_size, held, folds, cv_txt))
        fh.write("TOP FEATURES PER CLASS:\n%s\n\n" % json.dumps(feats, indent=1))
        fh.write("REPRODUCIBILITY STAMP (also in %s, and embedded in the model bundle):\n%s\n"
                 % (args.manifest, json.dumps(
                     {k: stamp[k] for k in ("created_utc", "argv", "python", "platform",
                                            "packages", "code_sha256", "data_sha256",
                                            "git_commit", "seed")}, indent=1)))

    print("\nPRIMARY (grouped by study, %s folds):\n%s" % (gfolds, grouped_txt), file=sys.stderr)
    print("optimistic random split, for comparison only:\n%s" % held, file=sys.stderr)
    print("wrote %s, %s, %s, %s" % (args.out, args.report, args.manifest, args.requirements),
          file=sys.stderr)
    print("pinned: %s" % ", ".join("%s==%s" % (k, v) for k, v in vers.items() if v),
          file=sys.stderr)
    print("keep the model, the manifest and the requirements file together -- the bundle is a\n"
          "pickle and is only meaningful in the environment those pins describe.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
