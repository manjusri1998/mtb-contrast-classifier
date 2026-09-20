#!/usr/bin/env python3
"""Acceptance test: does this model, on this machine, still predict what it predicted at fit time?

    python selftest_classifier.py --write-fixture      # once, right after training
    python selftest_classifier.py                      # thereafter, before trusting any run

Why a frozen fixture and not just a version check
-------------------------------------------------
provenance.py catches an environment that has *changed*. It cannot catch an environment that
differs in a way nobody thought to record, and it cannot catch the case where the versions match
but the behaviour does not. This does: it replays a small, frozen set of contrasts through the
bundle and compares the output to what was recorded when the fixture was written.

Two independent levels, because there are two independent things that can drift:

  LEVEL A -- prediction reproducibility.  Frozen FEATURE ROWS in, predictions out, compared to
             the frozen predictions. Hermetic: needs only the bundle, no GEO cache and no
             network. Catches scikit-learn/numpy pickle drift -- the failure that changes
             answers without raising.

  LEVEL B -- feature reproducibility.  Rebuilds the feature rows from the frozen registry slice
             and the GEO cache, and compares them to the frozen rows cell by cell. Catches an
             edit to contrast_features.py, a changed cache, or a GEO record that was revised
             upstream. Skipped (not failed) when the cache is unavailable.

Level A failing means the model is no longer the model. Level B failing while A passes means the
model is intact but is being fed different features than it was trained on -- which will show up
as silently worse accuracy on real data, never as an error.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contrast_features as cfx
import provenance as prov

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = "fixture_features.csv"
EXPECTED = "fixture_expected.csv"
REGISTRY = "fixture_registry.csv"
META = "fixture_meta.json"

KEY = ["study_id", "comparison_id"]
# every column build_features emits that feeds the model or explains a difference
# Every column build_features emits that feeds the model or explains a difference. Kept in
# sync with contrast_features deliberately: a new feature column that is not listed here is a
# column Level B would not notice changing.
COMPARE_COLS = [cfx.TEXT_COL] + cfx.CAT_COLS + ["name_text", "cond_text", "vocab_text",
                                                "vocab_varying_text", "summary_text",
                                                "diff_left_text", "diff_right_text", "diff_text",
                                                "arm_meta_diff_text", "arm_meta_resolved",
                                                "n_samples", "has_series_metadata"]


def write_fixture(args):
    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    reg = pd.read_csv(args.comparisons)
    for c in KEY:
        if c not in reg.columns:
            sys.exit("registry is missing required column '%s'" % c)
    if "technology" not in reg.columns:
        reg["technology"] = "unknown"

    # Stratify the sample over studies, not rows: contrasts within a series share nearly all
    # their text, so 20 rows from three series would test almost nothing.
    rng = np.random.RandomState(args.seed)
    studies = sorted(reg.study_id.unique())
    take = min(args.n, len(reg))
    picked = (reg.groupby("study_id", group_keys=False, sort=True)
                 .sample(1, random_state=args.seed)
                 .reset_index(drop=True))
    if len(picked) > take:
        picked = picked.sample(take, random_state=args.seed)
    elif len(picked) < take:
        rest = reg.merge(picked[KEY], on=KEY, how="left", indicator=True)
        rest = rest[rest._merge == "left_only"].drop(columns="_merge")
        picked = pd.concat([picked, rest.sample(take - len(picked), random_state=args.seed)])
    picked = picked.sort_values(KEY).reset_index(drop=True)

    X = cfx.build_features(picked, args.cache)
    proba = pipe.predict_proba(X)
    classes = np.array(bundle["classes"])
    top = proba.argmax(1)
    exp = pd.DataFrame({"study_id": X.study_id, "comparison_id": X.comparison_id,
                        "expected_category": classes[top],
                        "expected_probability": proba[np.arange(len(X)), top]})

    os.makedirs(args.fixture, exist_ok=True)
    picked.to_csv(os.path.join(args.fixture, REGISTRY), index=False)
    X.to_csv(os.path.join(args.fixture, FEATURES), index=False)
    exp.to_csv(os.path.join(args.fixture, EXPECTED), index=False)
    with open(os.path.join(args.fixture, META), "w") as fh:
        json.dump({"written_utc": prov.stamp(module_dir=HERE)["created_utc"],
                   "model_file": os.path.abspath(args.model),
                   "model_sha256": prov.file_sha256(args.model),
                   "model_provenance": bundle.get("provenance"),
                   "n_contrasts": int(len(X)),
                   "n_studies": int(X.study_id.nunique()),
                   "total_studies_available": len(studies),
                   "sample_seed": args.seed,
                   "cache_dir": os.path.abspath(args.cache)}, fh, indent=2, default=str)
    print("fixture written to %s/: %d contrasts from %d studies (seed %d)"
          % (args.fixture, len(X), X.study_id.nunique(), args.seed), file=sys.stderr)
    print("commit this directory alongside the model. Regenerate it only when you retrain "
          "deliberately -- never to make a failing test pass.", file=sys.stderr)
    return 0


def run_test(args):
    for f in (FEATURES, EXPECTED):
        p = os.path.join(args.fixture, f)
        if not os.path.exists(p):
            sys.exit("no fixture at %s -- run with --write-fixture first" % p)

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    prov.report(bundle, module_dir=HERE, strict=False)

    meta = {}
    if os.path.exists(os.path.join(args.fixture, META)):
        meta = json.load(open(os.path.join(args.fixture, META)))
    now_sha = prov.file_sha256(args.model)
    if meta.get("model_sha256") and meta["model_sha256"] != now_sha:
        print("\nNOTE: the model file has changed since the fixture was written "
              "(%s -> %s). A Level A failure below is then expected, and means the fixture "
              "must be regenerated deliberately -- not that the test is broken."
              % (meta["model_sha256"], now_sha), file=sys.stderr)

    frozen = pd.read_csv(os.path.join(args.fixture, FEATURES), keep_default_na=False)
    exp = pd.read_csv(os.path.join(args.fixture, EXPECTED))
    failures = []

    # ---- LEVEL A -----------------------------------------------------------------------
    proba = pipe.predict_proba(frozen)
    classes = np.array(bundle["classes"])
    top = proba.argmax(1)
    got_cat, got_p = classes[top], proba[np.arange(len(frozen)), top]
    chk = exp.assign(got_category=got_cat, got_probability=got_p)
    bad_cat = chk[chk.expected_category != chk.got_category]
    dp = (chk.expected_probability - chk.got_probability).abs()
    bad_p = chk[dp > args.tol]

    print("\nLEVEL A  prediction reproducibility (%d frozen contrasts)" % len(chk))
    print("  class agreement       %d/%d" % (len(chk) - len(bad_cat), len(chk)))
    print("  max |prob difference| %.3e   (tolerance %.0e)" % (dp.max(), args.tol))
    if len(bad_cat):
        failures.append("LEVEL A: %d/%d contrasts changed predicted class" % (len(bad_cat), len(chk)))
        print(bad_cat[KEY + ["expected_category", "got_category",
                             "expected_probability", "got_probability"]].to_string(index=False))
    elif len(bad_p):
        failures.append("LEVEL A: classes all agree but %d probabilities moved more than %g"
                        % (len(bad_p), args.tol))
        print(bad_p[KEY + ["expected_probability", "got_probability"]].to_string(index=False))
    else:
        print("  PASS")

    # ---- LEVEL B -----------------------------------------------------------------------
    reg_path = os.path.join(args.fixture, REGISTRY)
    have_cache = os.path.isdir(args.cache) and os.path.exists(reg_path)
    print("\nLEVEL B  feature reproducibility")
    if not have_cache:
        print("  SKIPPED (need %s and a GEO cache at %s). Level A alone does not prove the "
              "feature builder is unchanged." % (REGISTRY, args.cache))
    else:
        reg = pd.read_csv(reg_path)
        rebuilt = cfx.build_features(reg, args.cache)
        a = frozen.set_index(KEY).sort_index()
        b = rebuilt.astype(object).where(rebuilt.notna(), "").set_index(KEY).sort_index()
        b = b.reindex(a.index)
        diffs = []
        for c in COMPARE_COLS:
            if c not in a.columns or c not in b.columns:
                diffs.append((c, "column missing", len(a)))
                continue
            ne = a[c].astype(str) != b[c].astype(str)
            if ne.any():
                diffs.append((c, "; ".join("%s/%s" % k for k in a.index[ne][:3]), int(ne.sum())))
        print("  %d feature columns compared over %d contrasts" % (len(COMPARE_COLS), len(a)))
        if diffs:
            failures.append("LEVEL B: %d feature column(s) differ from the frozen fixture"
                            % len(diffs))
            for c, where, n in diffs:
                print("  DIFFERS  %-22s %4d row(s)  e.g. %s" % (c, n, where))
        else:
            print("  PASS")

    print("\n" + ("=" * 72))
    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print("  - " + f)
        print("\nDo not score new data with this bundle until the cause is understood. The\n"
              "likely causes, in order: a scikit-learn version change, an edit to\n"
              "contrast_features.py, a different GEO cache, or a model file that was retrained\n"
              "without the fixture being regenerated.")
        return 1
    print("SELFTEST PASSED -- this bundle reproduces its frozen predictions on this machine.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="contrast_classifier.joblib")
    ap.add_argument("--fixture", default="fixture")
    ap.add_argument("--cache", default="geo_cache")
    ap.add_argument("--write-fixture", action="store_true",
                    help="(re)generate the frozen fixture from the current model")
    ap.add_argument("--comparisons", default="comparison_registry.csv",
                    help="source registry to sample the fixture from (--write-fixture only)")
    ap.add_argument("--n", type=int, default=24, help="fixture size (--write-fixture only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="allowed drift in the top-class probability before Level A fails")
    args = ap.parse_args()
    sys.exit(write_fixture(args) if args.write_fixture else run_test(args))


if __name__ == "__main__":
    main()
