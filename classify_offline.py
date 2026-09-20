#!/usr/bin/env python3
"""Classify a contrast registry offline with the fitted model. No API key, no Anthropic.

    python classify_offline.py --comparisons big_registry.csv --model contrast_classifier.joblib \\
        --out big_categories.xlsx

Needs network only to fetch GEO sample metadata for series not already in the cache -- that is
a keyless public request, and cached series are reused. Pass --no-fetch to refuse the network
entirely; contrasts from uncached series are then classified from their name alone and flagged.

Outputs the same workbook shape as the teacher:
  categories : gse | comparison_id | perturbation_category
  full       : + probability, abstain flag, drug_class, matched agent, aeration, OOV share
  coverage   : how far the new file sits from the training vocabulary
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run from any cwd

import contrast_features as cfx
import provenance as prov

HERE = os.path.dirname(os.path.abspath(__file__))

# reuse the teacher's fetcher so the cache format is identical
try:
    from label_contrasts import load_cached
except Exception:                                                   # noqa: BLE001
    load_cached = None


def ensure_cache(registry, cache_dir, allow_fetch):
    os.makedirs(cache_dir, exist_ok=True)
    missing = [g for g in sorted(registry.study_id.unique())
               if not os.path.exists(os.path.join(cache_dir, g + ".pkl"))]
    if not missing:
        return []
    if not allow_fetch or load_cached is None:
        print("%d series not cached and fetching disabled: %s"
              % (len(missing), ", ".join(missing[:8])), file=sys.stderr)
        return missing
    print("fetching %d uncached series from GEO (keyless, no API cost)" % len(missing),
          file=sys.stderr)
    for i, g in enumerate(missing, 1):
        load_cached(g, cache_dir)
        if i % 10 == 0:
            print("  %d/%d" % (i, len(missing)), file=sys.stderr)
        time.sleep(0.34)
    return [g for g in missing
            if not os.path.exists(os.path.join(cache_dir, g + ".pkl"))]


def oov_share(pipe, texts):
    """Fraction of each text's tokens absent from the training vocabulary."""
    vec = pipe.named_steps["features"].named_transformers_["text"]
    vocab = set(vec.vocabulary_)
    analyze = vec.build_analyzer()
    out = []
    for t in texts:
        toks = [w for w in analyze(t) if " " not in w]
        out.append(float(np.mean([w not in vocab for w in toks])) if toks else 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparisons", required=True,
                    help="CSV with study_id, comparison_id, technology")
    ap.add_argument("--model", default="contrast_classifier.joblib")
    ap.add_argument("--cache", default="geo_cache")
    ap.add_argument("--lexicon", default="drug_lexicon.csv")
    ap.add_argument("--out", default="offline_categories.xlsx")
    ap.add_argument("--abstain-below", type=float, default=0.55,
                    help="flag predictions whose top-class probability is under this")
    ap.add_argument("--strict-provenance", action="store_true",
                    help="exit rather than predict if the environment differs from the one the "
                         "model was trained in (library versions, feature-module source, lexicon)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="never touch the network; uncached series are name-only and flagged")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    if not prov.report(bundle, module_dir=HERE,
                       data_files={"lexicon": cfx.resolve_path(args.lexicon)},
                       strict=args.strict_provenance):
        sys.exit(2)
    pipe = bundle["pipeline"]
    reg = pd.read_csv(args.comparisons)
    for col in ("study_id", "comparison_id"):
        if col not in reg.columns:
            sys.exit("input is missing required column '%s'" % col)
    if "technology" not in reg.columns:
        reg["technology"] = "unknown"
    print("input: %d contrasts across %d studies; model trained on %d"
          % (len(reg), reg.study_id.nunique(), bundle["trained_on"]), file=sys.stderr)

    still_missing = ensure_cache(reg, args.cache, allow_fetch=not args.no_fetch)
    X = cfx.build_features(reg, args.cache)

    proba = pipe.predict_proba(X)
    classes = np.array(bundle["classes"])
    top = proba.argmax(1)
    pred, prob = classes[top], proba[np.arange(len(X)), top]
    second = np.sort(proba, axis=1)[:, -2] if proba.shape[1] > 1 else np.zeros(len(X))

    lexicon = cfx.load_lexicon(args.lexicon)
    dc = [cfx.match_drug_class(t, lexicon) for t in X[cfx.TEXT_COL]]
    oov = oov_share(pipe, X[cfx.TEXT_COL])

    full = pd.DataFrame({
        "gse": X.study_id, "comparison_id": X.comparison_id,
        "perturbation_category": pred,
        "probability": prob.round(3),
        "margin_over_runner_up": (prob - second).round(3),
        "abstain": prob < args.abstain_below,
        "drug_class": [d[0] for d in dc],
        "matched_agent": [d[1] for d in dc],
        "n_drug_classes_matched": [d[2] for d in dc],
        "aeration": X.aeration, "technology": X.technology,
        "series_metadata_available": X.has_series_metadata.astype(bool),
        "oov_token_share": np.round(oov, 3),
    })
    full["needs_review"] = (full.abstain | ~full.series_metadata_available
                            | (full.n_drug_classes_matched > 1) | (full.oov_token_share > 0.5))
    simple = full[["gse", "comparison_id", "perturbation_category"]]

    coverage = pd.DataFrame({
        "metric": ["contrasts", "studies", "series without metadata", "abstained",
                   "needs review", "median OOV token share", "drug_class assigned"],
        "value": [len(full), full.gse.nunique(), int((~full.series_metadata_available).sum()),
                  int(full.abstain.sum()), int(full.needs_review.sum()),
                  round(float(np.median(oov)), 3), int(full.drug_class.notna().sum())],
    })

    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        simple.to_excel(xl, sheet_name="categories", index=False)
        full.to_excel(xl, sheet_name="full", index=False)
        coverage.to_excel(xl, sheet_name="coverage", index=False)
        # so a workbook found on disk later can be traced to the exact model and environment
        trained, now = bundle.get("provenance") or {}, prov.stamp(module_dir=HERE)
        pv = [("model_file", os.path.abspath(args.model)),
              ("model_sha256", prov.file_sha256(args.model)),
              ("model_trained_utc", trained.get("created_utc")),
              ("model_trained_on_n", bundle.get("trained_on")),
              ("training_argv", trained.get("argv")),
              ("scored_utc", now["created_utc"]),
              ("scoring_argv", now["argv"])]
        for k in sorted(set(list(trained.get("packages", {})) + list(now["packages"]))):
            pv.append(("pkg:" + k, "trained=%s scored=%s"
                       % (trained.get("packages", {}).get(k), now["packages"].get(k))))
        for k in sorted(set(list(trained.get("code_sha256", {})) + list(now["code_sha256"]))):
            pv.append(("code:" + k, "trained=%s scored=%s"
                       % (trained.get("code_sha256", {}).get(k), now["code_sha256"].get(k))))
        pd.DataFrame(pv, columns=["field", "value"]).to_excel(
            xl, sheet_name="provenance", index=False)
        if still_missing:
            pd.DataFrame({"uncached_series": still_missing}).to_excel(
                xl, sheet_name="uncached", index=False)

    print("\nwrote %s" % args.out, file=sys.stderr)
    print(coverage.to_string(index=False), file=sys.stderr)
    print("\ncategory counts:\n%s" % full.perturbation_category.value_counts().to_string(),
          file=sys.stderr)


if __name__ == "__main__":
    main()
