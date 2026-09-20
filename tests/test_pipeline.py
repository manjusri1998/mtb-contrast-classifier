"""End-to-end test on synthetic data: train -> stamp -> self-test -> score, plus drift detection.

Runs each stage as a subprocess, the way a user runs it, so the CLIs are covered and not just
the importable functions. Everything happens in a tmp directory built by make_demo_data.py --
no real GEO data, no network, no API key.
"""
import json
import os
import subprocess
import sys

import joblib
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args, cwd, expect=0):
    p = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    assert p.returncode == expect, (
        "expected exit %d, got %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (expect, p.returncode, p.stdout, p.stderr))
    return p


def script(name):
    return os.path.join(ROOT, name)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A demo dataset with a model, manifest, lock file and frozen fixture beside it."""
    d = str(tmp_path_factory.mktemp("pipeline"))
    run([script("make_demo_data.py"), "--out", "demo"], cwd=d)
    run([script("train_classifier.py"),
         "--labels", "demo/contrast_categories.xlsx",
         "--comparisons", "demo/comparison_registry.csv",
         "--cache", "demo/geo_cache",
         "--out", "model.joblib"], cwd=d)
    run([script("selftest_classifier.py"), "--write-fixture",
         "--model", "model.joblib",
         "--comparisons", "demo/comparison_registry.csv",
         "--cache", "demo/geo_cache"], cwd=d)
    return d


def test_training_writes_the_full_artefact_set(trained):
    for f in ("model.joblib", "training_manifest.json", "requirements-lock.txt",
              "training_report.txt"):
        assert os.path.exists(os.path.join(trained, f)), "missing " + f


def test_bundle_carries_a_provenance_stamp(trained):
    b = joblib.load(os.path.join(trained, "model.joblib"))
    p = b["provenance"]
    assert p["packages"]["scikit-learn"], "scikit-learn version not recorded"
    assert p["code_sha256"]["contrast_features.py"], "feature module not hashed"
    assert p["data_sha256"]["labels"], "label file not hashed"
    assert p["seed"] == 0 and p["n_train"] > 0 and p["schema_version"] >= 2


def test_manifest_is_valid_json_with_metrics(trained):
    m = json.load(open(os.path.join(trained, "training_manifest.json")))
    assert m["model_sha256"] and m["grouped_cv_folds"] >= 2
    assert set(m["class_counts"]) == set(m["grouped_cv_report"]) - {
        "accuracy", "macro avg", "weighted avg"}


def test_lock_file_pins_exactly(trained):
    pins = [l for l in open(os.path.join(trained, "requirements-lock.txt"))
            if l.strip() and not l.startswith("#")]
    assert pins, "no pins written"
    assert all("==" in p for p in pins), "lock file must pin exact versions"


def test_selftest_passes_on_a_clean_tree(trained):
    out = run([script("selftest_classifier.py"), "--model", "model.joblib",
               "--cache", "demo/geo_cache"], cwd=trained).stdout
    assert "SELFTEST PASSED" in out
    assert "LEVEL A" in out and "LEVEL B" in out


def test_offline_scoring_runs_and_stamps_the_workbook(trained):
    import pandas as pd
    run([script("classify_offline.py"), "--model", "model.joblib",
         "--comparisons", "demo/comparison_registry.csv",
         "--cache", "demo/geo_cache", "--no-fetch",
         "--out", "scored.xlsx"], cwd=trained)
    xl = pd.ExcelFile(os.path.join(trained, "scored.xlsx"))
    assert {"categories", "full", "provenance"} <= set(xl.sheet_names)
    prov = xl.parse("provenance").set_index("field")["value"]
    assert prov["model_sha256"] and prov["model_trained_utc"]


def test_strict_provenance_accepts_a_matching_environment(trained):
    run([script("classify_offline.py"), "--model", "model.joblib",
         "--comparisons", "demo/comparison_registry.csv",
         "--cache", "demo/geo_cache", "--no-fetch", "--strict-provenance",
         "--out", "strict.xlsx"], cwd=trained)


def test_feature_module_edit_is_caught(trained, tmp_path):
    """The failure this whole apparatus exists for: features change, model still 'works'.

    Copy the tree, perturb contrast_features.py, and confirm Level A still passes (the model
    is intact) while Level B fails (it is being fed different features) -- and that strict
    scoring refuses with exit 2.
    """
    import shutil
    d = str(tmp_path / "drift")
    shutil.copytree(trained, d)
    for f in ("contrast_features.py", "provenance.py", "selftest_classifier.py",
              "classify_offline.py", "train_classifier.py",
              "drug_lexicon.csv", "category_lexicon.csv"):
        shutil.copy(script(f), os.path.join(d, f))

    src = os.path.join(d, "contrast_features.py")
    text = open(src).read()
    assert '" || ".join' in text, "feature joiner not found; update this test"
    open(src, "w").write(text.replace('" || ".join', '" ~~ ".join'))

    out = run([sys.executable and os.path.join(d, "selftest_classifier.py"),
               "--model", "model.joblib", "--cache", "demo/geo_cache"],
              cwd=d, expect=1).stdout
    assert "SELFTEST FAILED" in out
    assert "LEVEL A" in out and "PASS" in out.split("LEVEL B")[0]
    assert "LEVEL B" in out and "DIFFERS" in out

    run([os.path.join(d, "classify_offline.py"), "--model", "model.joblib",
         "--comparisons", "demo/comparison_registry.csv", "--cache", "demo/geo_cache",
         "--no-fetch", "--strict-provenance", "--out", "x.xlsx"], cwd=d, expect=2)
