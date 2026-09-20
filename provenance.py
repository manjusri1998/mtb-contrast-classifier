"""Reproducibility stamp shared by train_classifier.py and classify_offline.py.

Imported by BOTH, for the same reason contrast_features.py is: the fields written into the
model bundle at training time and the fields checked when that bundle is loaded months later
must be defined in one place or they drift apart silently.

What this guards against
------------------------
A scikit-learn Pipeline is a pickle. Unpickling one under a different scikit-learn version
either raises (the loud, safe failure) or loads with changed estimator defaults and quietly
predicts something else (the failure you never notice). Neither the version nor the code that
built the feature columns is recorded inside a joblib file by default, so a bundle handed to a
future machine is unfalsifiable: you cannot tell whether it is behaving as it did when trained.

So we record, at fit time:
  packages   - exact versions of every library the pickle depends on to load and predict
  python     - interpreter version and platform
  code       - SHA-256 of each module whose behaviour is baked into the feature space
  data       - SHA-256 of the label file, the contrast registry and the drug lexicon
  argv       - the literal command that produced the bundle

and at load time compare the first three, reporting every mismatch. The check is advisory by
default (a numpy patch bump is not worth blocking a run over) and fatal under --strict-provenance.

A hash mismatch is not proof the model is wrong; it is proof you no longer know that it is right.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

# Libraries whose version can change what a loaded pipeline predicts, or whether it loads at all.
TRACKED_PACKAGES = ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "openpyxl"]

# Modules whose source defines the feature space or the bundle format.
TRACKED_MODULES = ["contrast_features.py", "provenance.py", "train_classifier.py"]

SCHEMA_VERSION = 2          # bundle layout; bump if the dict keys change meaning


def file_sha256(path, short=True):
    """Hash a file. Returns None for a path that does not exist, so callers can stay simple."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    return d[:16] if short else d


def package_versions(names=TRACKED_PACKAGES):
    """Installed version of each tracked distribution, or None where it is absent."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                              # py<3.8
        return {}
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def code_hashes(module_dir=None, names=TRACKED_MODULES):
    d = module_dir or os.path.dirname(os.path.abspath(__file__))
    return {n: file_sha256(os.path.join(d, n)) for n in names}


def git_commit(module_dir=None):
    """Commit of the repo the scripts live in, if they live in one. None otherwise."""
    d = module_dir or os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", d, "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:                                                # noqa: BLE001
        pass
    return None


def stamp(data_files=None, extra=None, module_dir=None):
    """Build the provenance record to embed in a model bundle.

    data_files: {label: path} for inputs whose content should be pinned by hash.
    extra:      anything else worth freezing (seed, hyperparameters, row counts).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "code_sha256": code_hashes(module_dir),
        "git_commit": git_commit(module_dir),
        "data_sha256": {k: file_sha256(v) for k, v in (data_files or {}).items()},
        "data_paths": {k: os.path.abspath(v) if v else None
                       for k, v in (data_files or {}).items()},
        **(extra or {}),
    }


def _cmp(kind, expected, actual):
    msgs = []
    for key, want in sorted((expected or {}).items()):
        got = (actual or {}).get(key)
        if want is None and got is None:
            continue
        if want != got:
            msgs.append("  %-14s %-24s trained: %-18s now: %s"
                        % (kind, key, want, got))
    return msgs


def verify(bundle, module_dir=None, data_files=None):
    """Compare a loaded bundle's stamp against the current environment.

    Returns (messages, severity) where severity is 'ok' | 'warn' | 'error'.
    'error' is reserved for scikit-learn, because that is the one whose version can change
    predictions from a pickle without raising.
    """
    p = bundle.get("provenance")
    if not p:
        return (["  bundle carries no provenance stamp -- it predates this check, so its "
                 "training environment is unknown and cannot be verified"], "warn")

    msgs = []
    msgs += _cmp("package", p.get("packages"), package_versions())
    msgs += _cmp("code", p.get("code_sha256"), code_hashes(module_dir))
    if data_files:
        msgs += _cmp("data", {k: v for k, v in (p.get("data_sha256") or {}).items()
                              if k in data_files},
                     {k: file_sha256(v) for k, v in data_files.items()})
    if p.get("python") != sys.version.split()[0]:
        msgs.append("  %-14s %-24s trained: %-18s now: %s"
                    % ("interpreter", "python", p.get("python"), sys.version.split()[0]))

    if not msgs:
        return [], "ok"
    sk_changed = (p.get("packages", {}).get("scikit-learn")
                  != package_versions(["scikit-learn"])["scikit-learn"])
    code_changed = p.get("code_sha256", {}).get("contrast_features.py") \
        != code_hashes(module_dir).get("contrast_features.py")
    return msgs, ("error" if (sk_changed or code_changed) else "warn")


def report(bundle, module_dir=None, data_files=None, strict=False, stream=sys.stderr):
    """Print the verification result. Returns True if the run should continue."""
    msgs, sev = verify(bundle, module_dir, data_files)
    if sev == "ok":
        print("provenance: environment matches the training stamp", file=stream)
        return True
    head = {"warn": "PROVENANCE WARNING", "error": "PROVENANCE MISMATCH"}[sev]
    print("\n%s -- this run differs from the environment the model was trained in:"
          % head, file=stream)
    print("\n".join(msgs), file=stream)
    if sev == "error":
        print("\nA scikit-learn version change or an edit to contrast_features.py can alter\n"
              "predictions without raising an error. Either recreate the training environment\n"
              "(pip install -r requirements.txt, pinned at training time) or retrain.",
              file=stream)
    if strict:
        print("--strict-provenance set: refusing to run.", file=stream)
        return False
    print("continuing anyway (pass --strict-provenance to make this fatal)\n", file=stream)
    return True


def write_requirements(path, names=TRACKED_PACKAGES):
    """Exact pins for the libraries needed to reload and run the bundle."""
    vers = package_versions(names)
    lines = ["# Pinned at training time by train_classifier.py.",
             "# Recreate with:  python -m venv .venv && . .venv/bin/activate",
             "#                 pip install -r %s" % os.path.basename(path),
             "# Interpreter used: python %s (%s)" % (sys.version.split()[0], platform.platform()),
             ""]
    lines += ["%s==%s" % (n, v) for n, v in vers.items() if v]
    missing = [n for n, v in vers.items() if not v]
    if missing:
        lines += ["", "# not installed at training time: " + ", ".join(missing)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return vers


def write_manifest(path, record):
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    return path
