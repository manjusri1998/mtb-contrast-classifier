"""Shared feature extraction for the contrast classifier.

Imported by BOTH train_classifier.py and classify_offline.py so the training and inference
feature spaces cannot drift apart. No API calls anywhere in this module.

A contrast's features are text plus two categorical signals:
  text      - contrast name, series title/summary, and the sample-group vocabulary of the series
              (source names + characteristics), which is what makes names like
              'gpl1343_cccp_vs_paired_reference_channel' interpretable
  aeration  - hypoxia_stated / aeration_stated / unstated, from the series growth protocol
  technology- microarray / rnaseq
"""
import collections
import os
import pickle
import re

import pandas as pd

AERATED = re.compile(r"\b(shaking|shaken|roller|rolling|aerat|aerobic|stirr|sparg|orbital)", re.I)
HYPOXIC = re.compile(r"\b(standing|static|sealed|unstirred|wayne|hypoxi|anaerob|anoxi|"
                     r"oxygen.{0,12}(depl|limit)|nitrogen.{0,6}flush)", re.I)

TEXT_COL = "text"
CAT_COLS = ["aeration", "technology"]


# --- cache access ----------------------------------------------------------------------------

def load_blob(gse, cache_dir):
    """Read one cached series. Returns an empty shell if the series was never fetched."""
    path = os.path.join(cache_dir, gse + ".pkl")
    if not os.path.exists(path):
        return {"head": {}, "samples": {}}
    blob = pickle.load(open(path, "rb"))
    if "error" in blob:
        return {"head": blob.get("head", {}), "samples": blob.get("samples", {})}
    return blob


def series_aeration(blob):
    txt = " ".join(
        [v.get("growth_protocol_ch1", [""])[0] for v in blob["samples"].values()
         if v.get("growth_protocol_ch1")]
        + [v.get("treatment_protocol_ch1", [""])[0] for v in blob["samples"].values()
           if v.get("treatment_protocol_ch1")])
    if HYPOXIC.search(txt):
        return "hypoxia_stated"
    if AERATED.search(txt):
        return "aeration_stated"
    return "unstated"


def series_vocabulary(blob, max_groups=60):
    """The distinct sample-group descriptors of a series, as one string."""
    seen, out = set(), []
    for v in blob["samples"].values():
        key = (v.get("source_name_ch1", [""])[0], "; ".join(v.get("characteristics_ch1", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(" ".join(filter(None, [key[0], key[1], v.get("title", [""])[0]])))
        if len(out) >= max_groups:
            break
    return " | ".join(out)


def varying_vocabulary(blob, max_groups=60):
    """Sample-group text with terms common to EVERY group removed.

    A term present in all of a series' sample groups cannot distinguish one arm from another --
    'genotype: wild-type' stated on every sample is background, not evidence of a genetic
    contrast. Dropping the series-constant terms leaves the vocabulary that actually varies,
    which is the closest series-level proxy for what differs between arms.
    """
    groups, seen = [], set()
    for v in blob["samples"].values():
        key = (v.get("source_name_ch1", [""])[0], "; ".join(v.get("characteristics_ch1", [])))
        if key in seen:
            continue
        seen.add(key)
        groups.append(set(re.findall(r"[a-z0-9.+-]+", " ".join(
            filter(None, [key[0], key[1], v.get("title", [""])[0]])).lower())))
        if len(groups) >= max_groups:
            break
    if len(groups) < 2:
        return series_vocabulary(blob, max_groups)
    common = set.intersection(*groups)
    return " | ".join(" ".join(sorted(g - common)) for g in groups)


# --- feature frame ----------------------------------------------------------------------------

def _norm(s):
    """Split on underscores/camel joins so 'T_24_vs_Control' and 'gpl1343_cccp' tokenise."""
    s = str(s).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def build_features(registry, cache_dir, study_registry=None):
    """registry: DataFrame with study_id, comparison_id, technology (extra columns are used
    if present: test_condition, control_condition). Returns a DataFrame of features, one row
    per contrast, in the registry's order."""
    blobs, rows = {}, []
    for r in registry.itertuples():
        gse = r.study_id
        if gse not in blobs:
            blob = load_blob(gse, cache_dir)
            blobs[gse] = {
                "aeration": series_aeration(blob),
                "vocab": series_vocabulary(blob),
                "vocab_varying": varying_vocabulary(blob),
                "title": (blob["head"].get("title") or [""])[0],
                "summary": (blob["head"].get("summary") or [""])[0][:1500],
                "n_samples": len(blob["samples"]),
            }
        b = blobs[gse]
        cond = ""
        for col in ("test_condition", "control_condition"):
            v = getattr(r, col, None)
            if isinstance(v, str):
                cond += " " + _norm(v)
        vocab, title, summary = b["vocab"], b["title"], b["summary"]
        rows.append({
            "study_id": gse,
            "comparison_id": r.comparison_id,
            "technology": getattr(r, "technology", "unknown"),
            "aeration": b["aeration"],
            "n_samples": b["n_samples"],
            "has_series_metadata": int(b["n_samples"] > 0),
            # separate sources, so a rule can prefer factual condition labels over the
            # series summary -- the summary describes the study's motivation, not the contrast
            "name_text": _norm(r.comparison_id),
            "cond_text": cond.strip(),
            "vocab_text": vocab,
            "vocab_varying_text": b["vocab_varying"],
            "summary_text": " ".join([title, summary]),
            TEXT_COL: " || ".join([_norm(r.comparison_id), cond.strip(), title, summary, vocab]),
        })
    return pd.DataFrame(rows)


# --- drug-class lexicon -------------------------------------------------------------------------

def resolve_path(path):
    """Resolve a data file that ships with the code, from any working directory.

    A bare default like 'drug_lexicon.csv' is looked up relative to the current directory
    first -- so a user-supplied copy still wins -- and falls back to the directory this module
    lives in, which is where the shipped lexicons sit. Returns the path unchanged if neither
    exists, so the caller still raises a normal FileNotFoundError naming what it wanted.
    """
    if os.path.exists(path):
        return path
    beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
    return beside if os.path.exists(beside) else path


def load_lexicon(path="drug_lexicon.csv"):
    lex = pd.read_csv(resolve_path(path))
    if "category_hint" not in lex.columns:
        lex["category_hint"] = "drug"
    return [(re.compile(p, re.I), a, c, h) for p, a, c, h
            in zip(lex.pattern, lex.canonical_agent, lex.drug_class, lex.category_hint)]


def match_drug_class(text, lexicon):
    """Return (drug_class, canonical_agent, n_distinct_classes_matched, category_hint).

    Deterministic lookup, not a learned label: 13 mechanism classes over a few hundred drug
    contrasts is too few examples per class to fit anything trustworthy, and the mapping from
    agent to mechanism is a fact rather than a pattern. Multiple hits are reported so genuinely
    ambiguous contrasts (a combination treatment) can be reviewed instead of silently reduced.
    """
    hits = [(a, c, h) for rx, a, c, h in lexicon if rx.search(text)]
    if not hits:
        return None, None, 0, None
    classes = collections.Counter(c for _, c, _ in hits)
    agents = sorted({a for a, _, _ in hits})
    hint = collections.Counter(h for _, _, h in hits).most_common(1)[0][0]
    return classes.most_common(1)[0][0], "+".join(agents), len(classes), hint
