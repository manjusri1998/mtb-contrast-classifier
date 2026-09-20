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


# --- arm difference from the contrast name / registry conditions --------------------------------

_SPLIT_VS = re.compile(r"_vs_|\bvs\b", re.I)
_TOKEN = re.compile(r"[a-z0-9.+]+")


def _side_tokens(s):
    return set(_TOKEN.findall(str(s).lower().replace("_", " ").replace("-", " ")))


def arm_diff(comparison_id, test_condition=None, control_condition=None):
    """Tokens unique to each arm. The category is a property of the difference, so this is the
    text a rule should match against: in 'H37Rv_Cholesterol_Rifampicin vs H37Rv_Glycerol_
    Rifampicin' the rifampicin is background on both arms and the difference is the carbon
    source. Prefers the registry's test/control strings; falls back to splitting the name.
    """
    if isinstance(test_condition, str) and isinstance(control_condition, str):
        lt, rt = _side_tokens(test_condition), _side_tokens(control_condition)
    else:
        parts = _SPLIT_VS.split(str(comparison_id), maxsplit=1)
        if len(parts) < 2:
            return _side_tokens(comparison_id), set()
        lt, rt = _side_tokens(parts[0]), _side_tokens(parts[1])
    return lt - rt, rt - lt


def sample_name_index(blob):
    """source_name/title -> that sample's characteristic text, for decoding contrast names.

    Many series build their contrast names out of sample names: GSE68856's 'KO_SDS_vs_KO_LOG_37'
    is 'KO_SDS_*' against 'KO_LOG_37.*', and each of those samples carries 'stress: SDS' /
    'stress: Log phase'. Prefix-matching the two sides of the name against this index recovers
    the arms' real metadata without needing recorded arm membership.
    """
    idx = []
    for v in blob["samples"].values():
        chars = "; ".join(v.get("characteristics_ch1", []))
        for f in ("source_name_ch1", "title"):
            for name in v.get(f, []):
                if name:
                    idx.append((_TOKEN.findall(name.lower().replace("_", " ").replace("-", " ")),
                                chars))
    return idx


GENERIC_SIDE = {"control", "ctrl", "con", "untreated", "mock", "reference", "ref", "paired",
                "channel", "none", "baseline", "wt", "parent"}


def arm_metadata(comparison_id, idx, sep_re=None, max_sets=2):
    """Characteristic text for each arm, found by token-prefix match on sample names.

    Two refusals, both necessary. A side whose tokens are all generic control words matches
    half the series ('..._vs_Control' would union every strain in it and manufacture a
    difference that is not there), and a side matching more than `max_sets` distinct
    characteristic strings has not been resolved to an arm at all. In both cases return
    nothing so the caller falls through to the contrast name.
    """
    sep_re = sep_re or _SPLIT_VS
    parts = sep_re.split(str(comparison_id), maxsplit=1)
    if len(parts) < 2:
        return "", ""
    out = []
    for side in parts[:2]:
        q = _TOKEN.findall(side.lower().replace("_", " ").replace("-", " "))
        if q and all(t in GENERIC_SIDE or t.isdigit() for t in q):
            return "", ""
        hits = set()
        for toks, chars in idx:
            if not (q and chars and len(toks) >= len(q)):
                continue
            # exact on all but the last query token; the last may be a prefix, because sample
            # names carry replicate suffixes ('KO_LOG_37' must match sample 'KO_LOG_37.1')
            if toks[:len(q) - 1] == q[:-1] and toks[len(q) - 1].startswith(q[-1]):
                hits.add(chars)
        if len(hits) > max_sets:
            return "", ""
        out.append(" | ".join(sorted(hits)))
    return out[0], out[1]


def metadata_diff(left_chars, right_chars):
    """Characteristic VALUES that differ between the arms (key-aware, so a shared
    'genotype: groEL1 KO' drops out and only 'stress: SDS' vs 'stress: Log phase' remains)."""
    def kv(text):
        d = {}
        for part in text.split("|"):
            for item in part.split(";"):
                k, _, v = item.partition(":")
                if v.strip():
                    d.setdefault(k.strip().lower(), set()).add(v.strip())
        return d
    L, R = kv(left_chars), kv(right_chars)
    out = []
    for k in sorted(set(L) | set(R)):
        lv, rv = L.get(k, set()), R.get(k, set())
        if lv != rv:
            out.append("%s: %s / %s" % (k, "+".join(sorted(lv)) or "-", "+".join(sorted(rv)) or "-"))
    return " | ".join(out)


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
                "idx": sample_name_index(blob),
                "aeration": series_aeration(blob),
                "vocab": series_vocabulary(blob),
                "vocab_varying": varying_vocabulary(blob),
                "title": (blob["head"].get("title") or [""])[0],
                "summary": (blob["head"].get("summary") or [""])[0][:1500],
                "n_samples": len(blob["samples"]),
            }
        b = blobs[gse]
        lchars, rchars = arm_metadata(r.comparison_id, b["idx"])
        dl, dr = arm_diff(r.comparison_id, getattr(r, "test_condition", None),
                          getattr(r, "control_condition", None))
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
            "diff_left_text": " ".join(sorted(dl)),
            "diff_right_text": " ".join(sorted(dr)),
            "diff_text": " ".join(sorted(dl | dr)),
            "arm_meta_diff_text": metadata_diff(lchars, rchars),
            "arm_meta_resolved": int(bool(lchars and rchars)),
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
