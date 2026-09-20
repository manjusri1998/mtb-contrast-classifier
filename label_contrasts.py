#!/usr/bin/env python3
"""Assign perturbation categories to individual atlas CONTRASTS (not studies).

The perturbation category is a property of the DIFFERENCE between a contrast's two arms.
This script pulls per-sample metadata from GEO for every series in the comparison registry,
compresses it into a decoding key for the contrast names, asks an LLM to label each contrast
against a fixed controlled vocabulary, then applies deterministic house rules on top.

Usage
-----
  export ANTHROPIC_API_KEY=...            # only needed when not using --dry-run

  # 1. inspect what would be sent, no API calls, no cost:
  python label_contrasts.py --dry-run

  # 2. tune: run one series and eyeball it
  python label_contrasts.py --only GSE1642 --out test.xlsx

  # 3. full run
  python label_contrasts.py --out contrast_categories.xlsx

Inputs  : comparison_registry.csv  (study_id, comparison_id, technology, test_condition, ...)
          study_registry.csv       (study_id, perturbation_broad_category, perturbation_subcategory, ...)
Outputs : <out>.xlsx  sheet 'categories'  -> gse | comparison_id | perturbation_category
                      sheet 'full'        -> every extracted field, flags, justification
                      sheet 'vs_study'    -> agreement against the study-level labels
          geo_cache/  one pickle per series, so re-runs are free
          prompts/    (--dry-run only) the exact prompt text per chunk

Tuning surface: everything under "CONTROLLED VOCABULARY" and PROMPT below. Edit those, re-run
--dry-run to read the prompt back, then re-run on one series before doing all of them.
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import pickle
import re
import sys
import time
import urllib.request

import pandas as pd

# =============================================================================================
# CONTROLLED VOCABULARY  (tuning surface -- edit freely)
# =============================================================================================

CATEGORIES = ["genetic", "drug", "environment", "nutrition", "infection"]

SUBCATS = {
    "drug": "antibiotics, or a more specific agent name",
    "environment": "hypoxia, pH, oxidative stress, nitrosative stress, heat, cold, "
                   "cell-envelope stress, DNA damage, dormancy/reaeration",
    "genetic": "knockout, deletion, overexpression, complementation, transposon mutant, "
               "clinical-isolate lineage",
    "nutrition": "carbon, lipid, nitrate, phosphate, potassium, iron, copper, vitamin C, "
                 "starvation, defined vs rich medium",
    "infection": "intracellular, clinical sample, animal, ex vivo sputum",
}

DRUG_CLASSES = {
    "cell_wall_synthesis_inhibitor":
        "isoniazid, ethionamide, ethambutol, D-cycloserine, beta-lactams (meropenem, cephalexin, "
        "amoxicillin/clavulanate), vancomycin, thiacetazone, SQ109 (MmpL3/mycolic-acid transport), "
        "delamanid/pretomanid (mycolic acid arm), triclosan/InhA inhibitors",
    "protein_synthesis_inhibitor":
        "streptomycin, amikacin, kanamycin, capreomycin, viomycin, linezolid/oxazolidinones, "
        "macrolides (erythromycin, clarithromycin), tetracyclines, chloramphenicol, spectinomycin, "
        "fusidic acid, puromycin",
    "rna_polymerase_inhibitor":
        "rifampicin, rifapentine, rifabutin, other rifamycins, streptolydigin",
    "dna_gyrase_inhibitor":
        "fluoroquinolones (moxifloxacin, levofloxacin, ciprofloxacin, ofloxacin, gatifloxacin, "
        "sparfloxacin), novobiocin, nalidixic acid",
    "dna_damaging_agent":
        "mitomycin C, mithramycin, alkylating agents (MMS), bleomycin, UV irradiation, hydroxyurea",
    "energy_metabolism_inhibitor":
        "bedaquiline (ATP synthase), telacebec/Q203 and other QcrB inhibitors, clofazimine, "
        "thioridazine, chlorpromazine, protonophores/uncouplers (CCCP, DNP, nigericin, valinomycin), "
        "cyanide, azide, rotenone",
    "folate_pathway_inhibitor":
        "para-aminosalicylic acid (PAS), sulfonamides/sulfamethoxazole, trimethoprim, dapsone",
    "membrane_disruptor":
        "SDS and other detergents, polymyxin, chlorhexidine, cationic peptides, cerulenin",
    "redox_or_oxidative_agent":
        "hydrogen peroxide, menadione, plumbagin, diamide, paraquat, cumene hydroperoxide, "
        "nitric-oxide donors (DETA/NO, GSNO), sodium nitrite, vitamin C / ascorbate",
    "efflux_pump_inhibitor":
        "verapamil, reserpine, piperine, carbonyl cyanide derivatives used as efflux inhibitors",
    "antimetabolite":
        "5-fluorouracil, 6-azauracil, purine/pyrimidine analogues, amino-acid analogues, "
        "nicotinamide and benzamide analogues used as antimetabolites",
    "other_defined_mechanism":
        "pyrazinamide (PanD / coenzyme A, pH-dependent), other agents with a named target "
        "not covered above",
    "unknown_mechanism":
        "a compound is present but its mechanism is not established or not identifiable here",
}

PROMPT = """You assign perturbation categories to the individual CONTRASTS of one GEO series from a
Mycobacterium tuberculosis expression atlas.

A contrast compares a treatment arm against a control arm. The category describes WHAT DIFFERS
BETWEEN THE ARMS, not what the study as a whole was about. A single series often contains contrasts
of several different categories -- that is the reason this task exists. If both arms were grown in
hypoxia and differ only in genotype, the category is genetic, not environment.

CATEGORY VOCABULARY (use these exact strings for primary_category):
{categories}

SUBCATEGORY guidance (match the atlas's existing vocabulary where possible):
{subcats}

HOUSE RULES, applied in this order:
1. If the arms differ ONLY in elapsed time -- no chemical agent, no genotype difference, no medium
   change, no host -- set arms_differ_only_in_time=true. The caller converts those to
   environment/hypoxia, so do not do it yourself; just report the structural fact honestly.
2. Metal availability (copper, iron, zinc) and vitamin C are NUTRITION in this atlas, not
   environment, unless the metadata frames them explicitly as toxicity or overload stress.
3. If a defined chemical or physical agent distinguishes the arms, also assign drug_class,
   EVEN IF primary_category ends up environment. Hydrogen peroxide is environment/oxidative stress
   with drug_class redox_or_oxidative_agent; UV is environment/DNA damage with drug_class
   dna_damaging_agent. Use null for drug_class only when no agent is involved at all.

DRUG_CLASS VOCABULARY (exact strings):
{drugclasses}

GROUNDING REQUIREMENT: for each contrast, name in arm_evidence which sample group(s) from the
SAMPLES block the treatment arm corresponds to. If you cannot identify the samples, say so there
and set confidence low. Do not infer a category from the series summary alone.

CONFIDENCE: high = the sample metadata states the perturbation explicitly; medium = inferable from
protocol free text or the series summary; low = the contrast name cannot be matched to samples, the
arms are ambiguous, or the metadata does not identify a perturbation.

The study-level label is supplied for REFERENCE ONLY. Where a contrast plainly differs from it, set
differs_from_study_label=true -- catching those is the entire purpose of this exercise. Never copy
the study label by default.

Return STRICT JSON: a list with EXACTLY one object per contrast, in the order given, each with keys:
"comparison_id", "primary_category", "subcategory", "drug_class" (or null), "agent_or_condition",
"dose" (or null), "duration" (or null), "control_type", "arms_differ_only_in_time" (true/false),
"arm_evidence", "confidence", "differs_from_study_label" (true/false), "justification" (<=25 words).
No prose outside the JSON.

=== SERIES ===
{series}

=== SAMPLES (grouped; this is the key for decoding contrast names) ===
{samples}

=== CONTRASTS TO LABEL ({n}) ===
{contrasts}
"""

# =============================================================================================
# GEO retrieval
# =============================================================================================

GEO_URL = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
           "?acc={acc}&targ={targ}&form=text&view=brief")

SAMPLE_FIELDS = ["title", "source_name_ch1", "characteristics_ch1", "growth_protocol_ch1",
                 "treatment_protocol_ch1", "description", "platform_id", "organism_ch1"]

AERATED = re.compile(r"\b(shaking|shaken|roller|rolling|aerat|aerobic|stirr|sparg|orbital)", re.I)
HYPOXIC = re.compile(r"\b(standing|static|sealed|unstirred|wayne|hypoxi|anaerob|anoxi|"
                     r"oxygen.{0,12}(depl|limit)|nitrogen.{0,6}flush)", re.I)


def _get(acc, targ, retries=3):
    last = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(GEO_URL.format(acc=acc, targ=targ),
                                          timeout=120).read().decode("utf-8", "replace")
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def fetch_series(gse):
    head = {}
    for line in _get(gse, "self").splitlines():
        m = re.match(r"!Series_(\w+) = (.*)", line)
        if m:
            head.setdefault(m.group(1), []).append(m.group(2))
    samples, cur = collections.OrderedDict(), None
    for line in _get(gse, "gsm").splitlines():
        m = re.match(r"\^SAMPLE = (GSM\d+)", line)
        if m:
            cur = m.group(1)
            samples[cur] = collections.defaultdict(list)
            continue
        m = re.match(r"!Sample_(\w+) = (.*)", line)
        if m and cur and m.group(1) in SAMPLE_FIELDS:
            samples[cur][m.group(1)].append(m.group(2))
    return head, {k: dict(v) for k, v in samples.items()}


def load_cached(gse, cache_dir, pause=0.4):
    path = os.path.join(cache_dir, gse + ".pkl")
    if os.path.exists(path):
        blob = pickle.load(open(path, "rb"))
        if "error" not in blob:
            return blob
    try:
        head, samples = fetch_series(gse)
        blob = {"head": head, "samples": samples}
    except Exception as exc:                                        # noqa: BLE001
        blob = {"error": str(exc), "head": {}, "samples": {}}
    pickle.dump(blob, open(path, "wb"))
    time.sleep(pause)
    return blob


# =============================================================================================
# Payload construction
# =============================================================================================

def series_protocols(blob):
    gp = sorted({v.get("growth_protocol_ch1", [""])[0]
                 for v in blob["samples"].values() if v.get("growth_protocol_ch1")})
    tp = sorted({v.get("treatment_protocol_ch1", [""])[0]
                 for v in blob["samples"].values() if v.get("treatment_protocol_ch1")})
    return gp, tp


def series_aeration(blob):
    gp, tp = series_protocols(blob)
    txt = " ".join(gp + tp)
    if HYPOXIC.search(txt):
        return "hypoxia_stated"
    if AERATED.search(txt):
        return "aeration_stated"
    return "unstated"


def group_lines(blob, max_groups=120):
    """Collapse samples into distinct (source_name, characteristics) groups.

    This is the decoding key for contrast names: a contrast called gpl1343_cccp_vs_... is only
    interpretable because a sample group carries source=CCCP.
    """
    groups = collections.OrderedDict()
    for gsm, v in blob["samples"].items():
        key = (v.get("source_name_ch1", [""])[0], "; ".join(v.get("characteristics_ch1", [])))
        groups.setdefault(key, []).append(v.get("title", [""])[0])
    lines = []
    for (src, ch), titles in list(groups.items())[:max_groups]:
        ex = "; ".join(titles[:3])[:220]
        lines.append("n=%d\tsource=%s\tchar=%s\tex_titles=%s" % (len(titles), src, ch, ex))
    if len(groups) > max_groups:
        lines.append("... %d further sample groups omitted" % (len(groups) - max_groups))
    return "\n".join(lines)


def build_tasks(comp, stud, cache_dir, chunk_size=20, only=None):
    tasks = []
    frame = comp if only is None else comp[comp.study_id.isin(only)]
    for gse, grp in frame.groupby("study_id"):
        blob = load_cached(gse, cache_dir)
        head = blob["head"]
        gp, tp = series_protocols(blob)
        srow = stud[stud.study_id == gse]
        ref = ("%s / %s" % (srow.perturbation_broad_category.iloc[0],
                            srow.perturbation_subcategory.iloc[0])) if len(srow) else "none"
        series_blk = (
            "accession: %s\ntitle: %s\nsummary: %s\ngrowth_protocol: %s\ntreatment_protocol: %s\n"
            "aeration_evidence: %s\nSTUDY-LEVEL REFERENCE LABEL (do not copy by default): %s"
            % (gse, (head.get("title") or [""])[0], (head.get("summary") or [""])[0][:1000],
               " | ".join(gp)[:600], " | ".join(tp)[:600], series_aeration(blob), ref))
        samples_blk = group_lines(blob)
        rows = list(grp.itertuples())
        for i in range(0, len(rows), chunk_size):
            batch = rows[i:i + chunk_size]
            clines = []
            for r in batch:
                extra = ""
                if isinstance(getattr(r, "test_condition", None), str):
                    extra = "\ttest=%s\tcontrol=%s" % (r.test_condition, r.control_condition)
                clines.append("%s%s" % (r.comparison_id, extra))
            tasks.append({
                "gse": gse,
                "ids": [r.comparison_id for r in batch],
                "aeration": series_aeration(blob),
                "study_label": ref,
                "prompt": PROMPT.format(
                    categories=", ".join(CATEGORIES),
                    subcats="\n".join("  %s: %s" % (k, v) for k, v in SUBCATS.items()),
                    drugclasses="\n".join("  %s: %s" % (k, v) for k, v in DRUG_CLASSES.items()),
                    series=series_blk, samples=samples_blk,
                    n=len(batch), contrasts="\n".join(clines)),
            })
    return tasks


# =============================================================================================
# LLM call
# =============================================================================================

def call_llm(prompt, model, max_tokens=6000, retries=3):
    import anthropic
    client = anthropic.Anthropic()
    last = None
    for attempt in range(retries):
        try:
            msg = client.messages.create(model=model, max_tokens=max_tokens,
                                         messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise last


def parse_json_list(text):
    s = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.S)
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("no JSON array in response (truncated?): %s" % s[-200:])
    return json.loads(s[start:end + 1])


# =============================================================================================
# Deterministic post-pass
# =============================================================================================

def apply_house_rule(call, aeration):
    """Arms differing only in elapsed time are hypoxia, unless the protocol states aeration."""
    call = dict(call)
    if not call.get("arms_differ_only_in_time"):
        call["house_rule"] = "not_applicable"
        return call
    if aeration == "aeration_stated":
        call["house_rule"] = "blocked:aeration_stated"
        call["confidence"] = "low"
        call["justification"] = ("time-only contrast but protocol states aeration - growth phase, "
                                 "not hypoxia; needs review")
        return call
    call["primary_category"] = "environment"
    call["subcategory"] = "hypoxia"
    call["agent_or_condition"] = "hypoxia (atlas convention for time-only contrasts)"
    call["drug_class"] = None
    call["confidence"] = "high" if aeration == "hypoxia_stated" else "medium"
    call["house_rule"] = "applied:time_only_to_hypoxia (%s)" % aeration
    return call


def validate(call):
    problems = []
    if call.get("primary_category") not in CATEGORIES:
        problems.append("category_off_vocabulary:%s" % call.get("primary_category"))
    dc = call.get("drug_class")
    if dc not in (None, "", "null") and dc not in DRUG_CLASSES:
        problems.append("drug_class_off_vocabulary:%s" % dc)
    if call.get("confidence") not in ("high", "medium", "low"):
        problems.append("confidence_off_vocabulary:%s" % call.get("confidence"))
    return "; ".join(problems)


# =============================================================================================
# Main
# =============================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparisons", default="comparison_registry.csv")
    ap.add_argument("--study", default="study_registry.csv")
    ap.add_argument("--cache", default="geo_cache")
    ap.add_argument("--out", default="contrast_categories.xlsx")
    ap.add_argument("--model", default="claude-sonnet-4-5",
                    help="Anthropic model id; check your account for the current name")
    ap.add_argument("--chunk", type=int, default=20, help="contrasts per LLM call")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these GSE accessions")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and save prompts, make no API calls")
    args = ap.parse_args()

    comp = pd.read_csv(args.comparisons)
    stud = pd.read_csv(args.study)
    os.makedirs(args.cache, exist_ok=True)

    print("registry: %d contrasts across %d studies"
          % (len(comp), comp.study_id.nunique()), file=sys.stderr)
    tasks = build_tasks(comp, stud, args.cache, args.chunk, args.only)
    print("built %d prompt chunks (~%d input tokens)"
          % (len(tasks), sum(len(t["prompt"]) for t in tasks) // 4), file=sys.stderr)

    if args.dry_run:
        os.makedirs("prompts", exist_ok=True)
        manifest = []
        for i, t in enumerate(tasks):
            path = "prompts/%03d_%s.txt" % (i, t["gse"])
            open(path, "w").write(t["prompt"])
            manifest.append({"chunk": i, "gse": t["gse"], "n_contrasts": len(t["ids"]),
                             "aeration": t["aeration"], "study_label": t["study_label"],
                             "prompt_chars": len(t["prompt"]),
                             "est_input_tokens": len(t["prompt"]) // 4, "path": path})
        pd.DataFrame(manifest).to_csv("prompt_manifest.csv", index=False)
        print("dry run: wrote prompts/ and prompt_manifest.csv; no API calls made", file=sys.stderr)
        return

    def run(task):
        try:
            return task, parse_json_list(call_llm(task["prompt"], args.model, args.max_tokens)), None
        except Exception as exc:                                    # noqa: BLE001
            return task, None, str(exc)

    records, failures = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (task, parsed, err) in enumerate(pool.map(run, tasks), 1):
            if err is not None:
                failures.append({"gse": task["gse"], "ids": ";".join(task["ids"]), "error": err})
                print("[%d/%d] %s FAILED: %s" % (n, len(tasks), task["gse"], err[:120]),
                      file=sys.stderr)
                continue
            by_id = {c.get("comparison_id"): c for c in parsed}
            for cid in task["ids"]:
                call = by_id.get(cid)
                if call is None:
                    failures.append({"gse": task["gse"], "ids": cid, "error": "missing from response"})
                    continue
                call = apply_house_rule(call, task["aeration"])
                call["gse"] = task["gse"]
                call["comparison_id"] = cid
                call["study_label"] = task["study_label"]
                call["schema_problems"] = validate(call)
                records.append(call)
            print("[%d/%d] %s ok" % (n, len(tasks), task["gse"]), file=sys.stderr)

    if not records:
        print("no records produced; see failures", file=sys.stderr)
        sys.exit(1)

    full = pd.DataFrame(records)
    cols = ["gse", "comparison_id", "primary_category", "subcategory", "drug_class",
            "agent_or_condition", "dose", "duration", "control_type",
            "arms_differ_only_in_time", "house_rule", "confidence",
            "differs_from_study_label", "study_label", "arm_evidence", "justification",
            "schema_problems"]
    full = full.reindex(columns=[c for c in cols if c in full.columns])
    simple = full[["gse", "comparison_id", "primary_category"]].rename(
        columns={"primary_category": "perturbation_category"})

    study_broad = full.study_label.str.split(" / ").str[0]
    vs = (pd.crosstab(study_broad, full.primary_category)
            .rename_axis(index="study_level_label", columns="contrast_level_label"))

    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        simple.to_excel(xl, sheet_name="categories", index=False)
        full.to_excel(xl, sheet_name="full", index=False)
        vs.to_excel(xl, sheet_name="vs_study")
        if failures:
            pd.DataFrame(failures).to_excel(xl, sheet_name="failures", index=False)

    agree = (study_broad == full.primary_category).mean()
    print("\nwrote %s" % args.out, file=sys.stderr)
    print("labelled %d/%d contrasts; %d chunk failures"
          % (len(full), len(comp) if args.only is None else len(full), len(failures)),
          file=sys.stderr)
    print("agreement with study-level label: %.1f%%" % (100 * agree), file=sys.stderr)
    print("confidence: %s" % full.confidence.value_counts().to_dict(), file=sys.stderr)
    print("drug_class breakdown:\n%s"
          % full[full.drug_class.notna()].drug_class.value_counts().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
