#!/usr/bin/env python3
"""Rule-based contrast categoriser. No model, no training, no API, no weights to go stale.

    python classify_rules.py --comparisons big_registry.csv --out rules_categories.xlsx

Why rules rather than a fitted model: the atlas has 604 contrasts but only 86 independent
studies, and what identifies a perturbation is study-specific vocabulary. A TF-IDF model scored
0.63 under grouped-by-study cross-validation and 0.39 on wholly unseen studies, because a new
study brings words it never saw. A lexicon has no such cliff -- it generalises exactly as far as
its vocabulary covers the new file, and when it doesn't cover something it says so instead of
guessing. It is also a CSV you can extend in a text editor.

Two-tier matching, because the category is a property of the DIFFERENCE between arms:
  tier 0a the ARM METADATA DIFFERENCE -- each side of the contrast name is prefix-matched against
          the series' sample names, and only the characteristic VALUES that differ between the two
          arms are matched. This decodes a series' own abbreviations from its own metadata:
          GSE68856's 'KO_CS_vs_KO_LOG_37' resolves to 'stress: Cold shock / Log phase', so CS is
          read as cold shock rather than guessed at                                       -> high
  tier 0b the ARM DIFFERENCE -- tokens present on one arm and absent from the other, taken from
          the registry's test/control strings or by splitting the name on _vs_. In
          'H37Rv_Cholesterol_Rifampicin vs H37Rv_Glycerol_Rifampicin' the rifampicin is on both
          arms and the difference is the carbon source                                   -> high
  tier 1  the whole contrast name, for contrasts whose difference cannot be isolated   -> medium
  tier 2  the series' sample-group vocabulary with series-constant terms removed, since a
          term stated on every sample cannot distinguish one arm from another            -> medium
  tier 3  the series title and summary, which describe the study's motivation rather than
          any one contrast, so a match here is reported at low confidence and flagged
A contrast whose name matches two categories is NOT forced into one: it is reported with both
and flagged, because the name alone genuinely cannot say which arm differs in what.

Precedence when several categories match in the same tier, applied in this order:
  infection > genetic > drug > nutrition > environment
Rationale: host context dominates whatever else is done inside it; a genotype difference makes a
contrast genetic even when both arms sit under a drug or a stress; a chemical agent outranks the
medium it was delivered in; hypoxia is the atlas's default reading for contrasts with no other
signal, so it sits last and catches the remainder via the time-only house rule.
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run from any cwd

import contrast_features as cfx

# Measured on 604 reviewed contrasts: this order beats drug-first (0.907 vs 0.881).
PRECEDENCE = ["infection", "genetic", "drug", "nutrition", "environment"]

TIME_ONLY = re.compile(r"^[^a-z]*(t|d|day|hr?|hour|time|tp)[ _-]?\d+(\.\d+)?[ _-]*(h|hr|hrs|d|min)?"
                       r"[ _-]*vs[ _-]*(t0|d0|control|con|ctrl|untreated|0h?|time ?0|reference)",
                       re.I)

# Reference strain names. A strain name is evidence of a GENETIC contrast only when the two arms
# carry DIFFERENT ones -- 'Beijing isolate vs control' varies the infection, not the genotype,
# whereas 'CDC1551 vs H37Rv at the same NO dose' varies nothing but the strain.
STRAIN_NAMES = {"h37rv", "h37ra", "cdc1551", "erdman", "beijing", "hn878", "mtb", "bcg",
                "cdc1550", "f11", "haarlem", "w4", "ra", "rv"}
CONTROL_TOKENS = {"control", "ctrl", "con", "untreated", "mock", "vehicle", "reference", "ref",
                  "parent", "wt", "wildtype", "none", "dmso", "t0", "d0", "0"}
TIME_TOKENS = re.compile(r"^\d+(\.\d+)?(h|hr|hrs|d|day|days|min|m)?$|^(t|d|day|tp)\d+$", re.I)
WT_TOKEN = re.compile(r"^(wt|wild|wildtype)$", re.I)


LOCUS_TAG = re.compile(r"^(rv|mra_|mt|mb)\d{3,4}[abc]?$", re.I)
GENOTYPE_MARKER = re.compile(r"\b(ko|knock ?out|mutant|delta|deleted|deletion|::|comp|complement)", re.I)


def differing_gene_identity(left, right, test_txt, ctrl_txt, known=()):
    """Both arms are mutants but of DIFFERENT genes -> genetic, by the arm-difference rule.

    'dosT_KO vs dosS_KO' shares the token KO, so a shared-token filter sees no difference; the
    difference is which gene is knocked out, and the arm-unique tokens are the gene identifiers
    (dost / doss, rv0307c / rv2680). This rule reads those: a genotype marker present on BOTH
    arms plus distinct identifier-like tokens unique to each arm.
    """
    if not (GENOTYPE_MARKER.search(test_txt) and GENOTYPE_MARKER.search(ctrl_txt)):
        return None, None
    def ids(toks):
        # a token already explained by a lexicon (sds, log, cu) is a condition, not a gene name
        toks = [t for t in toks if not any(rx.search(t) for rx in known)]
        return {t for t in toks
                if LOCUS_TAG.match(t) or (3 <= len(t) <= 7 and t.isalnum() and not t.isdigit()
                                          and not TIME_TOKENS.match(t)
                                          and t not in CONTROL_TOKENS
                                          and not GENOTYPE_MARKER.match(t))}
    li, ri = ids(left), ids(right)
    if li and ri and li != ri:
        return "knockout", "%s vs %s (same genotype marker on both arms)" % (
            "+".join(sorted(li)[:2]), "+".join(sorted(ri)[:2]))
    if li and not ri:
        return "complementation", "%s added on one arm only" % "+".join(sorted(li)[:2])
    return None, None


POINT_MUT = re.compile(r"^[a-z]\d{1,4}[a-z]$", re.I)          # i27f, k125a, r47e
DESIGNATION = re.compile(r"^[a-z]{1,6}\d{1,5}[a-z]?$", re.I)   # ASR0, ASR1, CDC1551, Rv1720c
GT_TEMPLATE = re.compile(r"genotype[_ ](?P<gt>.+?)[_ ]condition[_ ]factor[_ ](?P<cf>.+)$", re.I)


def template_genetic(comparison_id, known=()):
    """The RNA-seq registry naming convention, read structurally.

    Names of the form 'genotype_A_condition_factor_B_vs_genotype_C_condition_factor_D' encode
    the two arms explicitly. When the differing part is an allele code (i27f = Ile27Phe) or a
    strain designation that no lexicon explains, the contrast varies genotype. Values that a
    lexicon DOES explain (a carbon source, a drug) are left alone -- those are conditions.
    """
    parts = re.split(r"_vs_", str(comparison_id), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None, None
    m1, m2 = GT_TEMPLATE.match(parts[0].strip()), GT_TEMPLATE.match(parts[1].strip())
    if not (m1 and m2):
        return None, None
    for field in ("gt", "cf"):
        a, b = m1.group(field).lower(), m2.group(field).lower()
        if a == b:
            continue
        toks = [t for t in re.findall(r"[a-z0-9.]+", a + " " + b)
                if not any(rx.search(t) for rx in known)]
        if any(POINT_MUT.match(t) for t in toks):
            return "mutant", "%s vs %s (amino-acid substitution code)" % (a, b)
        # PI rule: strain designations that differ while the genotype field is IDENTICAL are
        # mutants of one background, so the contrast varies genotype. Requires the differing
        # token to look like a designation (letters then digits: ASR1, CDC1551), never a plain
        # word -- an unexplained WORD in this field is an unknown condition, not a strain.
        if (field == "cf" and m1.group("gt").lower() == m2.group("gt").lower()
                and any(DESIGNATION.match(t) for t in toks)
                and not any(rx.search(a) or rx.search(b) for rx in known)):
            return "mutant", "%s vs %s (strain designations, genotype field identical)" % (a, b)
    # Deliberately NO general fallback on 'this field differs and no lexicon explains it'.
    # Measured at 0.897 vs 0.930 when unguarded: an unexplained condition_factor is usually an
    # unknown CONDITION. The two branches above fire only on designation-shaped tokens.
    return None, None


def structural_genetic(left, right):
    """Genotype evidence that comes from the STRUCTURE of the arm difference, not a keyword.

    Two rules, both requiring the evidence to sit on opposite arms:
      - different reference strains on the two arms -> a strain comparison
      - wild-type named on one arm while the other names something that is not merely a control
        or a timepoint -> a genotype contrast (an allele code, a mutant id)
    Neither fires when the token appears on both arms, which is what makes this safe where a
    plain keyword match is not.
    """
    ls, rs = left & STRAIN_NAMES, right & STRAIN_NAMES
    if ls and rs and ls != rs:
        return "clinical-isolate lineage", "%s|%s" % ("+".join(sorted(ls)), "+".join(sorted(rs)))
    for a, b in ((left, right), (right, left)):
        if any(WT_TOKEN.match(t) for t in a):
            informative = [t for t in b
                           if t not in CONTROL_TOKENS and not TIME_TOKENS.match(t)]
            if informative:
                # an amino-acid substitution code is a point mutant, not a knockout; anything
                # that is not designation-shaped is a genotype contrast of unstated kind
                if any(POINT_MUT.match(t) for t in informative):
                    sub = "mutant"
                elif any(DESIGNATION.match(t) for t in informative):
                    sub = "mutant"
                else:
                    sub = "unspecified"
                return sub, "wt vs %s" % "+".join(sorted(informative)[:3])
    return None, None


def load_category_lexicon(path):
    lex = pd.read_csv(cfx.resolve_path(path))
    if "tiers" not in lex.columns:
        lex["tiers"] = "any"
    return [(re.compile(p, re.I), c, s, t)
            for p, c, s, t in zip(lex.pattern, lex.category, lex.subcategory, lex.tiers)]


def match_categories(text, cat_lex, drug_lex, tier="name"):
    """Return {category: subcategory} for every category with a hit in this text.

    Patterns marked tiers='name' in the lexicon are only consulted against the contrast name.
    Genotype and strain-lineage terms are marked that way: a deletion mutant or a reference
    strain appearing somewhere in a series' samples says nothing about whether the two arms of
    THIS contrast differ in genotype, and treating it as evidence re-imports the study-level
    error the whole pipeline exists to remove.
    """
    hits, ev = {}, {}
    for rx, cat, sub, allowed in cat_lex:
        if allowed == "name" and tier != "name":
            continue
        if allowed == "name_or_varying" and tier not in ("name", "varying_sample_vocabulary"):
            continue
        m = rx.search(text)
        if m:
            hits.setdefault(cat, sub)
            ev.setdefault(cat, m.group(0))
    dc, agent, _, hint = cfx.match_drug_class(text, drug_lex)
    if dc is not None:
        # the agent's mechanism (drug_class) and the category it implies are separate: an
        # oxidant is environment/oxidative stress, ascorbate is nutrition in this atlas, but
        # both still carry a mechanism class.
        cat = hint or "drug"
        sub = {"drug": "antibiotics", "environment": "oxidative stress",
               "nutrition": "vitamin C"}.get(cat, "antibiotics")
        if cat == "environment" and "detergent" in str(agent):
            sub = "cell-envelope stress"
        hits.setdefault(cat, sub)
        ev.setdefault(cat, agent)
    return hits, dc, agent, ev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparisons", required=True)
    ap.add_argument("--cache", default="geo_cache")
    ap.add_argument("--category-lexicon", default="category_lexicon.csv")
    ap.add_argument("--drug-lexicon", default="drug_lexicon.csv")
    ap.add_argument("--out", default="rules_categories.xlsx")
    ap.add_argument("--precedence", default=",".join(PRECEDENCE),
                    help="comma-separated category precedence when several match")
    ap.add_argument("--use-arm-diff", action="store_true",
                    help="opt-in: add a tier matching only tokens that differ between the arms. "
                         "Measured WORSE than the default on the reviewed labels (0.902 vs "
                         "0.907) -- kept because it should win once test/control strings are "
                         "filled for more than the RNA-seq rows.")
    ap.add_argument("--no-structural", action="store_true",
                    help="ablation: drop the strain-difference and wild-type-on-one-arm rules. "
                         "Costs 0.002 (0.949 -> 0.947). These measured WORSE than off (0.889) "
                         "until they were confined to the fallback path; the earlier damage was "
                         "precedence injection, not the rules themselves.")
    ap.add_argument("--no-summary-tier", action="store_true",
                    help="ablation: never fall back to the series title/summary. Measured much "
                         "worse (0.803): the summary tier is 80%% accurate, not 0%%.")
    ap.add_argument("--no-demote-shared-genotype", action="store_true",
                    help="ablation: keep a genetic call even when the genotype term appears on "
                         "both arms and a different category is supported by arm-unique tokens")
    ap.add_argument("--no-template-genetic", action="store_true",
                    help="ablation: stop reading allele codes (i27f, k125a) out of the "
                         "genotype_X_condition_factor_Y naming convention. Costs 0.007 (0.937 "
                         "-> 0.930) on the reviewed labels.")
    ap.add_argument("--use-gene-identity", action="store_true",
                    help="opt-in: when both arms carry a genotype marker but different gene "
                         "identifiers, call it genetic (dosT_KO vs dosS_KO)")
    ap.add_argument("--gold", default=None,
                    help="reviewed labels (xlsx with a 'full' or 'categories' sheet carrying "
                         "gse, comparison_id, perturbation_category) -- prints accuracy and an "
                         "error breakdown so a lexicon edit can be measured, not argued")
    ap.add_argument("--study", default=None,
                    help="optional study_registry.csv, to report agreement (not ground truth)")
    args = ap.parse_args()

    precedence = [c.strip() for c in args.precedence.split(",")]
    known_rx = [rx for rx, *_ in cfx.load_lexicon(args.drug_lexicon)] + \
               [rx for rx, *_ in load_category_lexicon(args.category_lexicon)]
    cat_lex = load_category_lexicon(args.category_lexicon)
    drug_lex = cfx.load_lexicon(args.drug_lexicon)
    reg = pd.read_csv(args.comparisons)
    if "technology" not in reg.columns:
        reg["technology"] = "unknown"
    X = cfx.build_features(reg, args.cache)

    rows = []
    for r in X.itertuples():
        tiers = [("arm_metadata", r.arm_meta_diff_text, "high"),
                 ("arm_difference", r.diff_text, "high"),
                 ("contrast_name", (r.name_text + " " + r.cond_text).strip(), "medium"),
                 ("varying_sample_vocabulary", r.vocab_varying_text, "medium"),
                 ("all_sample_vocabulary", r.vocab_text, "low"),
                 ("series_summary", r.summary_text, "low")]
        if not args.use_arm_diff:
            tiers = [t for t in tiers if t[0] != "arm_difference"]
        if args.no_summary_tier:
            tiers = [t for t in tiers if t[0] != "series_summary"]
        left = set(str(r.diff_left_text).split())
        right = set(str(r.diff_right_text).split())
        struct_sub, struct_ev = ((None, None) if args.no_structural
                                 else structural_genetic(left, right))
        if not args.no_template_genetic and not struct_sub:
            struct_sub, struct_ev = template_genetic(r.comparison_id, known=known_rx)
        if args.use_gene_identity and not struct_sub:
            struct_sub, struct_ev = differing_gene_identity(
                left, right, str(r.diff_left_text) + " " + str(r.name_text),
                str(r.diff_right_text) + " " + str(r.name_text), known=known_rx)

        hits, ev, dc, agent, tier, conf = {}, {}, None, None, "no_match", "abstain"
        for tname, ttext, tconf in tiers:
            h, d, a, e = match_categories(
                ttext, cat_lex, drug_lex,
                tier="name" if tname in ("arm_metadata", "arm_difference", "contrast_name")
                else tname)
            dc, agent = dc or d, agent or a
            if h:
                hits, ev, tier, conf = h, e, tname, tconf
                break

        # The PI's double-testing rule, in general form: a perturbation present on BOTH arms is
        # background for that contrast, so a category supported by arm-unique tokens outranks a
        # category whose only evidence is shared. 'Rv1720c.mutant_Cholesterol vs
        # Rv1720c.mutant_Glycerol' is carbon source; 'sigE_SDS vs wt_SDS' is genotype.
        if not args.no_demote_shared_genotype and hits and tier in ("contrast_name",
                                                                    "arm_difference"):
            whole = (str(r.name_text) + " " + str(r.cond_text)).lower()
            arm_unique = (str(r.diff_left_text) + " " + str(r.diff_right_text)).lower()
            shared = set()
            for cat, _sub in hits.items():
                hit_txt = str(ev.get(cat, "")).strip().lower()
                if (hit_txt and " " not in hit_txt and hit_txt in whole
                        and hit_txt not in arm_unique):
                    shared.add(cat)
            if shared and len(shared) == len(hits):
                h2, d2, a2, e2 = match_categories(r.diff_text, cat_lex, drug_lex, tier="name")
                gsub, gev = structural_genetic(left, right)
                if gsub and "genetic" not in h2:
                    h2, e2 = dict(h2, genetic=gsub), dict(e2, genetic=gev)
                alt = [c for c in precedence if c in h2 and c not in shared]
                if alt:
                    hits = {c: h2[c] for c in alt}
                    ev = {c: e2.get(c) for c in alt}
                    dc, agent = d2 or dc, a2 or agent
                    tier, conf = "arm_difference_over_shared_evidence", "high"

        # Structural/template genetic evidence is a FALLBACK, not a competitor: injecting it
        # into a hit set that already has condition evidence lets genetic win on precedence and
        # measured 0.929 vs 0.937. It applies only where nothing else explained the contrast.
        if struct_sub and not hits:
            hits["genetic"] = struct_sub
            ev["genetic"] = struct_ev
            tier, conf = "arm_structure", "high"

        ordered = [c for c in precedence if c in hits]
        if len(ordered) > 1:
            conf = "low"
        if tier in ("all_sample_vocabulary", "series_summary"):
            conf = "low"     # the summary is the study's motivation, not the contrast's content

        if ordered:
            cat, sub = ordered[0], hits[ordered[0]]
        elif TIME_ONLY.match(str(r.comparison_id)) and r.aeration != "aeration_stated":
            cat, sub, tier = "environment", "hypoxia", "house_rule_time_only"
            conf = "high" if r.aeration == "hypoxia_stated" else "medium"
        else:
            cat, sub, tier, conf = None, None, "no_match", "abstain"

        rows.append({
            "gse": r.study_id, "comparison_id": r.comparison_id,
            "perturbation_category": cat, "subcategory": sub,
            "drug_class": dc, "matched_agent": agent,
            "matched_on": tier, "confidence": conf,
            "matched_text": ev.get(ordered[0]) if ordered else None,
            "all_matches": "; ".join("%s=%s" % (k, v) for k, v in ev.items()),
            "all_categories_matched": "|".join(ordered),
            "n_categories_matched": len(ordered),
            "aeration": r.aeration, "technology": r.technology,
            "series_metadata_available": bool(r.has_series_metadata),
        })

    full = pd.DataFrame(rows)
    full["needs_review"] = (full.confidence.isin(["low", "abstain"])
                            | ~full.series_metadata_available)
    simple = full[["gse", "comparison_id", "perturbation_category"]]

    cover = pd.DataFrame({
        "metric": ["contrasts", "studies", "categorised", "abstained",
                   "matched on arm difference", "matched on contrast name",
                   "matched on arm structure", "matched on varying sample vocabulary",
                   "matched on all sample vocabulary (weak)",
                   "matched on series summary (weak)",
                   "via time-only house rule", "multi-category (flagged)",
                   "drug_class assigned", "needs review"],
        "value": [len(full), full.gse.nunique(), int(full.perturbation_category.notna().sum()),
                  int((full.confidence == "abstain").sum()),
                  int((full.matched_on == "arm_difference").sum()),
                  int((full.matched_on == "contrast_name").sum()),
                  int((full.matched_on == "arm_structure").sum()),
                  int((full.matched_on == "varying_sample_vocabulary").sum()),
                  int((full.matched_on == "all_sample_vocabulary").sum()),
                  int((full.matched_on == "series_summary").sum()),
                  int((full.matched_on == "house_rule_time_only").sum()),
                  int((full.n_categories_matched > 1).sum()),
                  int(full.drug_class.notna().sum()), int(full.needs_review.sum())],
    })

    vs = None
    if args.study:
        s = pd.read_csv(args.study)[["study_id", "perturbation_broad_category"]]
        j = full.merge(s, left_on="gse", right_on="study_id", how="left")
        vs = pd.crosstab(j.perturbation_broad_category.fillna("(none)"),
                         j.perturbation_category.fillna("(abstain)"))

    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        simple.to_excel(xl, sheet_name="categories", index=False)
        full.to_excel(xl, sheet_name="full", index=False)
        cover.to_excel(xl, sheet_name="coverage", index=False)
        if vs is not None:
            vs.to_excel(xl, sheet_name="vs_study")

    if args.gold:
        if str(args.gold).lower().endswith(".csv"):
            gsheet = pd.read_csv(args.gold)
        else:
            sheets = pd.read_excel(args.gold, sheet_name=None)
            gsheet = sheets.get("full", sheets.get("categories"))
        g = gsheet[["gse", "comparison_id", "perturbation_category"]].rename(
            columns={"perturbation_category": "reviewed"})
        g["reviewed"] = g.reviewed.astype(str).str.strip().str.lower()
        j = full.merge(g, on=["gse", "comparison_id"], how="inner")
        j["predicted"] = j.perturbation_category.fillna("(abstain)").str.lower()
        j["ok"] = j.predicted == j.reviewed
        print("\nGOLD EVALUATION on %d matched contrasts: accuracy %.3f"
              % (len(j), j.ok.mean()), file=sys.stderr)
        print("\nby tier:\n%s" % j.groupby("matched_on").ok.agg(["mean", "size"]).round(3)
              .to_string(), file=sys.stderr)
        print("\nerrors (rows predicted, cols reviewed):\n%s"
              % pd.crosstab(j[~j.ok].predicted, j[~j.ok].reviewed).to_string(), file=sys.stderr)

    print("wrote %s" % args.out, file=sys.stderr)
    print(cover.to_string(index=False), file=sys.stderr)
    print("\ncategory counts:\n%s"
          % full.perturbation_category.value_counts(dropna=False).to_string(), file=sys.stderr)
    if vs is not None:
        print("\nvs study-level label (reference, not ground truth):\n%s"
              % vs.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
