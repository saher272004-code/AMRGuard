"""
AMR Guard — FastAPI backend (REAL data, wired to AMRGuardPanel)
================================================================
Serves the real per-country recommender, with allergy filtering,
patient-condition caution flags, WHO AWaRe stewardship tiers, and a
discontinued-drug flag.

Run:
    pip install fastapi uvicorn --break-system-packages
    uvicorn main:app --reload --port 8000

Needs, in the same folder:
    - recommend.py      (AMRGuardPanel)
    - amr_models/       (trained models + metadata + species_prevalence.csv)

Endpoints:
    GET /api/countries
    GET /api/species/{country}
    GET /api/allergies
    GET /api/conditions
    GET /api/recommend-known?country=..&species=..&allergies=..&conditions=..
    GET /api/recommend-empiric?country=..&allergies=..&conditions=..

Caution data sources: FDA labels via DailyMed (US drugs); UK eMC/MHRA SmPC for
teicoplanin (not FDA-approved). This is decision-support only — a starting draft
requiring clinical review, NOT a substitute for a full drug reference.
"""

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from recommend import AMRGuardPanel

app = FastAPI(title="AMR Guard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load the real trained panel once at startup ----
MODELS_DIR = "amr_models"
panel = AMRGuardPanel(MODELS_DIR)

# ---- WHO AWaRe classification, 2023 (verified vs WHO-MHP-HPS-EML-2023.04) ----
AWARE = {
    "Ampicillin sulbactam": "Access", "Oxacillin": "Access", "Ampicillin": "Access",
    "Penicillin": "Access", "Amoxy/clav": "Access", "Amikacin": "Access",
    "Gentamicin": "Access", "Trimethoprim sulfa": "Access",
    "Ciprofloxacin": "Watch", "Levofloxacin": "Watch", "Pip/taz": "Watch",
    "Erythromycin": "Watch", "Minocycline": "Watch", "Moxifloxacin": "Watch",
    "Teicoplanin": "Watch", "Vancomycin": "Watch", "Azithromycin": "Watch",
    "Clarithromycin": "Watch", "Cefepime": "Watch", "Ceftazidime": "Watch",
    "Imipenem": "Watch", "Meropenem": "Watch", "Ceftriaxone": "Watch",
    "Doripenem": "Watch", "Ertapenem": "Watch", "Ceftibuten": "Watch",
    "Colistin": "Reserve", "Linezolid": "Reserve", "Ceftaroline": "Reserve",
    "Ceftazidime avibactam": "Reserve", "Ceftolozane tazobactam": "Reserve",
    "Meropenem vaborbactam": "Reserve", "Cefiderocol": "Reserve",
}

# ---- Antibiotics discontinued/withdrawn in many markets ----
# Flag but don't remove — they remain in the surveillance data, but are often
# not obtainable in practice.
DISCONTINUED = {
    "Doripenem": "Discontinued or withdrawn in several countries — verify local availability",
}

# ---- Allergy classes exposed to the frontend as checkboxes ----
ALLERGY_CLASSES = [
    {"id": "penicillins", "name": "Penicillins"},
    {"id": "cephalosporins", "name": "Cephalosporins"},
    {"id": "carbapenems", "name": "Carbapenems"},
    {"id": "sulfonamides", "name": "Sulfonamides (e.g. Trimethoprim-sulfa)"},
    {"id": "fluoroquinolones", "name": "Fluoroquinolones"},
    {"id": "macrolides", "name": "Macrolides"},
    {"id": "glycopeptides", "name": "Glycopeptides (Vancomycin/Teicoplanin)"},
    {"id": "aminoglycosides", "name": "Aminoglycosides"},
    {"id": "tetracyclines", "name": "Tetracyclines"},
]

# ---- Every antibiotic your models can output, mapped to a drug class ----
DRUG_TO_CLASS = {
    "Ampicillin sulbactam": "penicillins", "Ampicillin": "penicillins",
    "Penicillin": "penicillins", "Amoxy/clav": "penicillins",
    "Oxacillin": "penicillins", "Pip/taz": "penicillins",
    "Cefepime": "cephalosporins", "Ceftazidime": "cephalosporins",
    "Ceftriaxone": "cephalosporins", "Ceftibuten": "cephalosporins",
    "Ceftaroline": "cephalosporins", "Ceftazidime avibactam": "cephalosporins",
    "Ceftolozane tazobactam": "cephalosporins", "Cefiderocol": "cephalosporins",
    "Imipenem": "carbapenems", "Meropenem": "carbapenems",
    "Doripenem": "carbapenems", "Ertapenem": "carbapenems",
    "Meropenem vaborbactam": "carbapenems",
    "Ciprofloxacin": "fluoroquinolones", "Levofloxacin": "fluoroquinolones",
    "Moxifloxacin": "fluoroquinolones",
    "Amikacin": "aminoglycosides", "Gentamicin": "aminoglycosides",
    "Trimethoprim sulfa": "sulfonamides",
    "Erythromycin": "macrolides", "Azithromycin": "macrolides",
    "Clarithromycin": "macrolides",
    "Vancomycin": "glycopeptides", "Teicoplanin": "glycopeptides",
    "Minocycline": "tetracyclines",
    "Linezolid": "oxazolidinones",
    "Colistin": "polymyxins",
}

# ---- Patient conditions exposed to the frontend ----
CONDITIONS = [
    {"id": "renal", "name": "Renal impairment"},
    {"id": "hepatic", "name": "Hepatic impairment"},
    {"id": "pregnancy", "name": "Pregnancy"},
    {"id": "elderly", "name": "Elderly (65+)"},
]

# ---- Condition-based caution flags ----
# Sourced from FDA labels (DailyMed) and, for teicoplanin, the UK eMC/MHRA SmPC.
# NOT a blanket toxicity badge — only specific, documented cautions tied to a
# patient condition the prescriber selects. Decision-support draft; review before
# clinical use. Cross-reactivity flags (e.g. penicillin->cephalosporin) are
# deliberately excluded (evidence too imprecise for a flat class-to-class flag).
# drug -> { condition_id: reason }
def _c(level, reason):
    return {"level": level, "reason": reason}

DRUG_CAUTIONS = {
    # ---- Penicillins ----
    "Ampicillin sulbactam": {
        "renal":   _c("routine",  "Renally cleared — dose adjustment needed"),
        "hepatic": _c("moderate", "Hepatic dysfunction (hepatitis, cholestatic jaundice) reported — monitor"),
    },
    "Ampicillin": {
        "renal": _c("routine", "Renally cleared — dose adjustment needed"),
    },
    "Penicillin": {
        "renal":   _c("routine", "Renally cleared — dose adjustment needed"),
        "elderly": _c("routine", "Consider renal decline and sodium load"),
    },
    "Amoxy/clav": {
        "renal":   _c("routine",  "Renally cleared — dose adjustment needed"),
        "hepatic": _c("moderate", "Most commonly implicated antibiotic in drug-induced liver injury "
                                  "(rare in absolute terms, cholestatic pattern)"),
        "elderly": _c("routine",  "Consider renal decline"),
    },
    "Pip/taz": {
        "renal":   _c("moderate", "Renally cleared; added nephrotoxicity risk if combined with vancomycin"),
        "elderly": _c("routine",  "Consider renal decline"),
    },
    "Oxacillin": {
        "renal":   _c("moderate", "Interstitial nephritis reported"),
        "elderly": _c("routine",  "Consider renal decline and sodium load"),
    },

    # ---- Cephalosporins ----
    "Cefepime": {
        "renal":     _c("routine",  "Renally cleared — dose adjustment needed"),
        "elderly":   _c("serious",  "Neurotoxicity risk (confusion, myoclonus, seizures), amplified by reduced renal clearance"),
        "pregnancy": _c("routine",  "Category B — crosses placenta; use only if clearly needed"),
    },
    "Ceftazidime": {
        "renal": _c("moderate", "Renally cleared; elevated levels in renal impairment can cause seizures, encephalopathy"),
    },
    "Ceftriaxone": {},
    "Ceftibuten":             {"renal": _c("routine", "Renally cleared — dose adjustment needed")},
    "Ceftaroline":            {"renal": _c("routine", "Renally cleared — dose adjustment needed")},
    "Ceftazidime avibactam":  {"renal": _c("routine", "Renally cleared — dose adjustment needed")},
    "Ceftolozane tazobactam": {"renal": _c("moderate", "Renally cleared; reduced efficacy at CrCl 30–50, monitor daily")},
    "Cefiderocol":            {"renal": _c("routine", "Renally cleared — dose adjustment needed")},

    # ---- Carbapenems ----
    "Imipenem": {
        "renal":   _c("moderate", "Renally cleared; seizure risk rises in renal impairment/elderly"),
        "elderly": _c("moderate", "Seizure risk increases with age-related renal decline"),
    },
    "Meropenem": {
        "renal":   _c("routine", "Renally cleared — dose adjustment needed"),
        "elderly": _c("routine", "Sodium load relevant in heart failure"),
    },
    "Doripenem":             {"renal": _c("routine", "Renally cleared — dose adjustment needed")},
    "Ertapenem":             {"renal": _c("routine", "Renally cleared — dose adjustment needed")},
    "Meropenem vaborbactam": {
        "renal":     _c("routine", "Renally cleared — dose adjustment needed"),
        "pregnancy": _c("serious", "Vaborbactam caused fetal malformations in animals — advise of fetal risk"),
    },

    # ---- Aminoglycosides ----
    "Amikacin": {
        "renal":     _c("moderate", "Nephrotoxic — renal monitoring and dose adjustment required"),
        "pregnancy": _c("serious",  "Avoid unless severe/life-threatening — fetal ototoxicity risk"),
        "elderly":   _c("moderate", "Higher nephro/ototoxicity risk with age-related renal decline"),
    },
    "Gentamicin": {
        "renal":     _c("moderate", "Nephrotoxic — renal monitoring and dose adjustment required"),
        "pregnancy": _c("serious",  "Avoid unless severe/life-threatening — fetal ototoxicity risk"),
        "elderly":   _c("moderate", "Higher nephro/ototoxicity risk with age-related renal decline"),
    },

    # ---- Glycopeptides ----
    "Vancomycin": {
        "renal":   _c("moderate", "Nephrotoxic — requires level monitoring and dose adjustment"),
        "elderly": _c("moderate", "Higher nephrotoxicity risk with age-related renal decline"),
    },
    "Teicoplanin": {
        "renal":     _c("moderate", "Dose-dependent nephrotoxicity — renal monitoring and dose adjustment needed"),
        "pregnancy": _c("moderate", "Limited data — avoid unless clearly needed"),
        "elderly":   _c("moderate", "Higher nephrotoxicity risk with age-related renal decline"),
    },

    # ---- Polymyxins ----
    "Colistin": {
        "renal":     _c("serious",  "Dose-dependent nephrotoxicity + neurotoxicity (paresthesias; apnea at high levels)"),
        "pregnancy": _c("moderate", "Category C — fetal effects in animals; use only if benefit justifies risk"),
        "elderly":   _c("moderate", "Higher nephro/neurotoxicity risk with age-related renal decline"),
    },

    # ---- Fluoroquinolones (FDA boxed-warning class) ----
    "Ciprofloxacin": {
        "renal":     _c("routine",  "Renally cleared — dose adjustment needed"),
        "pregnancy": _c("moderate", "Generally avoided (cartilage/joint effects in animals; alternatives preferred)"),
        "hepatic":   _c("moderate", "Severe hepatotoxicity reported (rare, esp. >55 yr)"),
        "elderly":   _c("serious",  "Boxed warning: tendon rupture, QT prolongation, CNS effects, aortic aneurysm"),
    },
    "Levofloxacin": {
        "renal":     _c("routine",  "Renally cleared — dose adjustment needed"),
        "pregnancy": _c("moderate", "Category C — avoid unless benefit justifies risk"),
        "hepatic":   _c("moderate", "Severe hepatotoxicity reported (fatal cases, mostly >65 yr)"),
        "elderly":   _c("serious",  "Boxed warning: tendon rupture, QT prolongation, CNS effects, aortic aneurysm"),
    },
    "Moxifloxacin": {
        "pregnancy": _c("moderate", "Avoid — cartilage/joint effects in animals"),
        "hepatic":   _c("moderate", "Hepatotoxicity reported — use with caution"),
        "elderly":   _c("serious",  "Boxed warning: tendon rupture, greatest QT prolongation in class, CNS effects"),
    },

    # ---- Sulfonamides ----
    "Trimethoprim sulfa": {
        "renal":     _c("routine",  "Renally cleared — dose adjustment needed"),
        "hepatic":   _c("moderate", "Hepatotoxicity reported"),
        "pregnancy": _c("serious",  "Avoid 1st trimester (folate antagonism) and near term (kernicterus risk)"),
    },

    # ---- Macrolides ----
    "Erythromycin": {
        "hepatic": _c("moderate", "Cholestatic hepatitis reported"),
        "elderly": _c("moderate", "QT prolongation / torsades risk; major CYP3A4 interactions"),
    },
    "Clarithromycin": {
        "renal":     _c("routine",  "Dose-adjust in severe renal impairment"),
        "hepatic":   _c("moderate", "Hepatotoxicity reported (rare fatal)"),
        "pregnancy": _c("serious",  "Avoid in pregnancy unless no alternative (embryo-fetal effects in animals)"),
        "elderly":   _c("moderate", "QT prolongation / torsades risk; colchicine toxicity"),
    },
    "Azithromycin": {
        "hepatic": _c("moderate", "Hepatotoxicity reported (some fatal)"),
        "elderly": _c("moderate", "QT prolongation / torsades risk; short-term cardiovascular death signal"),
    },

    # ---- Tetracyclines ----
    "Minocycline": {
        "hepatic":   _c("moderate", "Hepatotoxicity reported, especially IV use"),
        "pregnancy": _c("serious",  "Contraindicated — fetal bone/teeth effects, maternal hepatic risk"),
    },

    # ---- Oxazolidinones ----
    "Linezolid": {
        "hepatic": _c("moderate", "Use with caution; lactic acidosis and hepatic effects reported"),
    },
}

def apply_condition_cautions(rows: list, condition_ids: list) -> list:
    """Add condition_cautions: [{condition, level, reason}, ...] to each row.
    Nothing is removed — this is informational. `level` is routine|moderate|serious."""
    for row in rows:
        cautions = DRUG_CAUTIONS.get(row["drug"], {})
        row["condition_cautions"] = [
            {"condition": cid, "level": cautions[cid]["level"], "reason": cautions[cid]["reason"]}
            for cid in condition_ids if cid in cautions
        ]
    return rows


def apply_allergy_filter(rows: list, allergy_ids: list, exclude: bool = False) -> tuple:
    """Flag (or optionally remove) drugs whose class matches a patient allergy.
    Returns (rows, excluded_count)."""
    if not allergy_ids:
        for row in rows:
            row["allergy_flag"] = False
        return rows, 0

    allergy_set = set(allergy_ids)
    result, excluded_count = [], 0
    for row in rows:
        is_match = DRUG_TO_CLASS.get(row["drug"]) in allergy_set
        row["allergy_flag"] = is_match
        if is_match and exclude:
            excluded_count += 1
            continue
        result.append(row)
    return result, excluded_count


def _df_to_json(df: pd.DataFrame, scol: str, allergy_ids: list = None,
                exclude_allergens: bool = False, condition_ids: list = None) -> dict:
    """Reshape a recommender DataFrame into the JSON the frontend renders."""
    rows = []
    for _, r in df.iterrows():
        n = r.get("n_isolates")
        rows.append({
            "drug": r["antibiotic"],
            "susceptibility_pct": round(float(r[scol]) * 100, 1),
            "confidence": r.get("confidence", "Unknown"),
            "n_isolates": int(n) if pd.notna(n) else None,
            "aware": AWARE.get(r["antibiotic"], ""),
            "discontinued": DISCONTINUED.get(r["antibiotic"]),
            "coverage_pct": (round(float(r["isolate_coverage"]) * 100, 0)
                             if "isolate_coverage" in r and pd.notna(r["isolate_coverage"])
                             else None),
        })

    rows, excluded_count = apply_allergy_filter(rows, allergy_ids or [], exclude=exclude_allergens)
    rows = apply_condition_cautions(rows, condition_ids or [])

    return {
        "year_used": int(df["year_used"].iloc[0]),
        "recommendations": rows,
        "excluded_for_allergy": excluded_count,
    }


def _parse_csv(value: str) -> list:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@app.get("/api/countries")
def get_countries():
    return [{"id": c, "name": c} for c in panel.available_countries()]


@app.get("/api/species/{country}")
def get_species(country: str):
    if country not in panel.meta:
        raise HTTPException(status_code=404, detail=f"Unknown country '{country}'")
    return sorted(panel.meta[country]["species_categories"])


@app.get("/api/allergies")
def get_allergies():
    return ALLERGY_CLASSES


@app.get("/api/conditions")
def get_conditions():
    return CONDITIONS


@app.get("/api/recommend-known")
def recommend_known(
    country: str = Query(...),
    species: str = Query(...),
    top_n: int = Query(5, ge=1, le=10),
    allergies: str = Query(None),
    exclude_allergens: bool = Query(False),
    conditions: str = Query(None),
):
    try:
        df = panel.recommend_known_pathogen(country, species, top_n=max(top_n * 3, 10))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    result = _df_to_json(df, "predicted_susceptibility", _parse_csv(allergies),
                         exclude_allergens, _parse_csv(conditions))
    result["recommendations"] = result["recommendations"][:top_n]
    result["country"] = country
    result["species"] = species
    return result


@app.get("/api/recommend-empiric")
def recommend_empiric(
    country: str = Query(...),
    top_n: int = Query(5, ge=1, le=10),
    allergies: str = Query(None),
    exclude_allergens: bool = Query(False),
    conditions: str = Query(None),
):
    try:
        df = panel.recommend_unknown_pathogen(country, top_n=max(top_n * 3, 10))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    result = _df_to_json(df, "expected_susceptibility", _parse_csv(allergies),
                         exclude_allergens, _parse_csv(conditions))
    result["recommendations"] = result["recommendations"][:top_n]
    result["country"] = country
    return result