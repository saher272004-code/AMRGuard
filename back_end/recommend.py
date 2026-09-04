"""
recommend.py — AMR Guard recommender (fixed paths + confidence)
================================================================
Loads the trained per-country panel from MODELS_DIR (default /content/amr_models)
and answers two use cases:

  1. recommend_known_pathogen(country, species)  -> rank antibiotics for a known bug
  2. recommend_unknown_pathogen(country)         -> empiric: rank antibiotics across
                                                    whatever bugs actually circulate

Every recommendation carries:
  - predicted susceptibility (model output, NOT an observed rate)
  - n_isolates  (how much data backs it)
  - confidence  (High/Moderate/Low from n_isolates)
  - year_used   (which year's resistance the prediction reflects)
  - data_quality_warning (country-level caveat, e.g. old/single-year data)

Requires in MODELS_DIR:
  - <Country>.json         (xgboost model)          from trainer
  - <Country>_meta.json    (metadata + combo_support) from trainer
  - species_prevalence.csv (country,species,prevalence) from trainer
"""

import json, os
import pandas as pd
import xgboost as xgb


# ---- confidence tier from isolate count ----
def confidence_from_n(n):
    if n is None:  return "Unknown"
    if n >= 100:   return "High"
    if n >= 30:    return "Moderate"
    if n >= 10:    return "Low"
    return "Very low"


class AMRGuardPanel:
    def __init__(self, models_dir="/content/amr_models"):
        self.models_dir = models_dir
        self.models = {}     # country -> xgb model
        self.meta = {}       # country -> metadata
        self._load_all()

        # species prevalence (for empiric mode)
        prev_path = os.path.join(models_dir, "species_prevalence.csv")
        if os.path.exists(prev_path):
            self.prevalence = pd.read_csv(prev_path)
        else:
            self.prevalence = None
            print(f"WARNING: {prev_path} not found — empiric mode "
                  f"(recommend_unknown_pathogen) will not work.")

        # tested (species, antibiotic) pairs per country
        self.tested_pairs = {
            c: set(tuple(p) for p in m.get("tested_pairs", []))
            for c, m in self.meta.items()
        }
        # per-combo isolate support: country -> {(species, antibiotic): N}
        self.combo_support = {
            c: {tuple(k.split("|||")): v for k, v in m.get("combo_support", {}).items()}
            for c, m in self.meta.items()
        }

    def _load_all(self):
        if not os.path.isdir(self.models_dir):
            raise FileNotFoundError(f"No models dir at {self.models_dir}. Run the trainer first.")
        meta_files = [f for f in os.listdir(self.models_dir) if f.endswith("_meta.json")]
        if not meta_files:
            raise FileNotFoundError(f"No *_meta.json in {self.models_dir}.")
        for mf in meta_files:
            with open(os.path.join(self.models_dir, mf)) as f:
                meta = json.load(f)
            country = meta["country"]
            self.meta[country] = meta
            safe = country.replace(" ", "_").replace(",", "")
            mpath = os.path.join(self.models_dir, f"{safe}.json")
            if os.path.exists(mpath):
                model = xgb.XGBClassifier()
                model.load_model(mpath)
                self.models[country] = model
            # countries with meta but no model were skipped (too little data)

    # ---- helpers ----
    def available_countries(self):
        return sorted(self.models.keys())

    def _check_country(self, country):
        if country not in self.meta:
            raise ValueError(f"'{country}' not in panel. Available: {self.available_countries()}")
        if country not in self.models:
            raise ValueError(f"'{country}' had too little data to train "
                             f"({self.meta[country].get('warning')})")

    def _default_year(self, country):
        return max(self.meta[country]["years"])

    def _combo_n(self, country, species, antibiotic):
        return self.combo_support.get(country, {}).get((species, antibiotic))

    def _predict_frame(self, country, species_list, antibiotic_list, year):
        meta = self.meta[country]
        sp_dtype  = pd.CategoricalDtype(categories=meta["species_categories"])
        abx_dtype = pd.CategoricalDtype(categories=meta["antibiotic_categories"])
        return pd.DataFrame({
            "species":    pd.Series(species_list, dtype=str).astype(sp_dtype),
            "antibiotic": pd.Series(antibiotic_list, dtype=str).astype(abx_dtype),
            "year":       year,
        })

    # ---- use case 1: pathogen known ----
    def recommend_known_pathogen(self, country, species, year=None, top_n=5):
        self._check_country(country)
        meta = self.meta[country]
        if species not in meta["species_categories"]:
            raise ValueError(f"'{species}' never tested in {country}. "
                             f"Known: {meta['species_categories'][:10]} ...")
        year = year or self._default_year(country)
        antibiotics = sorted({ab for (sp, ab) in self.tested_pairs[country] if sp == species})
        if not antibiotics:
            raise ValueError(f"'{species}' has no susceptibility tests on record for {country}.")

        X = self._predict_frame(country, [species]*len(antibiotics), antibiotics, year)
        proba = self.models[country].predict_proba(X)[:, 1]

        result = pd.DataFrame({
            "antibiotic": antibiotics,
            "predicted_susceptibility": proba,
        }).sort_values("predicted_susceptibility", ascending=False).reset_index(drop=True)

        result["n_isolates"] = result["antibiotic"].apply(
            lambda ab: self._combo_n(country, species, ab))
        result["confidence"] = result["n_isolates"].apply(confidence_from_n)
        result["country"]    = country
        result["species"]    = species
        result["year_used"]  = year
        result["rank"]       = result.index + 1
        result["data_quality_warning"] = meta.get("warning") or "None"

        cols = ["rank", "country", "species", "antibiotic", "predicted_susceptibility",
                "n_isolates", "confidence", "year_used", "data_quality_warning"]
        return result[cols].head(top_n)

    # ---- use case 2: pathogen unknown (empiric) ----
    def recommend_unknown_pathogen(self, country, year=None, top_n=5,
                                   min_species_coverage=0.0, min_isolate_coverage=0.3):
        self._check_country(country)
        if self.prevalence is None:
            raise RuntimeError("species_prevalence.csv not loaded — empiric mode unavailable.")
        meta = self.meta[country]
        year = year or self._default_year(country)
        antibiotics = meta["antibiotic_categories"]
        pairs = self.tested_pairs[country]

        prev = self.prevalence[self.prevalence["country"] == country].copy()
        prev = prev.sort_values("prevalence", ascending=False)
        if min_species_coverage > 0.0:
            prev["cum"] = prev["prevalence"].cumsum()
            cutoff = (prev["cum"] >= min_species_coverage).idxmax()
            prev = prev.loc[:cutoff]
        prev = prev[prev["species"].isin(meta["species_categories"])].copy()
        if prev.empty:
            raise ValueError(f"No species prevalence data for '{country}'.")
        pool_total = prev["prevalence"].sum()

        model = self.models[country]
        scores, total_n, species_used, coverage_map, skipped = {}, {}, {}, {}, []

        for ab in antibiotics:
            sub = prev[prev["species"].apply(lambda sp: (sp, ab) in pairs)]
            if sub.empty:
                continue
            coverage = sub["prevalence"].sum() / pool_total
            if coverage < min_isolate_coverage:
                skipped.append(ab); continue

            weights = (sub["prevalence"] / sub["prevalence"].sum()).to_numpy()
            species_list = sub["species"].tolist()
            X = self._predict_frame(country, species_list, [ab]*len(species_list), year)
            proba = model.predict_proba(X)[:, 1]
            scores[ab] = float((proba * weights).sum())
            species_used[ab] = len(species_list)
            coverage_map[ab] = float(coverage)
            ns = [self._combo_n(country, sp, ab) for sp in species_list]
            ns = [n for n in ns if n is not None]
            total_n[ab] = int(sum(ns)) if ns else None

        if not scores:
            raise ValueError(f"No antibiotic in '{country}' meets min_isolate_coverage="
                             f"{min_isolate_coverage}. ({len(skipped)} skipped.)")

        result = pd.DataFrame({
            "antibiotic": list(scores.keys()),
            "expected_susceptibility": list(scores.values()),
        })
        result["n_isolates"]        = result["antibiotic"].map(total_n)
        result["confidence"]        = result["n_isolates"].apply(confidence_from_n)
        result["species_considered"]= result["antibiotic"].map(species_used)
        result["isolate_coverage"]  = result["antibiotic"].map(coverage_map)
        result['empiric_score'] = (
            result['expected_susceptibility'] * result['isolate_coverage']
        )
        result = result.sort_values("empiric_score", ascending=False).reset_index(drop=True)
        result["country"]   = country
        result["year_used"] = year
        result["rank"]      = result.index + 1
        result["data_quality_warning"] = meta.get("warning") or "None"

        cols = ["rank", "country", "antibiotic", "expected_susceptibility",
                "empiric_score","n_isolates", "confidence", "species_considered", "isolate_coverage",
                "year_used", "data_quality_warning"]
        return result[cols].head(top_n)


if __name__ == "__main__":
    panel = AMRGuardPanel("/content/amr_models")
    print("Countries:", panel.available_countries())
    print("\n=== Known: Japan, Staphylococcus aureus ===")
    print(panel.recommend_known_pathogen("Japan", "Staphylococcus aureus", top_n=5))
    print("\n=== Empiric: Japan ===")
    print(panel.recommend_unknown_pathogen("Japan", top_n=5))