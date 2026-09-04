"""
AMRGuard — Model Evaluation
===========================
Validates the trained per-country XGBoost models THREE ways:

  1. Per-country baseline comparison
     (does the model beat a naive historical-average, per country?)

  2. Pooled sparse-vs-rich analysis
     (XGBoost vs baseline, split by how much training data each combo had —
      this is where the model's real value shows: on data-sparse combinations)

  3. Three-way comparison adding the Bayesian hierarchical model
     (XGBoost vs baseline vs Bayesian partial-pooling, on identical rows)

  4. Overfitting check
     (does XGBoost still perform on combinations it NEVER saw in training?)

------------------------------------------------------------------------------
PREREQUISITES (run these first, in the same session / notebook):
  - train.py         -> provides `wb`, `all_results`, `features`, MIN_ROWS_FOR_HOLDOUT
  - bayesian_model.py-> provides a fitted `idata` (the Bayesian posterior)
This is an analysis script meant to be run after the models exist. Sections 3
and 4 require the Bayesian `idata`; sections 1-2 do not.
------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ==========================================================================
# 1. PER-COUNTRY BASELINE COMPARISON
#    Baseline = historical weighted-average susceptibility per (species,
#    antibiotic), computed on the training years, applied to the holdout year.
# ==========================================================================
def baseline_for_country(dfc):
    years = sorted(dfc['year'].unique().tolist())
    if len(dfc) < MIN_ROWS_FOR_HOLDOUT or len(years) < 2:
        return None
    hy = years[-1]
    tr, te = dfc[dfc['year'] < hy], dfc[dfc['year'] == hy]
    if tr.empty or te.empty or te['label'].nunique() < 2:
        return None

    numerator   = (tr['label'] * tr['weight']).groupby([tr['species'], tr['antibiotic']]).sum()
    denominator = tr['weight'].groupby([tr['species'], tr['antibiotic']]).sum()
    combo_avg   = numerator / denominator
    fallback    = (tr['label'] * tr['weight']).sum() / tr['weight'].sum()

    baseline_preds = te.apply(
        lambda row: combo_avg.get((row['species'], row['antibiotic']), fallback),
        axis=1
    )
    return roc_auc_score(te['label'], baseline_preds, sample_weight=te['weight'])


def run_per_country_baseline(wb):
    results = []
    for country in sorted(wb['country'].unique().tolist()):
        dfc = wb[wb['country'] == country].copy()
        results.append({'country': country, 'baseline_auc': baseline_for_country(dfc)})
    return pd.DataFrame(results)


# ==========================================================================
# 2. POOLED SPARSE-vs-RICH: build one pooled test set with
#    model_pred, baseline_pred, and train_support per row.
# ==========================================================================
def build_pooled(wb, all_results, features):
    all_test_rows = []
    for country in all_results.keys():
        dfc = wb[wb['country'] == country].copy()
        years = sorted(dfc['year'].unique().tolist())
        if len(years) < 2 or len(dfc) < 200:
            continue
        hy = years[-1]
        tr = dfc[dfc['year'] < hy].copy()
        te = dfc[dfc['year'] == hy].copy()
        if tr.empty or te.empty or te['label'].nunique() < 2:
            continue

        species_cats = sorted(dfc['species'].unique().tolist())
        abx_cats = sorted(dfc['antibiotic'].unique().tolist())

        numerator   = (tr['label'] * tr['weight']).groupby([tr['species'], tr['antibiotic']]).sum()
        denominator = tr['weight'].groupby([tr['species'], tr['antibiotic']]).sum()
        combo_avg   = numerator / denominator
        fallback    = (tr['label'] * tr['weight']).sum() / tr['weight'].sum()

        te['baseline_pred'] = te.apply(
            lambda r: combo_avg.get((r['species'], r['antibiotic']), fallback), axis=1).values
        model = all_results[country]['model']
        te['model_pred'] = model.predict_proba(features(te, species_cats, abx_cats))[:, 1]
        te['train_support'] = te.apply(
            lambda r: denominator.get((r['species'], r['antibiotic']), 0), axis=1).values

        all_test_rows.append(te)

    pooled = pd.concat(all_test_rows, ignore_index=True)
    print(f"Total pooled test rows: {len(pooled)}")
    return pooled


def compare_model_vs_baseline(pooled):
    def compare_group(subset, name):
        if subset['label'].nunique() < 2:
            print(f"{name}: only one class (n={len(subset)})")
            return
        w = subset['weight']
        m = roc_auc_score(subset['label'], subset['model_pred'], sample_weight=w)
        b = roc_auc_score(subset['label'], subset['baseline_pred'], sample_weight=w)
        print(f"{name}: model={m:.3f}, baseline={b:.3f}, gap={m-b:+.3f} (n={len(subset)})")

    for thresh in [10, 30, 50]:
        print(f"\n--- threshold {thresh} ---")
        compare_group(pooled[pooled['train_support'] >= thresh], f"Rich (>={thresh})")
        compare_group(pooled[pooled['train_support'] <  thresh], f"Sparse (<{thresh})")


# ==========================================================================
# 3. THREE-WAY COMPARISON (adds Bayesian). Requires a fitted `idata`.
# ==========================================================================
def add_bayesian_predictions(pooled, idata):
    # posterior means (point estimates) for reconstructing predictions
    gi = idata.posterior["global_intercept"].mean().item()
    country_off = idata.posterior["country_offset"].mean(dim=["chain", "draw"]).values
    combo_off = idata.posterior["combo_offset"].mean(dim=["chain", "draw"]).values
    country_cats = list(idata.posterior.coords["country"].values)
    combo_cats = list(idata.posterior.coords["combo"].values)
    country_map = {c: i for i, c in enumerate(country_cats)}
    combo_map = {c: i for i, c in enumerate(combo_cats)}

    def predict_single(country, species, antibiotic):
        logit = gi
        ci = country_map.get(country)
        if ci is not None:
            logit += country_off[ci]
        combo_key = species + " ||| " + antibiotic
        combo_i = combo_map.get(combo_key)
        if combo_i is not None:
            logit += combo_off[combo_i]
        return 1.0 / (1.0 + np.exp(-logit))

    pooled = pooled.copy()
    pooled["bayes_pred"] = pooled.apply(
        lambda r: predict_single(r["country"], r["species"], r["antibiotic"]), axis=1)
    return pooled


def compare_three_way(pooled):
    def compare(subset, name):
        if subset["label"].nunique() < 2:
            print(f"{name}: only one class (n={len(subset)})")
            return
        w = subset["weight"]
        m = roc_auc_score(subset["label"], subset["model_pred"],    sample_weight=w)
        b = roc_auc_score(subset["label"], subset["baseline_pred"], sample_weight=w)
        y = roc_auc_score(subset["label"], subset["bayes_pred"],    sample_weight=w)
        print(f"{name:<16} n={len(subset):<5} XGB={m:.3f}  Baseline={b:.3f}  Bayesian={y:.3f}")

    print("\n=== THREE-WAY COMPARISON (identical splits + weights) ===")
    for thresh in [10, 30, 50]:
        print(f"\n--- training support threshold {thresh} ---")
        compare(pooled[pooled["train_support"] >= thresh], f"Rich (>={thresh})")
        compare(pooled[pooled["train_support"] <  thresh], f"Sparse (<{thresh})")
    print("\n=== OVERALL ===")
    compare(pooled, "All combos")


# ==========================================================================
# 4. OVERFITTING CHECK — truly-unseen vs seen-but-sparse combinations.
# ==========================================================================
def overfitting_check(pooled):
    truly_unseen = pooled[pooled["train_support"] == 0]
    seen_sparse  = pooled[(pooled["train_support"] > 0) & (pooled["train_support"] < 30)]
    rich         = pooled[pooled["train_support"] >= 30]

    print("\n=== OVERFITTING CHECK ===")
    print(f"Truly unseen (support=0): {len(truly_unseen)} rows")
    print(f"Seen-but-sparse (<30):    {len(seen_sparse)} rows")
    print(f"Rich (>=30):              {len(rich)} rows")

    def xgb_auc(subset, name):
        if len(subset) == 0 or subset["label"].nunique() < 2:
            print(f"{name}: n={len(subset)} (can't score)")
            return
        m = roc_auc_score(subset["label"], subset["model_pred"], sample_weight=subset["weight"])
        print(f"{name:<26} n={len(subset):<5} XGBoost AUC={m:.3f}")

    xgb_auc(truly_unseen, "Truly unseen (support=0)")
    xgb_auc(seen_sparse,  "Seen-but-sparse (<30)")
    xgb_auc(rich,         "Rich (>=30)")
    print("XGBoost performing well on TRULY UNSEEN combos = genuine generalization,")
    print("not overfitting (and this is a temporal holdout of unseen future data).")


# ==========================================================================
# RUN — assumes wb, all_results, features, MIN_ROWS_FOR_HOLDOUT exist (train.py),
# and idata exists (bayesian_model.py) for sections 3-4.
# ==========================================================================
if __name__ == "__main__":
    print("=== 1. PER-COUNTRY BASELINE ===")
    print(run_per_country_baseline(wb).to_string(index=False))

    print("\n=== 2. POOLED SPARSE vs RICH (model vs baseline) ===")
    pooled = build_pooled(wb, all_results, features)
    compare_model_vs_baseline(pooled)

    # sections 3 & 4 need the Bayesian idata
    try:
        pooled = add_bayesian_predictions(pooled, idata)
        compare_three_way(pooled)
        overfitting_check(pooled)
    except NameError:
        print("\n(Skipping Bayesian three-way + overfitting check: `idata` not in memory.")
        print(" Run bayesian_model.py first, then re-run sections 3-4.)")