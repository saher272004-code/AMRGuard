"""
AMRGuard — per-country antibiotic susceptibility model trainer.

Trains one XGBoost classifier per country on ATLAS surveillance data, using a
weighted-binary encoding of aggregate susceptibility. Validated by temporal
holdout (train on past years, test on the newest). Saves models, metadata
(including per-combo isolate support for confidence tiers), and species
prevalence for empiric-mode recommendations.

Usage:
    python train.py    # expects AMR_Combined_AllCountries.csv in the same folder
"""
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import os, json, pickle
import pandas as pd

CSV_PATH = "AMR_Combined_AllCountries.csv"
MODELS_DIR = "amr_models"
MIN_N = 10
MIN_ROWS_FOR_HOLDOUT = 200
MIN_ROWS_TO_TRAIN = 20

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    enable_categorical = True,
    tree_method = 'hist',
    eval_metric = 'logloss',
    random_state = 42,
)

def load_clean(path):
  df = pd.read_csv(path)
  need = {"country", "species", "antibiotic", "year", "N", "susceptible_pct", "intermediate_pct" , "resistant_pct"}
  miss = need - set(df.columns)
  if miss:
    raise ValueError(f"Missing colums: {miss}")

  EXCLUDE_COUNTRIES = ["Vietnam", "Indonesia"]
  df = df[~df["country"].isin(EXCLUDE_COUNTRIES)].copy()

  df = df.dropna(subset= ["country","species","antibiotic", "year", "N"]). copy()
  for c in ["country", "species", "antibiotic"]:
    df[c] = df[c].astype(str).str.strip()
  df["year"] = df["year"].astype(int)
  df["N"] = df["N"].astype(float)
  df = df[df["N"] >= MIN_N]
  for c in ["susceptible_pct", "intermediate_pct", "resistant_pct"]:
    df[c] = df[c].clip(0,100)
  return df.reset_index(drop=True)

  #step2:
def to_weighted_binary(df):
  n_sus = (df["N"] * df["susceptible_pct"] / 100.0).round()
  n_non = (df["N"] - n_sus).clip(lower=0)
  cols = ["country", "species", "antibiotic", "year", "N"]
  pos = df[cols].copy(); pos['label'] = 1; pos["weight"] = n_sus
  neg = df[cols].copy(); neg["label"] = 0; neg["weight"] = n_non
  out =pd.concat([pos, neg], ignore_index =True)
  return out[out["weight"] > 0].reset_index(drop=True)
def species_prevalence(df):
    grp = df.groupby(["country", "species"], observed=True)["N"].sum().reset_index()
    grp = grp.rename(columns={"N": "isolate_count"})
    grp["prevalence"] = grp.groupby("country", observed=True)["isolate_count"].transform(
        lambda s: s / s.sum())
    return grp

#step 3
def features(df, species_cats, abx_cats):
  return pd.DataFrame({
      "species": df["species"].astype(pd.CategoricalDtype(species_cats)),
      "antibiotic": df["antibiotic"].astype(pd.CategoricalDtype(abx_cats)),
      "year": df["year"].astype(int),
  })

#step4
def train_country(country, dfc):
  dfc=dfc.copy()
  species_cats=sorted(dfc['species'].unique().tolist())
  abx_cats=sorted(dfc['antibiotic'].unique().tolist())
  years=sorted(dfc['year'].unique().tolist())
  tested_pairs=sorted(set(zip(dfc['species'],dfc['antibiotic'])))
  support = (dfc.groupby(['species','antibiotic'], observed=True)['weight'].sum().round().astype(int))
  combo_support = {f'{sp}|||{ab}': int(n) for (sp, ab), n in support.items()}
  res={'country':country, 'n_rows':len(dfc),
       'n_species':len(species_cats),
       'n_antibiotics':len(abx_cats),
       'years':years,'species_categories':species_cats,
       'antibiotic_categories':abx_cats,
       'combo_support': combo_support,
       'tested_pairs':[list(p)for p in tested_pairs],
       'holdout_auc':None, 'holdout_brier':None, 'holdout_year':None,
       'warning':None, 'model':None
       }
  if len(dfc)<MIN_ROWS_TO_TRAIN:
    res['warning']=f'skipped: only {len(dfc)} rows (<{MIN_ROWS_TO_TRAIN})'
    return res

  do_holdout=len(dfc)>=MIN_ROWS_FOR_HOLDOUT and len(years)>=2
  if do_holdout:
    hy=years[-1]
    tr,te=dfc[dfc['year']<hy], dfc[dfc['year']==hy]
    if tr.empty or te.empty or te['label'].nunique()<2:
      do_holdout=False
    else:
      m=xgb.XGBClassifier(**XGB_PARAMS)
      m.fit(features(tr, species_cats, abx_cats), tr['label'], sample_weight=tr['weight'])
      p=m.predict_proba(features(te, species_cats, abx_cats))[:, 1]
      try:
        res['holdout_auc']=float(roc_auc_score(te['label'], p, sample_weight=te['weight']))
      except ValueError:
        res['holdout_auc']=None
      res['holdout_brier']=float(brier_score_loss(te['label'],p,sample_weight=te['weight']))
      res['holdout_year']=int(hy)
  if not do_holdout and res['warning'] is None:
    res['warning']=f'small/single-year ({len(dfc)} rows, {len(years)} yr): no holdout metric - use with caution'

  final=xgb.XGBClassifier(**XGB_PARAMS)
  final.fit(features(dfc, species_cats, abx_cats), dfc['label'],sample_weight=dfc['weight'])
  res['model']=final
  return res

raw=load_clean(CSV_PATH)
print(f'Clean rows: {len(raw):,}')
wb=to_weighted_binary(raw)
print(f'weighted-binary rows: {len(wb):,}')
os.makedirs(MODELS_DIR, exist_ok=True)

# save species prevalence for empiric-mode recommendations
prevalence = species_prevalence(raw)   
prevalence.to_csv(os.path.join(MODELS_DIR, "species_prevalence.csv"), index=False)
print(f"Saved species_prevalence.csv ({len(prevalence)} rows)")

all_results, report_rows={}, []
countries=sorted(wb['country'].unique().tolist())
print(f'Training {len(countries)} countries....\n')
for country in countries:
  dfc=wb[wb['country']==country].copy()
  res=train_country(country, dfc)
  all_results[country]=res

  report_rows.append({
      'country': country, 'n_rows': res['n_rows'],
      'n_species': res['n_species'], 'n_antibiotics': res['n_antibiotics'],
      'years': f"{min(res['years'])}-{max(res['years'])}" if res['years'] else "",
      'holdout_year': res['holdout_year'], 'holdout_auc': res['holdout_auc'],
      'holdout_brier': res['holdout_brier'], 'warning': res['warning']
  })
  if res['model'] is None:
    print(f'  [{country}] SKIPPED - {res["warning"]}')
    continue
  safe=country.replace(" ",'_').replace(',','')
  res['model'].save_model(os.path.join(MODELS_DIR, f"{safe}.json"))
  with open(os.path.join(MODELS_DIR, f"{safe}.pkl"), 'wb') as f:
    pickle.dump(res['model'], f)
  meta= {k:v for k, v in res.items() if k!='model'}
  with open(os.path.join(MODELS_DIR, f'{safe}_meta.json'),'w') as f:
    json.dump(meta, f, indent=2)
  auc = f"{res['holdout_auc']:.3f}" if res["holdout_auc"] is not None else "n/a"
  print(f"  [{country}] rows={res['n_rows']:>6} species={res['n_species']:>3} "
          f"abx={res['n_antibiotics']:>3} holdout_AUC={auc}")

training_report = pd.DataFrame(report_rows)
training_report.to_csv(os.path.join(MODELS_DIR, "training_report.csv"), index=False)
print(f"\nModels saved to {MODELS_DIR}/")
print(training_report.to_string(index=False))
