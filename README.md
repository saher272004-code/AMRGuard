**AMRGuard — Antibiotic Decision Support**  
Outsmarting resistance, one prescription at a time.  
AMRGuard is a clinician-facing decision-support tool that recommends antibiotics using real regional antimicrobial-resistance (AMR) surveillance data. It ranks antibiotics by predicted effectiveness for a given country and, optionally, a known pathogen, while surfacing the confidence, data recency, stewardship tier, and patient-specific cautions a clinician needs to judge how far to trust each recommendation.  
\> Decision support only. AMRGuard is designed to inform prescribing, not to  
\> replace clinical judgement, local antibiograms, or culture results.  
\---  
**The Problem**  
Antimicrobial resistance is one of the leading global health threats, associated with over a million deaths per year. A core driver is empiric prescribing, clinicians frequently must choose an antibiotic before culture and susceptibility results are available, effectively prescribing blind. Poor empiric choices harm patients and accelerate resistance. Local resistance patterns vary widely by country, yet clinicians often lack an accessible, data-driven way to inform that first decision.  
**The Solution**  
AMRGuard supports two real clinical workflows:  
**Known pathogen**: rank antibiotics for a specific identified organism in a given country.  
**Empiric (unknown pathogen):** rank antibiotics across the pathogens actually circulating in that country, weighted by how much of the local infection landscape each drug covers.  
**Each recommendation carries:**

* Predicted susceptibility (a model output, not a raw observed rate)  
* Confidence tier derived from the number of isolates behind the prediction  
* Data recency (which year the prediction reflects)  
* WHO AWaRe stewardship tier (Access / Watch / Reserve, 2023 classification)  
* Patient-condition cautions (renal / hepatic / pregnancy / elderly),  
  severity-tiered so routine notes stay quiet and serious warnings stand out  
* Allergy flagging by drug class

\---  
**Technical Approach**  
**Models**  
One XGBoost classifier per country (11 Asian countries), trained on a weighted-binary encoding of aggregate susceptibility data, using species, antibiotic, and year as features.  
**Validation**  
Models are validated by temporal holdout  trained on earlier years, tested on the most recent year (deployment-realistic, no leakage). We validated the models four ways:

* Per-country baseline comparison vs. a naive historical-average predictor.  
* Pooled sparse-vs-rich analysis: the model's advantage over the baseline  
  grows dramatically on data-sparse pathogen–antibiotic combinations.  
* Three-way benchmark against a Bayesian hierarchical (partial-pooling)  
  model we implemented specifically to test whether principled pooling could  
  beat gradient boosting on sparse data.  
* Overfitting check: performance on combinations never seen in training.

**Headline results (pooled, temporal holdout)**

|           Split |         XGBoost |  Naive baseline	 | Bayesian (pooling) |
| :---- | :---- | :---- | :---- |
| Sparse combos (\<10 isolates) | 0.919	 | 0.431 | 0.804 |
| Rich combos (≥30 isolates) | 0.867 | 0.858 | 0.847 |
| Overall	 | 0.875 | 0.849 | 0.847 |

**Interpretation:** on data-rich combinations all methods perform similarly. On data-sparse combinations the naive baseline collapses (near-random), the Bayesian model recovers through partial pooling, but XGBoost performs best which captures species–antibiotic–country interactions the additive pooling model cannot. The Bayesian model converged cleanly (r̂ ≤ 1.01) and serves as a rigorous benchmark confirming our deployment choice.  
**Not overfitting**: XGBoost achieves 0.919 AUC on combinations it never saw in training which demonstrates genuine generalization to novel cases, on unseen future-year data.  
\---  
**Project Structure**  
amrguard/  
├── README.md  
├── requirements.txt  
├── .gitignore  
├── train.py               \# per-country XGBoost trainer (temporal holdout)  
├── evaluation.py          \# baseline, sparse-vs-rich, three-way, overfitting checks  
├── bayesian.py            \# Bayesian hierarchical benchmark (PyMC)  
├── project\_summary.md  
├── back\_end/  
│   ├── main.py            \# FastAPI backend (serves the real models)  
│   ├── recommend.py       \# AMRGuardPanel — loads models, produces recommendations  
│   └── amr\_models/        \# trained models \+ metadata \+species\_prevalence.csv  
└── front\_end/  
    ├── index.html         \# two-path clinical UI  
    ├── app.js             \# talks to the API, renders recommendations  
    ├── style.css  
    └── CDK1.pdb           \# protein structure for the landing-page visual---  
**Data**  
AMRGuard is trained on antimicrobial resistance surveillance data from ATLAS:  
\> Pfizer. ATLAS (Antimicrobial Testing Leadership and Surveillance).  
\> https://atlas-surveillance.com \[Registration required\]. Accessed 17 August 2026\.  
The dataset is registration-gated and is not redistributed in this repository. To reproduce: register at ATLAS, download the per-country exports for the covered countries, combine them into **\`AMR\_Combined\_AllCountries.csv\`,** and run **\`train.py\`.** Trained models are provided in **\`amr\_models/\`** so the application runs without the raw data.  
Stewardship tiers use the WHO AWaRe classification of antibiotics (2023).  
Drug-caution data is compiled from FDA labelling (DailyMed) and, for teicoplanin, the UK eMC/MHRA SmPC — as a decision-support draft requiring clinical review.  
\---  
**Setup and Usage**  
**1\. Install dependencies**  
\`\`\`bash  
pip install \-r requirements.txt  
\`\`\`  
**2\. (Optional) Retrain the models**  
Only needed if reproducing from raw data. Place \`AMR\_Combined\_AllCountries.csv\`  
in the project root, then:  
\`\`\`bash  
python train.py  
\`\`\`  
This writes trained models, metadata, and \`species\_prevalence.csv\` to \`amr\_models/\`.  
**3\. Run the backend**  
\`\`\`bash  
uvicorn main:app \--reload \--port 8000  
\`\`\`  
Interactive API docs are available at \`http://localhost:8000/docs\`.  
**4\. Run the frontend**  
From the \`frontend/\` folder:  
\`\`\`bash  
python \-m http.server 5500  
\`\`\`  
Open \`http://localhost:5500\` in a browser (Chrome or Edge recommended). Keep the  
backend running in a separate terminal.  
\---  
**Limitations**

* Trained on aggregate surveillance data from 11 Asian countries; scope is limited to countries with sufficient, valid data (Vietnam and Indonesia were excluded for insufficient, single-year data).  
* Predictions reflect each country's most recent available data year, which is surfaced in the interface — some countries' data is older than others.  
* Drug-caution and stewardship data is a decision-support draft and must be confirmed against full prescribing information for the specific patient.  
* AMRGuard is a decision-support prototype, not a validated clinical device.

\---  
**Team**  
\[Team member names — Saher Abbas Fajar Abbas\]  
**Note**  
Built for **ALIBABA CLOUD AI HACKATHON PAKISTAN 2026** . AI coding assistance was used during development; all modelling decisions, methodology, validation, data sourcing, and clinical framing were directed and verified by the team.  
