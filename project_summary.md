# AMRGuard — Project Summary

## The Problem

Antimicrobial resistance (AMR) is one of the leading global health threats, associated with over a million deaths each year. A core driver is

**empiric prescribing**: clinicians frequently must choose an antibiotic *before* culture and susceptibility results are available, effectively prescribing blind. Poor empiric choices worsen patient outcomes and accelerate resistance. Local resistance patterns vary widely by country, yet clinicians often lack an accessible, data-driven way to inform that first decision.

## Our Solution

**AMRGuard** is a clinician-facing antibiotic decision-support tool. Given a country and optionally a known pathogen. It ranks antibiotics by their predicted effectiveness using real regional resistance surveillance data. It serves two real clinical workflows:

* **Known pathogen:** rank antibiotics for a specific identified organism.
* **Empiric (unknown pathogen):** rank antibiotics across the pathogens actually circulating in that country, weighted by how much of the local infection landscape each drug covers.

## The Need and Impact

AMRGuard supports more informed empiric prescribing where it is hardest; at the point of the first decision. It surfaces **confidence, data recency, antibiotic
stewardship tier (WHO AWaRe), and patient-specific cautions** so clinicians can see not just *what* is recommended but *how much to trust it*. By promoting
appropriate, resistance-aware choices, it supports antimicrobial stewardship; directly addressing the driver of the AMR crisis.

## Innovation and Technology

* **Validated per-country ML models** (gradient-boosted classifiers) trained on ATLAS surveillance data across 11 Asian countries, evaluated by **temporal holdout** (train on past years, test on the newest) a deployment-realistic test.
* **Rigorous benchmarking:** we compared our model against a naive historical-average baseline and a **Bayesian hierarchical (partial-pooling) model**. On data-sparse combinations, the baseline collapses (AUC 0.43), the Bayesian model recovers through pooling (0.80), and our gradient-boosted model performs best (0.92) and we verified this is **genuine generalization, not overfitting** (0.92 AUC on 222 combinations never seen in training).
* **Per-recommendation confidence** derived from isolate counts, **coverage-weighted empiric ranking**, **WHO AWaRe stewardship flags** (2023), and a **severity-tiered clinical caution system** (renal/hepatic/pregnancy/elderly) sourced from FDA labelling and equivalent references.
* **Full-stack, genuinely functional:** a FastAPI backend serving the real models, with a responsive web frontend not a mockup.

## Feasibility and What We Built

AMRGuard is a working end-to-end application: trained and validated models, a live API, and an interactive clinical interface with allergy filtering, patient-
condition cautions, stewardship flags, and adjustable output. It runs on commodity hardware and is honest about its scope and limitations. It is decision *support*, to be confirmed against local guidelines, never a replacement for clinical judgement.

