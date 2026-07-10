# Credit-Risk Default & Decisioning Engine

Predict a borrower's **probability of default (PD)** and turn that probability into a
**cost-optimal, calibrated, explainable** approve/decline decision — the kind a lending
risk team could actually ship.

> **TL;DR** — A calibrated LightGBM scores each applicant's PD. We **decline when
> PD ≥ 0.09** — a threshold chosen by *minimising expected loss*, not the naive 0.5.
> That operating point **approves ~70% of applicants while cutting expected loss ~42%**
> versus a 0.5 cutoff, and every decline ships with its **top-3 SHAP reason codes**.

---

## The problem

A digital lender (NBFC) must decide, for every incoming loan application, whether to
**approve or decline**. They want to:

1. **Know how risky each applicant is** — a calibrated probability of default, not a black-box yes/no.
2. **Draw the approve/decline line by money** — a threshold that minimizes expected business loss,
   defended with a cost matrix (not a naive 0.5 cutoff).
3. **Explain every decline** — per-applicant reason codes, because regulators require it.

Only ~8% of loans default, so this is a **severe class-imbalance** problem. That single fact drives
every choice below.

## Target, metric & decision rule (defined before any modeling)

- **Target:** `TARGET = 1` if the client had serious repayment difficulty (default), else `0`.
- **Primary metric:** **PR-AUC** (ranking quality on the rare positive class) + **expected cost at the
  chosen threshold t\*** (decision quality).
- **Secondary metrics:** ROC-AUC / Gini, KS statistic, Brier score (for calibration).
- **Decision rule:** approve if `PD < t*`, where `t*` minimizes `FN·C_fn + FP·C_fp`.
  Every decline carries its **top-3 SHAP reason codes**.

Accuracy is **deliberately not used** — a model that predicts "never defaults" would score ~92%
accuracy and approve every bad loan. That number is a trap on imbalanced data.

## Results (held-out validation, 61,503 applicants)

| Metric | Value | Reading |
|---|---|---|
| **PR-AUC** | **0.284** | ~3.5× the 0.081 base rate — real skill at finding rare defaulters |
| ROC-AUC | 0.785 | healthy; not suspiciously high (>0.90 would smell of leakage) |
| Gini | 0.571 | `2·AUC − 1`, the credit-industry scale |
| KS | 0.433 | good separation of defaulters vs repayers |
| Brier (calibrated) | 0.066 | down from 0.174 raw — the PD is a real probability |

**Decision at t\* = 0.09:** approve **70.2%**, expected loss **↓ ~42%** vs a 0.5 cutoff.
The choice is robust: re-picking t\* with the cost of a missed default scaled ±50% keeps it
between 0.04 and 0.12 — always far below 0.5. Full write-up in
[`reports/risk_report.md`](reports/risk_report.md).

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — ~307k loan
applications, ~8% default rate, plus relational satellite tables (bureau, previous applications,
installment payments, etc.) aggregated into applicant-level features (**204 features**).

**Scope control:** start with `application_train.csv` alone to get a full pipeline working
end-to-end, then add the satellite-table features and show each block lifts PR-AUC
(application 0.258 → +bureau → +previous → +installments → +pos → +credit_card 0.281).

### Key columns used

| Column | Meaning / use |
|---|---|
| `SK_ID_CURR` | Application ID — the join key to every satellite table. |
| `TARGET` | Label. `1` = serious repayment difficulty (default). |
| `AMT_CREDIT` | Loan amount — used as the exposure (EAD) in the cost math. |
| `AMT_INCOME_TOTAL` | Declared income — basis for credit-to-income ratio. |
| `AMT_ANNUITY` | Loan installment — basis for annuity-to-income and the FP cost. |
| `EXT_SOURCE_1/2/3` | Normalized external credit scores — the strongest predictors. |
| `DAYS_BIRTH` | Age in days (negative) → converted to `AGE_YEARS`. |
| `DAYS_EMPLOYED` | Employment length in days; `365243` is a "not employed" placeholder → cleaned to `NaN`. |
| `NAME_EDUCATION_TYPE` | Education level — lower education correlates with higher default. |
| `CODE_GENDER` | Applicant gender — used only for the fairness check, never as a model feature by policy. |
| `ORGANIZATION_TYPE` | Employer industry — high-cardinality categorical (frequency-encoded). |

## Pipeline

```
raw tables → feature engineering → imbalance-aware LightGBM → isotonic calibration
          → cost-based threshold t* → SHAP reason codes → risk report + Streamlit scorer
```

Every arrow is a step you can inspect: the `notebooks/` build it up in order, and the reusable
logic lives in `src/`.

## Key figures

| | |
|---|---|
| Feature blocks each lift PR-AUC | `reports/figures/11_feature_block_lift.png` |
| Calibration hugs the diagonal | `reports/figures/12_reliability_curve.png` |
| Expected cost is a U-shape, min at t\* | `reports/figures/13_cost_vs_threshold.png` |
| SHAP global drivers (external scores dominate) | `reports/figures/15_shap_global_bar.png` |
| Fairness by gender (four-fifths = 0.89, passes) | `reports/figures/18_fairness_by_gender.png` |

## The app

A small [Streamlit](https://credit-risk-default-engine.streamlit.app/) scorer (`app/streamlit_app.py`) with two modes:

- **Look up an existing applicant** → see their **calibrated default risk**, the
  **approve/decline** call at t\*, and the **top-3 reasons**;
- **Score a new applicant** → a short form of the handful of fields that actually drive the
  model (income, loan, payment, age, employment, external credit score, payment history); every
  other feature is auto-filled with the training median and the inputs flow through the *same*
  feature engineering, model and SHAP explainer;
- a sidebar **cost-ratio slider** changes how expensive a missed default is and **recomputes
  t\* and the approval rate live** — the core "how do we set the line?" trade-off, made tangible.

Everything reads in **plain language** — reason codes and the verdict are written for a
non-technical reader, with the exact PD/threshold numbers tucked into a "technical detail"
expander.

## Repository structure

```
Credit-Risk Default & Decisioning Engine/
├── data/          raw CSVs (not committed) + processed feature matrix
├── notebooks/     step-by-step build (EDA → FE → model → calibration → SHAP)
├── src/           reusable modules (data_prep, features, model, evaluate, explain, scoring)
├── app/           Streamlit applicant scorer
├── models/        trained LightGBM + calibrated model
├── reports/       risk-decision report + figures
└── tests/         sanity tests (shapes, decision rule, t* reproduces 0.09)
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Launch the scorer** (from the project root):

```bash
streamlit run app/streamlit_app.py
```

**Run the sanity tests:**

```bash
python -m pytest -q
```

The app **works straight from a fresh clone** — the trained models and the validation data it
needs are committed (≈ 21 MB). No rebuild step required.

To reproduce the full analysis from scratch, open the notebooks in `notebooks/` in order (01 → 05).

> **What's committed vs rebuilt:** the repo ships the two models, the full validation set
> (`valid.parquet`), the gender file for the fairness check, and a tiny precomputed
> `feature_medians.parquet` (which stands in for the 38 MB training set the app would otherwise
> only use to compute medians). The large `train.parquet` and the 2.5 GB `data/raw/` stay out of
> git — regenerate them with notebooks 02–04 if you want to retrain.

---

*Project 2 of my Data Science portfolio — predict the probability, decide by the cost, explain every call.*
