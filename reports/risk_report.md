# Credit-Risk Decision Report

**Scenario.** A digital lender must decide, for each incoming application, whether to
**approve** or **decline**. The goal is to minimise expected loss while keeping the
approval rate (revenue) healthy — and to explain every decline, because regulators
require it.

---

## TL;DR — the decision

> We score each applicant's **calibrated probability of default (PD)** with a LightGBM
> model, and **decline when PD ≥ 0.09** — a threshold chosen by minimising expected loss,
> not the default 0.5. At this operating point we **approve ~70% of applicants** and cut
> expected loss by **~42%** versus a 0.5 cutoff. Every decline carries its **top-3 SHAP
> reason codes**.

---

## 1. Model performance (validation set, 61,503 applicants)

Default base rate is **8.1%**, so accuracy is disqualified — a "approve everyone" model
would score 92% accuracy and lose money on every defaulter. We lead with **PR-AUC**.

| Metric | Value | Reading |
|---|---|---|
| **PR-AUC** | **0.284** | ~3.5× the 0.081 base rate — real skill at finding rare defaulters |
| ROC-AUC | 0.785 | healthy; not suspiciously high (>0.90 would signal leakage) |
| Gini | 0.571 | `2·AUC − 1`, the credit-industry scale |
| KS | 0.433 | good separation of defaulters vs repayers |

Ranking is unchanged by calibration (isotonic is monotonic), so these hold for the
deployed calibrated model.

---

## 2. Calibration — the PD is a real probability

Training with `scale_pos_weight` made the raw scores over-dramatic (mean predicted PD
0.37 vs true 0.08). We **isotonic-calibrated inside 5-fold CV** and the probabilities now
match reality — which matters because the PD feeds expected-loss maths, `EL = PD·LGD·EAD`.

| | Raw model | Calibrated |
|---|---|---|
| Mean predicted PD | 0.37 | **0.08** (= base rate) |
| **Brier score** | 0.174 | **0.066** |

See `figures/12_reliability_curve.png` — the calibrated curve hugs the diagonal.

---

## 3. The decision threshold — from a cost matrix, not 0.5

The two mistakes cost very different amounts, and the cost scales with the loan:

| Mistake | Meaning | Cost assumption |
|---|---|---|
| **False negative** | approve a defaulter | `C_fn = LGD × EAD = 0.60 × AMT_CREDIT` (median ≈ 307k) |
| **False positive** | decline a good customer | `C_fp ≈ AMT_ANNUITY` (median ≈ 25k) |

A missed default costs **~12× more** than a false decline, so the cost-minimising threshold
lands far below 0.5.

| Threshold | Approval rate | Expected cost | Defaulters caught |
|---|---|---|---|
| 0.50 (naïve) | 99.5% | 1.62 B | 183 / 4,965 |
| **0.09 (t\*)** | **70.2%** | **0.94 B** | **3,445 / 4,965** |

**Expected loss cut ≈ 42%.** See `figures/13_cost_vs_threshold.png` (a clean U-shape with
its minimum at t\*).

**Confusion matrix at t\*:**

| | approved | declined |
|---|---|---|
| **actually repaid** | 41,680 (TN) | 14,858 (FP) |
| **actually default** | 1,520 (FN) | 3,445 (TP) |

**Sensitivity.** We guessed the costs, so we re-picked t\* with the cost of a missed default
scaled ±50%: t\* moves only between **0.12 and 0.04** — always far below 0.5. The "decline
aggressively" conclusion is robust to the cost assumptions (`figures/14_threshold_sensitivity.png`).

---

## 4. Why each applicant is declined — SHAP reason codes

Global drivers (`figures/15_shap_global_bar.png`, `16_shap_beeswarm.png`): the external
credit scores **EXT_SOURCE_2/3/1** dominate, followed by the annuity/credit and
credit/goods ratios, employment length and past-loan payment behaviour — all sensible risk
factors, no leakage smell.

Every decline gets its **top-3 reason codes** (the features pushing that applicant's risk
up). Example — applicant 156227, PD 0.93, **DECLINE**:

1. low **external credit score 3**
2. low **external credit score 2**
3. high **average overdue amount (credit bureau)**

This is exactly the content an adverse-action notice needs (`figures/17_shap_waterfall.png`).

---

## 5. Fairness sanity check — by gender

Gender is **excluded from the features** (policy), but we still measure outcomes across
groups because correlated features can carry the signal.

| Group | n | Default rate | Approval rate | Missed defaults (FN) | False declines (FP) |
|---|---|---|---|---|---|
| Female | 40,561 | 7.0% | 73.1% | 33.9% | 24.0% |
| Male | 20,940 | 10.2% | 64.7% | 26.3% | 30.9% |

Men default more in the data, so the model approves them somewhat less. The
**four-fifths ratio is 0.89** (≥ 0.80 → **passes** the four-fifths rule). The error mix
differs by group — women see more missed defaults, men more false declines
(`figures/18_fairness_by_gender.png`). We **flag** this honestly rather than claiming the
model is bias-free; a production system would monitor it and consider group-aware
thresholds.

---

## 6. Caveats

- Costs are **stated assumptions** (LGD = 0.6; C_fp ≈ one annuity), not audited figures —
  the ±50% sensitivity check covers that uncertainty. The "42% loss cut" is relative, on
  the validation set, under these assumptions.
- The split is **stratified random** (no clean date field for a true out-of-time split);
  in production we would validate out-of-time and monitor drift (PSI).

---

*Pipeline: raw tables → applicant features → LightGBM (imbalance-aware) → isotonic
calibration → cost-based threshold → SHAP reason codes. Full details in `notebooks/`.*
