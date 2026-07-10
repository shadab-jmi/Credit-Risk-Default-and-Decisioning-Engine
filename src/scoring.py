"""End-to-end scoring pipeline behind the Streamlit app.

Loads the models and validation applicants, builds per-applicant costs, picks the
cost-optimal threshold t*, and scores a single applicant into a PD, an
approve/decline decision, and reason codes. The maths lives in evaluate and explain.
"""
import os

import joblib
import pandas as pd

from src import evaluate, explain, features


CALIBRATED_MODEL = "models/calibrated_model.joblib"
RAW_MODEL = "models/lgbm_model.joblib"
TRAIN_DATA = "data/processed/train.parquet"
VALID_DATA = "data/processed/valid.parquet"
SENSITIVE_DATA = "data/processed/sensitive.parquet"
MEDIANS_DATA = "data/processed/feature_medians.parquet"


def load_bundle():
    calibrated = joblib.load(CALIBRATED_MODEL)
    raw = joblib.load(RAW_MODEL)

    valid = pd.read_parquet(VALID_DATA)
    y = valid["TARGET"]
    X = valid.drop(columns=["TARGET"])

    gender = pd.read_parquet(SENSITIVE_DATA)["CODE_GENDER"]

    return {
        "calibrated": calibrated,
        "raw": raw,
        "X": X,
        "y": y,
        "gender": gender,
        "features": list(X.columns),
    }


def predicted_pd(calibrated, X):
    return calibrated.predict_proba(X)[:, 1]


def build_costs(X, cost_ratio=1.0, lgd=0.6):
    amt_credit = X["AMT_CREDIT"]
    amt_annuity = X["AMT_ANNUITY"].fillna(X["AMT_ANNUITY"].median())
    c_fn, c_fp = evaluate.applicant_costs(amt_credit, amt_annuity, lgd=lgd)
    return c_fn * cost_ratio, c_fp


def choose_threshold(y, pd_proba, c_fn, c_fp):
    t_star = evaluate.pick_threshold(y, pd_proba, c_fn, c_fp)
    approval = float((pd_proba < t_star).mean())
    return t_star, approval


def decide(pd_value, t_star):
    return "APPROVE" if pd_value < t_star else "DECLINE"


def explain_applicant(raw_model, x_row, feature_names, top_n=3):
    explainer = explain.make_explainer(raw_model)
    shap_values = explain.shap_matrix(explainer, x_row)
    return explain.reason_codes(shap_values[0], feature_names, top_n=top_n)


FORM_FIELDS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "AGE_YEARS", "YEARS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "INST_LATE_RATE", "CC_UTILIZATION_MEAN",
]


def training_medians(medians_path=MEDIANS_DATA, train_path=TRAIN_DATA):
    if os.path.exists(medians_path):
        return pd.read_parquet(medians_path)["median"]
    train = pd.read_parquet(train_path)
    return train.drop(columns=["TARGET"]).median(numeric_only=True)


def build_new_row(inputs, medians):
    row = medians.copy().astype(float)
    row["AMT_INCOME_TOTAL"] = inputs["income"]
    row["AMT_CREDIT"] = inputs["loan"]
    row["AMT_ANNUITY"] = inputs["annuity"]
    row["AMT_GOODS_PRICE"] = inputs["goods_price"]
    row["AGE_YEARS"] = inputs["age"]
    row["YEARS_EMPLOYED"] = inputs["years_employed"]
    # one slider drives all three external bureau scores
    row["EXT_SOURCE_1"] = inputs["ext_score"]
    row["EXT_SOURCE_2"] = inputs["ext_score"]
    row["EXT_SOURCE_3"] = inputs["ext_score"]
    row["INST_LATE_RATE"] = inputs["missed_payment_rate"]
    row["CC_UTILIZATION_MEAN"] = inputs["card_usage"]

    one_row = features.add_credit_ratios(row.to_frame().T)
    return one_row[medians.index]  # keep the model's original column order


def validate_new_inputs(inputs):
    problems = []
    for key, name in [("income", "annual income"), ("loan", "loan amount"),
                      ("annuity", "monthly payment"), ("goods_price", "item price")]:
        if inputs[key] <= 0:
            problems.append(f"{name} must be greater than 0.")
    if not (18 <= inputs["age"] <= 100):
        problems.append("age must be between 18 and 100.")
    if inputs["years_employed"] < 0 or inputs["years_employed"] > inputs["age"] - 16:
        problems.append("employment length must be between 0 and (age − 16) years.")
    if not (0.0 <= inputs["ext_score"] <= 1.0):
        problems.append("external credit score must be between 0 and 1.")
    if inputs["annuity"] > inputs["loan"]:
        problems.append("monthly payment can't be larger than the whole loan.")
    return problems
