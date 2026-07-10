"""SHAP explainability and per-applicant reason codes.

SHAP runs on the uncalibrated LightGBM; calibration is a monotonic transform,
so it does not change which features drive the risk.
"""
import numpy as np
import pandas as pd
import shap

FEATURE_LABELS = {
    "EXT_SOURCE_1": "external credit bureau score",
    "EXT_SOURCE_2": "external credit bureau score",
    "EXT_SOURCE_3": "external credit bureau score",
    "CREDIT_INCOME_RATIO": "loan amount vs. income",
    "ANNUITY_INCOME_RATIO": "monthly payment vs. income",
    "CREDIT_GOODS_RATIO": "loan amount vs. item value",
    "ANNUITY_CREDIT_RATIO": "monthly payment vs. loan size",
    "EMPLOYED_AGE_RATIO": "share of life spent employed",
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "loan amount",
    "AMT_ANNUITY": "monthly payment",
    "AMT_GOODS_PRICE": "price of the item financed",
    "AGE_YEARS": "age",
    "YEARS_EMPLOYED": "employment length",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "family size",
    "OWN_CAR_AGE": "age of owned car",
    "REGION_RATING_CLIENT": "home region risk rating",
    "REGION_RATING_CLIENT_W_CITY": "home region/city risk rating",
    "REGION_POPULATION_RELATIVE": "how urban the home region is",
    "DAYS_ID_PUBLISH": "time since ID was updated",
    "DAYS_REGISTRATION": "time since registration",
    "DAYS_LAST_PHONE_CHANGE": "time since last phone change",
    "OBS_30_CNT_SOCIAL_CIRCLE": "known contacts count",
    "OBS_60_CNT_SOCIAL_CIRCLE": "known contacts count",
    "DEF_30_CNT_SOCIAL_CIRCLE": "defaults among known contacts",
    "DEF_60_CNT_SOCIAL_CIRCLE": "defaults among known contacts",
    "OCCUPATION_TYPE_FREQ": "occupation type",
    "ORGANIZATION_TYPE_FREQ": "employer industry",
    "BURO_COUNT": "number of past loans (credit bureau)",
    "BURO_ACTIVE_COUNT": "number of currently active loans",
    "BURO_ACTIVE_RATIO": "share of loans still active",
    "BURO_AMT_CREDIT_SUM_MEAN": "average size of past loans (bureau)",
    "BURO_AMT_CREDIT_SUM_TOTAL": "total borrowed across past loans (bureau)",
    "BURO_AMT_DEBT_MEAN": "average outstanding debt (bureau)",
    "BURO_DAY_OVERDUE_MAX": "worst current overdue days (bureau)",
    "BURO_AMT_OVERDUE_MEAN": "average overdue amount (bureau)",
    "BURO_DAYS_CREDIT_MEAN": "how long ago past loans were taken (bureau)",
    "BURO_MAX_DPD": "worst days-past-due (bureau)",
    "BURO_DPD_MONTHS": "number of past-due months (bureau)",
    "PREV_COUNT": "number of past applications with us",
    "PREV_APPROVED_RATE": "past approval rate with us",
    "PREV_REFUSED_COUNT": "number of past refusals with us",
    "PREV_AMT_APPLICATION_MEAN": "average amount previously requested",
    "PREV_AMT_CREDIT_MEAN": "average amount previously granted",
    "PREV_CNT_PAYMENT_MEAN": "average length of past loans",
    "PREV_DAYS_DECISION_MEAN": "how long ago past decisions were made",
    "INST_COUNT": "number of past installments",
    "INST_LATE_RATE": "history of missed payments",
    "INST_DAYS_LATE_MEAN": "average days late on past payments",
    "INST_DAYS_LATE_MAX": "worst lateness on past payments",
    "INST_PAYMENT_RATIO_MEAN": "how fully past installments were paid",
    "POS_COUNT": "number of past POS/cash loan records",
    "POS_DPD_MEAN": "average days-past-due on POS/cash loans",
    "POS_DPD_MAX": "worst days-past-due on POS/cash loans",
    "POS_DPD_MONTHS": "past-due months on POS/cash loans",
    "POS_CNT_FUTURE_MEAN": "remaining installments on POS/cash loans",
    "CC_COUNT": "number of credit-card records",
    "CC_UTILIZATION_MEAN": "credit card usage level",
    "CC_DPD_MEAN": "average days-past-due on credit cards",
    "CC_DPD_MONTHS": "past-due months on credit cards",
}


ONEHOT_PREFIXES = {
    "NAME_CONTRACT_TYPE_": "loan type",
    "NAME_TYPE_SUITE_": "accompanied by",
    "NAME_INCOME_TYPE_": "income type",
    "NAME_EDUCATION_TYPE_": "education",
    "NAME_FAMILY_STATUS_": "family status",
    "NAME_HOUSING_TYPE_": "housing",
    "WEEKDAY_APPR_PROCESS_START_": "application weekday",
    "FONDKAPREMONT_MODE_": "building fund type",
    "HOUSETYPE_MODE_": "building type",
    "WALLSMATERIAL_MODE_": "wall material",
    "EMERGENCYSTATE_MODE_": "emergency-state flag",
    "FLAG_OWN_CAR_": "car ownership",
    "FLAG_OWN_REALTY_": "property ownership",
}


def make_explainer(model):
    return shap.TreeExplainer(model)


def shap_matrix(explainer, X):
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def global_importance(shap_values, feature_names, top_n=15):
    """Mean absolute SHAP value per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (pd.Series(mean_abs, index=feature_names)
            .sort_values(ascending=False)
            .head(top_n))


def label_for(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, category in ONEHOT_PREFIXES.items():
        if feature.startswith(prefix):
            value = feature[len(prefix):].replace("_", " ").lower()
            return f"{category}: {value}"
    return feature.replace("_", " ").lower()


def reason_codes(shap_row, feature_names, top_n=3):
    """Top features that pushed the applicant's risk up (the decline reasons)."""
    contrib = pd.Series(np.asarray(shap_row), index=feature_names)
    positive = contrib[contrib > 0].sort_values(ascending=False)

    reasons, seen = [], set()
    for feature, value in positive.items():
        label = label_for(feature)
        if label in seen:
            continue
        seen.add(label)
        reasons.append((label, float(value)))
        if len(reasons) == top_n:
            break
    return reasons
