"""Feature engineering: credit ratios and satellite-table aggregations.

Each aggregate_* function reads one satellite table (many rows per applicant)
and reduces it to one row per applicant, keyed by SK_ID_CURR. Columns are
prefixed by source (BURO_, PREV_, INST_, POS_, CC_).
"""
import numpy as np
import pandas as pd


def add_credit_ratios(df):
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"]
    return df


def aggregate_bureau(bureau_path, bureau_balance_path):
    # STATUS is '0'..'5' (months past due), 'C' (closed), 'X' (unknown)
    bb = pd.read_csv(bureau_balance_path, usecols=["SK_ID_BUREAU", "STATUS"])
    bb["DPD_LEVEL"] = pd.to_numeric(bb["STATUS"], errors="coerce")
    bb["IS_DPD"] = (bb["DPD_LEVEL"] > 0).astype(int)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MAX_DPD=("DPD_LEVEL", "max"),
        BB_DPD_MONTHS=("IS_DPD", "sum"),
    )

    bureau = pd.read_csv(bureau_path)
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)

    agg = bureau.groupby("SK_ID_CURR").agg(
        BURO_COUNT=("IS_ACTIVE", "count"),
        BURO_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BURO_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BURO_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BURO_AMT_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BURO_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BURO_AMT_OVERDUE_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BURO_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BURO_MAX_DPD=("BB_MAX_DPD", "max"),
        BURO_DPD_MONTHS=("BB_DPD_MONTHS", "sum"),
    )
    agg["BURO_ACTIVE_RATIO"] = agg["BURO_ACTIVE_COUNT"] / agg["BURO_COUNT"]
    return agg


def aggregate_previous(path):
    prev = pd.read_csv(path, usecols=[
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
        "AMT_CREDIT", "CNT_PAYMENT", "DAYS_DECISION",
    ])
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("IS_APPROVED", "count"),
        PREV_APPROVED_RATE=("IS_APPROVED", "mean"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
    )
    return agg


def aggregate_installments(path):
    inst = pd.read_csv(path, usecols=[
        "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT", "AMT_PAYMENT",
    ])
    # positive = payment came in after it was due (late)
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]

    agg = inst.groupby("SK_ID_CURR").agg(
        INST_COUNT=("IS_LATE", "count"),
        INST_LATE_RATE=("IS_LATE", "mean"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
    )
    return agg


def aggregate_pos(path):
    pos = pd.read_csv(path, usecols=[
        "SK_ID_CURR", "SK_DPD", "CNT_INSTALMENT_FUTURE",
    ])
    pos["IS_DPD"] = (pos["SK_DPD"] > 0).astype(int)

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("IS_DPD", "count"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MONTHS=("IS_DPD", "sum"),
        POS_CNT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    )
    return agg


def aggregate_credit_card(path):
    cc = pd.read_csv(path, usecols=[
        "SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD",
    ])
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
    cc["IS_DPD"] = (cc["SK_DPD"] > 0).astype(int)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("IS_DPD", "count"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
        CC_DPD_MONTHS=("IS_DPD", "sum"),
    )
    return agg
