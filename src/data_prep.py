"""Loading and cleaning of the main application table."""

import numpy as np
import pandas as pd


def load_application(path):
    return pd.read_csv(path)


def clean_application(df):
    df = df.copy()

    # 365243 is a placeholder for "not employed" -> treat as missing
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # DAYS_* are negative day-counts back from the application date
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365

    return df
