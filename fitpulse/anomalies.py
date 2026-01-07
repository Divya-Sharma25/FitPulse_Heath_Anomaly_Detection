import numpy as np
import pandas as pd
from typing import Tuple


def rule_based_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple rule
    df["rule_high_hr"] = df["heart_rate_bpm"] > 160
    df["rule_low_hr"] = df["heart_rate_bpm"] < 45
    df["rule_low_steps_day"] = (df["step_count"] < 100) & (df["activity_type"] != "Sleeping")
    df["rule_high_calories"] = df["calories_burned"] > df["calories_burned"].quantile(0.99)

    df["rule_anomaly"] = (
        df["rule_high_hr"]
        | df["rule_low_hr"]
        | df["rule_low_steps_day"]
        | df["rule_high_calories"]
    )

    return df


def prophet_residual_anomalies(hourly_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    merged = hourly_df.merge(
        forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        left_on="timestamp",
        right_on="ds",
        how="left",
    )

    merged["residual"] = merged["heart_rate_bpm"] - merged["yhat"]
    # Use robust threshold with median absolute deviation
    median_abs = np.median(np.abs(merged["residual"].dropna()))
    threshold = 3 * median_abs if median_abs > 0 else merged["residual"].std() * 3

    merged["model_anomaly"] = merged["residual"].abs() > threshold

    return merged


def cluster_anomalies(features: pd.DataFrame, dbscan_labels) -> pd.Series:
    return pd.Series(dbscan_labels == -1, index=features.index, name="cluster_anomaly")
