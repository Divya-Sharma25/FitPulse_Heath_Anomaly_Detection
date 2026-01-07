import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters


def prepare_tsfresh_input(df: pd.DataFrame) -> pd.DataFrame:
    # Group by day, treat each day as one time series (id)
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["id"] = df["date"].astype(str)

    # For TSFresh we typically need an integer sort index
    df["time"] = df["timestamp"].astype("int64") // 10**9

    # Long format: columns (id, time, kind, value)
    long_df = pd.melt(
        df,
        id_vars=["id", "time"],
        value_vars=["heart_rate_bpm", "step_count", "sleep_hours", "calories_burned"],
        var_name="kind",
        value_name="value",
    )

    return long_df


def extract_tsfresh_features(long_df: pd.DataFrame) -> pd.DataFrame:
    fc_params = EfficientFCParameters()
    features = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_params,
        impute_function=None,
    )
    # Drop columns and rows that are entirely / partially NaN
    features = features.dropna(axis=1, how="all")
    features = features.dropna(axis=0, how="any")
    return features
