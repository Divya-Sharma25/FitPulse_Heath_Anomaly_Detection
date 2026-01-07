import pandas as pd
from pathlib import Path


def load_data(path: str = "data/Sample_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_validation(df: pd.DataFrame) -> None:
    print("=== Dataset Info ===")
    print(df.info())
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Column names ===")
    print(df.columns)
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates
    df = df.drop_duplicates()

    # Timestamp normalization
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    numeric_cols = ["heart_rate_bpm", "sleep_hours", "step_count", "calories_burned"]
    df[numeric_cols] = df[numeric_cols].ffill()
    df["activity_type"] = df["activity_type"].ffill()

    print("\nMissing values after handling:")
    print(df.isnull().sum())

    # Outlier clipping (same as in notebook)
    df["heart_rate_bpm"] = df["heart_rate_bpm"].clip(lower=40, upper=180)
    df["step_count"] = df["step_count"].clip(lower=0, upper=300)
    df["calories_burned"] = df["calories_burned"].clip(lower=0)
    df["sleep_hours"] = df["sleep_hours"].clip(lower=0, upper=12)

    print("\nData ranges after outlier handling:")
    print(df[["heart_rate_bpm", "step_count", "calories_burned", "sleep_hours"]].agg(["min", "max"]))

    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.set_index("timestamp", inplace=True)
    numeric_cols = ["heart_rate_bpm", "sleep_hours", "step_count", "calories_burned"]

    hourly_numeric = df[numeric_cols].resample("1h").mean()
    hourly_activity = df["activity_type"].resample("1h").agg(lambda x: x.mode()[0])

    hourly_df = pd.concat([hourly_numeric, hourly_activity], axis=1)
    hourly_df.reset_index(inplace=True)

    print("\nResampled data (first rows):")
    print(hourly_df.head())

    return hourly_df


def quality_report(df: pd.DataFrame) -> None:
    print(f"\nTotal records: {len(df)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nNumeric column statistics:")
    print(df.describe())
    print("\nActivity type counts:")
    print(df["activity_type"].value_counts())
    if df.isnull().sum().sum() == 0:
        print("\nNo missing values left — Data quality looks good!")
    else:
        print("\nWarning: There are still missing values!")


if __name__ == "__main__":
    raw_df = load_data()
    basic_validation(raw_df)
    clean_df = preprocess_raw(raw_df)
    hourly_df = resample_hourly(clean_df)
    quality_report(clean_df)
