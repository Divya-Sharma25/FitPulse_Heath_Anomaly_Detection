from typing import Tuple
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN


def fit_prophet_model(hourly_df: pd.DataFrame, target_col: str = "heart_rate_bpm") -> Prophet:
    df = hourly_df[["timestamp", target_col]].rename(
        columns={"timestamp": "ds", target_col: "y"}
    )
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
    )
    model.fit(df)
    return model


def prophet_forecast(model: Prophet, periods: int = 24) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq="H")
    forecast = model.predict(future)
    return forecast


def cluster_behaviors(features: pd.DataFrame, n_clusters: int = 3):
    # Drop rows with NaNs
    clean = features.dropna(axis=0, how="any")

    if clean.shape[0] < n_clusters:
        # Not enough samples to cluster into n_clusters
        # Return dummy labels (-1) and None models
        dummy_labels = np.full(clean.shape[0], -1, dtype=int)
        return dummy_labels, dummy_labels, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    return kmeans_labels, dbscan_labels, scaler, kmeans, dbscan

from sklearn.cluster import KMeans  # file ke top me ya yahin add kar sakte ho


def add_kmeans_clusters(df, n_clusters: int = 3):
    """
    Add a 'kmeans_cluster' column to the dataframe using
    heart_rate_bpm and step_count as features.
    """
    data = df.copy()

    # Agar columns missing hon to error avoid karne ke liye simple guard
    if "heart_rate_bpm" not in data.columns or "step_count" not in data.columns:
        return data

    X = data[["heart_rate_bpm", "step_count"]].values

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    data["kmeans_cluster"] = labels

    return data

def add_dbscan_clusters(df, eps: float = 2.5, min_samples: int = 5):
    """
    Add a 'dbscan_cluster' column to the dataframe using
    heart_rate_bpm and step_count as features.
    Cluster label -1 means 'noise' according to DBSCAN.
    """
    data = df.copy()

    if "heart_rate_bpm" not in data.columns or "step_count" not in data.columns:
        return data

    X = data[["heart_rate_bpm", "step_count"]].values

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    data["dbscan_cluster"] = labels

    return data