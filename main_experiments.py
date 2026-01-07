from fitpulse.data_generation import save_sample_data
from fitpulse.preprocessing import (
    load_data,
    basic_validation,
    preprocess_raw,
    resample_hourly,
    quality_report,
)
from fitpulse.features import prepare_tsfresh_input, extract_tsfresh_features
from fitpulse.modeling import fit_prophet_model, prophet_forecast, cluster_behaviors
from fitpulse.anomalies import rule_based_anomalies, prophet_residual_anomalies, cluster_anomalies
from fitpulse.visualizations import plot_time_series_with_anomalies


def main():
    # Milestone 1: Data creation + preprocessing
    save_sample_data()
    raw_df = load_data()
    basic_validation(raw_df)
    clean_df = preprocess_raw(raw_df)
    hourly_df = resample_hourly(clean_df)
    quality_report(clean_df)

    # Milestone 2: Feature extraction & modeling
    long_df = prepare_tsfresh_input(hourly_df)
    ts_features = extract_tsfresh_features(long_df)

    prophet_model = fit_prophet_model(hourly_df)
    forecast = prophet_forecast(prophet_model, periods=24)

    kmeans_labels, dbscan_labels, scaler, kmeans, dbscan = cluster_behaviors(ts_features)

    if dbscan_labels is not None and len(dbscan_labels)==len(ts_features):
        from fitpulse.anomalies import cluster_anomalies
        cluster_flags=cluster_anomalies(ts_features,dbscan_labels)
    else:
        cluster_flags=None

    # Milestone 3: Anomaly detection
    from fitpulse.anomalies import rule_based_anomalies, prophet_residual_anomalies
    rule_df = rule_based_anomalies(hourly_df)
    model_df = prophet_residual_anomalies(rule_df, forecast)
    if cluster_flags is not None:
        model_df["cluster_anomaly"]=cluster_flags.reindex(model_df.index, fill_value=False)

    # Simple visualization
    plot_time_series_with_anomalies(model_df, output_path="heart_rate_anomalies.png")


if __name__ == "__main__":
    main()
