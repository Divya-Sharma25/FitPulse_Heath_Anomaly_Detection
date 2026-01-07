import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def plot_time_series_with_anomalies(df: pd.DataFrame, output_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["heart_rate_bpm"], label="Heart Rate")

    if "rule_anomaly" in df.columns:
        anomalies = df[df["rule_anomaly"]]
        ax.scatter(
            anomalies["timestamp"],
            anomalies["heart_rate_bpm"],
            color="red",
            label="Rule Anomaly",
        )

    if "model_anomaly" in df.columns:
        anomalies2 = df[df["model_anomaly"]]
        ax.scatter(
            anomalies2["timestamp"],
            anomalies2["heart_rate_bpm"],
            color="orange",
            label="Model Anomaly",
            marker="x",
        )

    ax.set_title("Heart Rate with Anomalies")
    ax.set_xlabel("Time")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_plotly_time_series(df: pd.DataFrame):
    
    data = df.copy()
    if "rule_anomaly" in data.columns:
        data["anomaly_flag"] = data["rule_anomaly"].map({True: "Anomaly", False: "Normal"})
        color_col = "anomaly_flag"
    else:
        color_col = None

    fig = px.scatter(
        data,
        x="timestamp",
        y="heart_rate_bpm",
        color=color_col,
        title="Heart Rate with Rule-based Anomalies",
    )
    fig.update_traces(mode="lines+markers")
    return fig
