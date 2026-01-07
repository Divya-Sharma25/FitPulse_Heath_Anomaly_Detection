import sys
from pathlib import Path
import io
import base64

import pandas as pd
import streamlit as st
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitpulse.preprocessing import preprocess_raw, resample_hourly
from fitpulse.features import prepare_tsfresh_input, extract_tsfresh_features
from fitpulse.modeling import (
    fit_prophet_model,
    prophet_forecast,
    add_kmeans_clusters,
    add_dbscan_clusters,
)
from fitpulse.anomalies import rule_based_anomalies, prophet_residual_anomalies
from fitpulse.visualizations import plot_plotly_time_series


# ---------- Page config & CSS ----------
st.set_page_config(page_title="FitPulse Health Anomaly Detection", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f9ff 0%, #ffffff 45%, #f0fbff 100%);
    }
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    .milestone-card {
        padding: 1.2rem 1.5rem;
        border-radius: 0.9rem;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.0rem;
    }
    .stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #0ea5e9;
    }
    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #0369a1;
    }
    .stButton>button {
        background-color: #0ea5e9;
        color: #ffffff;
        border-radius: 999px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# upload box – inner text darker (but keep sidebar dark)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button *,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] svg {
        color: #0f172a !important;
        fill: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("FitPulse Controls")

uploaded_file = st.sidebar.file_uploader("Upload fitness CSV", type=["csv"])
forecast_hours = st.sidebar.slider("Forecast hours ahead (Prophet)", 0, 72, 24)

milestone = st.sidebar.radio(
    "Select view",
    [
        "Overview",
        "Milestone 1 – Preprocessing",
        "Milestone 2 – Features & Modeling",
        "Milestone 3 – Anomalies & Clusters",
        "Milestone 4 – Insights & PDF",
    ],
)

show_full_raw = st.sidebar.checkbox("Show full raw data", value=False)
show_features = st.sidebar.checkbox("Show TSFresh features", value=False)

# ---------- Hero ----------
st.markdown(
    '<div class="hero-title">FitPulse – Health Anomaly Detection from Fitness Devices</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-subtitle">
    End‑to‑end time‑series pipeline: generate or ingest fitness data, clean it, extract features
    with TSFresh, model normal behaviour using Prophet, and detect anomalies in heart rate,
    steps, sleep, and calories.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Project milestones & tech stack", expanded=(milestone == "Overview")):
    st.markdown(
        """
        **Milestone 1 – Data Collection & Preprocessing**  
        • Clean and resample heart rate, steps, sleep, calories to hourly time‑series.  

        **Milestone 2 – Feature Extraction & Modeling**  
        • Use TSFresh to compute statistical features per day.  
        • Forecast heart rate using Prophet.  

        **Milestone 3 – Anomaly Detection & Clustering**  
        • Rule‑based and residual anomalies.  
        • K‑Means and DBSCAN clustering to group behavioural patterns.  

        **Milestone 4 – Dashboard Insights & Report**  
        • Summarise anomalies and export a PDF report.  
        """
    )

# ---------- Pipeline ----------
@st.cache_data
def run_pipeline(file_bytes: bytes, fhours: int):
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
    clean_df = preprocess_raw(raw_df)
    hourly_df = resample_hourly(clean_df)

    long_df = prepare_tsfresh_input(hourly_df)
    ts_features = extract_tsfresh_features(long_df)

    model = fit_prophet_model(hourly_df)
    forecast = prophet_forecast(model, periods=fhours)

    # 1) Rule‑based anomalies
    rule_df = rule_based_anomalies(hourly_df)

    # 2) Model‑based anomalies (Prophet residuals)
    model_df = prophet_residual_anomalies(rule_df, forecast)

    # 3) Clustering (K‑Means + DBSCAN)
    model_df = add_kmeans_clusters(model_df, n_clusters=3)
    model_df = add_dbscan_clusters(model_df, eps=5.0, min_samples=3)

    return raw_df, clean_df, hourly_df, ts_features, forecast, model_df


if uploaded_file is None:
    if milestone != "Overview":
        st.warning("Upload a CSV on the left to explore milestones.")
else:
    raw_df, clean_df, hourly_df, ts_features, forecast, model_df = run_pipeline(
        uploaded_file.getvalue(), forecast_hours
    )

# ---------- Helper: model‑based anomaly plot ----------
def plot_model_based_anomalies(model_df: pd.DataFrame, forecast: pd.DataFrame):
    df = model_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    f = forecast[["ds", "yhat"]].copy()
    f.rename(columns={"ds": "timestamp", "yhat": "pred_hr"}, inplace=True)

    merged = pd.merge(df, f, on="timestamp", how="left")
    anoms = merged[merged["model_anomaly"] == 1]

    fig = go.Figure()

    # actual HR
    fig.add_trace(
        go.Scatter(
            x=merged["timestamp"],
            y=merged["heart_rate_bpm"],
            mode="lines",
            name="Actual HR",
            line=dict(color="#0f172a"),
        )
    )

    # predicted HR
    fig.add_trace(
        go.Scatter(
            x=merged["timestamp"],
            y=merged["pred_hr"],
            mode="lines",
            name="Prophet prediction",
            line=dict(color="#0ea5e9", dash="dash"),
        )
    )

    # model‑based anomalies
    fig.add_trace(
        go.Scatter(
            x=anoms["timestamp"],
            y=anoms["heart_rate_bpm"],
            mode="markers",
            name="Model-based anomalies",
            marker=dict(color="#ef4444", size=8),
        )
    )

    fig.update_layout(
        title="Model-based anomalies (Prophet residuals)",
        xaxis_title="Time",
        yaxis_title="Heart rate (bpm)",
        height=450,
    )
    return fig

# ---------- Overview ----------
if milestone == "Overview":
    st.markdown('<div class="milestone-card">', unsafe_allow_html=True)
    st.subheader("Welcome to the FitPulse dashboard")

    if uploaded_file is None:
        st.markdown(
            "- Start by uploading a CSV file from your fitness device in the **left panel**.\n"
            "- Expected columns: `timestamp`, `heart_rate_bpm`, `sleep_hours`, "
            "`step_count`, `calories_burned`, `activity_type`."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw data (head)**")
            st.dataframe(raw_df.head(), use_container_width=True)
        with c2:
            st.markdown("**Cleaned numeric summary**")
            st.dataframe(clean_df.describe(), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Milestone 1 – Preprocessing ----------
if milestone == "Milestone 1 – Preprocessing" and uploaded_file is not None:
    st.markdown('<div class="milestone-card">', unsafe_allow_html=True)
    st.subheader("Milestone 1 – Data Collection & Preprocessing")
    st.caption("Cleaning, validation, and hourly resampling.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw data sample**")
        st.dataframe(raw_df.head(10), use_container_width=True)
    with c2:
        st.markdown("**Missing values per column (after cleaning)**")
        st.write(clean_df.isna().sum())

    if show_full_raw:
        st.markdown("**Full raw dataset**")
        st.dataframe(raw_df, use_container_width=True, height=260)

    st.markdown("**Heart rate analysis (hourly)**")
    st.line_chart(hourly_df.set_index("timestamp")["heart_rate_bpm"])

    st.markdown("**Sleep duration analysis (hourly)**")
    st.line_chart(hourly_df.set_index("timestamp")["sleep_hours"])

    st.markdown("**Step count analysis (hourly)**")
    st.line_chart(hourly_df.set_index("timestamp")["step_count"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Milestone 2 – Features & Modeling ----------
if milestone == "Milestone 2 – Features & Modeling" and uploaded_file is not None:
    st.markdown('<div class="milestone-card">', unsafe_allow_html=True)
    st.subheader("Milestone 2 – Feature Extraction & Modeling")
    st.caption("TSFresh feature matrix and Prophet forecast.")

    if show_features:
        st.markdown("**TSFresh feature matrix (first 10 rows)**")
        st.dataframe(ts_features.head(10), use_container_width=True)
    else:
        st.info("Enable **Show TSFresh features** in the sidebar to inspect feature matrix.")

    st.markdown("**Prophet forecast for heart rate**")
    forecast_plot = forecast[["ds", "yhat"]].copy()
    forecast_plot.rename(columns={"ds": "timestamp", "yhat": "predicted_hr"}, inplace=True)
    st.line_chart(forecast_plot.set_index("timestamp"), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Milestone 3 – Anomalies & Clusters ----------
if milestone == "Milestone 3 – Anomalies & Clusters" and uploaded_file is not None:
    st.markdown('<div class="milestone-card">', unsafe_allow_html=True)
    st.subheader("Milestone 3 – Anomaly Detection, K‑Means & DBSCAN")

    total_points = len(model_df)
    rule_anoms = int(model_df["rule_anomaly"].sum())
    model_anoms = int(model_df["model_anomaly"].sum())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total hourly points", total_points)
    with k2:
        st.metric("Rule-based anomalies", rule_anoms)
    with k3:
        st.metric("Model-based anomalies", model_anoms)

    st.markdown("**Heart rate timeline with anomalies (rule-based)**")
    fig_ts = plot_plotly_time_series(model_df)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("**Model-based anomalies (Prophet residuals)**")
    fig_model = plot_model_based_anomalies(model_df, forecast)
    st.plotly_chart(fig_model, use_container_width=True)

    st.markdown("**K‑Means clustering (HR vs steps)**")
    st.scatter_chart(
        model_df[["heart_rate_bpm", "step_count", "kmeans_cluster"]]
        .rename(columns={"kmeans_cluster": "cluster"})
    )

    st.markdown("**DBSCAN clustering (HR vs steps)**")

    dbscan_df = model_df[["heart_rate_bpm", "step_count", "dbscan_cluster"]].copy()
    dbscan_df["cluster_label"] = dbscan_df["dbscan_cluster"].astype(str)
    dbscan_df.loc[dbscan_df["dbscan_cluster"] == -1, "cluster_label"] = "noise"

    fig_db = px.scatter(
        dbscan_df,
        x="heart_rate_bpm",
        y="step_count",
        color="cluster_label",
        color_discrete_map={"noise": "#ef4444"},
        title="DBSCAN clusters (red = noise)",
    )
    fig_db.update_traces(marker=dict(size=8, opacity=0.8))
    st.plotly_chart(fig_db, use_container_width=True)

    st.caption(
        "DBSCAN is density‑based: points labelled 'noise' (red) do not belong to any dense cluster."
    )

    st.markdown("**Hourly data with anomaly + cluster labels**")
    st.dataframe(model_df, use_container_width=True, height=280)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Milestone 4 – Insights & PDF ----------
def make_pdf_summary(rule_anoms, model_anoms, total_points) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "FitPulse Anomaly Report", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        8,
        f"Total hourly points: {total_points}\n"
        f"Rule-based anomalies: {rule_anoms}\n"
        f"Model-based anomalies: {model_anoms}\n",
    )
    pdf.ln(4)
    pdf.multi_cell(
        0,
        8,
        "Notes:\n"
        "- Rule-based anomalies flag extreme heart rate, suspicious steps, or calories.\n"
        "- Model-based anomalies indicate unexpected deviations from Prophet forecast.\n",
    )
    pdf_bytes = pdf.output(dest="S")
    return bytes(pdf_bytes)


def download_pdf_button(data: bytes, filename: str):
    b64 = base64.b64encode(data).decode("latin-1")
    href = (
        f'<a href="data:application/octet-stream;base64,{b64}" '
        f'download="{filename}">📄 Download PDF report</a>'
    )
    st.markdown(href, unsafe_allow_html=True)


if milestone == "Milestone 4 – Insights & PDF" and uploaded_file is not None:
    st.markdown('<div class="milestone-card">', unsafe_allow_html=True)
    st.subheader("Milestone 4 – Insights & Exportable Report")

    rule_anoms = int(model_df["rule_anomaly"].sum())
    model_anoms = int(model_df["model_anomaly"].sum())
    total_points = len(model_df)

    st.markdown("### Key observations")
    st.markdown(
        f"- **{rule_anoms}** of **{total_points}** hourly points are flagged by rule-based logic.\n"
        f"- **{model_anoms}** points are additionally flagged by Prophet residuals.\n"
        "- Anomalies often cluster around high‑intensity activities."
    )

    st.markdown("### Export report")
    if st.button("Generate PDF report"):
        pdf_bytes = make_pdf_summary(rule_anoms, model_anoms, total_points)
        download_pdf_button(pdf_bytes, "fitpulse_anomaly_report.pdf")

    st.markdown("</div>", unsafe_allow_html=True)
