# Milestone 2 — Feature Extraction & Modeling  
FitPulse: Health Anomaly Detection from Fitness Devices  
Infosys Springboard Internship 6.0

---

## 1. Objective of Milestone 2
The goal of Milestone 2 is to convert cleaned time-series data into meaningful, model-ready signals using:

1. TSFresh statistical feature extraction  
2. Seasonal trend modeling using Prophet  
3. Behavioral pattern clustering using KMeans / DBSCAN  

This milestone transforms raw numeric sequences into feature-rich representations suitable for anomaly detection in the next phase.

---

## 2. Tasks Completed in This Milestone

### 2.1 TSFresh Feature Extraction  
Using the clean 5-minute heart-rate data from Milestone 1:

- Sliding windows of 12 data points (≈1 hour) were created  
- For each window, TSFresh extracted statistical features including:  
  - Mean  
  - Median  
  - Standard deviation  
  - Minimum / Maximum  
  - Skewness  
  - Kurtosis  
  - Autocorrelation  
  - Linear trend  
  - Energy, entropy, absolute variations  
- Missing values were imputed  
- Constant (zero-variance) features were removed  
- Result: a numerical feature matrix capturing the statistical behavior of the heart-rate signal over time

---

### 2.2 Trend Modeling (Using Rolling Median Baseline)
Prophet installation was not compatible in this environment  
so an alternative **rolling-median trend modeling** method was implemented.

Steps:

- Heart-rate time series resampled at 5-min intervals  
- 60-minute rolling median computed  
- Residuals calculated as:  
  `residual = actual_value – rolling_baseline`  
- Anomalies identified using the **3-sigma rule**  
  - Any point where |residual| > 3 × standard deviation was flagged  
- Trend and baseline were plotted to visualize deviations  

This method successfully highlights sharp spikes or unusual dips even without Prophet.

---

### 2.3 Behavioral Clustering (KMeans)
To group similar heart-rate patterns:

- Extracted TSFresh features were standardized  
- PCA (2-component) was used for visualization  
- KMeans clustering performed (3 clusters)  
- A scatter plot shows clusters representing different physiological behavior groups  

Result:  
Different periods of heart-rate behavior were grouped together, enabling pattern recognition and anomaly analysis.

---

## 3. Code Summary (Executed in Milestone 2 Notebook)

### Implemented functionality:
- Sliding window generation  
- TSFresh feature extraction  
- Rolling baseline trend modeling  
- 3-sigma anomaly detection  
- KMeans clustering  
- PCA visualization  

### Files Produced:
- `tsfresh_features.csv` (feature matrix)  
- `clusters_pca.csv` (PCA + cluster labels)  
- `anomalies.csv` (3-sigma anomalies)  

All files can be used for analytics or fed into Streamlit in later milestones.

---

## 4. Outputs Generated

### 4.1 TSFresh Features  
A comprehensive feature matrix capturing statistical properties of the heart-rate signal.

### 4.2 Trend Plot (Baseline vs Actual)  
Visualization of rolling-median trend and anomalies.

### 4.3 Clustering Visualization  
PCA-based scatter plot showing distinct behavioral zones.

---

