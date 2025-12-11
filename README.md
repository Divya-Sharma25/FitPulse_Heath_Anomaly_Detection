# FitPulse — Health Anomaly Detection from Fitness Devices  
**Infosys Springboard Internship 6.0**

---

## 🔍 Project Problem Statement
Fitness devices generate a **large volume of noisy and irregular time-series data** (heart rate, sleep cycles, steps, calories, activity patterns).  
This data often contains:

- Irregular timestamps  
- Missing values  
- Sudden spikes & outliers  
- Non-uniform sampling rates  
- Activity-dependent variations  

These inconsistencies can **hide early health warning signals**, making it difficult to detect abnormalities or unusual physiological patterns.

---

## 🎯 Project Objective
To develop a **robust, end-to-end pipeline** that ingests, cleans, processes, and analyzes wearable fitness-device data in order to:

- Detect early **health anomalies**  
- Identify unusual activity or sleep patterns  
- Understand behavior using clustering and trends  
- Present real-time insights through a Streamlit dashboard  

The system enables accurate anomaly detection for **preventive healthcare and early intervention**.

---

# 🚀 Project Pipeline

The FitPulse pipeline contains **4 major components**:

---

## 1️⃣ Data Collection & Preprocessing

### ✔ Import data from wearable devices
- Accept CSV or JSON exported from fitness trackers  
- Support for heart rate, step count, sleep duration  

### ✔ Clean and standardize data  
- Convert timestamps to datetime  
- Fix missing values (forward fill, interpolation)  
- Remove outliers using realistic physiological ranges  

### ✔ Align time intervals  
- Resample everything into consistent 5-minute intervals  
- Convert cumulative sleep hours → sleep duration per interval  

### **Output:**  
A clean, uniform dataset ready for feature extraction and modeling.

---

## 2️⃣ Feature Extraction & Modeling

### ✔ TSFresh Feature Extraction  
Automatically compute 100+ statistical features like:

- Mean, median, variance  
- Skewness, kurtosis  
- Entropy, energy  
- Autocorrelation  
- Trend features  

### ✔ Seasonal Trend Modeling using Facebook Prophet  
Prophet is used to understand:

- Daily cycles  
- Weekly habits  
- Long-term patterns  
- Residuals for anomaly detection  

### ✔ Behavioral Clustering  
Using **KMeans** and **DBSCAN** to group similar patterns:

- Stable HR windows  
- High-stress periods  
- Irregular behavior segments  

### **Output:**  
Feature vectors, Prophet trend curves, and cluster labels.

---

## 3️⃣ Anomaly Detection & Visualization

### ✔ Types of detected anomalies

**Point anomalies:**  
Single unusual spikes  
Example: Sudden HR jump to 180 bpm while resting.

**Contextual anomalies:**  
Unusual only in specific context  
Example: High heart rate during sleep.

**Collective anomalies:**  
Abnormal patterns over time  
Example: Low sleep (<3 hrs) for multiple days.

---

### ✔ Rule-based anomaly detection  
- Thresholds  
- Standard deviation (3-sigma)  
- Rolling median baseline  

### ✔ Model-based anomaly detection  
- Prophet residuals  
- Cluster outliers  
- Feature-based deviations  

### ✔ Visualizations  
Built using Plotly/Matplotlib:

- Heart rate trends  
- Seasonal patterns  
- Cluster maps (PCA)  
- Highlighted anomaly points  

---

## 4️⃣ Streamlit Dashboard for Interactive Insights  
*(Built in the final milestone)*

### Features:
- Upload fitness device data  
- Run preprocessing + anomaly detection  
- Visualize trends and dangerous spikes  
- See cluster behaviors  
- Export report as PDF/CSV  

This enables healthcare-style analysis for ordinary wearable users.

---

# 🔧 Tools & Technologies Used

### **Programming Language**
- Python 3.x

### **Libraries**
- Pandas, NumPy — data processing  
- Matplotlib, Plotly — visualization  
- TSFresh — time-series feature engineering  
- Facebook Prophet — seasonality modeling  
- Scikit-learn — clustering (KMeans, DBSCAN), scaling, PCA  
- Streamlit — dashboard/UI  

### **Data Formats**
- CSV  
- JSON (exported from fitness trackers)

---
