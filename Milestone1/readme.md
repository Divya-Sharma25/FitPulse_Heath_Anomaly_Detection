# Milestone 1 — Data Collection & Preprocessing  
FitPulse: Health Anomaly Detection from Fitness Devices  
Infosys Springboard Internship 6.0

---

## 1. Problem Statement
Wearable fitness devices generate large volumes of noisy, irregular time-series data.  
This noise and inconsistency can hide important health-related anomalies, making it difficult to detect unusual patterns early.

Milestone 1 focuses on building a reliable preprocessing pipeline that prepares this data for modeling.

---

## 2. Objective of Milestone 1
The goal of this milestone is to create a complete data ingestion and preprocessing pipeline that can:

- Load raw fitness device data  
- Validate and clean all columns  
- Handle missing values and duplicates  
- Normalize timestamps  
- Remove outliers  
- Convert sleep hours into meaningful metrics  
- Resample all data to a uniform 5-minute frequency  
- Produce modeling-ready datasets for further analysis in Milestone 2  

---

## 3. Dataset Description
The dataset (`Sample_data.csv`) contains realistic synthetic readings from wearable fitness devices.

| Column Name       | Description |
|-------------------|-------------|
| timestamp         | Timestamp of reading |
| heart_rate_bpm    | Heart rate in beats per minute |
| step_count        | Steps recorded in the interval |
| sleep_hours       | Cumulative sleep hours counter |
| calories_burned   | Calories burned (approx) |
| activity_type     | Activity label (Sleeping/Walking/Workout/etc.) |

---

## 4. Preprocessing Workflow

### 4.1 Data Loading & Validation
- Loaded CSV using pandas  
- Checked dataset structure: columns, datatypes, missing values  
- Removed duplicate rows  
- Printed statistical summary

---

### 4.2 Timestamp Normalization
- Converted `timestamp` column into datetime format  
- Sorted data chronologically  
- Ensured proper indexing for time-series analysis

---

### 4.3 Handling Missing Values
- Forward-filled numeric columns (heart rate, steps, calories, sleep)  
- Forward-filled `activity_type`  
- Verified no missing values remain

---

### 4.4 Outlier Detection & Treatment
Applied realistic bounds to remove noise:

| Metric          | Applied Range |
|-----------------|----------------|
| heart_rate_bpm  | 40 to 180 bpm |
| step_count      | 0 to 300 |
| sleep_hours     | 0 to 12 hours |
| calories_burned | >= 0 |

Values outside these ranges were clipped to reduce unrealistic spikes.

---

### 4.5 Resampling to Uniform 5-Minute Frequency
To make the time-series consistent:

- Numeric features were resampled using mean aggregation  
- Categorical `activity_type` was resampled using statistical mode  
- Result: a clean, evenly spaced dataset ideal for modeling

---

### 4.6 Sleep Data Transformation
The `sleep_hours` column represented cumulative sleep.  
It was transformed into **per-interval sleep duration (in minutes)**:

- Calculated differences between time steps  
- Reset when not sleeping  
- Clipped negative values  
- Produced interpretable "sleep duration per interval"

---

### 4.7 Final Outputs Generated
Three clean, modeling-ready datasets were generated:

| Dataset | Columns |
|---------|---------|
| Heart Rate | timestamp, heart_rate |
| Steps | timestamp, step_count |
| Sleep | timestamp, duration_minutes |

All aligned to the same 5-minute frequency.

---
