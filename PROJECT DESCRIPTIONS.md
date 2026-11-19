# 📊 Applied Data Science Portfolio — Shehram Khan
**Course:** DSC680 — Applied Data Science  
**Author:** Shehram Khan  

### 🌐 **[Click Here to View My Live Portfolio Website](https://shehramkhan3.github.io/DSC680/)**

---

## 📂 Repository Structure & Project Inventory
This repository contains 10 end-to-end data science and generative AI projects audited for strict validation hygiene, reproducibility, and business impact. Each project resides in its own dedicated subfolder containing source code, data assets, and dedicated documentation.

| Project # | Project Title | Primary Domain | Key Metric / Outcome | Folder Link |
| :---: | :--- | :--- | :--- | :--- |
| **01** | Predictive Maintenance Telemetry Pipeline | Industrial ML & IoT | **$2.37M** net downtime savings (77.4% reduction) | [`/Predictive Maintenance`](https://github.com/shehramkhan3/DSC680/tree/main/Predictive%20Maintenance) |
| **02** | MNIST Handwritten Digit Classification | Deep Learning / CV | **98.89%** test accuracy (Conv2D CNN) | [`/MNIST CNN CLASSIFICATION`](https://github.com/shehramkhan3/DSC680/tree/main/MNIST%20CNN%20CLASSIFICATION) |
| **03** | Housing Price Regression & PCA | Dimensionality Reduction | **$R^2$ 0.6478 $\rightarrow$ 0.8409** (PCA compression) | [`/Housing Price Regression Model`](https://github.com/shehramkhan3/DSC680/tree/main/Housing%20Price%20Regression%20Model) |
| **04** | U.S. Retail Sales Time-Series Forecasting | Macroeconomic Forecasting | **5.69% MAPE** (Holt-Winters Exponential Smoothing) | [`/Retail Sales Holt Winter`](https://github.com/shehramkhan3/DSC680/tree/main/Retail%20Sales%20Holt%20Winter) |
| **05** | Loan Approval Ensemble Classification | Supervised Classification | **83.33% accuracy** (Tuned Logistic Regression vs KNN) | [`/Loan Approval Grid Sea`](https://github.com/shehramkhan3/DSC680/tree/main/Loan%20Approval%20Grid%20Sea) |
| **06** | Childcare Burden Economic Analysis | Data Storytelling & EDA | **Empirical regional analysis** (23,593 records analyzed) | [`/Childcare Burden`](https://github.com/shehramkhan3/DSC680/tree/main/Childcare%20Burden) |
| **07** | Video Game Sales Exploratory Data Analysis | Exploratory Data Hygiene | **16,719 records audited** & missing values imputed | [`/Video Games Sales EDA`](https://github.com/shehramkhan3/DSC680/tree/main/Video%20Games%20Sales%20EDA) |
| **08** | Machine Learning Telemetry Classifier | Industrial IoT & ML | **Optimized XGBoost** with safety-first threshold tuning | [`/Building a Machine Learning Classifier...`](https://github.com/shehramkhan3/DSC680/tree/main/Building%20a%20Machine%20Learning%20Classifier%20for%20Early%20Anomaly%20Detection%20in%20Industrial%20Machinery%20Telemetry) |
| **09** | CSCS AI Personal Trainer & Periodization Engine | Generative AI & Streamlit | **100% JSON schema & orthopedic safety** (SFT + Grounding) | [`/CSCS AI Personal Trailer`](https://github.com/shehramkhan3/DSC680/tree/main/CSCS%20AI%20Personal%20Trailer) |
| **10** | Subscriber Churn & Wearable Biometrics Analytics | Behavioral & Biometric ML | **High-recall churn intervention** & heart-rate segmentation | [`/Subscriber Churn Wearable Biometrics`](https://github.com/shehramkhan3/DSC680/tree/main/Subscriber%20Churn%20Wearable%20Biometrics) |

---

## 📋 Detailed Project Descriptions

### 01. Predictive Maintenance Telemetry Pipeline
* **Folder:** [`/Predictive Maintenance`](https://github.com/shehramkhan3/DSC680/tree/main/Predictive%20Maintenance)
* **Overview:** Engineered an end-to-end industrial machine learning pipeline across 10,000 continuous sensor observations (process/air temperature, rotational speed, torque, and tool wear) to predict equipment failures prior to breakdown. Quarantined post-failure leakage flags, standardized features strictly within training folds, and trained an XGBoost classifier achieving **98.60% accuracy** and **86.8% failure recall**—cutting simulated unplanned downtime expenses by **77.4% ($2,368,500)**.

### 02. MNIST Handwritten Digit Classification CNN
* **Folder:** [`/MNIST CNN CLASSIFICATION`](https://github.com/shehramkhan3/DSC680/tree/main/MNIST%20CNN%20CLASSIFICATION)
* **Overview:** Designed and trained a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify 70,000 grayscale handwritten digit images. Normalized 2D pixel tensors, stacked `Conv2D` and `MaxPooling2D` spatial feature extractors with dropout regularization, and converged in 5 epochs to **98.89% test accuracy**, backed by Seaborn confusion-matrix diagnostics to isolate visually ambiguous digit pairs.

### 03. Housing Price Regression & Dimensionality Reduction (PCA)
* **Folder:** [`/Housing Price Regression Model`](https://github.com/shehramkhan3/DSC680/tree/main/Housing%20Price%20Regression%20Model)
* **Overview:** Optimized a high-dimensional residential real estate pricing model containing 236 encoded structural and neighborhood features. Systematically pruned high-missingness columns, imputed partition-safe medians and modes, and applied Principal Component Analysis (PCA) to compress the feature space into 127 orthogonal components—eliminating multicollinearity and boosting **$R^2$ from 0.6478 to 0.8409**.

### 04. U.S. Retail Sales Time-Series Forecasting
* **Folder:** [`/Retail Sales Holt Winter`](https://github.com/shehramkhan3/DSC680/tree/main/Retail%20Sales%20Holt%20Winter)
* **Overview:** Modeled three decades of monthly U.S. Census retail sales data capturing secular growth, 12-month holiday seasonality, and structural regime breaks (the 2008 Financial Crisis and 2020 COVID-19 shock). Reshaped wide historical matrices into a continuous `DatetimeIndex` and trained an Additive Holt-Winters Exponential Smoothing model achieving **5.69% Mean Absolute Percentage Error (MAPE)**.

### 05. Loan Approval Ensemble Classification
* **Folder:** [`/Loan Approval Grid Sea`](https://github.com/shehramkhan3/DSC680/tree/main/Loan%20Approval%20Grid%20Sea)
* **Overview:** Constructed multi-model binary classification pipelines to automate consumer loan underwriting decisions. Benchmarked K-Nearest Neighbors (KNN), Random Forest, and L2-regularized Logistic Regression using stratified 5-fold `GridSearchCV`, diagnosing high-dimensional distance decay in KNN and selecting regularized Logistic Regression at **83.33% out-of-sample test accuracy**.

### 06. Childcare Burden Economic Analysis & Storytelling
* **Folder:** [`/Childcare Burden`](https://github.com/shehramkhan3/DSC680/tree/main/Childcare%20Burden)
* **Overview:** Built a programmatic data storytelling pipeline analyzing county-level U.S. childcare affordability burdens across **23,593 records**. Modeled log-normal cost burden distributions, automated Kernel Density Estimation (KDE) plots and regional outlier boxplots in Seaborn/Matplotlib, and compiled an executive-ready PowerPoint briefing automatically via `python-pptx`.

### 07. Video Game Sales Exploratory Data Analysis (EDA)
* **Folder:** [`/Video Games Sales EDA`](https://github.com/shehramkhan3/DSC680/tree/main/Video%20Games%20Sales%20EDA)
* **Overview:** Performed comprehensive exploratory data analysis and data hygiene across **16,719 global video game releases**. Standardized mixed-type critic and user score strings (`'tbd'` to `NaN`), imputed 6,704 missing `User_Score` records using platform-genre distributions, and engineered North American regional revenue-share metrics to rank multi-decade publisher market dominance.

### 08. Machine Learning Classifier for Early Anomaly Detection
* **Folder:** [`/Building a Machine Learning Classifier...`](https://github.com/shehramkhan3/DSC680/tree/main/Building%20a%20Machine%20Learning%20Classifier%20for%20Early%20Anomaly%20Detection%20in%20Industrial%20Machinery%20Telemetry)
* **Overview:** Executed an end-to-end anomaly classification workflow on Kaggle AI4I 2020 plant telemetry ($N=10,000$) to detect subtle mechanical degradation before failure. Engineered physics-guided interaction features (thermal differential, mechanical power, wear-torque ratio), mitigated severe class imbalance via dynamic class weighting, and tuned probability decision thresholds ($\ge 0.32$) using 5-fold cross-validation to minimize uncaught line stoppages.

### 09. CSCS AI Personal Trainer & Clinical Periodization Engine
* **Folder:** [`/CSCS AI Personal Trailer`](https://github.com/shehramkhan3/DSC680/tree/main/CSCS%20AI%20Personal%20Trailer)
* **Overview:** Architected a production **Streamlit** web application powered by a supervised fine-tuning (SFT) and schema-grounded OpenAI inference pipeline. Eliminated three zero-shot LLM failure modes: enforced **100% parseable JSON** across multi-turn gym-floor equipment swaps, blocked axial-loading contraindications (e.g., Standing Calf Raises) for acute L5-S1 lumbar disc herniations, and automated multi-lift 4-week progressive overload and deload cycles.

### 10. Subscriber Churn & Wearable Biometrics Analytics
* **Folder:** [`/Subscriber Churn Wearable Biometrics`](https://github.com/shehramkhan3/DSC680/tree/main/Subscriber%20Churn%20Wearable%20Biometrics)
* **Overview:** Conducted end-to-end predictive modeling across Model Fitness subscriber records and Endomondo wearable workout telemetry to forecast member attrition. Engineered longitudinal session frequency, contract tenure, and cardiovascular heart-rate exertion features to deliver high-recall churn risk segmentation and actionable retention intervention thresholds supported by an executive whitepaper.

---

## 🛠️ Global Tools & Tech Stack
* **Programming Languages:** Python 3.x, R, SQL
* **Machine Learning & Deep Learning:** `scikit-learn`, `XGBoost`, `LightGBM`, `TensorFlow`, `Keras`
* **Generative AI & LLM Engineering:** `OpenAI API` (Supervised Fine-Tuning, Structured Outputs, Schema Grounding)
* **Data Manipulation & Analysis:** `pandas`, `numpy`, `statsmodels`
* **Data Visualization & Web Apps:** `matplotlib`, `seaborn`, `Power BI`, `Streamlit`, `python-pptx`
