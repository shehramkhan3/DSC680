# ⚙️ Building a Machine Learning Classifier for Early Anomaly Detection in Industrial Machinery Telemetry

**Author:** Shehram Khan  
**Course:** DSC680 — Applied Data Science  
**Domain:** Industrial Machine Learning & Predictive Maintenance  

---

## 📌 Project Objective
Unplanned equipment downtime is one of the most significant cost drivers in modern manufacturing. This project executes an end-to-end predictive maintenance machine learning workflow designed to detect early anomalies and predict catastrophic machine failures from continuous physical sensor telemetry (`N = 10,000` observations from the Kaggle AI4I 2020 Predictive Maintenance dataset).

The primary goals of this pipeline are:
* **RQ1 / Hypothesis 1 ($H_{01}$):** Benchmark supervised machine learning architectures against traditional single-variable rule-based plant alarms, targeting a test set **ROC-AUC $\ge$ 0.85** and **Failure Recall $\ge$ 0.85**.
* **RQ2 / Hypothesis 2 ($H_{02}$):** Quantify the predictive gain of physics-guided interaction features over raw, isolated sensor streams.

---

## 🛠️ Methodology & Pipeline Architecture

### 1. Data Hygiene & Leakage Prevention
* Loaded the 10,000-record AI4I 2020 telemetry dataset containing air temperature, process temperature, rotational speed (RPM), torque (Nm), tool wear (minutes), and product quality tier (`Type` L, M, H).
* Quarantined and dropped primary identifiers (`UDI`, `Product_ID`) alongside post-hoc multi-class failure flags (`Failure_Type`) to prevent target leakage.
* Executed an **80/20 stratified train-test split** (`8,000` train / `2,000` test) with a fixed global random seed (`SEED = 42`) for 100% execution reproducibility.
* Addressed severe class imbalance (~3.24% failure rate) by computing a dynamic positive class weighting ratio (`scale_pos_weight = 29.89`) and applying `StandardScaler` exclusively on training partitions for linear models.

### 2. Physics-Guided Feature Engineering
Engineered three domain-specific physical interaction features to capture thermo-mechanical stress on the assembly tooling:
* **`Temp_Difference`:** $\text{Process\_Temp\_K} - \text{Air\_Temp\_K}$ (captures thermal dissipation gradients).
* **`Mechanical_Power`:** $\text{Torque\_Nm} \times \left(\frac{2\pi \times \text{Rotational\_Speed\_RPM}}{60}\right)$ (converts rotational kinematics into physical power in Watts).
* **`Wear_Torque_Ratio`:** $\text{Tool\_Wear\_Min} \times \text{Torque\_Nm}$ (models cumulative mechanical strain on degraded cutting tools).

### 3. Model Training & Hyperparameter Optimization
Using **5-Fold Stratified Cross-Validation**, five distinct classification approaches were benchmarked:
1. **Simple Threshold-Alarm Baseline:** Single-variable plant rule (`Tool_Wear_Min > 200` OR `Process_Temp_K > 310`).
2. **Logistic Regression:** Linear baseline with balanced class weights (`class_weight='balanced'`).
3. **Random Forest Classifier:** Bagging ensemble (`n_estimators=100`, `class_weight='balanced'`).
4. **LightGBM Classifier:** Gradient boosted decision trees weighted by `scale_pos_weight`.
5. **Optimized XGBoost Classifier:** Tuned via `GridSearchCV` optimizing for `roc_auc` across learning rate, tree depth, estimator count, and subsample ratio (`learning_rate: 0.1`, `max_depth: 5`, `n_estimators: 200`, `subsample: 0.8`).

### 4. Safety-First Decision Threshold Tuning
To prioritize plant safety and minimize costly false negatives (uncaught equipment breakdowns), the classification probability threshold was shifted from the default `0.50` to **`0.32`**. This threshold was calibrated by analyzing Precision-Recall curves exclusively on the validation folds during 5-fold cross-validation, keeping the holdout test set completely unseen.

---

## 📊 Key Findings & Outcomes
* **Superiority Over Rule-Based Alarms:** Multi-variable ensemble classifiers significantly outperform static single-sensor threshold alarms in separating normal operating telemetry from imminent breakdown states.
* **Dominant Feature Importance:** Feature importance gain analysis confirmed that the engineered thermo-mechanical interaction terms (`Temp_Difference`, `Wear_Torque_Ratio`, and `Mechanical_Power`) alongside `Tool_Wear_Min` rank among the top predictors of equipment failure, validating the physics-informed feature engineering approach.
* **Visual Diagnostics:** Includes automated generation of Receiver Operating Characteristic (ROC-AUC) curves comparing the tuned XGBoost classifier against the random baseline.

---

## 💻 Tech Stack & Dependencies
* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`

### How to Run
1. Clone this repository and navigate to the project directory.
2. Ensure `ai4i2020.csv` is placed in the root of this project folder.
3. Launch JupyterLab / Jupyter Notebook and run all cells sequentially.
