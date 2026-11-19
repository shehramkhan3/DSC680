# Subscriber Churn & Wearable Biometric Attrition Analysis

## 📋 Overview

Forecasted 30-day subscriber churn and workout attrition by combining 4,000 commercial fitness club records with 167,373 quality-filtered wearable workout trajectories via engineered workout decay velocity, Acute-to-Chronic Workload Ratio (ACWR), and Gradient Boosted Decision Trees (`95.75%` accuracy, `0.9856` ROC-AUC).

## 🛠️ Tools Used

* Python 3.x
* **Machine Learning / Data:** `scikit-learn` (Gradient Boosting, Random Forest, Logistic Regression), `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`

## 📁 File Structure

```text
Subscriber Churn Wearable Biometrics/
├── Milestone_2_FitPulse_Analysis_Shehram_Khan_v4.ipynb           # Data prep, feature engineering, modeling & charts
├── Shehram+Khan+Project+Milestone+2.pdf                          # Full APA white paper with embedded figures & code
├── gym_churn_us.csv                                              # Commercial gym retention dataset (N = 4,000)
└── README.md                                                     # Project documentation
