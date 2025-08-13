# HVAC-Factory-Supply-Chain
# HVAC Factory Supply Chain KPI Dashboard

## Overview
This project analyzes supply chain KPIs for an HVAC manufacturing facility using both Power BI and Python.  
It focuses on On-Time In-Full (OTIF) performance, scrap rates, downtime analysis, loss hierarchy drill-downs, and **machine learning prediction** for OTIF performance.

## Requirements

### Required Libraries
Install the following Python libraries before running the analysis:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

**Detailed Requirements:**
- `pandas >= 1.3.0` - Data manipulation and analysis
- `numpy >= 1.21.0` - Numerical computing
- `scikit-learn >= 1.0.0` - Machine learning algorithms
- `joblib >= 1.0.0` - Model serialization
- `matplotlib >= 3.4.0` - Data visualization (optional)
- `seaborn >= 0.11.0` - Statistical data visualization (optional)

### Python Version
- **Python 3.7+** (Recommended: Python 3.8 or higher)

## Installation

### 1. Clone Repository
```bash
git clone [your-repository-url]
cd otif-ml-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create requirements.txt
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Dataset Description
1. **Inbound_Deliveries.csv** – Supplier deliveries, planned vs actual, delay reasons.
2. **Outbound_Shipments.csv** – Customer shipments, planned vs actual, late reasons.
3. **Production_Output.csv** – Daily production volumes, scrap, downtime, downtime reasons.
4. **Loss_Hierarchy.csv** – Level 1–3 classification of downtime/loss causes.

## KPIs
- **OTIF Inbound (Qty)**: % of inbound deliveries received on time and in full.
- **OTIF Outbound (Qty)**: % of outbound shipments delivered on time and in full.
- **Scrap Rate %**: Scrap units as a percentage of total units produced.
- **Total Downtime Hours**: Total production downtime converted from minutes.
- **Downtime per Unit Produced**: Indicator of operational efficiency.
- **OTIF Prediction**: Machine learning model to predict delivery performance.

## Why DOH Inventory Was Not Calculated
Inventory Days on Hand (DOH) requires stock balance or average inventory data, which is missing in the provided datasets.

## Machine Learning Analysis
Advanced predictive modeling for OTIF performance using historical shipment data with feature engineering and multiple algorithm comparison.

## How to Run the Project

### Prerequisites
Install required Python libraries:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### Steps
1. Place the CSV files in the project directory.
2. Update file path in `ml_otif_final.py`:
   ```python
   df = pd.read_csv("outbound-shipments.csv", parse_dates=["Planned Ship Date", "Actual Ship Date"])
   ```
3. Run the Python script for EDA and ML analysis:
   ```bash
   python ml_otif_final.py
   ```
4. Open the `.pbix` Power BI file to view the interactive dashboard.

### Expected Output
- Model performance comparison
- Feature importance analysis
- Best model selection and saving
- OTIF prediction capabilities

## Code Structure

### Main Script: `ml_otif_final.py`

```python
# ml_otif_final.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# [Full code implementation as provided in the artifact above]
```

### Key Functions

#### 1. Data Loading & Preprocessing
```python
# Load data with date parsing
df = pd.read_csv("path/to/data.csv", parse_dates=["Planned Ship Date", "Actual Ship Date"])

# Create target variable
df["OnTime"] = np.where(
    df["Actual Ship Date"].notna() &
    df["Planned Ship Date"].notna() &
    (df["Actual Ship Date"] <= df["Planned Ship Date"]), 1, 0
)
```

#### 2. Feature Engineering
```python
# Time-based features
df["PlannedWeekday"] = df["Planned Ship Date"].dt.dayofweek
df["IsWeekend"] = df["PlannedWeekday"].isin([5, 6]).astype(int)

# Performance encoding
customer_ontime_rate = df.groupby("Customer")["OnTime"].mean()
df["CustomerOntimeRate"] = df["Customer"].map(customer_ontime_rate)
```

#### 3. Model Training & Evaluation
```python
# Multiple models comparison
models = {
    "Logistic Regression": Pipeline([...]),
    "Random Forest (Conservative)": Pipeline([...]),
    "Random Forest (Balanced)": Pipeline([...])
}

# Cross-validation and evaluation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5))
    # ... evaluation metrics
```

## Results Interpretation

### Performance Metrics
- **ROC AUC > 0.7**: Good predictive performance
- **ROC AUC 0.5-0.7**: Moderate performance
- **ROC AUC < 0.5**: Poor performance (worse than random)

### Overfitting Detection
- **Gap < 10%**: Good generalization
- **Gap 10-20%**: Moderate overfitting
- **Gap > 20%**: High overfitting risk

### Feature Importance
Top features typically include:
1. Customer historical performance
2. Product delivery patterns
3. Temporal factors (weekday, season)
4. Volume characteristics

## Troubleshooting

### Common Issues

#### 1. Library Version Conflicts
```bash
# Update scikit-learn for OneHotEncoder compatibility
pip install --upgrade scikit-learn
```

#### 2. Date Parsing Errors
```python
# Ensure proper date format in CSV
df = pd.read_csv("data.csv", parse_dates=["Planned Ship Date", "Actual Ship Date"])
```

#### 3. Memory Issues
```python
# Reduce data size for large datasets
df_sample = df.sample(n=10000, random_state=42)
```

#### 4. Categorical Data Issues
```python
# Handle categorical columns properly
if X[col].dtype.name == 'category':
    X[col] = X[col].cat.add_categories("Unknown").fillna("Unknown")
```

## Tools Used
- **Power BI** for interactive dashboards and KPI visualizations.
- **Python (Pandas, Seaborn, Matplotlib)** for data cleaning and exploratory analysis.
- **Scikit-learn** for machine learning model development and evaluation.
- **Joblib** for model serialization and deployment.

## Insights from Analysis
- High OTIF scores indicate strong supplier and logistics performance.
- Low scrap rate (0.02%) signals high manufacturing quality.
- Downtime hours (220) still represent potential efficiency gains.
- Loss hierarchy drill-down helps identify key downtime causes.
- **ML Model**: Achieves predictive accuracy for OTIF performance with key features being customer history and product patterns.

## Author
Mostafa Karam Mohamed AbdElmohsen
