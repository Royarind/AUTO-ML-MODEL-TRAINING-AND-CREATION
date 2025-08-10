# Auto ML Streamlit App

An interactive Auto Machine Learning web application built with [Streamlit](https://streamlit.io/) that allows you to:

- Upload datasets or use sample datasets
- Perform Exploratory Data Analysis (EDA)
- Preprocess and transform data
- Train multiple ML models in parallel
- Tune hyperparameters with Grid Search, Random Search, or Optuna
- Compare model performance in a metrics table
- View Learning Curves and SHAP explainability
- Save, download, and reuse trained models
- Make predictions from any trained model

---

## Features

### Data Loading
- Upload CSV files or select from built-in datasets
- Preview dataset and basic statistics

### Exploratory Data Analysis (EDA)
- Automated profiling with `ydata-profiling`
- Summary stats, correlation heatmaps, groupby analysis
- Missing value detection

### Preprocessing
- Handle missing values
- Encode categorical variables
- Scale/normalize features
- Feature selection methods:
  - Variance threshold
  - Correlation filtering
  - SelectKBest
  - Recursive Feature Elimination (RFE)

### Model Training
- Select multiple models to train simultaneously
- Supports:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Gradient Boosting (XGBoost, LightGBM)
  - KNN, SVM, etc.
- Cross-validation support
- Hyperparameter tuning:
  - Grid Search
  - Random Search
  - Optuna Bayesian Optimization

### Model Evaluation
- Performance metrics table (best model highlighted)
- Interactive model selection
- Learning curves
- SHAP explainability (global & per-feature)

### Model Management
- Save trained models to disk
- Download pickle files
- Load previously trained models into session

### Prediction
- Select any trained model and input features
- Get predictions and probabilities

---

## Installation

### Requirements
- Python 3.11

### Steps
```bash
# Clone repository
git clone https://github.com/yourusername/auto-ml-streamlit.git
cd auto-ml-streamlit

# Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt


## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
