# app.py
# Full merged Streamlit ML app with EDA profiling, feature engineering, multi-model training,
# Optuna tuning, SHAP explainability, learning curves, ROC/PR, model registry and deployment

import os
import io
import json
import time
import joblib
import pickle
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn imports
import sklearn
from packaging import version
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, VarianceThreshold
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve)
# models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import learning_curve

# optional libs (non-critical)
try:
    from ydata_profiling import ProfileReport
    _HAS_PROFILING = True
except Exception:
    _HAS_PROFILING = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

# xgboost optional
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# app config
st.set_page_config(page_title="ML Model Builder", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore")

# Create models dir
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------
# Helper functions
# -----------------------
def get_one_hot_encoder():
    try:
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except Exception:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data
def get_seaborn_datasets() -> List[str]:
    try:
        return sns.get_dataset_names()
    except Exception:
        return []


def load_seaborn_dataset(name: str) -> pd.DataFrame:
    return sns.load_dataset(name)


def read_uploaded_file(uploaded) -> pd.DataFrame:
    try:
        if hasattr(uploaded, "type") and (uploaded.type == "text/csv" or getattr(uploaded, "name", "").lower().endswith(".csv")):
            return pd.read_csv(uploaded)
        elif getattr(uploaded, "name", "").lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded)
        else:
            return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return pd.DataFrame()


def show_df_overview(df: pd.DataFrame):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(50))


def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], factor=1.5) -> pd.DataFrame:
    df_clean = df.copy()
    for col in numeric_cols:
        try:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col].isna()) | ((df_clean[col] >= lower) & (df_clean[col] <= upper))]
        except Exception:
            continue
    return df_clean


def encode_target(y: pd.Series) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    try:
        if y.dtype == "object" or y.dtype.name == "category" or not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            return y_enc, le
        else:
            return y.values, None
    except Exception:
        try:
            y_num = pd.to_numeric(y, errors="coerce").values
            return y_num, None
        except Exception:
            return y.values, None


def build_preprocessor(df: pd.DataFrame, selected_imputation: str, fill_value, encode_method: str, scaling_method: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
    transformers = []
    # numeric pipeline
    if numeric_cols:
        if selected_imputation in ["mean", "median"]:
            num_imputer = SimpleImputer(strategy=selected_imputation)
        elif selected_imputation == "most_frequent":
            num_imputer = SimpleImputer(strategy="most_frequent")
        elif selected_imputation == "constant":
            num_imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            num_imputer = SimpleImputer(strategy="mean")

        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = None

        if scaler is not None:
            transformers.append(("num", Pipeline([("imputer", num_imputer), ("scaler", scaler)]), numeric_cols))
        else:
            transformers.append(("num", Pipeline([("imputer", num_imputer)]), numeric_cols))
    # categorical pipeline
    if categorical_cols:
        if encode_method == "onehot":
            ohe = get_one_hot_encoder()
            cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
            transformers.append(("cat", cat_transformer, categorical_cols))
        elif encode_method == "label":
            cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
            transformers.append(("cat", cat_transformer, categorical_cols))
        else:
            cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
            transformers.append(("cat", cat_transformer, categorical_cols))

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    else:
        preprocessor = None
    return preprocessor, numeric_cols, categorical_cols


def apply_label_encoding(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    df2 = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col].astype(str))
            encoders[col] = le
        except Exception:
            continue
    return df2, encoders


def correlation_feature_selection(df: pd.DataFrame, target: str, threshold: float) -> List[str]:
    try:
        corr = df.corr(numeric_only=True)
        if target not in corr.columns:
            return df.columns.drop(target).tolist()
        corrs = corr[target].abs().sort_values(ascending=False)
        selected = corrs[corrs >= threshold].index.drop(target).tolist()
        return selected
    except Exception:
        try:
            return df.columns.drop(target).tolist()
        except Exception:
            return []


def model_options_for_task(task: str) -> Dict[str, object]:
    if task == "Regression":
        opts = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42, n_jobs=-1),
            "ExtraTreesRegressor": ExtraTreesRegressor(random_state=42, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "SVR": SVR()
        }
        if _HAS_XGB:
            opts["XGBRegressor"] = XGBRegressor(eval_metric='rmse', verbosity=0)
        return opts
    else:
        opts = {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42, n_jobs=-1),
            "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42, n_jobs=-1),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "SVC": SVC(probability=True),
            "GaussianNB": GaussianNB(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis()
        }
        if _HAS_XGB:
            opts["XGBClassifier"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        return opts


def default_param_grid(name: str, task: str) -> Dict[str, List[Any]]:
    grids = {
        "RandomForestClassifier": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        "GradientBoostingClassifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "GradientBoostingRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "DecisionTreeClassifier": {"max_depth": [None, 5, 10]},
        "DecisionTreeRegressor": {"max_depth": [None, 5, 10]},
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "Lasso": {"alpha": [0.001, 0.01, 0.1]},
        "SVC": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "KNeighborsClassifier": {"n_neighbors": [3, 5, 7]},
        "KNeighborsRegressor": {"n_neighbors": [3, 5, 7]},
        "XGBClassifier": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
        "XGBRegressor": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]}
    }
    return grids.get(name, {})


def apply_per_feature_imputation(df: pd.DataFrame, per_feature_strategy: Dict[str, str], per_feature_constant: Dict[str, object], global_strategy: str, global_fill_value):
    df_loc = df.copy()
    for col, strat in per_feature_strategy.items():
        try:
            strat_to_apply = strat if strat else "use_global"
            if strat_to_apply == "drop_rows":
                df_loc = df_loc[df_loc[col].notna()]
            elif strat_to_apply in ("ffill", "bfill"):
                df_loc[col].fillna(method='ffill' if strat_to_apply == "ffill" else 'bfill', inplace=True)
            elif strat_to_apply == "mean":
                try:
                    mean_val = pd.to_numeric(df_loc[col], errors='coerce').mean()
                    df_loc[col].fillna(mean_val, inplace=True)
                except Exception:
                    if not df_loc[col].mode().empty:
                        df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
            elif strat_to_apply == "median":
                try:
                    med_val = pd.to_numeric(df_loc[col], errors='coerce').median()
                    df_loc[col].fillna(med_val, inplace=True)
                except Exception:
                    if not df_loc[col].mode().empty:
                        df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
            elif strat_to_apply in ("mode", "most_frequent"):
                if not df_loc[col].mode().empty:
                    df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
                else:
                    df_loc[col].fillna(global_fill_value, inplace=True)
            elif strat_to_apply == "constant":
                val = per_feature_constant.get(col, global_fill_value)
                df_loc[col].fillna(val, inplace=True)
            elif strat_to_apply == "none":
                pass
            elif strat_to_apply == "use_global":
                if pd.api.types.is_numeric_dtype(df_loc[col]):
                    if global_strategy == "mean":
                        df_loc[col].fillna(df_loc[col].mean(), inplace=True)
                    elif global_strategy == "median":
                        df_loc[col].fillna(df_loc[col].median(), inplace=True)
                    elif global_strategy == "most_frequent":
                        if not df_loc[col].mode().empty:
                            df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
                        else:
                            df_loc[col].fillna(global_fill_value, inplace=True)
                    elif global_strategy == "constant":
                        df_loc[col].fillna(global_fill_value, inplace=True)
                    else:
                        try:
                            df_loc[col].fillna(df_loc[col].mean(), inplace=True)
                        except Exception:
                            if not df_loc[col].mode().empty:
                                df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
                else:
                    if global_strategy in ("most_frequent", "mode"):
                        if not df_loc[col].mode().empty:
                            df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
                        else:
                            df_loc[col].fillna(global_fill_value, inplace=True)
                    elif global_strategy == "constant":
                        df_loc[col].fillna(global_fill_value, inplace=True)
                    elif global_strategy in ("mean", "median"):
                        if not df_loc[col].mode().empty:
                            df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
                        else:
                            df_loc[col].fillna(global_fill_value, inplace=True)
                    else:
                        if not df_loc[col].mode().empty:
                            df_loc[col].fillna(df_loc[col].mode().iloc[0], inplace=True)
                        else:
                            df_loc[col].fillna(global_fill_value, inplace=True)
            else:
                pass
        except Exception as e:
            st.warning(f"Imputation failed for {col} with strategy {strat}: {e}")
    return df_loc


# --------------------------
# Model / pipeline registry helpers
# --------------------------
def save_pipeline(name: str, pipeline_obj: object, metadata: dict):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.joblib"
    ppath = MODELS_DIR / filename
    joblib.dump(pipeline_obj, ppath)
    meta = metadata.copy()
    meta.update({"pipeline_file": str(ppath), "created_at": timestamp})
    metafile = MODELS_DIR / f"{name}_{timestamp}.meta.json"
    with open(metafile, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return str(ppath), str(metafile)


def list_saved_models():
    metas = list(MODELS_DIR.glob("*.meta.json"))
    entries = []
    for m in metas:
        try:
            with open(m, "r") as f:
                md = json.load(f)
                entries.append(md)
        except Exception:
            continue
    entries = sorted(entries, key=lambda x: x.get("created_at", ""), reverse=True)
    return entries


def load_pipeline_from_file(pipeline_file_path: str):
    return joblib.load(pipeline_file_path)


# --------------------------
# Profiling / SHAP / Optuna helpers
# --------------------------
def generate_profile_report(df: pd.DataFrame, sample: int = 5000):
    if not _HAS_PROFILING:
        raise RuntimeError("ydata-profiling not installed. pip install ydata-profiling")
    if len(df) > sample:
        df_sample = df.sample(sample, random_state=42)
    else:
        df_sample = df
    profile = ProfileReport(df_sample, title="Data profiling", explorative=True)
    fname = MODELS_DIR / f"profile_{time.strftime('%Y%m%d_%H%M%S')}.html"
    profile.to_file(str(fname))
    return str(fname)


def optuna_search_sklearn(estimator_cls, param_space_func, X, y, cv=3, n_trials=30, direction="maximize", scoring=None, random_state=42):
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna not installed.")
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    def objective(trial):
        params = param_space_func(trial)
        try:
            est = estimator_cls(**params) if callable(estimator_cls) else estimator_cls.set_params(**params)
        except Exception:
            est = estimator_cls
        try:
            scores = cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))
        except Exception:
            return -1.0
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def compute_shap_values(model, X_background, X_explain):
    if not _HAS_SHAP:
        raise RuntimeError("shap not installed.")
    try:
        explainer = shap.Explainer(model, X_background)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_background, 100))
    shap_values = explainer(X_explain)
    return explainer, shap_values


def plot_learning_curve(estimator, X, y, cv=5, scoring=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes_res, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots()
    ax.plot(train_sizes_res, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes_res, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    ax.plot(train_sizes_res, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes_res, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(loc='best')
    return fig


def plot_roc_pr(model, X_test, y_test, pos_label=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        prob = model.predict_proba(X_test)
        if prob.shape[1] > 1:
            y_score = prob[:, 1]
        else:
            y_score = prob.ravel()
    except Exception:
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            raise RuntimeError("Model does not provide predict_proba or decision_function for ROC/PR.")
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=pos_label)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, label=f"AUPR={pr_auc:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    return fig


# --------------------------
# Session state initialization
# --------------------------
def ensure_session_state_keys():
    keys_defaults = {
        "raw_df": None,
        "proc_df": None,
        "preprocessor": None,
        "models": {},  # dictionary name->pipeline or estimator
        "model": None,
        "label_encoder": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "X_df": None,
        "encoding_method": None,
        "scaling_method": None,
        "preprocessor_options": None,
        "download_buffer": None,
        "model_metrics": None
    }
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_session_state_keys()

# --------------------------
# Multi-page navigation
# --------------------------
pages = ["Data", "EDA", "Preprocessing", "Modeling", "Evaluation", "Export & Download", "Model Registry & Deploy", "Prediction"]
page = st.sidebar.radio("Navigate", pages)

# --------------------------
# Page: Data
# --------------------------
if page == "Data":
    st.header("1 — Data: Load & Explore")
    st.markdown("Load a Seaborn dataset or upload a CSV/XLSX file. The loaded data will be available in other pages.")

    with st.sidebar:
        st.subheader("Data Input")
        data_source = st.radio("Choose data source", ["Seaborn dataset", "Upload file"], key="data_source_radio")
        if data_source == "Seaborn dataset":
            seaborn_list = get_seaborn_datasets()
            sel_ds = st.selectbox("Pick a seaborn dataset", seaborn_list, key="data_ds_select")
            if sel_ds:
                try:
                    loaded = load_seaborn_dataset(sel_ds)
                    st.session_state["raw_df"] = loaded.copy()
                    st.success(f"Loaded seaborn dataset {sel_ds} (shape: {loaded.shape})")
                except Exception as e:
                    st.error(f"Failed to load {sel_ds}: {e}")
        else:
            uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"], key="file_uploader")
            if uploaded is not None:
                df_read = read_uploaded_file(uploaded)
                if not df_read.empty:
                    st.session_state["raw_df"] = df_read.copy()
                    st.success(f"Uploaded dataset {getattr(uploaded, 'name', 'uploaded file')} (shape: {df_read.shape})")
                else:
                    st.error("Uploaded file could not be read or is empty.")

    st.markdown("---")
    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("No dataset loaded yet. Load a Seaborn dataset or upload a file from the sidebar.")
    else:
        st.subheader("Preview & Quick EDA")
        show_df_overview(raw_df)
        with st.expander("More EDA: dtypes, description, missing values"):
            try:
                dtypes = pd.DataFrame({"column": raw_df.columns, "dtype": [str(raw_df[c].dtype) for c in raw_df.columns]})
                st.dataframe(dtypes)
            except Exception:
                pass
            if st.checkbox("Show descriptive stats (all)"):
                st.write(raw_df.describe(include="all").T)
            if st.checkbox("Show missing value summary"):
                missing = raw_df.isna().sum()
                st.write(missing[missing > 0])
            if st.checkbox("Show correlation heatmap (numeric only)"):
                num_df = raw_df.select_dtypes(include=[np.number])
                if num_df.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Need at least 2 numeric columns for correlation heatmap.")

# --------------------------
# Page: EDA
# --------------------------
if page == "EDA":
    st.header("2 — Exploratory Data Analysis (EDA)")
    df = st.session_state.get("raw_df")
    if df is None or df.empty:
        st.warning("No dataset loaded. Load a Seaborn dataset or upload a file on the Data page first.")
        st.stop()

    st.subheader("Quick overview")
    show_df_overview(df)

    st.markdown("---")
    st.subheader("Column summary & missingness")
    with st.expander("Column datatypes & missing"):
        dtypes = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
        st.dataframe(dtypes)
        missing = df.isna().sum()
        st.write("Missing values (top):")
        st.write(missing[missing > 0].sort_values(ascending=False))

    st.markdown("---")
    st.subheader("Automated profile report")
    if _HAS_PROFILING:
        if st.button("Generate profile report (may take time for large datasets)"):
            try:
                profile_path = generate_profile_report(df, sample=5000)
                st.success("Profile generated.")
                with open(profile_path, "rb") as f:
                    st.download_button("Download profile HTML", data=f.read(), file_name=os.path.basename(profile_path), mime="text/html")
            except Exception as e:
                st.error(f"Profile generation failed: {e}")
    else:
        st.info("Install ydata-profiling for automated profile reports (pip install ydata-profiling)")

    st.markdown("---")
    st.subheader("Univariate exploration")
    col = st.selectbox("Select a column", df.columns.tolist(), key="eda_col_select")
    if col:
        st.write("Dtype:", str(df[col].dtype))
        if pd.api.types.is_numeric_dtype(df[col]):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=ax[0])
            sns.boxplot(x=df[col].dropna(), ax=ax[1])
            st.pyplot(fig)
            st.write(df[col].describe())
        else:
            vc = df[col].value_counts().head(40)
            st.bar_chart(vc)
            st.write("Top value counts:")
            st.write(vc)

    st.markdown("---")
    st.subheader("Groupby & pivot helpers")
    with st.expander("Groupby / Aggregate example"):
        group_cols = st.multiselect("Group by columns", df.columns.tolist(), key="gb_group_cols")
        agg_col = st.selectbox("Aggregate column (numeric)", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], key="gb_agg_col")
        agg_func = st.selectbox("Agg function", ["mean", "sum", "median", "count", "std"], index=0, key="gb_agg_func")
        if st.button("Run groupby aggregation"):
            if group_cols and agg_col:
                try:
                    gb = df.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
                    st.dataframe(gb)
                except Exception as e:
                    st.error(f"Groupby failed: {e}")
            else:
                st.info("Select at least one group column and one numeric aggregate column.")

    st.markdown("---")
    st.subheader("Correlation & pairwise")
    if st.checkbox("Show correlation heatmap (numeric only)"):
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns for correlation heatmap.")
    if st.checkbox("Show pairplot (may be slow)"):
        pair_cols = st.multiselect("Columns for pairplot (max 6)", df.select_dtypes(include=[np.number]).columns.tolist(), max_selections=6)
        if pair_cols:
            fig = sns.pairplot(df[pair_cols].dropna().sample(min(500, len(df))), diag_kind="kde")
            st.pyplot(fig)

# --------------------------
# Page: Preprocessing
# --------------------------
elif page == "Preprocessing":
    st.header("3 — Preprocessing: dtypes, missing values, encoding, scaling")
    df_source = st.session_state.get("raw_df")
    if df_source is None:
        st.warning("No dataset loaded. Please go to Data page and load a dataset first.")
        st.stop()

    st.subheader("Preview of loaded data")
    show_df_overview(df_source)

    st.markdown("---")
    st.subheader("A) Change data types (optional)")
    dtype_map_ui: Dict[str, str] = {}
    with st.expander("Change column dtypes", expanded=False):
        for col in df_source.columns:
            current_dtype = str(df_source[col].dtype)
            dtype_choice = st.selectbox(f"{col} (current: {current_dtype})", ["leave", "int", "float", "category", "object", "bool", "datetime"], index=0, key=f"dtype_change_{col}")
            dtype_map_ui[col] = dtype_choice

    st.markdown("---")
    st.subheader("B) Missing values: global & per-feature")
    imputation_method = st.selectbox("Global missing value strategy", ["mean", "median", "most_frequent", "constant"], key="global_imputation")
    fill_value_global = None
    if imputation_method == "constant":
        fill_value_global = st.text_input("Global fill value (constant)", value="", key="global_fill_val")
        try:
            fill_value_global = float(fill_value_global) if fill_value_global != "" else 0.0
        except Exception:
            pass

    per_feature_strategy: Dict[str, str] = {}
    per_feature_constants: Dict[str, Any] = {}
    enable_pf = st.checkbox("Enable per-feature missing-value strategies", value=False, key="enable_pf")
    if enable_pf:
        with st.expander("Set per-feature strategies", expanded=True):
            numeric_cols_all = df_source.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols_all = df_source.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            if numeric_cols_all:
                st.write("Numeric columns")
                for col in numeric_cols_all:
                    sel = st.selectbox(f"Numeric: {col}", ["use_global", "drop_rows", "mean", "median", "constant", "ffill", "bfill", "none"], key=f"pf_num_{col}")
                    per_feature_strategy[col] = sel
                    if sel == "constant":
                        v = st.text_input(f"Constant fill for {col}", key=f"pf_num_const_{col}")
                        try:
                            v_conv = float(v) if v != "" else 0.0
                        except Exception:
                            v_conv = v
                        per_feature_constants[col] = v_conv
            if categorical_cols_all:
                st.write("Categorical columns")
                for col in categorical_cols_all:
                    sel = st.selectbox(f"Categorical: {col}", ["use_global", "drop_rows", "mode", "most_frequent", "constant", "ffill", "bfill", "none"], key=f"pf_cat_{col}")
                    per_feature_strategy[col] = sel
                    if sel == "constant":
                        v = st.text_input(f"Constant fill for {col}", key=f"pf_cat_const_{col}")
                        per_feature_constants[col] = v

    st.markdown("---")
    st.subheader("C) Encoding & Scaling defaults")
    encoding_method = st.selectbox("Default categorical encoding method", ["onehot", "label", "none"], index=0, key="sidebar_encoding_method")
    scaling_method = st.selectbox("Default scaling method", ["none", "standard", "minmax"], index=0, key="sidebar_scaling_method")

    st.session_state["encoding_method"] = encoding_method
    st.session_state["scaling_method"] = scaling_method

    st.markdown("---")
    if st.button("Apply preprocessing and save processed dataframe"):
        df_proc = df_source.copy()
        st.write("Applying dtype conversions...")
        for col, dtype_choice in dtype_map_ui.items():
            if dtype_choice and dtype_choice != "leave":
                try:
                    if dtype_choice == "int":
                        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").astype("Int64")
                    elif dtype_choice == "float":
                        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
                    elif dtype_choice == "category":
                        df_proc[col] = df_proc[col].astype("category")
                    elif dtype_choice == "object":
                        df_proc[col] = df_proc[col].astype("object")
                    elif dtype_choice == "bool":
                        df_proc[col] = df_proc[col].astype("boolean")
                    elif dtype_choice == "datetime":
                        df_proc[col] = pd.to_datetime(df_proc[col], errors="coerce")
                except Exception as e:
                    st.warning(f"Could not convert {col} to {dtype_choice}: {e}")

        st.write("Applying missing value strategies...")
        all_cols_for_impute = df_proc.columns.tolist()
        per_feature_strategy_final = {}
        per_feature_constants_final = {}
        for col in all_cols_for_impute:
            per_feature_strategy_final[col] = per_feature_strategy.get(col, "use_global") if enable_pf else "use_global"
            if per_feature_strategy_final[col] == "constant":
                per_feature_constants_final[col] = per_feature_constants.get(col, fill_value_global)
        try:
            df_proc = apply_per_feature_imputation(df_proc, per_feature_strategy_final, per_feature_constants_final, imputation_method, fill_value_global)
        except Exception as e:
            st.warning(f"Per-feature imputation failed: {e}")

        st.session_state["proc_df"] = df_proc.copy()
        st.success("Processed dataframe saved to session (proc_df).")
        show_df_overview(df_proc)

# --------------------------
# Page: Modeling
# --------------------------
elif page == "Modeling":
    st.header("4 — Modeling: Build preprocessor, transform data & train (multi-model)")
    df_proc = st.session_state.get("proc_df") if st.session_state.get("proc_df") is not None else st.session_state.get("raw_df")
    if df_proc is None:
        st.warning("No dataset available. Load data and apply preprocessing first.")
        st.stop()

    show_df_overview(df_proc)
    st.markdown("---")
    st.subheader("Select target and features")
    all_cols = df_proc.columns.tolist()
    target_col = st.selectbox("Select target column", [""] + all_cols, key="mod_target")
    feature_cols = st.multiselect("Select feature columns (leave empty to use all except target)", all_cols, key="mod_features")
    if feature_cols == [] and target_col:
        feature_cols = [c for c in all_cols if c != target_col]
    if not target_col:
        st.info("Select a target column to continue.")
        st.stop()

    st.markdown("---")
    task = st.radio("Task", ["Regression", "Classification"], index=1 if st.session_state.get("last_task") == "Classification" else 0, key="mod_task")
    st.session_state["last_task"] = task

    # Preprocessor options
    imputation_method = st.selectbox("Imputation strategy (for numeric)", ["mean", "median", "most_frequent", "constant"], key="prep_impute_multi")
    fill_value = None
    if imputation_method == "constant":
        fill_value = st.text_input("Preprocessor constant fill value", value="", key="prep_fill_multi")
        try:
            fill_value = float(fill_value) if fill_value != "" else 0.0
        except Exception:
            pass
    encoding_method = st.selectbox("Categorical encoding (in preprocessor)", ["onehot", "label", "none"], index=0, key="prep_encoding_multi")
    scaling_method = st.selectbox("Scaling method (in preprocessor)", ["none", "standard", "minmax"], index=0, key="prep_scaling_multi")
    st.session_state["preprocessor_options"] = {"encoding_method": encoding_method, "scaling_method": scaling_method, "imputation_method": imputation_method}

    st.markdown("---")
    test_size = st.slider("Test set size (%)", min_value=5, max_value=50, value=20, key="split_test_size_multi") / 100.0
    random_state = int(st.number_input("Random seed", step=1, value=42, key="split_random_state_multi"))
    cv_folds = int(st.number_input("CV folds (for tuning / CV scoring)", min_value=2, max_value=10, value=5, key="cv_folds"))

    st.markdown("---")
    st.subheader("Choose algorithms to train (multi-select)")
    models_map = model_options_for_task(task)
    available_models = list(models_map.keys())
    selected_models = st.multiselect("Select models", available_models, default=available_models[:2], key="multi_model_select")

    st.markdown("Hyperparameter tuning & search")
    global_tune = st.checkbox("Enable tuning for selected models (Grid / Randomized)", value=True, key="global_tune_multi")
    tune_method = st.selectbox("Tuning method (applies when tuning enabled)", ["GridSearchCV", "RandomizedSearchCV"], key="global_tune_method")
    n_iter = None
    if tune_method == "RandomizedSearchCV":
        n_iter = int(st.number_input("n_iter (RandomizedSearch)", min_value=1, max_value=200, value=20, key="global_tune_niter"))

    use_optuna = False
    if _HAS_OPTUNA:
        use_optuna = st.checkbox("Use Optuna for Bayesian hyperparameter tuning (instead of Grid/Random)", value=False, key="use_optuna")
        if use_optuna:
            optuna_trials = int(st.number_input("Optuna n_trials", min_value=10, max_value=500, value=50, key="optuna_trials"))
            optuna_direction = st.selectbox("Optuna optimize direction", ["maximize", "minimize"], index=0, key="optuna_direction")
    else:
        st.info("Install optuna to enable Bayesian hyperparameter tuning (pip install optuna)")

    use_cv_scoring = st.checkbox("Use cross-validation (CV folds) to compute final scores for each model", value=True, key="use_cv_scoring")
    allow_download = st.checkbox("Allow model download after training", value=True, key="allow_download_multi")

    if task == "Regression":
        scoring_choices = ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"]
        default_score = "r2"
    else:
        scoring_choices = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        default_score = "f1_weighted"
    chosen_scoring = st.selectbox("Primary scoring metric (used to pick best model)", scoring_choices, index=scoring_choices.index(default_score))

    # Feature selection options
    with st.expander("Feature selection (optional)"):
        use_variance = st.checkbox("Variance threshold filter", value=False)
        var_thresh = st.number_input("Variance threshold", value=0.0, step=0.01) if use_variance else None
        use_corr_filter = st.checkbox("Drop highly correlated features", value=False)
        corr_thresh = st.slider("Correlation threshold (abs)", 0.5, 0.99, 0.95) if use_corr_filter else None
        use_kbest = st.checkbox("SelectKBest (univariate)", value=False)
        kbest_k = st.number_input("k for SelectKBest (int)", min_value=1, max_value=max(1, len(feature_cols)), value=min(10, max(1, len(feature_cols)))) if use_kbest else None

    if st.button("Train selected models"):
        if not selected_models:
            st.error("No models selected.")
            st.stop()
        if not feature_cols:
            st.error("No feature columns selected.")
            st.stop()

        preprocessor, numeric_cols, categorical_cols = build_preprocessor(df_proc[feature_cols], imputation_method, fill_value, encoding_method, scaling_method)
        st.session_state["preprocessor"] = preprocessor

        X = df_proc[feature_cols].copy()
        y_series = df_proc[target_col].copy()

        label_encoder = None
        if task == "Classification":
            y, label_encoder = encode_target(y_series)
            st.session_state["label_encoder"] = label_encoder
        else:
            y = pd.to_numeric(y_series, errors="coerce").values

        feature_label_encoders = {}
        if encoding_method == "label":
            cat_cols_to_encode = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            if cat_cols_to_encode:
                X, feature_label_encoders = apply_label_encoding(X, cat_cols_to_encode)
                st.session_state["feature_label_encoders"] = feature_label_encoders

        # Fit and transform X using preprocessor
        if preprocessor is not None:
            try:
                X_trans = preprocessor.fit_transform(X)
                col_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == "num":
                        col_names += list(cols)
                    elif name == "cat":
                        if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                            ohe = trans.named_steps["onehot"]
                            if hasattr(ohe, "categories_"):
                                for i, orig_col in enumerate(cols):
                                    cats = ohe.categories_[i]
                                    col_names += [f"{orig_col}__{str(cat)}" for cat in cats]
                            else:
                                for i, orig_col in enumerate(cols):
                                    col_names.append(f"{orig_col}__ohe_{i}")
                        else:
                            col_names += list(cols)
                if len(col_names) == 0:
                    X_df = pd.DataFrame(X_trans)
                elif X_trans.shape[1] == len(col_names):
                    X_df = pd.DataFrame(X_trans, columns=col_names)
                else:
                    X_df = pd.DataFrame(X_trans, columns=[f"f_{i}" for i in range(X_trans.shape[1])])
            except Exception as e:
                st.warning(f"Preprocessor transform failed: {e}. Falling back to numeric-only features.")
                X_df = pd.DataFrame(X.select_dtypes(include=[np.number])).fillna(0)
        else:
            X_df = X.copy()

        # Apply feature selection if chosen
        X_sel = X_df.copy()
        if use_variance and var_thresh is not None:
            try:
                vt = VarianceThreshold(threshold=var_thresh)
                X_sel = pd.DataFrame(vt.fit_transform(X_sel), columns=[f for i, f in enumerate(X_sel.columns) if vt.variances_[i] > var_thresh])
            except Exception as e:
                st.warning(f"Variance filter failed: {e}")
        if use_corr_filter and corr_thresh is not None:
            try:
                corr = X_sel.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
                X_sel.drop(columns=to_drop, inplace=True)
            except Exception as e:
                st.warning(f"Correlation filter failed: {e}")
        if use_kbest and kbest_k:
            try:
                sel = SelectKBest(k=int(kbest_k))
                sel_fit = sel.fit(X_sel.fillna(0), y)
                support = sel_fit.get_support(indices=True)
                X_sel = pd.DataFrame(sel_fit.transform(X_sel.fillna(0)), columns=[X_sel.columns[i] for i in support])
            except Exception as e:
                st.warning(f"SelectKBest failed: {e}")

        # train/test split
        stratify_arg = y if (task == "Classification" and len(np.unique(y)) > 1) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)
        except Exception:
            # fallback without stratify
            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=test_size, random_state=random_state)

        trained_models = {}
        metrics_rows = []

        progress = st.progress(0)
        total = len(selected_models)
        for i, mname in enumerate(selected_models, start=1):
            st.write("=== Training:", mname)
            base_model = models_map.get(mname)
            if base_model is None:
                st.warning(f"Model {mname} not available. Skipping.")
                continue
            model_to_train = base_model
            final_model = None
            best_params = {}

            # Optuna
            if use_optuna:
                grid = default_param_grid(mname, task)
                if not grid:
                    st.warning(f"No param space defined for Optuna for {mname}, skipping Optuna.")
                else:
                    try:
                        # build a quick param_space_func for optuna from grid (discrete choices)
                        def param_space(trial):
                            params = {}
                            for k, v in grid.items():
                                if isinstance(v, list):
                                    # try categorical sampling
                                    if all(isinstance(x, (int, float)) for x in v):
                                        params[k] = trial.suggest_categorical(k, v)
                                    else:
                                        params[k] = trial.suggest_categorical(k, v)
                                else:
                                    params[k] = trial.suggest_float(k, min(v), max(v)) if isinstance(v, (list, tuple)) else v
                            return params
                        study = optuna_search_sklearn(type(base_model), param_space, X_train, y_train, cv=cv_folds, n_trials=optuna_trials, direction=optuna_direction, scoring=chosen_scoring)
                        st.write("Optuna best params:", study.best_params)
                        best_params = study.best_params
                        try:
                            final_model = base_model.set_params(**best_params)
                            final_model.fit(X_train, y_train)
                        except Exception:
                            final_model = base_model
                            final_model.fit(X_train, y_train)
                    except Exception as e:
                        st.warning(f"Optuna failed for {mname}: {e}. Falling back.")
                        try:
                            model_to_train.fit(X_train, y_train)
                            final_model = model_to_train
                        except Exception as e2:
                            st.error(f"Training failed for {mname}: {e2}")
                            continue
            else:
                # Grid / Randomized or plain fit
                grid = default_param_grid(mname, task) if global_tune else {}
                if global_tune and grid:
                    st.write(f"Running tune for {mname} with grid: {grid}")
                    try:
                        if tune_method == "GridSearchCV":
                            gs = GridSearchCV(model_to_train, grid, cv=cv_folds, scoring=chosen_scoring, n_jobs=-1, verbose=0)
                            with st.spinner(f"GridSearchCV for {mname}..."):
                                gs.fit(X_train, y_train)
                            final_model = gs.best_estimator_
                            best_params = gs.best_params_
                        else:
                            rs = RandomizedSearchCV(model_to_train, grid, cv=cv_folds, n_iter=n_iter or 20, scoring=chosen_scoring, n_jobs=-1, verbose=0, random_state=42)
                            with st.spinner(f"RandomizedSearchCV for {mname}..."):
                                rs.fit(X_train, y_train)
                            final_model = rs.best_estimator_
                            best_params = rs.best_params_
                        st.success(f"{mname} tuning done.")
                    except Exception as e:
                        st.warning(f"Tuning failed for {mname}: {e}. Will train base model.")
                        try:
                            model_to_train.fit(X_train, y_train)
                            final_model = model_to_train
                        except Exception as e2:
                            st.error(f"Training failed for {mname}: {e2}")
                            continue
                else:
                    with st.spinner(f"Fitting {mname}"):
                        try:
                            model_to_train.fit(X_train, y_train)
                            final_model = model_to_train
                            best_params = {}
                        except Exception as e:
                            st.error(f"Training failed for {mname}: {e}")
                            continue

            # evaluate
            try:
                preds = final_model.predict(X_test)
            except Exception as e:
                st.warning(f"Prediction failed for {mname} on test set: {e}")
                preds = None

            row = {"model": mname, "best_params": best_params}
            if preds is not None:
                if task == "Regression":
                    row.update({
                        "R2": r2_score(y_test, preds),
                        "MAE": mean_absolute_error(y_test, preds),
                        "MSE": mean_squared_error(y_test, preds),
                        "RMSE": mean_squared_error(y_test, preds, squared=False)
                    })
                else:
                    row.update({
                        "Accuracy": accuracy_score(y_test, preds),
                        "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                        "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                        "F1": f1_score(y_test, preds, average="weighted", zero_division=0)
                    })
            else:
                row.update({k: None for k in (["R2", "MAE", "MSE", "RMSE"] if task == "Regression" else ["Accuracy", "Precision", "Recall", "F1"])})

            if use_cv_scoring:
                try:
                    cv_scores = cross_val_score(final_model, X_sel, y, cv=cv_folds, scoring=chosen_scoring, n_jobs=-1)
                    row[f"cv_{chosen_scoring}_mean"] = float(np.mean(cv_scores))
                    row[f"cv_{chosen_scoring}_std"] = float(np.std(cv_scores))
                except Exception as e:
                    st.warning(f"CV scoring failed for {mname}: {e}")
                    row[f"cv_{chosen_scoring}_mean"] = None
                    row[f"cv_{chosen_scoring}_std"] = None

            # save pipeline (preprocessor + model) as joblib
            try:
                full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", final_model)])
                metadata = {
                    "model_name": mname,
                    "task": task,
                    "features": list(X_sel.columns),
                    "scoring": chosen_scoring,
                    "best_params": best_params,
                    "metrics": row,
                    "sklearn_version": sklearn.__version__
                }
                ppath, mpath = save_pipeline(mname, full_pipeline, metadata)
                st.write(f"Saved pipeline: {ppath}")
            except Exception as e:
                st.warning(f"Saving pipeline failed for {mname}: {e}")

            trained_models[mname] = final_model
            metrics_rows.append(row)
            st.success(f"{mname} done.")
            progress.progress(i / total)

        # store results
        st.session_state["models"] = trained_models
        try:
            st.session_state["model_metrics"] = pd.DataFrame(metrics_rows).set_index("model")
        except Exception:
            st.session_state["model_metrics"] = pd.DataFrame(metrics_rows)
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["X_df"] = X_sel
        st.success("All selected models trained and saved to session_state['models'].")

    # show existing models
    if st.session_state.get("models"):
        st.subheader("📦 Trained Models in Session")

        models_dict = st.session_state.get("models", {})

        if models_dict:
            # Create a DataFrame from models dict keys
            models_df = pd.DataFrame({
                "Model Name": list(models_dict.keys())
            })

        # Style the table for readability
        st.dataframe(
            models_df.style.set_properties(**{
                'text-align': 'left',
                'font-weight': 'bold'
         }),
        use_container_width=True
    )
    else:
        st.info("No trained models available in session. Train a model first.")

# --------------------------
# Page: Evaluation
# --------------------------
elif page == "Evaluation":
    st.header("5 — Evaluation: Compare trained models")
    models_dict = st.session_state.get("models")
    metrics_df = st.session_state.get("model_metrics")
    X_df = st.session_state.get("X_df")
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    X_train = st.session_state.get("X_train")
    if not models_dict or metrics_df is None:
        st.warning("No trained models found. Train models on the Modeling page first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Metrics table", "Plots & Curves", "Explainability (SHAP)"])

    with tab1:
        st.subheader("Metrics table (all trained models)")
        metric_candidates = [c for c in metrics_df.columns if c not in ("best_params",)]
        highlight_metric = st.selectbox("Highlight best model by metric", metric_candidates, index=0)
        ascending = st.checkbox("Smaller = better (choose when metric is loss-like e.g. MSE)", value=False)
        styled = metrics_df.copy()
        try:
            best_idx = styled[highlight_metric].astype(float).idxmax() if not ascending else styled[highlight_metric].astype(float).idxmin()
        except Exception:
            best_idx = None
        def highlight_best(row):
            return ['background-color: #b3ffb3' if row.name == best_idx else '' for _ in row]
        try:
            st.dataframe(styled.style.format(precision=4).apply(highlight_best, axis=1))
        except Exception:
            st.write(styled)

    with tab2:
        st.subheader("Model inspection plots")
        model_to_plot = st.selectbox("Select model to inspect", list(models_dict.keys()))
        model = models_dict.get(model_to_plot)
        if model is None:
            st.warning("Pick a model.")
        else:
            try:
                preds = model.predict(X_test)
                if pd.api.types.is_numeric_dtype(preds) and pd.api.types.is_numeric_dtype(y_test):
                    st.subheader("Actual vs Predicted (sample)")
                    sample_df = pd.DataFrame({"actual": y_test, "predicted": preds})
                    st.dataframe(sample_df.head(50))
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, preds, alpha=0.6)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    st.pyplot(fig)
                    st.subheader("Learning curve")
                    try:
                        fig_lc = plot_learning_curve(model, X_df.fillna(0), st.session_state["y_train"], cv=5, scoring=None)
                        st.pyplot(fig_lc)
                    except Exception as e:
                        st.warning(f"Learning curve failed: {e}")
                else:
                    st.subheader("Confusion Matrix / Classification report (if classification)")
                    cm = confusion_matrix(y_test, preds)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                    st.pyplot(fig)
                    st.text(classification_report(y_test, preds, zero_division=0))
                    st.subheader("ROC & Precision-Recall")
                    try:
                        fig = plot_roc_pr(model, X_test, y_test)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"ROC/PR failed: {e}")
            except Exception as e:
                st.warning(f"Failed to compute plots/metrics for {model_to_plot}: {e}")

    with tab3:
        st.subheader("SHAP explainability (global & local)")
        if not _HAS_SHAP:
            st.warning("Install shap to enable explainability (pip install shap).")
        else:
            model_to_explain = st.selectbox("Select model for SHAP", list(models_dict.keys()), key="shap_model_select")
            model_obj = models_dict.get(model_to_explain)
            if model_obj is None:
                st.warning("No model available.")
            else:
                try:
                    Xbg = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=X_df.columns)
                    Xex = X_test.head(50) if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=X_df.columns).head(50)
                    explainer, shap_vals = compute_shap_values(model_obj, Xbg, Xex)
                    plt.figure()
                    shap.summary_plot(shap_vals, Xex, show=False)
                    st.pyplot(plt.gcf())
                    idx = st.number_input("SHAP: choose row index (0..n-1)", min_value=0, max_value=max(0, len(Xex)-1))
                    plt.figure()
                    try:
                        shap.plots.waterfall(shap_vals[idx], show=False)
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.warning(f"Per-sample SHAP waterfall failed: {e}")
                except Exception as e:
                    st.warning(f"SHAP explanation failed: {e}")

# --------------------------
# Page: Export & Download
# --------------------------
elif page == "Export & Download":
    st.header("6 — Export & Download trained model(s)")
    models_dict = st.session_state.get("models", {})
    preprocessor = st.session_state.get("preprocessor")
    label_encoder = st.session_state.get("label_encoder")
    feature_label_encoders = st.session_state.get("feature_label_encoders", {})

    if not models_dict:
        st.warning("No trained models to export. Train models on the Modeling page first.")
        st.stop()

    st.write("Trained models:")
    model_names = list(models_dict.keys())
    to_export = st.multiselect("Select models to export", model_names, default=model_names)
    export_name = st.text_input("Zip filename (without extension)", value="trained_models_bundle")

    import zipfile
    if st.button("Prepare & Download selected models (.zip)"):
        if not to_export:
            st.error("Select at least one model to export.")
        else:
            try:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name in to_export:
                        model_obj = models_dict[name]
                        package = {
                            "model": model_obj,
                            "preprocessor": preprocessor,
                            "label_encoder": label_encoder,
                            "feature_label_encoders": feature_label_encoders,
                            "feature_names": list(st.session_state.get("X_df").columns) if st.session_state.get("X_df") is not None else [],
                            "sklearn_version": sklearn.__version__
                        }
                        pickled = pickle.dumps(package)
                        zf.writestr(f"{name}.pkl", pickled)
                buf.seek(0)
                st.download_button("Download .zip", data=buf, file_name=f"{export_name}.zip", mime="application/zip")
                st.success("Prepared zip for download.")
            except Exception as e:
                st.error(f"Failed to prepare zip: {e}")

# --------------------------
# Page: Model Registry & Deploy
# --------------------------
elif page == "Model Registry & Deploy":
    st.header("7 — Model Registry & Deployment")
    entries = list_saved_models()
    if not entries:
        st.info("No saved pipelines found in models/. Train and save a model first.")
    else:
        for md in entries:
            st.subheader(f"{md.get('model_name','unknown')} — {md.get('created_at','')}")
            st.json(md)
            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("Load into session", key=f"load_{md.get('pipeline_file','')}_{md.get('created_at','')}"):
                try:
                    pipe = load_pipeline_from_file(md["pipeline_file"])
                    # if pipeline -> store full pipeline under name
                    st.session_state["models"][md["model_name"]] = pipe
                    st.success("Pipeline loaded into session['models']")
                except Exception as e:
                    st.error(f"Load failed: {e}")
            if c2.button("Download pipeline", key=f"dl_{md.get('pipeline_file','')}_{md.get('created_at','')}"):
                try:
                    with open(md["pipeline_file"], "rb") as f:
                        st.download_button("Download", data=f.read(), file_name=os.path.basename(md["pipeline_file"]))
                except Exception as e:
                    st.error(f"Download failed: {e}")
            if c3.button("Generate FastAPI scaffold", key=f"api_{md.get('pipeline_file','')}_{md.get('created_at','')}"):
                def generate_fastapi_scaffold(meta):
                    return f'''# FastAPI scaffold for model {meta['model_name']}
from fastapi import FastAPI
import joblib
import pandas as pd
app = FastAPI()
model = joblib.load(r"{meta['pipeline_file']}")
@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    preds = model.predict(df)
    return {{"prediction": preds.tolist()}}
'''
                def generate_dockerfile(meta):
                    return f'''FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
'''
                api_code = generate_fastapi_scaffold(md)
                dockerfile = generate_dockerfile(md)
                st.download_button("Download FastAPI .py", data=api_code.encode("utf-8"), file_name=f"{md['model_name']}_api.py")
                st.download_button("Download Dockerfile", data=dockerfile.encode("utf-8"), file_name="Dockerfile")

# --------------------------
# Page: Prediction
# --------------------------
elif page == "Prediction":
    st.header("8 — Make Predictions")
    models_dict = st.session_state.get("models", {})
    if not models_dict:
        st.warning("No trained models available. Train at least one model on the Modeling page.")
        st.stop()

    selected_model_name = st.selectbox("Select trained model for prediction", list(models_dict.keys()))
    model_entry = models_dict[selected_model_name]

    # If stored item is a pipeline, use it directly for preprocessing+predict
    is_pipeline = isinstance(model_entry, Pipeline) or (hasattr(model_entry, "named_steps") and "preprocessor" in getattr(model_entry, "named_steps", {}))
    pipeline_obj = model_entry if is_pipeline else None
    model = pipeline_obj.named_steps["model"] if pipeline_obj else model_entry
    preprocessor = pipeline_obj.named_steps["preprocessor"] if pipeline_obj else st.session_state.get("preprocessor")
    label_encoder = st.session_state.get("label_encoder")
    feature_label_encoders = st.session_state.get("feature_label_encoders", {})
    X_df = st.session_state.get("X_df")
    original_df = None
    if "proc_df" in st.session_state and isinstance(st.session_state["proc_df"], pd.DataFrame) and not st.session_state["proc_df"].empty:
        original_df = st.session_state["proc_df"]
    elif "raw_df" in st.session_state and isinstance(st.session_state["raw_df"], pd.DataFrame) and not st.session_state["raw_df"].empty:
        original_df = st.session_state["raw_df"]
    if original_df is None:
        st.warning("No data available. Load data first on the Data page.")
        st.stop()

    target_col = st.session_state.get("target_col")
    input_values = {}
    with st.form("prediction_inputs"):
        st.subheader("Enter Feature Values")
        feature_names = [col for col in original_df.columns if col != target_col]
        for col in feature_names:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                try:
                    val_min = float(original_df[col].min())
                    val_max = float(original_df[col].max())
                    val_median = float(original_df[col].median())
                except Exception:
                    val_min, val_max, val_median = 0.0, 1.0, 0.0
                input_values[col] = st.number_input(f"{col} (range: {val_min:.2f} to {val_max:.2f})", value=val_median)
            elif col in feature_label_encoders:
                options = list(feature_label_encoders[col].classes_)
                input_values[col] = st.selectbox(f"{col} (categorical)", options)
            else:
                unique_vals = original_df[col].dropna().unique()
                if len(unique_vals) > 50 or len(unique_vals) == 0:
                    input_values[col] = st.text_input(f"{col} (categorical free text)")
                else:
                    input_values[col] = st.selectbox(f"{col} (categorical)", unique_vals)
        submitted = st.form_submit_button("Make Prediction")

    if submitted:
        try:
            input_df = pd.DataFrame([input_values])
            for col, le in feature_label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
            if pipeline_obj is not None:
                pred = pipeline_obj.predict(input_df)
            else:
                if preprocessor is not None:
                    processed_input = preprocessor.transform(input_df)
                else:
                    processed_input = input_df.values
                pred = model.predict(processed_input)
            st.success("## Prediction Result")
            if label_encoder is not None:
                decoded_pred = label_encoder.inverse_transform(np.array(pred).astype(int))
                st.metric("Predicted Class", decoded_pred[0])
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(processed_input)[0]
                    proba_df = pd.DataFrame({"Class": label_encoder.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
                    st.write("Class Probabilities:")
                    st.dataframe(proba_df)
                    fig, ax = plt.subplots()
                    sns.barplot(data=proba_df, x="Class", y="Probability", ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            else:
                st.metric("Predicted Value", float(pred[0]))
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Debug Info - Input Features:")
            st.json(input_values)

# Footer tips
st.sidebar.markdown("---")
st.sidebar.markdown("#### Notes & Tips")
st.sidebar.markdown("""
- This app is educational; test with small datasets first.
- ydata-profiling and SHAP may need a lot of memory; sample if dataset is large.
""")