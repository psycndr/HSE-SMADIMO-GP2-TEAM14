import os
import json
import traceback
import difflib
import pickle
import re
import datetime
import math
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

STAGE = "05_modelling"
BUSINESS_TASK = "Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?"
REQUESTED_TARGET_COLUMN = "Positive Feedback Count"
INPUT_CSV_PATH = "/files/reviews_data_prepared.csv"
MODELLING_REPORT_PATH = "/files/modelling.md"
MEMORY_DIR = "/files/memory"
BEST_MODEL_PATH = "/files/memory/best_model.pkl"
BEST_METRICS_PATH = "/files/memory/best_metircs.json"
PREDICTIONS_PATH = "/files/predictions.csv"
MEMORY_PREDICTIONS_PATH = "/files/memory/predictions.csv"

def safe_float(value):
    try:
        if value is None:
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

def write_report(content):
    try:
        with open(MODELLING_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass

def normalize_column_name(name):
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())

def find_target_column(df_columns, requested_target):
    warnings_list = []
    # Exact match
    if requested_target in df_columns:
        return requested_target, warnings_list
    # Normalized match
    normalized_requested = normalize_column_name(requested_target)
    for col in df_columns:
        if normalize_column_name(col) == normalized_requested:
            warnings_list.append(f"Target column found by normalized name match: '{col}'")
            return col, warnings_list
    # Similarity match
    matches = difflib.get_close_matches(requested_target, df_columns, n=1, cutoff=0.35)
    if matches:
        warnings_list.append(f"Target column found by similarity match: '{matches[0]}'")
        return matches[0], warnings_list
    return None, warnings_list

def safe_rmse(y_true, y_pred):
    try:
        mse = mean_squared_error(y_true, y_pred)
        if math.isnan(mse) or math.isinf(mse):
            return None
        return math.sqrt(mse)
    except Exception:
        return None

def safe_mae(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        if math.isnan(mae) or math.isinf(mae):
            return None
        return mae
    except Exception:
        return None

def safe_r2(y_true, y_pred):
    try:
        if len(np.unique(y_true)) == 1:
            return None
        r2 = r2_score(y_true, y_pred)
        if math.isnan(r2) or math.isinf(r2):
            return None
        return r2
    except Exception:
        return None

def safe_rank_correlation(y_true, y_pred):
    # Spearman rank correlation without scipy
    try:
        if len(y_true) != len(y_pred):
            return None, "Length mismatch in rank correlation"
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            return None, "Constant array in rank correlation"
        def rankdata(a):
            temp = np.argsort(a)
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(a))
            return ranks
        rank_true = rankdata(np.array(y_true))
        rank_pred = rankdata(np.array(y_pred))
        n = len(y_true)
        d = rank_true - rank_pred
        d_squared_sum = np.sum(d**2)
        spearman = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        if math.isnan(spearman) or math.isinf(spearman):
            return None, "NaN or Inf in rank correlation"
        return spearman, None
    except Exception as e:
        return None, f"Exception in rank correlation: {str(e)}"

def safe_top_10_helpful_rate(y_true, y_pred):
    try:
        n = len(y_true)
        if n == 0:
            return None
        top_10_count = max(1, n // 10)
        # Indices of top 10% predicted
        top_pred_indices = np.argsort(y_pred)[-top_10_count:]
        # Actual values at those indices
        actual_top = np.array(y_true)[top_pred_indices]
        # Count how many actual are >0 (helpful)
        helpful_count = np.sum(actual_top > 0)
        rate = helpful_count / top_10_count
        if math.isnan(rate) or math.isinf(rate):
            return None
        return rate
    except Exception:
        return None

def main():
    warnings_list = []
    model_errors = []
    model_results = []
    model_names = []
    model_error_messages = {}
    model_error_tracebacks = {}
    model_trained_flags = {}
    result = {}
    # Check input file existence
    if not os.path.isfile(INPUT_CSV_PATH):
        msg = f"Input CSV file not found at path: {INPUT_CSV_PATH}"
        warnings_list.append(msg)
        report_content = f"# Modelling Report\n\n## Error\n\n{msg}\n"
        write_report(report_content)
        result = {
            "status": "error",
            "message": "input_file_not_found",
            "warnings": warnings_list,
            "stage": STAGE,
            "input_path": INPUT_CSV_PATH,
            "report_path": MODELLING_REPORT_PATH,
            "best_model_path": BEST_MODEL_PATH,
            "best_metrics_path": BEST_METRICS_PATH,
            "predictions_path": PREDICTIONS_PATH,
            "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
        }
        print(json.dumps(sanitize_for_json(result), ensure_ascii=False))
        return

    # Read CSV
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except Exception as e:
        tb_str = traceback.format_exc()
        warnings_list.append(f"Failed to read CSV: {str(e)}")
        report_content = f"# Modelling Report\n\n## Error reading CSV\n\n```\n{tb_str}\n```\n"
        write_report(report_content)
        result = {
            "status": "error",
            "message": "failed_to_read_csv",
            "warnings": warnings_list,
            "stage": STAGE,
            "input_path": INPUT_CSV_PATH,
            "report_path": MODELLING_REPORT_PATH,
            "best_model_path": BEST_MODEL_PATH,
            "best_metrics_path": BEST_METRICS_PATH,
            "predictions_path": PREDICTIONS_PATH,
            "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
        }
        print(json.dumps(sanitize_for_json(result), ensure_ascii=False))
        return

    df_columns = list(df.columns)

    # Remove technical columns from features
    technical_cols = [col for col in df_columns if col == "Unnamed: 0" or col.startswith("Unnamed:")]
    # Find target column
    actual_target_column, find_warnings = find_target_column(df_columns, REQUESTED_TARGET_COLUMN)
    warnings_list.extend(find_warnings)
    if actual_target_column is None:
        msg = f"Target column '{REQUESTED_TARGET_COLUMN}' not found in data columns."
        warnings_list.append(msg)
        report_content = f"# Modelling Report\n\n## Error\n\n{msg}\n"
        write_report(report_content)
        result = {
            "status": "error",
            "message": "target_column_not_found",
            "warnings": warnings_list,
            "stage": STAGE,
            "input_path": INPUT_CSV_PATH,
            "report_path": MODELLING_REPORT_PATH,
            "best_model_path": BEST_MODEL_PATH,
            "best_metrics_path": BEST_METRICS_PATH,
            "predictions_path": PREDICTIONS_PATH,
            "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
        }
        print(json.dumps(sanitize_for_json(result), ensure_ascii=False))
        return

    # Prepare target
    y_raw = df[actual_target_column]
    y_numeric = pd.to_numeric(y_raw, errors="coerce")
    valid_target_mask = y_numeric.notna()
    y_numeric = y_numeric[valid_target_mask]
    if y_numeric.shape[0] < 50:
        msg = f"Not enough valid target rows after filtering: {y_numeric.shape[0]} rows found, minimum 50 required."
        warnings_list.append(msg)
        report_content = f"# Modelling Report\n\n## Error\n\n{msg}\n"
        write_report(report_content)
        result = {
            "status": "error",
            "message": "not_enough_valid_target_rows",
            "warnings": warnings_list,
            "stage": STAGE,
            "input_path": INPUT_CSV_PATH,
            "report_path": MODELLING_REPORT_PATH,
            "best_model_path": BEST_MODEL_PATH,
            "best_metrics_path": BEST_METRICS_PATH,
            "predictions_path": PREDICTIONS_PATH,
            "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
        }
        print(json.dumps(sanitize_for_json(result), ensure_ascii=False))
        return

    problem_type = "regression"

    # Log transform target
    y = np.log1p(y_numeric)

    # Prepare features
    df_valid = df.loc[valid_target_mask].copy()
    X = df_valid.drop(columns=technical_cols + [actual_target_column], errors='ignore')

    # Identify feature types
    text_columns = []
    # Priority for text columns: combined_text, else Review Text or Title
    if "combined_text" in X.columns:
        text_columns = ["combined_text"]
    else:
        if "Review Text" in X.columns:
            text_columns = ["Review Text"]
        elif "Title" in X.columns:
            text_columns = ["Title"]
    if not text_columns:
        warnings_list.append("No text columns found for modeling.")

    # Numeric columns: all numeric except technical and target
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    # Categorical columns: object, category, bool excluding text columns
    cat_columns_all = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Only low cardinality categorical columns: Division Name, Department Name, Class Name
    allowed_cat_cols = {"Division Name", "Department Name", "Class Name"}
    categorical_columns = []
    for col in cat_columns_all:
        if col in text_columns:
            continue
        if col in allowed_cat_cols:
            unique_count = X[col].nunique(dropna=False)
            if unique_count > 100:
                warnings_list.append(f"Categorical column '{col}' has high cardinality ({unique_count}), excluded.")
            else:
                categorical_columns.append(col)

    # If no numeric features, add fallback constant feature
    if len(numeric_columns) == 0:
        X["constant_feature"] = 1
        numeric_columns = ["constant_feature"]

    # Split train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
    except Exception as e:
        # fallback without stratify param if error
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        except Exception as e2:
            msg = f"Failed to split data: {str(e2)}"
            warnings_list.append(msg)
            report_content = f"# Modelling Report\n\n## Error\n\n{msg}\n"
            write_report(report_content)
            result = {
                "status": "error",
                "message": "failed_to_split_data",
                "warnings": warnings_list,
                "stage": STAGE,
                "input_path": INPUT_CSV_PATH,
                "report_path": MODELLING_REPORT_PATH,
                "best_model_path": BEST_MODEL_PATH,
                "best_metrics_path": BEST_METRICS_PATH,
                "predictions_path": PREDICTIONS_PATH,
                "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
            }
            print(json.dumps(sanitize_for_json(result), ensure_ascii=False))
            return

    # Define models to train
    models_to_train = []

    # 1. DummyRegressor numeric only
    numeric_transformer_dummy = SimpleImputer(strategy="median")
    pipeline_dummy = Pipeline([
        ("imputer", numeric_transformer_dummy),
        ("model", DummyRegressor(strategy="mean"))
    ])
    models_to_train.append(("DummyRegressor_numeric", pipeline_dummy, numeric_columns, [], []))

    # 2. Ridge numeric only
    numeric_transformer_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pipeline_ridge = Pipeline([
        ("preprocessor", numeric_transformer_ridge),
        ("model", Ridge(random_state=42))
    ])
    models_to_train.append(("Ridge_numeric", pipeline_ridge, numeric_columns, [], []))

    # 3. RandomForestRegressor numeric only
    numeric_transformer_rf = SimpleImputer(strategy="median")
    pipeline_rf = Pipeline([
        ("imputer", numeric_transformer_rf),
        ("model", RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1
        ))
    ])
    models_to_train.append(("RandomForest_numeric", pipeline_rf, numeric_columns, [], []))

    # 4. Ridge numeric + text + categorical (if text present)
    if text_columns:
        # Text transformer
        min_df_val = 2 if len(X_train) >= 10 else 1
        text_transformer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 1),
            min_df=min_df_val,
            dtype=np.float32
        )
        # Numeric transformer
        numeric_transformer_full = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        # Categorical transformer
        categorical_transformer_full = None
        if categorical_columns:
            try:
                categorical_transformer_full = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except Exception as e:
                warnings_list.append(f"Failed to create OneHotEncoder: {str(e)}")
                categorical_transformer_full = None

        # Compose preprocessor
        transformers = []
        if numeric_columns:
            transformers.append(("num", numeric_transformer_full, numeric_columns))
        if categorical_transformer_full and categorical_columns:
            transformers.append(("cat", categorical_transformer_full, categorical_columns))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

        # Full pipeline with text and numeric/categorical
        class TextSelector:
            def __init__(self, key):
                self.key = key
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X[self.key].fillna("").values

        from sklearn.base import BaseEstimator, TransformerMixin

        class TextTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, vectorizer):
                self.vectorizer = vectorizer
            def fit(self, X, y=None):
                self.vectorizer.fit(X)
                return self
            def transform(self, X):
                return self.vectorizer.transform(X)

        # Build pipeline with FeatureUnion for text and other features
        try:
            from sklearn.pipeline import FeatureUnion
            text_pipe = Pipeline([
                ("selector", TextSelector(text_columns[0])),
                ("tfidf", text_transformer)
            ])
            if transformers:
                full_preprocessor = FeatureUnion([
                    ("text", text_pipe),
                    ("other", preprocessor)
                ])
            else:
                full_preprocessor = FeatureUnion([
                    ("text", text_pipe)
                ])
            pipeline_full = Pipeline([
                ("features", full_preprocessor),
                ("model", Ridge(random_state=42))
            ])
            models_to_train.append(("Ridge_numeric_text_cat", pipeline_full, numeric_columns, categorical_columns, text_columns))
        except Exception as e:
            warnings_list.append(f"Failed to create full pipeline with text and categorical: {str(e)}")

    # Train models
    trained_models = {}
    trained_metrics = {}
    for model_name, pipeline, num_cols, cat_cols, txt_cols in models_to_train:
        model_trained_flags[model_name] = False
        try:
            # Prepare data for this model
            X_train_sub = pd.DataFrame()
            X_test_sub = pd.DataFrame()
            if txt_cols:
                # For text pipeline, pass full X_train and X_test
                X_train_sub = X_train.copy()
                X_test_sub = X_test.copy()
            else:
                # Numeric only
                X_train_sub = X_train[num_cols].copy()
                X_test_sub = X_test[num_cols].copy()
            # Fit
            pipeline.fit(X_train_sub, y_train)
            model_trained_flags[model_name] = True
            # Predict
            y_pred_log = pipeline.predict(X_test_sub)
            # Metrics on log scale
            rmse_log = safe_rmse(y_test, y_pred_log)
            mae_log = safe_mae(y_test, y_pred_log)
            r2_log = safe_r2(y_test, y_pred_log)
            # Back transform predictions
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_test)
            # Clip negative predictions to zero
            y_pred_clip = np.clip(y_pred, 0, None)
            # Metrics on original scale
            rmse = safe_rmse(y_true, y_pred_clip)
            mae = safe_mae(y_true, y_pred_clip)
            r2 = safe_r2(y_true, y_pred_clip)
            rank_corr, rank_corr_warn = safe_rank_correlation(y_true, y_pred_clip)
            if rank_corr_warn:
                warnings_list.append(f"Model {model_name} rank correlation warning: {rank_corr_warn}")
            top_10_rate = safe_top_10_helpful_rate(y_true, y_pred_clip)
            trained_models[model_name] = pipeline
            trained_metrics[model_name] = {
                "rmse_log": rmse_log,
                "mae_log": mae_log,
                "r2_log": r2_log,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "rank_correlation": rank_corr,
                "top_10_percent_helpful_rate": top_10_rate
            }
        except Exception as e:
            err_msg = str(e)
            tb = traceback.format_exc()
            model_errors.append(f"Model {model_name} training failed: {err_msg}")
            model_error_messages[model_name] = err_msg
            model_error_tracebacks[model_name] = tb

    # Check if any model trained
    trained_model_names = [name for name, trained in model_trained_flags.items() if trained]
    if len(trained_model_names) == 0:
        msg = "No models were successfully trained."
        warnings_list.append(msg)
        report_content = f"# Modelling Report\n\n## Model Training Errors\n\n"
        for mn in model_error_messages:
            report_content += f"### {mn}\n\nError message:\n```\n{model_error_messages[mn]}\n```\n\nTraceback (last 10 lines):\n```\n{''.join(model_error_tracebacks[mn].splitlines()[-10:])}\n```\n\n"
        write_report(report_content)
        result = {
            "status": "error",
            "message": "no_models_trained",
            "warnings": warnings_list,
            "model_errors_count": len(model_error_messages),
            "first_model_error": next(iter(model_error_messages.values())) if model_error_messages else None,
            "stage": STAGE,
            "input_path": INPUT_CSV_PATH,
            "report_path": MODELLING_REPORT_PATH,
            "best_model_path": BEST_MODEL_PATH,
            "best_metrics_path": BEST_METRICS_PATH,
            "predictions_path": PREDICTIONS_PATH,
            "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
        }
        print(json.dumps(sanitize_for_json(result), ensure_ascii=False))
        return

    # Select best model by rmse (original scale)
    def model_sort_key(mn):
        m = trained_metrics[mn]
        # Use large number if metric is None
        rmse = m.get("rmse") if m.get("rmse") is not None else float("inf")
        mae = m.get("mae") if m.get("mae") is not None else float("inf")
        rank_corr = m.get("rank_correlation") if m.get("rank_correlation") is not None else -float("inf")
        top10 = m.get("top_10_percent_helpful_rate") if m.get("top_10_percent_helpful_rate") is not None else -float("inf")
        return (rmse, mae, -rank_corr, -top10)

    best_model_name = min(trained_model_names, key=model_sort_key)
    best_metrics = trained_metrics[best_model_name]
    best_model = trained_models[best_model_name]

    # Refit best model on all valid data
    refit_warning = None
    try:
        if text_columns:
            X_full = X.copy()
        else:
            X_full = X[numeric_columns].copy()
        best_model.fit(X_full, y)
    except Exception as e:
        refit_warning = f"Refit of best model '{best_model_name}' on full data failed: {str(e)}"
        warnings_list.append(refit_warning)

    # Prepare predictions DataFrame
    try:
        if text_columns:
            X_pred = X.copy()
        else:
            X_pred = X[numeric_columns].copy()
        y_pred_log_full = best_model.predict(X_pred)
        y_pred_full = np.expm1(y_pred_log_full)
        y_true_full = np.expm1(y)
        y_pred_full_clip = np.clip(y_pred_full, 0, None)
        residual = y_true_full - y_pred_full_clip
        abs_error = np.abs(residual)
        # Create DataFrame
        pred_df = pd.DataFrame({
            "row_index": y.index,
            "actual_target": y_true_full,
            "predicted_target": y_pred_full_clip,
            "predicted_target_log": y_pred_log_full,
            "residual": residual,
            "abs_error": abs_error,
        }, index=y.index)
        # predicted_rank descending
        pred_df["predicted_rank"] = pred_df["predicted_target"].rank(method="min", ascending=False).astype(int)
        # predicted_percentile
        pred_df["predicted_percentile"] = pred_df["predicted_target"].rank(pct=True)
        pred_df["model_name"] = best_model_name
        pred_df["prediction_scope"] = "all_valid_rows"
        # Add extra columns if present
        extra_cols = ["Clothing ID", "Title", "Review Text", "Rating", "Recommended IND", "Division Name", "Department Name", "Class Name"]
        for col in extra_cols:
            if col in df_valid.columns:
                pred_df[col] = df_valid[col]
        # Reset index for saving
        pred_df_reset = pred_df.reset_index(drop=True)
    except Exception as e:
        warnings_list.append(f"Failed to prepare predictions DataFrame: {str(e)}")
        pred_df_reset = pd.DataFrame()

    # Save predictions to /files/predictions.csv
    predictions_save_status = "failed"
    try:
        pred_df_reset.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
        predictions_save_status = "saved"
    except Exception as e:
        warnings_list.append(f"Failed to save predictions to {PREDICTIONS_PATH}: {str(e)}")

    # Create MEMORY_DIR if not exists
    model_memory_save_status = "skipped"
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
    except Exception as e:
        warnings_list.append(f"Failed to create memory directory {MEMORY_DIR}: {str(e)}")

    # Save copy of predictions to memory
    try:
        pred_df_reset.to_csv(MEMORY_PREDICTIONS_PATH, index=False, encoding="utf-8")
    except Exception as e:
        warnings_list.append(f"Failed to save memory predictions to {MEMORY_PREDICTIONS_PATH}: {str(e)}")

    # Load historical model and metrics if exist
    previous_best_metric_value = None
    is_new_model_better = True
    historical_model = None
    historical_metrics = None
    try:
        if os.path.isfile(BEST_MODEL_PATH) and os.path.isfile(BEST_METRICS_PATH):
            with open(BEST_METRICS_PATH, "r", encoding="utf-8") as f:
                historical_metrics = json.load(f)
            with open(BEST_MODEL_PATH, "rb") as f:
                historical_model = pickle.load(f)
            prev_rmse = historical_metrics.get("best_metric_value")
            if prev_rmse is not None and best_metrics.get("rmse") is not None:
                previous_best_metric_value = prev_rmse
                if best_metrics["rmse"] >= prev_rmse:
                    is_new_model_better = False
    except Exception as e:
        warnings_list.append(f"Failed to load historical model or metrics: {str(e)}")
        is_new_model_better = True

    # Save model and metrics if new model better or no historical
    try:
        if is_new_model_better:
            model_info = {
                "model": best_model,
                "model_name": best_model_name,
                "target_column": actual_target_column,
                "problem_type": problem_type,
                "trained_at": datetime.datetime.now().isoformat(),
                "feature_info": {
                    "text_columns": text_columns,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                },
                "metrics": best_metrics,
                "predictions_path": MEMORY_PREDICTIONS_PATH,
            }
            with open(BEST_MODEL_PATH, "wb") as f:
                pickle.dump(model_info, f)
            with open(BEST_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json({
                    "stage": STAGE,
                    "business_task": BUSINESS_TASK,
                    "target_column": actual_target_column,
                    "problem_type": problem_type,
                    "best_model_name": best_model_name,
                    "best_metric_name": "rmse",
                    "best_metric_value": best_metrics.get("rmse"),
                    "all_metrics": best_metrics,
                    "trained_at": model_info["trained_at"],
                    "input_path": INPUT_CSV_PATH,
                    "predictions_path": MEMORY_PREDICTIONS_PATH,
                }), f, ensure_ascii=False)
            model_memory_save_status = "saved"
        else:
            model_memory_save_status = "skipped"
    except Exception as e:
        warnings_list.append(f"Failed to save best model or metrics: {str(e)}")
        model_memory_save_status = "failed"

    # Compose markdown report
    report_lines = []
    report_lines.append(f"# Modelling Report")
    report_lines.append(f"## Context")
    report_lines.append(f"- Stage: {STAGE}")
    report_lines.append(f"- Business task: {BUSINESS_TASK}")
    report_lines.append(f"- Input file: {INPUT_CSV_PATH}")
    report_lines.append(f"## Data Check")
    report_lines.append(f"- Total rows: {len(df)}")
    report_lines.append(f"- Valid target rows: {len(y_numeric)}")
    report_lines.append(f"- Target column used: {actual_target_column}")
    report_lines.append(f"## Feature Types")
    report_lines.append(f"- Numeric columns ({len(numeric_columns)}): {', '.join(numeric_columns)}")
    report_lines.append(f"- Categorical columns ({len(categorical_columns)}): {', '.join(categorical_columns)}")
    report_lines.append(f"- Text columns ({len(text_columns)}): {', '.join(text_columns)}")
    report_lines.append(f"## Data Split")
    report_lines.append(f"- Train rows: {len(X_train)}")
    report_lines.append(f"- Test rows: {len(X_test)}")
    report_lines.append(f"## Model Training")
    for mn in model_error_messages:
        report_lines.append(f"### {mn} - Failed")
        report_lines.append(f"Error message:\n```\n{model_error_messages[mn]}\n```\n")
        report_lines.append(f"Traceback (last 10 lines):\n```\n{''.join(model_error_tracebacks[mn].splitlines()[-10:])}\n```\n")
    for mn in trained_model_names:
        m = trained_metrics[mn]
        report_lines.append(f"### {mn} - Trained")
        report_lines.append(f"- RMSE (log scale): {m.get('rmse_log')}")
        report_lines.append(f"- MAE (log scale): {m.get('mae_log')}")
        report_lines.append(f"- R2 (log scale): {m.get('r2_log')}")
        report_lines.append(f"- RMSE (original scale): {m.get('rmse')}")
        report_lines.append(f"- MAE (original scale): {m.get('mae')}")
        report_lines.append(f"- R2 (original scale): {m.get('r2')}")
        report_lines.append(f"- Rank correlation: {m.get('rank_correlation')}")
        report_lines.append(f"- Top 10% helpful rate: {m.get('top_10_percent_helpful_rate')}")
    report_lines.append(f"## Best Model")
    report_lines.append(f"- Model name: {best_model_name}")
    report_lines.append(f"- RMSE (original scale): {best_metrics.get('rmse')}")
    report_lines.append(f"- MAE (original scale): {best_metrics.get('mae')}")
    report_lines.append(f"- Rank correlation: {best_metrics.get('rank_correlation')}")
    report_lines.append(f"- Top 10% helpful rate: {best_metrics.get('top_10_percent_helpful_rate')}")
    if refit_warning:
        report_lines.append(f"## Refit Warning\n- {refit_warning}")
    report_lines.append(f"## Predictions")
    report_lines.append(f"- Predictions saved to: {PREDICTIONS_PATH}")
    report_lines.append(f"## Historical Model Comparison")
    if previous_best_metric_value is not None:
        report_lines.append(f"- Previous best RMSE: {previous_best_metric_value}")
        report_lines.append(f"- New model better: {is_new_model_better}")
    else:
        report_lines.append(f"- No historical model found or failed to load.")
    report_lines.append(f"## Warnings")
    if warnings_list:
        for w in warnings_list:
            report_lines.append(f"- {w}")
    else:
        report_lines.append("- None")
    report_lines.append(f"## Conclusion")
    if len(trained_model_names) == 0:
        report_lines.append("No models were successfully trained.")
    else:
        report_lines.append(f"Best model '{best_model_name}' selected and saved.")
    write_report("\n".join(report_lines))

    # Compose final result JSON
    status = "success"
    if warnings_list:
        status = "warning"
    if len(trained_model_names) == 0:
        status = "error"
    result = {
        "status": status,
        "stage": STAGE,
        "input_path": INPUT_CSV_PATH,
        "report_path": MODELLING_REPORT_PATH,
        "best_model_path": BEST_MODEL_PATH,
        "best_metrics_path": BEST_METRICS_PATH,
        "predictions_path": PREDICTIONS_PATH,
        "memory_predictions_path": MEMORY_PREDICTIONS_PATH,
        "target_column": actual_target_column,
        "problem_type": problem_type,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "models_trained_count": len(trained_model_names),
        "best_new_model_name": best_model_name if trained_model_names else None,
        "best_new_metric_name": "rmse",
        "best_new_metric_value": best_metrics.get("rmse") if trained_model_names else None,
        "previous_best_metric_value": previous_best_metric_value,
        "is_new_model_better": is_new_model_better,
        "model_memory_save_status": model_memory_save_status,
        "predictions_save_status": predictions_save_status,
        "predictions_rows": len(pred_df_reset) if not pred_df_reset.empty else 0,
        "warnings": warnings_list,
    }
    if status == "error" and len(trained_model_names) == 0:
        result["model_errors_count"] = len(model_error_messages)
        result["first_model_error"] = next(iter(model_error_messages.values())) if model_error_messages else None
        result["message"] = "no_models_trained"
    print(json.dumps(sanitize_for_json(result), ensure_ascii=False))

if __name__ == "__main__":
    main()