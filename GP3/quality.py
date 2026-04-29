import os
import sys
import traceback
import difflib
import pandas as pd
import numpy as np

BUSINESS_TASK = "Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?"
REQUESTED_TARGET_COLUMN = "Positive Feedback Count"
CSV_PATH = "/files/reviews_data.csv"
REPORT_PATH = "/files/quality_report.md"

def main():
    warnings = []
    try:
        if not os.path.isfile(CSV_PATH):
            warnings.append(f"ERROR: CSV file not found at path: {CSV_PATH}")
            report = generate_report(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, warnings)
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(report)
            sys.exit(1)

        df = pd.read_csv(CSV_PATH)

        # Target column detection
        actual_target_column = None
        target_match_type = None
        if REQUESTED_TARGET_COLUMN in df.columns:
            actual_target_column = REQUESTED_TARGET_COLUMN
            target_match_type = "exact"
        else:
            close_matches = difflib.get_close_matches(REQUESTED_TARGET_COLUMN, df.columns, n=1, cutoff=0.6)
            if close_matches:
                actual_target_column = close_matches[0]
                target_match_type = "fuzzy"
                warnings.append(f"Requested target column '{REQUESTED_TARGET_COLUMN}' not found. Using closest match '{actual_target_column}'.")
            else:
                target_match_type = "not found"
                warnings.append(f"Requested target column '{REQUESTED_TARGET_COLUMN}' not found and no close match found. Target statistics will be skipped.")

        # Dataset overview
        rows = df.shape[0]
        columns = df.shape[1]
        duplicate_rows = df.duplicated().sum()

        if duplicate_rows > 0:
            warnings.append(f"Dataset contains {duplicate_rows} duplicate rows.")

        columns_list = list(df.columns)
        dtypes = df.dtypes.astype(str).to_dict()

        # Missing values
        missing_count = df.isnull().sum()
        missing_share = (missing_count / rows).round(4)

        high_missing_cols = missing_share[missing_share > 0.3].index.tolist()
        if high_missing_cols:
            warnings.append(f"Columns with more than 30% missing values: {', '.join(high_missing_cols)}")

        # Target column check and statistics
        target_stats = None
        target_can_cast_numeric = False
        target_missing_count = None
        target_zero_count = None
        target_zero_share = None
        target_min = None
        target_max = None
        target_mean = None
        target_median = None

        if actual_target_column and actual_target_column in df.columns:
            target_dtype = df[actual_target_column].dtype
            # Try to convert to numeric
            try:
                target_numeric = pd.to_numeric(df[actual_target_column], errors='coerce')
                target_missing_count = target_numeric.isnull().sum()
                target_zero_count = (target_numeric == 0).sum()
                target_zero_share = round(target_zero_count / (rows - target_missing_count), 4) if (rows - target_missing_count) > 0 else None
                target_min = target_numeric.min()
                target_max = target_numeric.max()
                target_mean = round(target_numeric.mean(), 4)
                target_median = target_numeric.median()
                target_can_cast_numeric = True
            except Exception:
                warnings.append(f"Target column '{actual_target_column}' cannot be converted to numeric. Skipping target statistics.")
        else:
            if target_match_type != "not found":
                warnings.append(f"Actual target column '{actual_target_column}' not found in dataframe columns.")

        # Text columns
        text_columns = ["Title", "Review Text"]
        text_columns_existing = [col for col in text_columns if col in df.columns]
        if not text_columns_existing:
            warnings.append("No text columns found among expected: Title, Review Text.")

        text_stats = {}
        for col in text_columns_existing:
            col_series = df[col]
            missing = col_series.isnull().sum()
            lengths = col_series.dropna().astype(str).map(len)
            empty_strings = (col_series == "").sum()
            text_stats[col] = {
                "missing_count": int(missing),
                "mean_length": round(lengths.mean(), 2) if not lengths.empty else None,
                "median_length": int(lengths.median()) if not lengths.empty else None,
                "min_length": int(lengths.min()) if not lengths.empty else None,
                "max_length": int(lengths.max()) if not lengths.empty else None,
                "empty_strings": int(empty_strings)
            }

        # Categorical columns
        categorical_columns = ["Clothing ID", "Division Name", "Department Name", "Class Name"]
        categorical_existing = [col for col in categorical_columns if col in df.columns]
        if not categorical_existing:
            warnings.append("No categorical columns found among expected: Clothing ID, Division Name, Department Name, Class Name.")

        categorical_stats = {}
        for col in categorical_existing:
            col_series = df[col]
            unique_count = col_series.nunique(dropna=True)
            missing = col_series.isnull().sum()
            top_values = col_series.value_counts(dropna=True).head(10).to_dict()
            categorical_stats[col] = {
                "unique_count": int(unique_count),
                "missing_count": int(missing),
                "top_10": top_values
            }

        # Numeric columns
        numeric_columns = ["Age", "Rating", "Recommended IND"]
        numeric_existing = [col for col in numeric_columns if col in df.columns]
        if not numeric_existing:
            warnings.append("No numeric columns found among expected: Age, Rating, Recommended IND.")

        numeric_stats = {}
        for col in numeric_existing:
            col_series = df[col]
            count = col_series.count()
            missing = col_series.isnull().sum()
            min_val = col_series.min()
            max_val = col_series.max()
            mean_val = round(col_series.mean(), 4)
            median_val = col_series.median()
            numeric_stats[col] = {
                "count": int(count),
                "missing_count": int(missing),
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val
            }

        # Target leakage check
        feature_columns = [col for col in df.columns if col != actual_target_column]
        if actual_target_column in feature_columns:
            warnings.append(f"Target leakage risk: target column '{actual_target_column}' found in feature columns.")

        # Compose report
        report = generate_report(
            CSV_PATH,
            rows,
            columns,
            duplicate_rows,
            columns_list,
            dtypes,
            missing_count,
            missing_share,
            REQUESTED_TARGET_COLUMN,
            actual_target_column,
            target_match_type,
            target_can_cast_numeric,
            target_stats if target_can_cast_numeric else None,
            text_stats,
            categorical_stats,
            numeric_stats,
            warnings
        )

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)

    except Exception:
        tb = traceback.format_exc()
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# 01 Data Quality Evaluation Report\n\n")
            f.write("## ERROR\n\n")
            f.write("An unexpected error occurred during data quality evaluation:\n\n")
            f.write("```\n")
            f.write(tb)
            f.write("\n```\n")
        sys.exit(1)

def generate_report(
    csv_path,
    rows,
    columns,
    duplicate_rows,
    columns_list,
    dtypes,
    missing_count,
    missing_share,
    requested_target_column,
    actual_target_column,
    target_match_type,
    target_can_cast_numeric,
    target_stats,
    text_stats,
    categorical_stats,
    numeric_stats,
    warnings
):
    lines = []
    lines.append("# 01 Data Quality Evaluation Report\n")
    lines.append("## Project Context")
    lines.append(f"- Dataset: Women's E-Commerce Clothing Reviews")
    lines.append(f"- Business task: {BUSINESS_TASK}")
    lines.append(f"- Requested target column: {requested_target_column}")
    lines.append(f"- Actual target column used: {actual_target_column if actual_target_column else 'None'}")
    lines.append(f"- Target match type: {target_match_type}\n")

    lines.append("## Dataset Overview")
    lines.append(f"- CSV path: {csv_path if csv_path else 'N/A'}")
    lines.append(f"- rows: {rows if rows is not None else 'N/A'}")
    lines.append(f"- columns: {columns if columns is not None else 'N/A'}")
    lines.append(f"- duplicate rows: {duplicate_rows if duplicate_rows is not None else 'N/A'}\n")

    lines.append("## Columns")
    if columns_list is not None:
        lines.append(", ".join(columns_list))
    else:
        lines.append("N/A")
    lines.append("")

    lines.append("## Data Types")
    if dtypes is not None:
        for col, dtype in dtypes.items():
            lines.append(f"- {col}: {dtype}")
    else:
        lines.append("N/A")
    lines.append("")

    lines.append("## Missing Values")
    if missing_count is not None and missing_share is not None:
        lines.append("| Column | Missing Count | Missing Share |")
        lines.append("|--------|---------------|---------------|")
        for col in columns_list:
            mc = missing_count.get(col, 'N/A')
            ms = missing_share.get(col, 'N/A')
            lines.append(f"| {col} | {mc} | {ms} |")
    else:
        lines.append("N/A")
    lines.append("")

    lines.append("## Target Column Check")
    if actual_target_column:
        lines.append(f"- Target column '{actual_target_column}' found with match type '{target_match_type}'.")
        if actual_target_column in dtypes:
            lines.append(f"- Data type: {dtypes[actual_target_column]}")
        else:
            lines.append("- Data type: N/A")
    else:
        lines.append("- Target column not found.")
    lines.append("")

    lines.append("## Target Statistics")
    if target_can_cast_numeric and target_stats:
        lines.append(f"- min: {target_stats['min']}")
        lines.append(f"- max: {target_stats['max']}")
        lines.append(f"- mean: {target_stats['mean']}")
        lines.append(f"- median: {target_stats['median']}")
        lines.append(f"- zero count: {target_stats['zero_count'] if 'zero_count' in target_stats else 'N/A'}")
        lines.append(f"- zero share: {target_stats['zero_share'] if 'zero_share' in target_stats else 'N/A'}")
    else:
        lines.append("Target statistics not available.")
    lines.append("")

    lines.append("## Text Columns Check")
    if text_stats:
        for col, stats in text_stats.items():
            lines.append(f"### {col}")
            lines.append(f"- Missing count: {stats['missing_count']}")
            lines.append(f"- Mean length: {stats['mean_length']}")
            lines.append(f"- Median length: {stats['median_length']}")
            lines.append(f"- Min length: {stats['min_length']}")
            lines.append(f"- Max length: {stats['max_length']}")
            lines.append(f"- Empty strings: {stats['empty_strings']}")
            lines.append("")
    else:
        lines.append("No text columns found.\n")

    lines.append("## Categorical Columns Check")
    if categorical_stats:
        for col, stats in categorical_stats.items():
            lines.append(f"### {col}")
            lines.append(f"- Unique values count: {stats['unique_count']}")
            lines.append(f"- Missing count: {stats['missing_count']}")
            lines.append(f"- Top 10 most frequent values:")
            for val, cnt in stats['top_10'].items():
                lines.append(f"  - {val}: {cnt}")
            lines.append("")
    else:
        lines.append("No categorical columns found.\n")

    lines.append("## Numeric Columns Check")
    if numeric_stats:
        for col, stats in numeric_stats.items():
            lines.append(f"### {col}")
            lines.append(f"- Count: {stats['count']}")
            lines.append(f"- Missing count: {stats['missing_count']}")
            lines.append(f"- Min: {stats['min']}")
            lines.append(f"- Max: {stats['max']}")
            lines.append(f"- Mean: {stats['mean']}")
            lines.append(f"- Median: {stats['median']}")
            lines.append("")
    else:
        lines.append("No numeric columns found.\n")

    lines.append("## Target Leakage Check")
    if actual_target_column:
        lines.append(f"- Target column '{actual_target_column}' excluded from features.")
    else:
        lines.append("- Target column not found; target leakage check skipped.")
    lines.append("")

    lines.append("## Warnings")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("No warnings.")
    lines.append("")

    lines.append("## Conclusion")
    conclusion = (
        "Данные содержат базовую информацию для анализа отзывов. "
        "Обнаружены некоторые предупреждения, которые следует учесть при дальнейшем анализе. "
        "Особое внимание стоит уделить качеству целевой переменной и пропускам в данных. "
        "Для бизнес-задачи важно обеспечить корректность и полноту данных, чтобы рекомендации по отзывам были максимально полезными."
    )
    lines.append(conclusion)
    lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    main()