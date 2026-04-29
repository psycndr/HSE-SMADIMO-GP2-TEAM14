import os
import json
import traceback
import difflib
import re
import pandas as pd
import numpy as np

STAGE = "04_feature_engineering"
BUSINESS_TASK = "Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?"
REQUESTED_TARGET_COLUMN = "Positive Feedback Count"
CLEAN_CSV_PATH = "/files/review_data_clean.csv"
PREPARED_CSV_PATH = "/files/reviews_data_prepared.csv"
FEATURE_REPORT_PATH = "/files/feature_engeneering_report.md"

def main():
    warnings = []
    try:
        if not os.path.isfile(CLEAN_CSV_PATH):
            warnings.append(f"Input file not found: {CLEAN_CSV_PATH}")
            report_content = f"# {STAGE} Отчет по обработке признаков\n\n" \
                             f"Ошибка: входной файл не найден: {CLEAN_CSV_PATH}\n\n" \
                             f"Предупреждения:\n"
            for w in warnings:
                report_content += f"- {w}\n"
            with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(report_content)
            result = {
                "status": "error",
                "stage": STAGE,
                "message": "input_file_not_found",
                "input_path": CLEAN_CSV_PATH,
                "prepared_csv_path": None,
                "report_path": FEATURE_REPORT_PATH,
                "requested_target_column": REQUESTED_TARGET_COLUMN,
                "actual_target_column": None,
                "target_match_type": "not_found",
                "input_rows": 0,
                "output_rows": 0,
                "created_features_count": 0,
                "warnings": warnings
            }
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return

        try:
            df = pd.read_csv(CLEAN_CSV_PATH)
        except Exception as e:
            tb = traceback.format_exc()
            report_content = f"# {STAGE} Отчет по обработке признаков\n\n" \
                             f"Ошибка при чтении CSV:\n```\n{tb}\n```\n\n" \
                             f"Предупреждения:\n"
            for w in warnings:
                report_content += f"- {w}\n"
            with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(report_content)
            result = {
                "status": "error",
                "stage": STAGE,
                "message": "failed_to_read_csv",
                "input_path": CLEAN_CSV_PATH,
                "prepared_csv_path": None,
                "report_path": FEATURE_REPORT_PATH,
                "requested_target_column": REQUESTED_TARGET_COLUMN,
                "actual_target_column": None,
                "target_match_type": "not_found",
                "input_rows": 0,
                "output_rows": 0,
                "created_features_count": 0,
                "warnings": warnings
            }
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return

        input_rows = df.shape[0]
        input_columns_count = df.shape[1]
        real_columns = list(df.columns)

        def normalize_colname(name):
            if not isinstance(name, str):
                name = str(name)
            name = name.strip()
            name = re.sub(r"\s+", " ", name)
            return name.lower()

        normalized_columns = [normalize_colname(c) for c in real_columns]
        requested_target_norm = normalize_colname(REQUESTED_TARGET_COLUMN)

        actual_target_column = None
        target_match_type = "not_found"

        # Exact match
        if REQUESTED_TARGET_COLUMN in real_columns:
            actual_target_column = REQUESTED_TARGET_COLUMN
            target_match_type = "exact"
        else:
            # Normalized match
            if requested_target_norm in normalized_columns:
                idx = normalized_columns.index(requested_target_norm)
                actual_target_column = real_columns[idx]
                target_match_type = "normalized"
            else:
                # Fuzzy match
                close_matches = difflib.get_close_matches(requested_target_norm, normalized_columns, n=1, cutoff=0.35)
                if close_matches:
                    idx = normalized_columns.index(close_matches[0])
                    actual_target_column = real_columns[idx]
                    target_match_type = "fuzzy"

        if actual_target_column is None:
            warnings.append("Requested target column was not found; feature engineering continued without target-specific checks")

        # If target found, convert to numeric and check NaNs
        if actual_target_column is not None:
            df[actual_target_column] = pd.to_numeric(df[actual_target_column], errors="coerce")
            if df[actual_target_column].isnull().any():
                warnings.append(f"NaN values found in target column '{actual_target_column}' after conversion to numeric")

        # Create working copy
        df_work = df.copy()

        # combined_text creation
        has_title_col = "Title" in df_work.columns
        has_review_text_col = "Review Text" in df_work.columns

        if has_title_col:
            title_series = df_work["Title"].fillna("").astype(str)
        else:
            title_series = pd.Series([""] * len(df_work))
            warnings.append("Column 'Title' not found; combined_text will use only 'Review Text' or be empty")

        if has_review_text_col:
            review_text_series = df_work["Review Text"].fillna("").astype(str)
        else:
            review_text_series = pd.Series([""] * len(df_work))
            warnings.append("Column 'Review Text' not found; combined_text will use only 'Title' or be empty")

        if not has_title_col and not has_review_text_col:
            warnings.append("Both 'Title' and 'Review Text' columns are missing; combined_text will be empty strings")

        combined_text = None
        if has_title_col and has_review_text_col:
            combined_text = title_series.str.strip() + " " + review_text_series.str.strip()
            combined_text = combined_text.str.strip()
        elif has_title_col:
            combined_text = title_series.str.strip()
        elif has_review_text_col:
            combined_text = review_text_series.str.strip()
        else:
            combined_text = pd.Series([""] * len(df_work))

        df_work["combined_text"] = combined_text.fillna("")

        # Text features
        if has_review_text_col:
            review_text_len_chars = review_text_series.str.len().fillna(0).astype(int)
            review_text_word_count = review_text_series.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            review_text_len_chars = pd.Series([0] * len(df_work))
            review_text_word_count = pd.Series([0] * len(df_work))
            warnings.append("Column 'Review Text' missing; review_text_len_chars and review_text_word_count set to 0")

        if has_title_col:
            title_len_chars = title_series.str.len().fillna(0).astype(int)
            title_word_count = title_series.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            title_len_chars = pd.Series([0] * len(df_work))
            title_word_count = pd.Series([0] * len(df_work))
            warnings.append("Column 'Title' missing; title_len_chars and title_word_count set to 0")

        df_work["review_text_len_chars"] = review_text_len_chars
        df_work["review_text_word_count"] = review_text_word_count
        df_work["title_len_chars"] = title_len_chars
        df_work["title_word_count"] = title_word_count

        # Binary features for presence of text
        def has_text(series):
            return series.fillna("").astype(str).str.strip().astype(bool).astype(int)

        df_work["has_title"] = has_text(df_work["Title"]) if has_title_col else pd.Series([0]*len(df_work))
        df_work["has_review_text"] = has_text(df_work["Review Text"]) if has_review_text_col else pd.Series([0]*len(df_work))

        # Rating features
        if "Rating" in df_work.columns:
            df_work["Rating"] = pd.to_numeric(df_work["Rating"], errors="coerce")
            df_work["rating_is_low"] = (df_work["Rating"] <= 2).fillna(False).astype(int)
            df_work["rating_is_high"] = (df_work["Rating"] >= 4).fillna(False).astype(int)
        else:
            df_work["rating_is_low"] = pd.Series([0]*len(df_work))
            df_work["rating_is_high"] = pd.Series([0]*len(df_work))
            warnings.append("Column 'Rating' missing; rating_is_low and rating_is_high set to 0")

        # Keyword features on combined_text
        combined_text_lower = df_work["combined_text"].str.lower()

        fit_keywords = ["fit", "fits", "fitting", "size", "small", "large", "tight", "loose", "petite", "waist", "hips"]
        quality_keywords = ["quality", "fabric", "material", "cheap", "expensive", "comfortable", "uncomfortable"]
        size_related_keywords = ["size", "small", "large", "tight", "loose", "petite", "waist", "hips"]

        def count_keywords(text_series, keywords):
            pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
            return text_series.str.count(pattern)

        df_work["fit_keyword_count"] = count_keywords(combined_text_lower, fit_keywords).fillna(0).astype(int)
        df_work["quality_keyword_count"] = count_keywords(combined_text_lower, quality_keywords).fillna(0).astype(int)

        pattern_size_flag = r"\b(" + "|".join(re.escape(k) for k in size_related_keywords) + r")\b"
        df_work["size_keyword_flag"] = combined_text_lower.str.contains(pattern_size_flag).fillna(False).astype(int)

        # Numeric columns processing
        for col in ["Age", "Rating", "Recommended IND"]:
            if col in df_work.columns:
                df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
            else:
                warnings.append(f"Column '{col}' missing; skipped numeric conversion")

        # Categorical columns processing
        for col in ["Clothing ID", "Division Name", "Department Name", "Class Name"]:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna("Unknown").astype(str)
            else:
                warnings.append(f"Column '{col}' missing; skipped categorical processing")

        # New features list
        new_features = [
            "combined_text",
            "review_text_len_chars",
            "review_text_word_count",
            "title_len_chars",
            "title_word_count",
            "has_title",
            "has_review_text",
            "rating_is_low",
            "rating_is_high",
            "fit_keyword_count",
            "quality_keyword_count",
            "size_keyword_flag"
        ]

        created_features_count = len(new_features)
        if created_features_count < 2:
            warnings.append("Меньше 2 новых признаков создано")

        # Feature columns
        if actual_target_column is not None:
            feature_columns = [c for c in df_work.columns if c != actual_target_column]
        else:
            feature_columns = list(df_work.columns)

        # Target leakage check
        if actual_target_column is not None:
            if actual_target_column in feature_columns:
                feature_columns.remove(actual_target_column)
                warnings.append(f"Целевая колонка '{actual_target_column}' удалена из признаков (target leakage)")
        else:
            warnings.append("Целевая колонка не найдена; проверка target leakage пропущена")

        # Save prepared CSV
        df_work.to_csv(PREPARED_CSV_PATH, index=False, encoding="utf-8")

        # Compose markdown report
        report_lines = [
            f"# 04 Feature Engineering Report",
            "",
            f"## Project Context",
            f"- Dataset: Women's E-Commerce Clothing Reviews",
            f"- Business task: {BUSINESS_TASK}",
            f"- Requested target column: {REQUESTED_TARGET_COLUMN}",
            f"- Actual target column used: {actual_target_column if actual_target_column is not None else 'None'}",
            f"- Target match type: {target_match_type}",
            f"- Input file: {CLEAN_CSV_PATH}",
            f"- Output file: {PREPARED_CSV_PATH}",
            "",
            f"## Feature Engineering Summary",
            f"- input rows: {input_rows}",
            f"- output rows: {df_work.shape[0]}",
            f"- input columns count: {input_columns_count}",
            f"- output columns count: {df_work.shape[1]}",
            f"- new features count: {created_features_count}",
            "",
            f"## Created Features",
            ", ".join(new_features),
            "",
            f"## Text Feature Logic",
            "Создан признак combined_text как объединение Title и Review Text (если обе колонки есть).",
            "Текстовые признаки: длина и количество слов в Review Text и Title.",
            "",
            f"## Keyword Feature Logic",
            "Подсчет количества упоминаний ключевых слов, связанных с посадкой (fit), качеством (quality) и размером (size) в combined_text.",
            "size_keyword_flag указывает на наличие size-related ключевых слов.",
            "",
            f"## Target Handling",
            f"- requested target column: {REQUESTED_TARGET_COLUMN}",
            f"- actual target column: {actual_target_column if actual_target_column is not None else 'None'}",
            f"- target match type: {target_match_type}",
        ]
        if actual_target_column is None:
            report_lines.append("- feature engineering продолжен без target-specific checks")
        report_lines.append("")
        report_lines.append("## Target Leakage Check")
        if actual_target_column is not None:
            report_lines.append("- target не использовался как признак")
        else:
            report_lines.append("- проверка target leakage пропущена")
        report_lines.append("")
        report_lines.append("## Warnings")
        if warnings:
            for w in warnings:
                report_lines.append(f"- {w}")
        else:
            report_lines.append("- нет предупреждений")
        report_lines.append("")
        report_lines.append("## Generated Artifacts")
        report_lines.append(f"- {PREPARED_CSV_PATH}")
        report_lines.append(f"- {FEATURE_REPORT_PATH}")

        report_content = "\n".join(report_lines)
        with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_content)

        status = "success" if not warnings else "warning"

        result = {
            "status": status,
            "stage": STAGE,
            "input_path": CLEAN_CSV_PATH,
            "prepared_csv_path": PREPARED_CSV_PATH,
            "report_path": FEATURE_REPORT_PATH,
            "requested_target_column": REQUESTED_TARGET_COLUMN,
            "actual_target_column": actual_target_column if actual_target_column is not None else None,
            "target_match_type": target_match_type,
            "input_rows": input_rows,
            "output_rows": df_work.shape[0],
            "created_features_count": created_features_count,
            "warnings": warnings
        }
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    except Exception as e:
        tb = traceback.format_exc()
        warnings.append(str(e))
        report_content = f"# {STAGE} Отчет по обработке признаков\n\n" \
                         f"Исключение:\n```\n{tb}\n```\n\n" \
                         f"Предупреждения:\n"
        for w in warnings:
            report_content += f"- {w}\n"
        with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_content)
        result = {
            "status": "error",
            "stage": STAGE,
            "message": str(e),
            "input_path": CLEAN_CSV_PATH,
            "prepared_csv_path": None,
            "report_path": FEATURE_REPORT_PATH,
            "requested_target_column": REQUESTED_TARGET_COLUMN,
            "actual_target_column": None,
            "target_match_type": "not_found",
            "input_rows": 0,
            "output_rows": 0,
            "created_features_count": 0,
            "warnings": warnings
        }
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return

if __name__ == "__main__":
    main()