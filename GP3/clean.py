import os
import sys
import json
import traceback
import difflib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE = "03_eda"
BUSINESS_TASK = "Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?"
REQUESTED_TARGET_COLUMN = "Positive Feedback Count"
CLEAN_CSV_PATH = "/files/data_reviews_clean.csv"
EDA_REPORT_PATH = "/files/eda_report.md"
OUTPUT_DIR = "/files"

def save_plot(fig, filename, warnings):
    try:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        warnings.append(f"Ошибка при сохранении графика {filename}: {str(e)}")
        return False

def plot_histogram(series, title, xlabel, ylabel, filename, warnings, bins=30):
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        ax.hist(series.dropna(), bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        saved = save_plot(fig, filename, warnings)
        return saved
    except Exception as e:
        warnings.append(f"Ошибка при построении гистограммы {filename}: {str(e)}")
        plt.close(fig)
        return False

def plot_bar(top_series, title, xlabel, ylabel, filename, warnings):
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        top_series.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        saved = save_plot(fig, filename, warnings)
        return saved
    except Exception as e:
        warnings.append(f"Ошибка при построении столбчатой диаграммы {filename}: {str(e)}")
        plt.close(fig)
        return False

def plot_scatter(x, y, title, xlabel, ylabel, filename, warnings):
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        ax.scatter(x, y, alpha=0.5, s=10)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        saved = save_plot(fig, filename, warnings)
        return saved
    except Exception as e:
        warnings.append(f"Ошибка при построении scatter plot {filename}: {str(e)}")
        plt.close(fig)
        return False

def plot_binned_agg(df, x_col, y_col, bins, title, xlabel, ylabel, filename, warnings):
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        df = df[[x_col, y_col]].dropna()
        df['bin'] = pd.cut(df[x_col], bins=bins)
        agg = df.groupby('bin')[y_col].mean()
        agg.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        saved = save_plot(fig, filename, warnings)
        return saved
    except Exception as e:
        warnings.append(f"Ошибка при построении binned aggregation {filename}: {str(e)}")
        plt.close(fig)
        return False

def plot_correlation_heatmap(df, filename, warnings):
    fig, ax = plt.subplots(figsize=(10,8))
    try:
        corr = df.corr()
        cax = ax.matshow(corr, cmap='coolwarm')
        fig.colorbar(cax)
        ticks = np.arange(0,len(corr.columns),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        ax.set_title("Корреляционная матрица числовых признаков", pad=20)
        saved = save_plot(fig, filename, warnings)
        return saved
    except Exception as e:
        warnings.append(f"Ошибка при построении корреляционной матрицы {filename}: {str(e)}")
        plt.close(fig)
        return False

def main():
    result = {
        "status": None,
        "stage": STAGE,
        "business_task": BUSINESS_TASK,
        "requested_target_column": REQUESTED_TARGET_COLUMN,
        "actual_target_column": None,
        "target_match_type": "not found",
        "input_path": CLEAN_CSV_PATH,
        "report_path": EDA_REPORT_PATH,
        "rows": 0,
        "columns": 0,
        "plots": [],
        "plots_count": 0,
        "warnings": []
    }
    warnings = result["warnings"]
    plots = result["plots"]

    try:
        if not os.path.isfile(CLEAN_CSV_PATH):
            result["status"] = "error"
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            sys.exit(1)

        df = pd.read_csv(CLEAN_CSV_PATH)
        rows, cols = df.shape
        result["rows"] = rows
        result["columns"] = cols

        actual_target_column = None
        target_match_type = "not found"
        if REQUESTED_TARGET_COLUMN in df.columns:
            actual_target_column = REQUESTED_TARGET_COLUMN
            target_match_type = "exact"
        else:
            close_matches = difflib.get_close_matches(REQUESTED_TARGET_COLUMN, df.columns, n=1, cutoff=0.6)
            if close_matches:
                actual_target_column = close_matches[0]
                target_match_type = "fuzzy"
                warnings.append(f"Использован fuzzy match для target колонки: '{actual_target_column}' вместо '{REQUESTED_TARGET_COLUMN}'")
            else:
                warnings.append(f"Target колонка '{REQUESTED_TARGET_COLUMN}' не найдена в датасете")

        result["actual_target_column"] = actual_target_column
        result["target_match_type"] = target_match_type

        # Dataset overview
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Найдено {duplicate_count} дубликатов строк")

        missing_info = df.isnull().sum()
        missing_ratio = missing_info / rows
        high_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
        if high_missing_cols:
            warnings.append(f"Колонки с высокой долей пропусков (>30%): {', '.join(high_missing_cols)}")

        # Data types
        dtypes = df.dtypes.astype(str).to_dict()

        # Target analysis
        target_stats = {}
        target_numeric = False
        if actual_target_column:
            df[actual_target_column] = pd.to_numeric(df[actual_target_column], errors="coerce")
            if df[actual_target_column].isnull().all():
                warnings.append(f"Target колонка '{actual_target_column}' не может быть приведена к числовому типу")
            else:
                target_numeric = True
                target_col = df[actual_target_column]
                count = target_col.count()
                missing_count = target_col.isnull().sum()
                min_val = target_col.min()
                max_val = target_col.max()
                mean_val = target_col.mean()
                median_val = target_col.median()
                std_val = target_col.std()
                zero_count = (target_col == 0).sum()
                zero_share = zero_count / count if count > 0 else 0
                skewness = None
                try:
                    skewness = target_col.skew()
                except Exception:
                    skewness = None
                target_stats = {
                    "count": count,
                    "missing_count": missing_count,
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "zero_count": zero_count,
                    "zero_share": zero_share,
                    "skewness": skewness
                }
                # Plot target distribution
                if plot_histogram(target_col, f"Распределение {actual_target_column}", actual_target_column, "Частота", "eda_target_distribution.png", warnings):
                    plots.append("eda_target_distribution.png")

        # Text columns analysis
        text_columns = []
        for col in ["Title", "Review Text"]:
            if col in df.columns:
                text_columns.append(col)
        if not text_columns:
            warnings.append("Отсутствуют колонки Title и Review Text для анализа текста")

        text_stats = {}
        for col in text_columns:
            series = df[col]
            missing_count = series.isnull().sum()
            empty_string_count = (series.fillna("").str.strip() == "").sum()
            lengths_chars = series.fillna("").str.len()
            lengths_words = series.fillna("").str.split().apply(len)
            mean_len = lengths_chars.mean()
            median_len = lengths_chars.median()
            min_len = lengths_chars.min()
            max_len = lengths_chars.max()
            text_stats[col] = {
                "missing_count": missing_count,
                "empty_string_count": empty_string_count,
                "mean_length_chars": mean_len,
                "median_length_chars": median_len,
                "min_length_chars": min_len,
                "max_length_chars": max_len
            }
            # Plot length distribution
            filename = f"eda_{col.lower().replace(' ', '_')}_length_distribution.png"
            if plot_histogram(lengths_chars, f"Распределение длины текста в символах ({col})", "Длина текста (символы)", "Частота", filename, warnings):
                plots.append(filename)

        # combined_text analysis
        combined_text = None
        if "Title" in df.columns and "Review Text" in df.columns:
            combined_text = df["Title"].fillna("") + " " + df["Review Text"].fillna("")
        elif "Title" in df.columns:
            combined_text = df["Title"].fillna("")
        elif "Review Text" in df.columns:
            combined_text = df["Review Text"].fillna("")
        else:
            combined_text = pd.Series([""] * rows)
            warnings.append("Отсутствуют колонки Title и Review Text, combined_text не создан")

        combined_lengths = combined_text.str.len()
        combined_words = combined_text.str.split().apply(len)
        combined_stats = {
            "mean_length_chars": combined_lengths.mean(),
            "median_length_chars": combined_lengths.median(),
            "min_length_chars": combined_lengths.min(),
            "max_length_chars": combined_lengths.max()
        }
        # Plot combined_text length distribution
        if combined_text is not None and not combined_text.empty:
            if plot_histogram(combined_lengths, "Распределение длины combined_text (Title + Review Text)", "Длина текста (символы)", "Частота", "eda_combined_text_length_distribution.png", warnings):
                plots.append("eda_combined_text_length_distribution.png")

        # Categorical columns analysis
        categorical_columns = ["Clothing ID", "Division Name", "Department Name", "Class Name"]
        existing_categorical = [col for col in categorical_columns if col in df.columns]
        if not existing_categorical:
            warnings.append("Отсутствуют ожидаемые категориальные колонки для анализа")

        categorical_stats = {}
        for col in existing_categorical:
            series = df[col]
            missing_count = series.isnull().sum()
            unique_count = series.nunique(dropna=True)
            top10 = series.value_counts(dropna=True).head(10)
            categorical_stats[col] = {
                "missing_count": missing_count,
                "unique_count": unique_count,
                "top10": top10
            }
            filename = f"eda_top_{col.lower().replace(' ', '_').replace('id','id')}.png"
            if plot_bar(top10, f"Топ-10 значений колонки {col}", col, "Частота", filename, warnings):
                plots.append(filename)

        # Numeric columns analysis
        numeric_columns = ["Age", "Rating", "Recommended IND"]
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        if not existing_numeric:
            warnings.append("Отсутствуют ожидаемые числовые колонки для анализа")

        numeric_stats = {}
        for col in existing_numeric:
            series = pd.to_numeric(df[col], errors="coerce")
            count = series.count()
            missing_count = series.isnull().sum()
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()
            numeric_stats[col] = {
                "count": count,
                "missing_count": missing_count,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val,
                "std": std_val
            }
            filename = f"eda_{col.lower().replace(' ', '_')}_distribution.png"
            if plot_histogram(series, f"Распределение {col}", col, "Частота", filename, warnings):
                plots.append(filename)

        # Rating distribution and target by rating
        if "Rating" in df.columns:
            rating_series = pd.to_numeric(df["Rating"], errors="coerce")
            filename_rating_dist = "eda_rating_distribution.png"
            if plot_histogram(rating_series, "Распределение Rating", "Rating", "Частота", filename_rating_dist, warnings):
                if "eda_rating_distribution.png" not in plots:
                    plots.append("eda_rating_distribution.png")
            if target_numeric:
                df_rating_target = df[[actual_target_column, "Rating"]].copy()
                df_rating_target = df_rating_target.dropna(subset=[actual_target_column, "Rating"])
                mean_target_by_rating = df_rating_target.groupby("Rating")[actual_target_column].mean()
                filename_target_by_rating = "eda_target_by_rating.png"
                if plot_bar(mean_target_by_rating, f"Средний {actual_target_column} по Rating", "Rating", f"Средний {actual_target_column}", filename_target_by_rating, warnings):
                    plots.append(filename_target_by_rating)

        # Age distribution and target by age
        if "Age" in df.columns:
            age_series = pd.to_numeric(df["Age"], errors="coerce")
            filename_age_dist = "eda_age_distribution.png"
            if plot_histogram(age_series, "Распределение Age", "Age", "Частота", filename_age_dist, warnings):
                if "eda_age_distribution.png" not in plots:
                    plots.append("eda_age_distribution.png")
            if target_numeric:
                df_age_target = df[[actual_target_column, "Age"]].copy()
                df_age_target = df_age_target.dropna(subset=[actual_target_column, "Age"])
                # Use binned aggregation for age vs target
                bins = 10
                filename_age_target = "eda_target_by_age.png"
                if plot_binned_agg(df_age_target, "Age", actual_target_column, bins, f"Средний {actual_target_column} по возрастным группам", "Возраст", f"Средний {actual_target_column}", filename_age_target, warnings):
                    plots.append(filename_age_target)

        # Review Text length vs target
        if "Review Text" in df.columns and target_numeric:
            review_text = df["Review Text"].fillna("")
            review_len = review_text.str.len()
            df_review_target = pd.DataFrame({actual_target_column: df[actual_target_column], "ReviewTextLength": review_len})
            df_review_target = df_review_target.dropna(subset=[actual_target_column])
            filename_review_target = "eda_target_by_review_text_length.png"
            if plot_scatter(df_review_target["ReviewTextLength"], df_review_target[actual_target_column], f"Связь длины Review Text и {actual_target_column}", "Длина Review Text (символы)", actual_target_column, filename_review_target, warnings):
                plots.append(filename_review_target)

        # Title length vs target
        if "Title" in df.columns and target_numeric:
            title_text = df["Title"].fillna("")
            title_len = title_text.str.len()
            df_title_target = pd.DataFrame({actual_target_column: df[actual_target_column], "TitleLength": title_len})
            df_title_target = df_title_target.dropna(subset=[actual_target_column])
            filename_title_target = "eda_target_by_title_length.png"
            if plot_scatter(df_title_target["TitleLength"], df_title_target[actual_target_column], f"Связь длины Title и {actual_target_column}", "Длина Title (символы)", actual_target_column, filename_title_target, warnings):
                plots.append(filename_title_target)

        # Correlation matrix
        numeric_corr_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_corr_cols.append(col)
        if actual_target_column and target_numeric and actual_target_column not in numeric_corr_cols:
            numeric_corr_cols.append(actual_target_column)
        if len(numeric_corr_cols) >= 2:
            corr_df = df[numeric_corr_cols].copy()
            for col in numeric_corr_cols:
                corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")
            filename_corr = "eda_numeric_correlation.png"
            if plot_correlation_heatmap(corr_df, filename_corr, warnings):
                plots.append(filename_corr)
        else:
            warnings.append("Недостаточно числовых колонок для построения корреляционной матрицы")

        # Формирование markdown отчета
        with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# 03 EDA Report\n\n")
            f.write("## Project Context\n")
            f.write("- Dataset: Women's E-Commerce Clothing Reviews\n")
            f.write(f"- Business task: {BUSINESS_TASK}\n")
            f.write(f"- Requested target column: {REQUESTED_TARGET_COLUMN}\n")
            f.write(f"- Actual target column used: {actual_target_column if actual_target_column else 'null'}\n")
            f.write(f"- Target match type: {target_match_type}\n")
            f.write(f"- Input file: {CLEAN_CSV_PATH}\n\n")

            f.write("## Dataset Overview\n")
            f.write(f"- rows: {rows}\n")
            f.write(f"- columns: {cols}\n")
            f.write(f"- duplicate rows: {duplicate_count}\n")
            f.write(f"- column list: {', '.join(df.columns)}\n\n")

            f.write("## Data Types\n")
            for col, dtype in dtypes.items():
                f.write(f"- {col}: {dtype}\n")
            f.write("\n")

            f.write("## Missing Values\n")
            for col in df.columns:
                miss = missing_info[col]
                ratio = missing_ratio[col]
                f.write(f"- {col}: {miss} пропусков, {ratio:.2%} доля\n")
            f.write("\n")

            f.write("## Target Analysis\n")
            if target_numeric:
                f.write(f"- count: {target_stats['count']}\n")
                f.write(f"- missing_count: {target_stats['missing_count']}\n")
                f.write(f"- min: {target_stats['min']}\n")
                f.write(f"- max: {target_stats['max']}\n")
                f.write(f"- mean: {target_stats['mean']:.4f}\n")
                f.write(f"- median: {target_stats['median']}\n")
                f.write(f"- std: {target_stats['std']:.4f}\n")
                f.write(f"- zero_count: {target_stats['zero_count']}\n")
                f.write(f"- zero_share: {target_stats['zero_share']:.4f}\n")
                if target_stats['skewness'] is not None:
                    f.write(f"- skewness: {target_stats['skewness']:.4f}\n")
                f.write(f"![Target Distribution](eda_target_distribution.png)\n\n")
            else:
                f.write("Target analysis не выполнен (target не найден или не числовой)\n\n")

            f.write("## Text Columns Analysis\n")
            for col in text_columns:
                stats = text_stats.get(col, {})
                f.write(f"### {col}\n")
                f.write(f"- missing_count: {stats.get('missing_count', 'N/A')}\n")
                f.write(f"- empty_string_count: {stats.get('empty_string_count', 'N/A')}\n")
                f.write(f"- mean_length_chars: {stats.get('mean_length_chars', 'N/A'):.2f}\n")
                f.write(f"- median_length_chars: {stats.get('median_length_chars', 'N/A')}\n")
                f.write(f"- min_length_chars: {stats.get('min_length_chars', 'N/A')}\n")
                f.write(f"- max_length_chars: {stats.get('max_length_chars', 'N/A')}\n")
                filename = f"eda_{col.lower().replace(' ', '_')}_length_distribution.png"
                if filename in plots:
                    f.write(f"![{col} Length Distribution]({filename})\n")
                f.write("\n")
            if combined_text is not None and not combined_text.empty:
                f.write("### combined_text (Title + Review Text)\n")
                f.write(f"- mean_length_chars: {combined_stats['mean_length_chars']:.2f}\n")
                f.write(f"- median_length_chars: {combined_stats['median_length_chars']}\n")
                f.write(f"- min_length_chars: {combined_stats['min_length_chars']}\n")
                f.write(f"- max_length_chars: {combined_stats['max_length_chars']}\n")
                if "eda_combined_text_length_distribution.png" in plots:
                    f.write("![Combined Text Length Distribution](eda_combined_text_length_distribution.png)\n")
                f.write("\n")

            f.write("## Categorical Columns Analysis\n")
            for col in existing_categorical:
                stats = categorical_stats.get(col, {})
                f.write(f"### {col}\n")
                f.write(f"- missing_count: {stats.get('missing_count', 'N/A')}\n")
                f.write(f"- unique_count: {stats.get('unique_count', 'N/A')}\n")
                f.write("- Top 10 values:\n")
                top10 = stats.get("top10", pd.Series())
                for val, cnt in top10.items():
                    f.write(f"  - {val}: {cnt}\n")
                filename = f"eda_top_{col.lower().replace(' ', '_').replace('id','id')}.png"
                if filename in plots:
                    f.write(f"![Top 10 {col}]({filename})\n")
                f.write("\n")

            f.write("## Numeric Columns Analysis\n")
            for col in existing_numeric:
                stats = numeric_stats.get(col, {})
                f.write(f"### {col}\n")
                f.write(f"- count: {stats.get('count', 'N/A')}\n")
                f.write(f"- missing_count: {stats.get('missing_count', 'N/A')}\n")
                f.write(f"- min: {stats.get('min', 'N/A')}\n")
                f.write(f"- max: {stats.get('max', 'N/A')}\n")
                f.write(f"- mean: {stats.get('mean', 'N/A'):.4f}\n")
                f.write(f"- median: {stats.get('median', 'N/A')}\n")
                f.write(f"- std: {stats.get('std', 'N/A'):.4f}\n")
                filename = f"eda_{col.lower().replace(' ', '_')}_distribution.png"
                if filename in plots:
                    f.write(f"![{col} Distribution]({filename})\n")
                f.write("\n")

            f.write("## Target Relationships\n")
            if "Rating" in df.columns:
                if "eda_rating_distribution.png" in plots:
                    f.write("![Rating Distribution](eda_rating_distribution.png)\n")
                if target_numeric and "eda_target_by_rating.png" in plots:
                    f.write(f"![Средний {actual_target_column} по Rating](eda_target_by_rating.png)\n")
            if "Age" in df.columns:
                if "eda_age_distribution.png" in plots:
                    f.write("![Age Distribution](eda_age_distribution.png)\n")
                if target_numeric and "eda_target_by_age.png" in plots:
                    f.write(f"![Средний {actual_target_column} по возрастным группам](eda_target_by_age.png)\n")
            if "Review Text" in df.columns and target_numeric:
                if "eda_target_by_review_text_length.png" in plots:
                    f.write(f"![Связь длины Review Text и {actual_target_column}](eda_target_by_review_text_length.png)\n")
            if "Title" in df.columns and target_numeric:
                if "eda_target_by_title_length.png" in plots:
                    f.write(f"![Связь длины Title и {actual_target_column}](eda_target_by_title_length.png)\n")
            f.write("\n")

            f.write("## Correlation Analysis\n")
            if "eda_numeric_correlation.png" in plots:
                f.write("![Корреляционная матрица](eda_numeric_correlation.png)\n\n")
            else:
                f.write("Корреляционная матрица не построена (недостаточно числовых колонок)\n\n")

            f.write("## Business Insights\n")
            f.write("Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?\n")
            f.write("- Анализ целевой переменной показывает распределение положительных отзывов, что поможет выделить наиболее полезные отзывы.\n")
            f.write("- Длина текста отзывов и заголовков может влиять на восприятие, стоит учитывать при сортировке.\n")
            f.write("- Категориальные признаки помогут сегментировать отзывы по отделам и классам товаров.\n")
            f.write("- Корреляционный анализ числовых признаков выявляет взаимосвязи, которые могут быть полезны для построения моделей.\n\n")

            f.write("## Warnings\n")
            if warnings:
                for w in warnings:
                    f.write(f"- {w}\n")
            else:
                f.write("Предупреждений нет.\n")
            f.write("\n")

            f.write("## Generated Artifacts\n")
            f.write(f"- {os.path.basename(EDA_REPORT_PATH)}\n")
            for plot in plots:
                f.write(f"- {plot}\n")

        result["status"] = "success"
        result["plots"] = plots
        result["plots_count"] = len(plots)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        tb = traceback.format_exc()
        result["status"] = "error"
        result["warnings"].append(tb)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        sys.exit(1)

if __name__ == "__main__":
    main()