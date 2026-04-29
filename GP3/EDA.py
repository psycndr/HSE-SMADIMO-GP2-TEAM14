import os
import sys
import json
import traceback
import difflib
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE = "03_eda"
BUSINESS_TASK = "Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?"
REQUESTED_TARGET_COLUMN = "Positive Feedback Count"
CLEAN_CSV_PATH = "/files/review_data_clean.csv"
EDA_REPORT_PATH = "/files/eda_report.md"
OUTPUT_DIR = "/files"

def safe_savefig(fig, path, warnings):
    try:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        warnings.append(f"Ошибка при сохранении графика {os.path.basename(path)}: {str(e)}")
        return False

def plot_histogram(series, title, xlabel, ylabel, path, warnings, bins=30):
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        ax.hist(series.dropna(), bins=bins, color='blue', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return safe_savefig(fig, path, warnings)
    except Exception as e:
        warnings.append(f"Ошибка при построении гистограммы {title}: {str(e)}")
        plt.close(fig)
        return False

def plot_bar_top_values(series, title, path, warnings):
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        top_values = series.value_counts(dropna=False).head(10)
        top_values.plot(kind='bar', ax=ax, color='green', alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.set_xlabel(series.name)
        plt.xticks(rotation=45, ha='right')
        return safe_savefig(fig, path, warnings)
    except Exception as e:
        warnings.append(f"Ошибка при построении bar chart {title}: {str(e)}")
        plt.close(fig)
        return False

def plot_scatter_or_bin_agg(df, x_col, y_col, xlabel, ylabel, title, path, warnings, bins=20):
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        if df[x_col].dtype.kind in 'biufc' and df[y_col].dtype.kind in 'biufc':
            # bin aggregation
            df_clean = df[[x_col, y_col]].dropna()
            if df_clean.empty:
                warnings.append(f"Нет данных для построения графика {title}")
                plt.close(fig)
                return False
            bin_labels = range(bins)
            df_clean['bin'] = pd.cut(df_clean[x_col], bins=bins, labels=bin_labels)
            agg = df_clean.groupby('bin')[y_col].mean()
            agg.index = agg.index.astype(str)
            agg.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
            ax.set_xlabel(xlabel + " (binned)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
        else:
            # scatter plot
            ax.scatter(df[x_col], df[y_col], alpha=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        return safe_savefig(fig, path, warnings)
    except Exception as e:
        warnings.append(f"Ошибка при построении графика {title}: {str(e)}")
        plt.close(fig)
        return False

def plot_heatmap(corr, title, path, warnings):
    fig, ax = plt.subplots(figsize=(10,8))
    try:
        cax = ax.matshow(corr, cmap='coolwarm')
        fig.colorbar(cax)
        ticks = np.arange(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        ax.set_title(title, pad=20)
        plt.tight_layout()
        return safe_savefig(fig, path, warnings)
    except Exception as e:
        warnings.append(f"Ошибка при построении тепловой карты корреляций: {str(e)}")
        plt.close(fig)
        return False

def main():
    warnings = []
    result = {
        "status": "success",
        "stage": STAGE,
        "business_task": BUSINESS_TASK,
        "requested_target_column": REQUESTED_TARGET_COLUMN,
        "actual_target_column": None,
        "target_match_type": None,
        "input_path": CLEAN_CSV_PATH,
        "report_path": EDA_REPORT_PATH,
        "rows": 0,
        "columns": 0,
        "plots": [],
        "plots_count": 0,
        "warnings": warnings
    }
    if not os.path.isfile(CLEAN_CSV_PATH):
        result.update({
            "status": "error",
            "message": f"Файл {CLEAN_CSV_PATH} не найден.",
            "warnings": warnings
        })
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    try:
        df = pd.read_csv(CLEAN_CSV_PATH)
    except Exception as e:
        result.update({
            "status": "error",
            "message": f"Ошибка при чтении файла {CLEAN_CSV_PATH}: {str(e)}",
            "warnings": warnings
        })
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    result["rows"], result["columns"] = df.shape
    actual_target_column = None
    target_match_type = None
    columns = list(df.columns)
    if REQUESTED_TARGET_COLUMN in columns:
        actual_target_column = REQUESTED_TARGET_COLUMN
        target_match_type = "exact"
    else:
        close_matches = difflib.get_close_matches(REQUESTED_TARGET_COLUMN, columns, n=1, cutoff=0.6)
        if close_matches:
            actual_target_column = close_matches[0]
            target_match_type = "fuzzy"
            warnings.append(f"Целевая колонка '{REQUESTED_TARGET_COLUMN}' не найдена, использована похожая '{actual_target_column}'.")
        else:
            warnings.append(f"Целевая колонка '{REQUESTED_TARGET_COLUMN}' не найдена и похожих колонок нет. Анализ target пропущен.")
    result["actual_target_column"] = actual_target_column
    result["target_match_type"] = target_match_type
    # Базовый обзор
    dtypes = df.dtypes.astype(str).to_dict()
    duplicates_count = df.duplicated().sum()
    missing_counts = df.isnull().sum()
    missing_shares = (missing_counts / len(df)).to_dict()
    high_missing_cols = [col for col, share in missing_shares.items() if share > 0.5]
    if high_missing_cols:
        warnings.append(f"Колонки с высокой долей пропусков (>50%): {', '.join(high_missing_cols)}")
    if duplicates_count > 0:
        warnings.append(f"Найдено полных дубликатов строк: {duplicates_count}")
    # Анализ target
    target_stats = {}
    target_numeric = None
    if actual_target_column:
        try:
            target_numeric = pd.to_numeric(df[actual_target_column], errors="coerce")
            if target_numeric.isnull().all():
                warnings.append(f"Колонку target '{actual_target_column}' не удалось привести к числовому типу.")
                target_numeric = None
            else:
                target_stats = {
                    "count": int(target_numeric.count()),
                    "missing_count": int(target_numeric.isnull().sum()),
                    "min": float(target_numeric.min()),
                    "max": float(target_numeric.max()),
                    "mean": float(target_numeric.mean()),
                    "median": float(target_numeric.median()),
                    "std": float(target_numeric.std()),
                    "zero_count": int((target_numeric == 0).sum()),
                    "zero_share": float((target_numeric == 0).mean()),
                    "skewness": float(target_numeric.skew())
                }
                # График распределения target
                path = os.path.join(OUTPUT_DIR, "eda_target_distribution.png")
                if plot_histogram(target_numeric, f"Распределение {actual_target_column}", actual_target_column, "Count", path, warnings):
                    result["plots"].append("eda_target_distribution.png")
        except Exception as e:
            warnings.append(f"Ошибка при анализе target: {str(e)}")
    # Анализ текстовых колонок
    text_columns = []
    for col in ["Title", "Review Text"]:
        if col in df.columns:
            text_columns.append(col)
    if not text_columns:
        warnings.append("Отсутствуют ожидаемые текстовые колонки Title и Review Text.")
    text_stats = {}
    for col in text_columns:
        try:
            s = df[col].astype(str)
            length_chars = s.str.len()
            length_words = s.str.split().apply(len)
            missing_count = df[col].isnull().sum()
            empty_string_count = (s.str.strip() == "").sum()
            text_stats[col] = {
                "missing_count": int(missing_count),
                "empty_string_count": int(empty_string_count),
                "length_chars_mean": float(length_chars.mean()),
                "length_chars_median": float(length_chars.median()),
                "length_chars_min": int(length_chars.min()),
                "length_chars_max": int(length_chars.max()),
                "length_words_mean": float(length_words.mean()),
                "length_words_median": float(length_words.median()),
                "length_words_min": int(length_words.min()),
                "length_words_max": int(length_words.max())
            }
            # График распределения длины текста
            path = os.path.join(OUTPUT_DIR, f"eda_{col.lower().replace(' ', '_')}_length_distribution.png")
            fig, ax = plt.subplots(figsize=(8,5))
            ax.hist(length_chars.dropna(), bins=30, color='orange', alpha=0.7)
            ax.set_title(f"Распределение длины текста (символы) в колонке {col}")
            ax.set_xlabel("Длина текста (символы)")
            ax.set_ylabel("Количество")
            if safe_savefig(fig, path, warnings):
                result["plots"].append(os.path.basename(path))
        except Exception as e:
            warnings.append(f"Ошибка при анализе текстовой колонки {col}: {str(e)}")
    # combined_text
    combined_text = ""
    if "Title" in df.columns and "Review Text" in df.columns:
        combined_text = df["Title"].fillna("").astype(str).str.strip() + " " + df["Review Text"].fillna("").astype(str).str.strip()
    elif "Title" in df.columns:
        combined_text = df["Title"].fillna("").astype(str).str.strip()
    elif "Review Text" in df.columns:
        combined_text = df["Review Text"].fillna("").astype(str).str.strip()
    else:
        warnings.append("Отсутствуют обе колонки Title и Review Text для объединения.")
    # Анализ категориальных колонок
    categorical_columns = ["Clothing ID", "Division Name", "Department Name", "Class Name"]
    existing_cat_cols = [col for col in categorical_columns if col in df.columns]
    if not existing_cat_cols:
        warnings.append("Отсутствуют ожидаемые категориальные колонки Clothing ID, Division Name, Department Name, Class Name.")
    cat_stats = {}
    for col in existing_cat_cols:
        try:
            s = df[col]
            unique_count = s.nunique(dropna=True)
            top10 = s.value_counts(dropna=False).head(10)
            missing_count = s.isnull().sum()
            cat_stats[col] = {
                "unique_count": int(unique_count),
                "missing_count": int(missing_count),
                "top10": top10.to_dict()
            }
            path = os.path.join(OUTPUT_DIR, f"eda_top_{col.lower().replace(' ', '_')}.png")
            if plot_bar_top_values(s, f"Топ-10 значений колонки {col}", path, warnings):
                result["plots"].append(os.path.basename(path))
        except Exception as e:
            warnings.append(f"Ошибка при анализе категориальной колонки {col}: {str(e)}")
    # Анализ числовых колонок
    numeric_columns = ["Age", "Rating", "Recommended IND"]
    existing_num_cols = [col for col in numeric_columns if col in df.columns]
    if not existing_num_cols:
        warnings.append("Отсутствуют ожидаемые числовые колонки Age, Rating, Recommended IND.")
    num_stats = {}
    for col in existing_num_cols:
        try:
            s = pd.to_numeric(df[col], errors="coerce")
            count = int(s.count())
            missing_count = int(s.isnull().sum())
            min_v = float(s.min())
            max_v = float(s.max())
            mean_v = float(s.mean())
            median_v = float(s.median())
            std_v = float(s.std())
            num_stats[col] = {
                "count": count,
                "missing_count": missing_count,
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "median": median_v,
                "std": std_v
            }
            path = os.path.join(OUTPUT_DIR, f"eda_{col.lower().replace(' ', '_')}_distribution.png")
            if plot_histogram(s, f"Распределение колонки {col}", col, "Count", path, warnings):
                result["plots"].append(os.path.basename(path))
        except Exception as e:
            warnings.append(f"Ошибка при анализе числовой колонки {col}: {str(e)}")
    # Дополнительный анализ для Rating
    if "Rating" in existing_num_cols:
        try:
            rating = pd.to_numeric(df["Rating"], errors="coerce")
            path = os.path.join(OUTPUT_DIR, "eda_rating_distribution.png")
            if plot_histogram(rating, "Распределение Rating", "Rating", "Count", path, warnings):
                result["plots"].append("eda_rating_distribution.png")
            if target_numeric is not None:
                df_rating_target = pd.DataFrame({"Rating": rating, "Target": target_numeric})
                mean_target_by_rating = df_rating_target.groupby("Rating")["Target"].mean()
                fig, ax = plt.subplots(figsize=(8,5))
                mean_target_by_rating.plot(kind='bar', ax=ax, color='brown', alpha=0.7)
                ax.set_title("Средний target по Rating")
                ax.set_xlabel("Rating")
                ax.set_ylabel(f"Средний {actual_target_column}")
                path = os.path.join(OUTPUT_DIR, "eda_target_by_rating.png")
                if safe_savefig(fig, path, warnings):
                    result["plots"].append("eda_target_by_rating.png")
        except Exception as e:
            warnings.append(f"Ошибка при анализе Rating: {str(e)}")
    # Анализ Age и target
    if "Age" in existing_num_cols:
        try:
            age = pd.to_numeric(df["Age"], errors="coerce")
            path = os.path.join(OUTPUT_DIR, "eda_age_distribution.png")
            if plot_histogram(age, "Распределение Age", "Age", "Count", path, warnings):
                result["plots"].append("eda_age_distribution.png")
            if target_numeric is not None:
                df_age_target = pd.DataFrame({"Age": age, "Target": target_numeric})
                path = os.path.join(OUTPUT_DIR, "eda_target_by_age.png")
                if plot_scatter_or_bin_agg(df_age_target, "Age", "Target", "Age", f"Средний {actual_target_column}", "Связь Age и target", path, warnings):
                    result["plots"].append("eda_target_by_age.png")
        except Exception as e:
            warnings.append(f"Ошибка при анализе Age: {str(e)}")
    # Анализ длины Review Text и target
    if "Review Text" in df.columns and target_numeric is not None:
        try:
            review_text_len = df["Review Text"].fillna("").astype(str).str.len()
            df_review_target = pd.DataFrame({"ReviewTextLen": review_text_len, "Target": target_numeric})
            path = os.path.join(OUTPUT_DIR, "eda_target_by_review_text_length.png")
            if plot_scatter_or_bin_agg(df_review_target, "ReviewTextLen", "Target", "Длина Review Text (символы)", f"Средний {actual_target_column}", "Связь длины Review Text и target", path, warnings):
                result["plots"].append("eda_target_by_review_text_length.png")
        except Exception as e:
            warnings.append(f"Ошибка при анализе длины Review Text и target: {str(e)}")
    # Анализ длины Title и target
    if "Title" in df.columns and target_numeric is not None:
        try:
            title_len = df["Title"].fillna("").astype(str).str.len()
            df_title_target = pd.DataFrame({"TitleLen": title_len, "Target": target_numeric})
            path = os.path.join(OUTPUT_DIR, "eda_target_by_title_length.png")
            if plot_scatter_or_bin_agg(df_title_target, "TitleLen", "Target", "Длина Title (символы)", f"Средний {actual_target_column}", "Связь длины Title и target", path, warnings):
                result["plots"].append("eda_target_by_title_length.png")
        except Exception as e:
            warnings.append(f"Ошибка при анализе длины Title и target: {str(e)}")
    # Корреляционный анализ
    try:
        numeric_cols_for_corr = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols_for_corr.append(col)
        if actual_target_column and actual_target_column not in numeric_cols_for_corr and target_numeric is not None:
            numeric_cols_for_corr.append(actual_target_column)
        corr_df = df[numeric_cols_for_corr].copy()
        if actual_target_column and target_numeric is not None:
            corr_df[actual_target_column] = target_numeric
        corr = corr_df.corr()
        path = os.path.join(OUTPUT_DIR, "eda_numeric_correlation.png")
        if plot_heatmap(corr, "Корреляционная матрица числовых колонок", path, warnings):
            result["plots"].append("eda_numeric_correlation.png")
    except Exception as e:
        warnings.append(f"Ошибка при построении корреляционной матрицы: {str(e)}")
    # Формирование markdown отчета
    try:
        with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(f"# EDA отчет\n\n")
            f.write(f"## Контекст проекта\n\n{BUSINESS_TASK}\n\n")
            f.write(f"## Обзор датасета\n\n")
            f.write(f"- Количество строк: {result['rows']}\n")
            f.write(f"- Количество колонок: {result['columns']}\n")
            f.write(f"- Список колонок: {', '.join(columns)}\n\n")
            f.write(f"## Типы данных колонок\n\n")
            for col, dt in dtypes.items():
                f.write(f"- {col}: {dt}\n")
            f.write(f"\n## Пропуски\n\n")
            for col, cnt in missing_counts.items():
                share = missing_shares[col]
                f.write(f"- {col}: пропусков {cnt} ({share:.2%})\n")
            f.write(f"\n## Анализ target\n\n")
            if target_stats:
                for k,v in target_stats.items():
                    f.write(f"- {k}: {v}\n")
                f.write(f"\n![Target Distribution](eda_target_distribution.png)\n\n")
            else:
                f.write("Целевая колонка не найдена или не числовая, анализ пропущен.\n\n")
            f.write(f"## Анализ текстовых колонок\n\n")
            if text_stats:
                for col, stats in text_stats.items():
                    f.write(f"### Колонка {col}\n")
                    for k,v in stats.items():
                        f.write(f"- {k}: {v}\n")
                    img_name = f"eda_{col.lower().replace(' ', '_')}_length_distribution.png"
                    if img_name in result["plots"]:
                        f.write(f"![Распределение длины текста {col}]({img_name})\n\n")
            else:
                f.write("Отсутствуют текстовые колонки для анализа.\n\n")
            f.write(f"## Анализ категориальных колонок\n\n")
            if cat_stats:
                for col, stats in cat_stats.items():
                    f.write(f"### Колонка {col}\n")
                    f.write(f"- Уникальных значений: {stats['unique_count']}\n")
                    f.write(f"- Пропущенных значений: {stats['missing_count']}\n")
                    f.write(f"- Топ-10 значений:\n")
                    for val, cnt in stats['top10'].items():
                        f.write(f"  - {val}: {cnt}\n")
                    img_name = f"eda_top_{col.lower().replace(' ', '_')}.png"
                    if img_name in result["plots"]:
                        f.write(f"![Топ-10 значений {col}]({img_name})\n\n")
            else:
                f.write("Отсутствуют категориальные колонки для анализа.\n\n")
            f.write(f"## Анализ числовых колонок\n\n")
            if num_stats:
                for col, stats in num_stats.items():
                    f.write(f"### Колонка {col}\n")
                    for k,v in stats.items():
                        f.write(f"- {k}: {v}\n")
                    img_name = f"eda_{col.lower().replace(' ', '_')}_distribution.png"
                    if img_name in result["plots"]:
                        f.write(f"![Распределение {col}]({img_name})\n\n")
            else:
                f.write("Отсутствуют числовые колонки для анализа.\n\n")
            f.write(f"## Связи с target\n\n")
            if "eda_target_by_rating.png" in result["plots"]:
                f.write("### Средний target по Rating\n")
                f.write("![Средний target по Rating](eda_target_by_rating.png)\n\n")
            if "eda_target_by_age.png" in result["plots"]:
                f.write("### Связь Age и target\n")
                f.write("![Связь Age и target](eda_target_by_age.png)\n\n")
            if "eda_target_by_review_text_length.png" in result["plots"]:
                f.write("### Связь длины Review Text и target\n")
                f.write("![Связь длины Review Text и target](eda_target_by_review_text_length.png)\n\n")
            if "eda_target_by_title_length.png" in result["plots"]:
                f.write("### Связь длины Title и target\n")
                f.write("![Связь длины Title и target](eda_target_by_title_length.png)\n\n")
            f.write(f"## Корреляционный анализ\n\n")
            if "eda_numeric_correlation.png" in result["plots"]:
                f.write("![Корреляционная матрица числовых колонок](eda_numeric_correlation.png)\n\n")
            else:
                f.write("Корреляционный анализ не выполнен.\n\n")
            f.write(f"## Бизнес-инсайты\n\n")
            f.write("Отчет подготовлен для понимания ключевых характеристик отзывов и факторов, влияющих на положительный отклик покупателей.\n\n")
            f.write(f"## Предупреждения\n\n")
            if warnings:
                for w in warnings:
                    f.write(f"- {w}\n")
            else:
                f.write("Предупреждений нет.\n")
            f.write(f"\n## Созданные артефакты\n\n")
            f.write(f"- {EDA_REPORT_PATH}\n")
            for plot in result["plots"]:
                f.write(f"- /files/{plot}\n")
    except Exception as e:
        result.update({
            "status": "error",
            "message": f"Ошибка при формировании отчета: {str(e)}",
            "warnings": warnings
        })
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    result["plots_count"] = len(result["plots"])
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()