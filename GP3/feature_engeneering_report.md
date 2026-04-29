# 04 Feature Engineering Report

## Project Context
- Dataset: Women's E-Commerce Clothing Reviews
- Business task: Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?
- Requested target column: Positive Feedback Count
- Actual target column used: Positive Feedback Count
- Target match type: exact
- Input file: /files/review_data_clean.csv
- Output file: /files/reviews_data_prepared.csv

## Feature Engineering Summary
- input rows: 23486
- output rows: 23486
- input columns count: 11
- output columns count: 23
- new features count: 12

## Created Features
combined_text, review_text_len_chars, review_text_word_count, title_len_chars, title_word_count, has_title, has_review_text, rating_is_low, rating_is_high, fit_keyword_count, quality_keyword_count, size_keyword_flag

## Text Feature Logic
Создан признак combined_text как объединение Title и Review Text (если обе колонки есть).
Текстовые признаки: длина и количество слов в Review Text и Title.

## Keyword Feature Logic
Подсчет количества упоминаний ключевых слов, связанных с посадкой (fit), качеством (quality) и размером (size) в combined_text.
size_keyword_flag указывает на наличие size-related ключевых слов.

## Target Handling
- requested target column: Positive Feedback Count
- actual target column: Positive Feedback Count
- target match type: exact

## Target Leakage Check
- target не использовался как признак

## Warnings
- нет предупреждений

## Generated Artifacts
- /files/reviews_data_prepared.csv
- /files/feature_engeneering_report.md