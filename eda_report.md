# EDA отчет

## Контекст проекта

Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?

## Обзор датасета

- Количество строк: 23486
- Количество колонок: 11
- Список колонок: Unnamed: 0, Clothing ID, Age, Title, Review Text, Rating, Recommended IND, Positive Feedback Count, Division Name, Department Name, Class Name

## Типы данных колонок

- Unnamed: 0: int64
- Clothing ID: int64
- Age: int64
- Title: object
- Review Text: object
- Rating: int64
- Recommended IND: int64
- Positive Feedback Count: int64
- Division Name: object
- Department Name: object
- Class Name: object

## Пропуски

- Unnamed: 0: пропусков 0 (0.00%)
- Clothing ID: пропусков 0 (0.00%)
- Age: пропусков 0 (0.00%)
- Title: пропусков 3810 (16.22%)
- Review Text: пропусков 845 (3.60%)
- Rating: пропусков 0 (0.00%)
- Recommended IND: пропусков 0 (0.00%)
- Positive Feedback Count: пропусков 0 (0.00%)
- Division Name: пропусков 0 (0.00%)
- Department Name: пропусков 0 (0.00%)
- Class Name: пропусков 0 (0.00%)

## Анализ target

- count: 23486
- missing_count: 0
- min: 0.0
- max: 122.0
- mean: 2.535936302478072
- median: 1.0
- std: 5.7022015020339385
- zero_count: 11176
- zero_share: 0.4758579579323853
- skewness: 6.47299772950122

![Target Distribution](eda_target_distribution.png)

## Анализ текстовых колонок

### Колонка Title
- missing_count: 3810
- empty_string_count: 0
- length_chars_mean: 16.485480711913482
- length_chars_median: 15.0
- length_chars_min: 2
- length_chars_max: 52
- length_words_mean: 2.96427659030912
- length_words_median: 3.0
- length_words_min: 1
- length_words_max: 12
![Распределение длины текста Title](eda_title_length_distribution.png)

### Колонка Review Text
- missing_count: 845
- empty_string_count: 0
- length_chars_mean: 297.6896023162735
- length_chars_median: 292.0
- length_chars_min: 3
- length_chars_max: 508
- length_words_mean: 58.066848335178406
- length_words_median: 57.0
- length_words_min: 1
- length_words_max: 115
![Распределение длины текста Review Text](eda_review_text_length_distribution.png)

## Анализ категориальных колонок

### Колонка Clothing ID
- Уникальных значений: 1206
- Пропущенных значений: 0
- Топ-10 значений:
  - 1078: 1024
  - 862: 806
  - 1094: 756
  - 1081: 582
  - 872: 545
  - 829: 527
  - 1110: 480
  - 868: 430
  - 895: 404
  - 936: 358
![Топ-10 значений Clothing ID](eda_top_clothing_id.png)

### Колонка Division Name
- Уникальных значений: 4
- Пропущенных значений: 0
- Топ-10 значений:
  - General: 13850
  - General Petite: 8120
  - Initmates: 1502
  - Unknown: 14
![Топ-10 значений Division Name](eda_top_division_name.png)

### Колонка Department Name
- Уникальных значений: 7
- Пропущенных значений: 0
- Топ-10 значений:
  - Tops: 10468
  - Dresses: 6319
  - Bottoms: 3799
  - Intimate: 1735
  - Jackets: 1032
  - Trend: 119
  - Unknown: 14
![Топ-10 значений Department Name](eda_top_department_name.png)

### Колонка Class Name
- Уникальных значений: 21
- Пропущенных значений: 0
- Топ-10 значений:
  - Dresses: 6319
  - Knits: 4843
  - Blouses: 3097
  - Sweaters: 1428
  - Pants: 1388
  - Jeans: 1147
  - Fine gauge: 1100
  - Skirts: 945
  - Jackets: 704
  - Lounge: 691
![Топ-10 значений Class Name](eda_top_class_name.png)

## Анализ числовых колонок

### Колонка Age
- count: 23486
- missing_count: 0
- min: 18.0
- max: 99.0
- mean: 43.198543813335604
- median: 41.0
- std: 12.279543615591493
![Распределение Age](eda_age_distribution.png)

### Колонка Rating
- count: 23486
- missing_count: 0
- min: 1.0
- max: 5.0
- mean: 4.196031678446734
- median: 5.0
- std: 1.1100307198243897
![Распределение Rating](eda_rating_distribution.png)

### Колонка Recommended IND
- count: 23486
- missing_count: 0
- min: 0.0
- max: 1.0
- mean: 0.8223622583666865
- median: 1.0
- std: 0.38221563891455684
![Распределение Recommended IND](eda_recommended_ind_distribution.png)

## Связи с target

### Средний target по Rating
![Средний target по Rating](eda_target_by_rating.png)

### Связь Age и target
![Связь Age и target](eda_target_by_age.png)

### Связь длины Review Text и target
![Связь длины Review Text и target](eda_target_by_review_text_length.png)

### Связь длины Title и target
![Связь длины Title и target](eda_target_by_title_length.png)

## Корреляционный анализ

![Корреляционная матрица числовых колонок](eda_numeric_correlation.png)

## Бизнес-инсайты

Отчет подготовлен для понимания ключевых характеристик отзывов и факторов, влияющих на положительный отклик покупателей.

## Предупреждения

Предупреждений нет.

## Созданные артефакты

- /files/eda_report.md
- /files/eda_target_distribution.png
- /files/eda_title_length_distribution.png
- /files/eda_review_text_length_distribution.png
- /files/eda_top_clothing_id.png
- /files/eda_top_division_name.png
- /files/eda_top_department_name.png
- /files/eda_top_class_name.png
- /files/eda_age_distribution.png
- /files/eda_rating_distribution.png
- /files/eda_recommended_ind_distribution.png
- /files/eda_rating_distribution.png
- /files/eda_target_by_rating.png
- /files/eda_age_distribution.png
- /files/eda_target_by_age.png
- /files/eda_target_by_review_text_length.png
- /files/eda_target_by_title_length.png
- /files/eda_numeric_correlation.png
