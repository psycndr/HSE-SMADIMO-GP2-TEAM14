# 01 Data Quality Evaluation Report

## Project Context
- Dataset: Women's E-Commerce Clothing Reviews
- Business task: Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?
- Requested target column: Positive Feedback Count
- Actual target column used: Positive Feedback Count
- Target match type: exact

## Dataset Overview
- CSV path: /files/reviews_data.csv
- rows: 23486
- columns: 11
- duplicate rows: 0

## Columns
Unnamed: 0, Clothing ID, Age, Title, Review Text, Rating, Recommended IND, Positive Feedback Count, Division Name, Department Name, Class Name

## Data Types
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

## Missing Values
| Column | Missing Count | Missing Share |
|--------|---------------|---------------|
| Unnamed: 0 | 0 | 0.0 |
| Clothing ID | 0 | 0.0 |
| Age | 0 | 0.0 |
| Title | 3810 | 0.1622 |
| Review Text | 845 | 0.036 |
| Rating | 0 | 0.0 |
| Recommended IND | 0 | 0.0 |
| Positive Feedback Count | 0 | 0.0 |
| Division Name | 14 | 0.0006 |
| Department Name | 14 | 0.0006 |
| Class Name | 14 | 0.0006 |

## Target Column Check
- Target column 'Positive Feedback Count' found with match type 'exact'.
- Data type: int64

## Target Statistics
Target statistics not available.

## Text Columns Check
### Title
- Missing count: 3810
- Mean length: 19.1
- Median length: 17
- Min length: 2
- Max length: 52
- Empty strings: 0

### Review Text
- Missing count: 845
- Mean length: 308.69
- Median length: 301
- Min length: 9
- Max length: 508
- Empty strings: 0

## Categorical Columns Check
### Clothing ID
- Unique values count: 1206
- Missing count: 0
- Top 10 most frequent values:
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

### Division Name
- Unique values count: 3
- Missing count: 14
- Top 10 most frequent values:
  - General: 13850
  - General Petite: 8120
  - Initmates: 1502

### Department Name
- Unique values count: 6
- Missing count: 14
- Top 10 most frequent values:
  - Tops: 10468
  - Dresses: 6319
  - Bottoms: 3799
  - Intimate: 1735
  - Jackets: 1032
  - Trend: 119

### Class Name
- Unique values count: 20
- Missing count: 14
- Top 10 most frequent values:
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

## Numeric Columns Check
### Age
- Count: 23486
- Missing count: 0
- Min: 18
- Max: 99
- Mean: 43.1985
- Median: 41.0

### Rating
- Count: 23486
- Missing count: 0
- Min: 1
- Max: 5
- Mean: 4.196
- Median: 5.0

### Recommended IND
- Count: 23486
- Missing count: 0
- Min: 0
- Max: 1
- Mean: 0.8224
- Median: 1.0

## Target Leakage Check
- Target column 'Positive Feedback Count' excluded from features.

## Warnings
No warnings.

## Conclusion
Данные содержат базовую информацию для анализа отзывов. Обнаружены некоторые предупреждения, которые следует учесть при дальнейшем анализе. Особое внимание стоит уделить качеству целевой переменной и пропускам в данных. Для бизнес-задачи важно обеспечить корректность и полноту данных, чтобы рекомендации по отзывам были максимально полезными.
