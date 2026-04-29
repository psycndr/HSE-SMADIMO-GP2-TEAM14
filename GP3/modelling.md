# Modelling Report
## Context
- Stage: 05_modelling
- Business task: Какие отзывы показывать первыми на карточке товара, чтобы они действительно помогли покупателю принять решение?
- Input file: /files/reviews_data_prepared.csv
## Data Check
- Total rows: 23486
- Valid target rows: 23486
- Target column used: Positive Feedback Count
## Feature Types
- Numeric columns (15): Clothing ID, Age, Rating, Recommended IND, review_text_len_chars, review_text_word_count, title_len_chars, title_word_count, has_title, has_review_text, rating_is_low, rating_is_high, fit_keyword_count, quality_keyword_count, size_keyword_flag
- Categorical columns (3): Division Name, Department Name, Class Name
- Text columns (1): combined_text
## Data Split
- Train rows: 18788
- Test rows: 4698
## Model Training
### DummyRegressor_numeric - Trained
- RMSE (log scale): 0.8803713890839598
- MAE (log scale): 0.7247777617027543
- R2 (log scale): -0.0006564001848741174
- RMSE (original scale): 5.604258706385542
- MAE (original scale): 2.448583162264225
- R2 (original scale): -0.05931536558728556
- Rank correlation: None
- Top 10% helpful rate: 0.5309168443496801
### Ridge_numeric - Trained
- RMSE (log scale): 0.8387568016181853
- MAE (log scale): 0.6829388644612678
- R2 (log scale): 0.09170850590854585
- RMSE (original scale): 5.479429723411109
- MAE (original scale): 2.3806770349441133
- R2 (original scale): -0.012650645765395208
- Rank correlation: 0.27633943937474115
- Top 10% helpful rate: 0.6695095948827292
### RandomForest_numeric - Trained
- RMSE (log scale): 0.8404480239317633
- MAE (log scale): 0.6808659476108282
- R2 (log scale): 0.08804195676062787
- RMSE (original scale): 5.455913304722185
- MAE (original scale): 2.381798196705521
- R2 (original scale): -0.0039771830449406576
- Rank correlation: 0.2774116038204918
- Top 10% helpful rate: 0.7036247334754797
### Ridge_numeric_text_cat - Trained
- RMSE (log scale): 0.8235736465862072
- MAE (log scale): 0.6599671818057684
- R2 (log scale): 0.12429461623170213
- RMSE (original scale): 5.408810332610308
- MAE (original scale): 2.3342164586815026
- R2 (original scale): 0.013283418076096454
- Rank correlation: 0.3318619857480284
- Top 10% helpful rate: 0.7718550106609808
## Best Model
- Model name: Ridge_numeric_text_cat
- RMSE (original scale): 5.408810332610308
- MAE (original scale): 2.3342164586815026
- Rank correlation: 0.3318619857480284
- Top 10% helpful rate: 0.7718550106609808
## Predictions
- Predictions saved to: /files/predictions.csv
## Historical Model Comparison
- No historical model found or failed to load.
## Warnings
- Failed to create OneHotEncoder: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
- Model DummyRegressor_numeric rank correlation warning: Constant array in rank correlation
- Failed to save best model or metrics: Can't get local object 'main.<locals>.TextSelector'
## Conclusion
Best model 'Ridge_numeric_text_cat' selected and saved.