## What this is for and what it does
- Prediction of binary or continuous target traits
- Utilizes scikit-learn's Random Forest for both classification and regression tasks
- Feature importance ranking using built-in scikit-learn
- Performance evaluation from scikit-learn.metrics

## Models

### 1. scikit-learn RandomForestClassifier
- For predicting binary target traits
- Metrics: Accuracy, AUC-ROC, Classification Report
- Implementation: `sklearn.ensemble.RandomForestClassifier`

### 2. scikit-learn RandomForestRegressor
- For predicting continuous target traits
- Metrics: Mean Absolute Error, Mean Squared Error, R-squared
- Implementation: `sklearn.ensemble.RandomForestRegressor`

## Example data

- `binary_data.csv`: For classification tasks
- `continuous_data.csv`: For regression tasks

Data is split into training and test sets using `sklearn.model_selection.train_test_split`.

## Usage

1. For binary trait prediction:
   ```
   python RandomForestClassifier_example.py
   ```

2. For continuous trait prediction:
   ```
   python RandomForestRegressor_example.py
   ```

## Output

Both scripts generate:
- `ranked_feature_importance.csv`: CSV file with ranked feature importances
- `*_predictions.csv`: CSV file with model predictions
  - `binary_TargetTrait_predictions.csv` for classification
  - `TargetTrait_predictions.csv` for regression
- Console output with:
  - Model performance metrics (using scikit-learn.metrics)
  - Top important features (using the `feature_importances_` attribute of scikit-learn's Random Forest models)

## Dependencies

- scikit-learn
- pandas
- numpy

To install the required packages:
```
pip install scikit-learn pandas numpy
```

## You should know

- Don't go around directly reporting these feature importance scores and saying Tina told you to. I definitely did not. You really should use a model explainer like SHAPELY or LIME if you're going to dive into feature importance. Throw some cross validation in here too; it's good for you. 
- Also, you really should be doing a lil hyperparameter tuning, but some of you live more dangerously than I do.
