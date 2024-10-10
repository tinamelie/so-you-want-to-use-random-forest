import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load data
data = pd.read_csv("binary_data.csv", index_col='Species')
X = data.drop(['TargetTrait'], axis=1)
y = data['TargetTrait']
print(f"Predicting binary TargetTrait for {len(data)} species using {len(X.columns)} features.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# These ideally should be selected via hyperparameter tuning but this is an example
# Train model with specified hyperparameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Get feature importances and rank them
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
importances['rank'] = importances.index + 1
print("\nRanked Feature Importances for predicting binary TargetTrait:")
print(importances)

# Make predictions on test set
predictions = model.predict(X_test)
prediction_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Display results
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions, 'Probability': prediction_proba}, index=X_test.index)
print("\nActual vs Predicted binary TargetTrait (Test Set):")
print(results)

# Save results
importances.to_csv("ranked_feature_importance.csv", index=False)
results.to_csv("binary_TargetTrait_predictions.csv")

# Print top 3 most important features
print("\nTop 3 Most Important Features for predicting binary TargetTrait:")
print(importances.head(3)[['rank', 'feature', 'importance']])

# Calculate and print model performance on test set
accuracy = accuracy_score(y_test, predictions)
auc_roc = roc_auc_score(y_test, prediction_proba)
print(f"\nModel Performance for binary TargetTrait Prediction (Test Set):")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))