import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("continuous_data.csv", index_col='Species')
X = data.drop(['TargetTrait'], axis=1)
y = data['TargetTrait']
print(f"Predicting TargetTrait for {len(data)} species using {len(X.columns)} features.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with specified hyperparameters
# These ideally should be selected via hyperparameter tuning but this is an example

model = RandomForestRegressor(
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
print("\nRanked Feature Importances for predicting TargetTrait:")
print(importances)

# Make predictions on test set
predictions = model.predict(X_test)

# Display results
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=X_test.index)
print("\nActual vs Predicted TargetTrait (Test Set):")
print(results)

# Save results
importances.to_csv("ranked_feature_importance.csv", index=False)
results.to_csv("TargetTrait_predictions.csv")

# Print top 3 most important features
print("\nTop 3 Most Important Features for predicting TargetTrait:")
print(importances.head(3)[['rank', 'feature', 'importance']])

# Calculate and print model performance on test set
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"\nModel Performance for TargetTrait Prediction (Test Set):")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")