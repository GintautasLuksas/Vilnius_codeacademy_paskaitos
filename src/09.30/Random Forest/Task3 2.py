import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Prepare the data
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

# Handle missing values by replacing 0 with NaN, then filling with median
X.replace(0, np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Create a parameter grid for tuning
param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 2, 3],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

# Fit the model
grid_search.fit(X_resampled, y_resampled)

# Output the best parameters
print("Best parameters from GridSearch:", grid_search.best_params_)

# Use the best model for predictions
best_xgb_model = grid_search.best_estimator_
y_pred_best_xgb = best_xgb_model.predict(X_test)

# Evaluate the best model
accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)
cm_best_xgb = confusion_matrix(y_test, y_pred_best_xgb)

print("Best XGBoost model accuracy:", accuracy_best_xgb)
print("Best XGBoost confusion matrix:\n", cm_best_xgb)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Best XGBoost Model')
plt.show()

# Feature importance
xgb_importance = best_xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Best XGBoost Model')
plt.show()
