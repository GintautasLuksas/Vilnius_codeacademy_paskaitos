import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn import tree  # Importing tree module

# 1. Load the dataset
df = pd.read_csv('diabetes.csv')

# 2. Select features and target variable
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

# 3. Handle missing values (0 considered as missing)
X.replace(0, np.nan, inplace=True)

# Fill missing values with medians
X.fillna(X.median(), inplace=True)

# 4. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 6. Define the Random Forest model
rf_clf = RandomForestClassifier(random_state=42)

# 7. Set up the hyperparameter grid for Random Forest with expanded options
rf_param_grid = {
    'n_estimators': [2, 5, 10, ],  # More options for n_estimators
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced']
}

# 8. Use GridSearchCV to find the best parameters for Random Forest
rf_grid_search = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_resampled, y_resampled)

# 9. Best parameters for Random Forest
print("Best parameters from GridSearch (Random Forest):", rf_grid_search.best_params_)

# 10. Use the best Random Forest model
best_rf = rf_grid_search.best_estimator_

# 11. Make predictions with the best Random Forest model
y_pred_rf = best_rf.predict(X_test)

# 12. Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest model accuracy after GridSearch:", accuracy_rf)
print("Random Forest confusion matrix after GridSearch:\n", cm_rf)

# 13. Feature importance for Random Forest
print("Feature importance (Random Forest):")
for feature, importance in zip(features, best_rf.feature_importances_):
    print(f"{feature}: {importance}")

# 14. Visualize one of the Random Forest trees
plt.figure(figsize=(20, 10))
tree.plot_tree(best_rf.estimators_[0], feature_names=features, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title('One decision tree from optimized Random Forest')
plt.show()

