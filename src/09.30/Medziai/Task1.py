import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Select relevant features
features = ['Age', 'Sex', 'Pclass', 'Fare']
X = df[features]
y = df['Survived']

# Handle missing values
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)

# Encode categorical variables
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Decision Tree with Gini Index
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_gini.fit(X_train, y_train)

# Decision Tree with Entropy
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_entropy.fit(X_train, y_train)

# Predictions with Gini model
y_pred_gini = model_gini.predict(X_test)

# Predictions with Entropy model
y_pred_entropy = model_entropy.predict(X_test)

# Evaluation for Gini model
accuracy_gini = accuracy_score(y_test, y_pred_gini)
cm_gini = confusion_matrix(y_test, y_pred_gini)

print("Gini Model Accuracy:", accuracy_gini)
print("Confusion Matrix:\n", cm_gini)

# Evaluation for Entropy model
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
cm_entropy = confusion_matrix(y_test, y_pred_entropy)

print("\nEntropy Model Accuracy:", accuracy_entropy)
print("Confusion Matrix:\n", cm_entropy)

# Plotting the Gini model
plt.figure(figsize=(20,10))
tree.plot_tree(model_gini, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True)
plt.title('Decision Tree using Gini Index')
plt.show()

# Plotting the Entropy model
plt.figure(figsize=(20,10))
tree.plot_tree(model_entropy, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True)
plt.title('Decision Tree using Entropy')
plt.show()

# Compare the models
print("Accuracy using Gini Index:", accuracy_gini)
print("Accuracy using Entropy:", accuracy_entropy)

if accuracy_gini > accuracy_entropy:
    print("\nThe model using Gini Index has higher accuracy.")
elif accuracy_gini < accuracy_entropy:
    print("\nThe model using Entropy has higher accuracy.")
else:
    print("\nBoth models have the same accuracy.")
