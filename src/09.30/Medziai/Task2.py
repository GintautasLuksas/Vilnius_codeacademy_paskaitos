import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Užkrauname Iris duomenų rinkinį
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Padalijame duomenis į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Sprendimo medis su Gini indeksu
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_gini.fit(X_train, y_train)

# Sprendimo medis su entropija
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_entropy.fit(X_train, y_train)

# Prognozės
y_pred_gini = model_gini.predict(X_test)
y_pred_entropy = model_entropy.predict(X_test)

# Gini modelio įvertinimas
accuracy_gini = accuracy_score(y_test, y_pred_gini)
cm_gini = confusion_matrix(y_test, y_pred_gini)

print("Gini modelio tikslumas:", accuracy_gini)
print("Klaidų matrica:\n", cm_gini)

# Entropijos modelio įvertinimas
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
cm_entropy = confusion_matrix(y_test, y_pred_entropy)

print("\nEntropijos modelio tikslumas:", accuracy_entropy)
print("Klaidų matrica:\n", cm_entropy)

# Gini modelio vizualizacija
plt.figure(figsize=(20,10))
tree.plot_tree(model_gini, feature_names=feature_names, class_names=class_names, filled=True)
plt.title('Sprendimo medis su Gini indeksu')
plt.show()

# Entropijos modelio vizualizacija
plt.figure(figsize=(20,10))
tree.plot_tree(model_entropy, feature_names=feature_names, class_names=class_names, filled=True)
plt.title('Sprendimo medis su entropija')
plt.show()

# Modelių palyginimas
print("Gini modelio tikslumas:", accuracy_gini)
print("Entropijos modelio tikslumas:", accuracy_entropy)

if accuracy_gini > accuracy_entropy:
    print("\nGini indeksu paremtas modelis yra tikslesnis.")
elif accuracy_gini < accuracy_entropy:
    print("\nEntropijos kriterijumi paremtas modelis yra tikslesnis.")
else:
    print("\nAbu modeliai turi vienodą tikslumą.")
