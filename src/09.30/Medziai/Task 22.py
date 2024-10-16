import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Įkelkite ranka rašytų skaitmenų duomenų rinkinį
digits = load_digits()
X = digits.data
y = digits.target

# 2. Padalinkite duomenis į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Sukurkite ir apmokykite modelius
# Gini modelis
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)
clf_gini.fit(X_train, y_train)

# Entropijos modelis
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
clf_entropy.fit(X_train, y_train)

# 4. Atlikite prognozes ir įvertinkite modelius
# Gini
y_pred_gini = clf_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)
print(f"Gini modelio tikslumas: {accuracy_gini}")

# Entropija
y_pred_entropy = clf_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
print(f"Entropijos modelio tikslumas: {accuracy_entropy}")

# 5. Kryžminis patikrinimas
# Gini
cv_scores_gini = cross_val_score(clf_gini, X, y, cv=5)
print(f"Kryžminio patikrinimo rezultatai su Gini indeksu: {cv_scores_gini}")
print(f"Vidutinis kryžminio patikrinimo rezultatas su Gini indeksu: {np.mean(cv_scores_gini)}")

# Entropija
cv_scores_entropy = cross_val_score(clf_entropy, X, y, cv=5)
print(f"Kryžminio patikrinimo rezultatai su Entropija: {cv_scores_entropy}")
print(f"Vidutinis kryžminio patikrinimo rezultatas su Entropija: {np.mean(cv_scores_entropy)}")

# 6. Vizualizacija
# Gini medis
plt.figure(figsize=(15,10))
tree.plot_tree(clf_gini, filled=True, feature_names=digits.feature_names, class_names=[str(i) for i in range(10)])
plt.title('Sprendimo medis su Gini indeksu')
plt.show()

# Entropijos medis
plt.figure(figsize=(15,10))
tree.plot_tree(clf_entropy, filled=True, feature_names=digits.feature_names, class_names=[str(i) for i in range(10)])
plt.title('Sprendimo medis su Entropija')
plt.show()
