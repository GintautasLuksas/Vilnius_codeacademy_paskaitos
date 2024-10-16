import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Įkelkite duomenų rinkinį
df = pd.read_csv('diabetes.csv')

# 2. Pasirinkite požymius ir tikslinį kintamąjį
features = ['Age', 'BMI', 'BloodPressure', 'Insulin']
X = df[features]
y = df['Outcome']

# 3. Tvarkykite trūkstamas reikšmes
X['BloodPressure'].replace(0, np.nan, inplace=True)
X['BMI'].replace(0, np.nan, inplace=True)
X['Insulin'].replace(0, np.nan, inplace=True)

X['BloodPressure'].fillna(X['BloodPressure'].median(), inplace=True)
X['BMI'].fillna(X['BMI'].median(), inplace=True)
X['Insulin'].fillna(X['Insulin'].median(), inplace=True)

# 4. Padalinkite duomenis į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Sukurkite ir apmokykite modelius
# Gini modelis
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf_gini.fit(X_train, y_train)

# Entropijos modelis
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_entropy.fit(X_train, y_train)

# 6. Atlikite prognozes
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)

# 7. Įvertinkite modelius
accuracy_gini = accuracy_score(y_test, y_pred_gini)
cm_gini = confusion_matrix(y_test, y_pred_gini)

accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
cm_entropy = confusion_matrix(y_test, y_pred_entropy)

print("Tikslumas su Gini indeksu:", accuracy_gini)
print("Klaidų matrica:\n", cm_gini)

print("\nTikslumas su Entropija:", accuracy_entropy)
print("Klaidų matrica:\n", cm_entropy)

# 8. Vizualizuokite sprendimo medžius
# Gini medis
plt.figure(figsize=(20,10))
tree.plot_tree(clf_gini, feature_names=features, class_names=['Neturi diabeto', 'Turi diabetą'], filled=True)
plt.title('Sprendimo medis su Gini indeksu')
plt.show()

# Entropijos medis
plt.figure(figsize=(20,10))
tree.plot_tree(clf_entropy, feature_names=features, class_names=['Neturi diabeto', 'Turi diabetą'], filled=True)
plt.title('Sprendimo medis su Entropija')
plt.show()

# 9. Palyginkite modelius
print("Tikslumas su Gini indeksu:", accuracy_gini)
print("Tikslumas su Entropija:", accuracy_entropy)

if accuracy_gini > accuracy_entropy:
    print("\nModelis su Gini indeksu turi didesnį tikslumą.")
elif accuracy_gini < accuracy_entropy:
    print("\nModelis su Entropija turi didesnį tikslumą.")
else:
    print("\nAbu modeliai turi vienodą tikslumą.")
