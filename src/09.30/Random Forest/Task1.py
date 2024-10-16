import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Įkelkite Titanic duomenų rinkinį
df = pd.read_csv('titanic.csv')

# Pasirinkite aktualius požymius
features = ['Age', 'Sex', 'Pclass', 'Fare']
X = df[features]
y = df['Survived']

# Tvarkykite trūkstamas reikšmes
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)

# Užkoduokite kategorinius kintamuosius
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Padalinkite duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sukurkite Random Forest modelį su 100 medžių
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, random_state=42)

# Apmokykite modelį
rf_model.fit(X_train, y_train)

# Prognozuokite su Random Forest modeliu
y_pred_rf = rf_model.predict(X_test)

# Apskaičiuokite tikslumą ir klaidų matricą
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest modelio tikslumas:", accuracy_rf)
print("Random Forest klaidų matrica:\n", cm_rf)

# Vizualizuokite pirmą sprendimo medį iš Random Forest modelio
tree1 = rf_model.estimators_[0]
plt.figure(figsize=(20, 10))
tree.plot_tree(tree1, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True)
plt.title('Vienas sprendimo medis iš Random Forest modelio')
plt.show()

# Kintamųjų svarba
print("\nKintamųjų svarba:")
for feature, importance in zip(features, rf_model.feature_importances_):
    print(f"{feature}: {importance}")
