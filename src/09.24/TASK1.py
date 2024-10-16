import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Mokymosi_valandos': [10, 5, 12, 8, 15, 4, 6, 11, 7, 9, 14, 3, 2, 13, 8, 10],
    'Pazymiu_vidurkis': [6.5, 4.0, 7.5, 6.0, 8.0, 3.5, 5.5, 7.0, 5.0, 6.5, 8.5, 2.0, 4.5, 7.5, 6.0, 7.0],
    'Dalyvavimas': [80, 60, 90, 70, 95, 50, 65, 85, 75, 80, 92, 40, 30, 88, 70, 80],
    'Islaikes': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Mokymosi_valandos', 'Pazymiu_vidurkis', 'Dalyvavimas']]
Y = df[['Islaikes']]

# Sukuriame du duomenu rinkinius
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Implementuojame modeli
model = LogisticRegression()
model.fit(X_train, Y_train)

threshold = 0.75


# Prognozuojame
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

y_cursom_th = []
for i in y_proba:
    if i >= threshold:
        y_cursom_th.append(1)
    else:
        y_cursom_th.append(0)


print(y_pred, y_proba)

acc = accuracy_score(y_test, y_cursom_th)
cm = confusion_matrix(y_test, y_cursom_th)

print(acc)
print(cm)


