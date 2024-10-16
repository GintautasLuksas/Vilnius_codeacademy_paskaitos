import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Duomenys
data = {
    'age': [25, 40, 35, 50, 23, 45, 30, 31, 34, 40, 28, 37, 24, 29, 55, 42],
    'salary': [30000, 50000, 45000, 60000, 28000, 58000, 35000, 36000, 40000, 52000,
               33000, 49000, 31000, 42000, 70000, 48000],
    'satisfaction': [7, 9, 6, 5, 4, 6, 7, 1, 9, 4, 1, 8, 9, 6, 5, 7],
    'experience': [30, 28, 55, 12, 20, 30, 45, 38, 40, 10, 35, 37, 25, 41, 13, 25],
    'is_loyal': [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
}

# Duomenų rėmelis
df = pd.DataFrame(data)

# Pasirenkame nepriklausomus kintamuosius (X) ir tikslinį kintamąjį (Y)
X = df[['age', 'salary', 'satisfaction', 'experience']]
Y = df['is_loyal']

threshold = 0.0000000006

# Skalavimas naudojant StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Padalijame duomenis į mokymo ir testavimo rinkinius
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.5, random_state=42)

# Sukuriame logistinės regresijos modelį ir padidiname iteracijų skaičių
model = LogisticRegression(max_iter=1000)

# Modelio treniravimas
model.fit(X_train, Y_train)

# Prognozuojame testavimo rinkinio rezultatus
Y_pred = model.predict(X_test)

# Apskaičiuojame tikslumą
accuracy = accuracy_score(Y_test, Y_pred)

# Sudarome sumaišymo matricą
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Spausdiname rezultatus
print(f"Prognozių tikslumas: {accuracy}")
print("Sumaišymo matrica:")
print(conf_matrix)
