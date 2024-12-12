import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Duomenų įkėlimas iš sklearn
diabetes = load_diabetes()
X = diabetes.data  # Features
y = diabetes.target  # Target (nuolatinė reikšmė)

# Kadangi tai regresijos užduotis, mes galime pritaikyti binarinę klasifikaciją.
# Pažymime diabetą, jei reikšmė viršija tam tikrą slenkstį (pvz., 150):
y = (y > 150).astype(int)  # Diabetas (1) arba ne (0)

# Duomenų normalizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tinklo įėjimo formos keitimas (seka)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # Keičiam formą, kad būtų tinkama RNN

# Mokymo ir testavimo duomenų atskyrimas
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# RNN modelio kūrimas
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 1 neuroną su sigmoid aktyvacija, nes tai binarinė klasifikacija

# Modelio kompiliavimas
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelio treniravimas
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# Prognozės ir ataskaita
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Sigmoid išvestis, paverčiame į 0 arba 1
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
