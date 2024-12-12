import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Duomenų įkėlimas
wine = load_wine()
X = wine.data  # Features
y = wine.target  # Target (klasės)

# Duomenų normalizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tinklo įėjimo formos keitimas (seka)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Žymių kodavimas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Mokymo ir testavimo duomenų atskyrimas
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# RNN modelio kūrimas
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3 klasės Wine rinkinyje

# Modelio kompiliavimas
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelio treniravimas
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Prognozės ir ataskaita
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=wine.target_names))
