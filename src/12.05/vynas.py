import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Duomenų įkėlimas
data = load_wine()
X = data.data
y = data.target

# Kategorijų kodavimas į One-Hot formą
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
print(y_onehot)

# Duomenų padalijimas į mokymo, validacijos ir testavimo rinkinius
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Duomenų normalizavimas (Standard Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Modelio kūrimas
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Mokymosi greičio grafikas (Learning Rate Schedule)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10,
    decay_rate=0.96
)

# Optimizatorius
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Modelio kompiliavimas
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ankstyvasis stabdymas (Early Stopping)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modelio mokymas
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=128, verbose=1,
                    callbacks=[early_stopping])

# Mokymo istorijos grafikai
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Mokymo nuostolis')
plt.plot(history.history['val_loss'], label='Validacijos nuostolis')
plt.xlabel('Epochs')
plt.ylabel('Nuostolis')
plt.title('Nuostolis mokymo metu')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Mokymo tikslumas')
plt.plot(history.history['val_accuracy'], label='Validacijos tikslumas')
plt.xlabel('Epochs')
plt.ylabel('Tikslumas')
plt.title('Tikslumas mokymo metu')
plt.legend()

plt.show()

# Modelio vertinimas
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Tikslumo skaičiavimas
acc = accuracy_score(y_test_labels, y_pred)
print('Tikslumas:', acc)

# Klaidų matrica
cm = confusion_matrix(y_test_labels, y_pred)
print('Klaidų matrica:\n', cm)

# Žingsnių per epochą skaičius
steps_per_epoch = X_train.shape[0] / 16
print('Žingsniai per epochą:', steps_per_epoch)
