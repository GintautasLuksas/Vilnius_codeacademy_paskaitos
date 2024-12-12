import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# List of optimizers to try
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adam']
histories = {}

# Model training loop for each optimizer
for opt in optimizers:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Single output for binary classification
    ])

    model.compile(optimizer=opt.lower(), loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Training with {opt} optimizer...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=1)
    histories[opt] = history

# Plot the training and validation accuracy for each optimizer
fig, axes = plt.subplots(len(optimizers), 1, figsize=(10, 14))
fig.suptitle('Training and Validation Accuracy Comparison')

for i, opt in enumerate(optimizers):
    history = histories[opt]
    axes[i].plot(history.history['accuracy'], label='Training Accuracy')
    axes[i].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[i].set_title(f'{opt} Accuracy')
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Accuracy')
    axes[i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Testing the model with the best-performing optimizer (Adam as an example)
best_optimizer = 'Adam'  # Choose based on visual inspection of graphs
model.compile(optimizer=best_optimizer.lower(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_val, y_train_val, epochs=100, batch_size=16, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss with {best_optimizer}:', test_loss)
print(f'Test Accuracy with {best_optimizer}:', test_accuracy)
