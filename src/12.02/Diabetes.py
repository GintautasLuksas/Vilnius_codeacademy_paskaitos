import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the diabetes dataset
data = load_diabetes()
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
        Dense(1)  # Single output for regression
    ])

    model.compile(optimizer=opt.lower(), loss='mean_squared_error', metrics=['mae'])

    print(f"Training with {opt} optimizer...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=1)
    histories[opt] = history

# Plot the training and validation loss and MAE for each optimizer
fig, axes = plt.subplots(len(optimizers), 2, figsize=(14, 10))
fig.suptitle('Training and Validation Metrics Comparison')

for i, opt in enumerate(optimizers):
    history = histories[opt]
    # Plot Loss
    axes[i, 0].plot(history.history['loss'], label='Training Loss')
    axes[i, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[i, 0].set_title(f'{opt} Loss')
    axes[i, 0].set_xlabel('Epochs')
    axes[i, 0].set_ylabel('Loss')
    axes[i, 0].legend()

    # Plot MAE
    axes[i, 1].plot(history.history['mae'], label='Training MAE')
    axes[i, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[i, 1].set_title(f'{opt} MAE')
    axes[i, 1].set_xlabel('Epochs')
    axes[i, 1].set_ylabel('MAE')
    axes[i, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Testing the model with the Adam optimizer as an example
best_optimizer = 'Adam'  # Choose based on visual inspection of graphs
model.compile(optimizer=best_optimizer.lower(), loss='mean_squared_error', metrics=['mae'])
model.fit(X_train_val, y_train_val, epochs=100, batch_size=16, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss with {best_optimizer}:', test_loss)
print(f'Test MAE with {best_optimizer}:', test_mae)
