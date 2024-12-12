import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Load and preprocess the data
data = load_iris()
X = data.data
y = data.target

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Prepare data loaders for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class SimpleModel(nn.Module):
    """
    A simple neural network for multi-class classification.

    Attributes:
        fn (nn.Sequential): A sequential container for the layers of the neural network.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initializes the neural network layers.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output classes.
        """
        super(SimpleModel, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(input_size, 64),  # Fully connected layer: input_size -> 64 neurons
            nn.ReLU(),  # Activation function
            nn.Linear(64, 32),  # Fully connected layer: 64 -> 32 neurons
            nn.ReLU(),  # Activation function
            nn.Linear(32, output_size),  # Fully connected layer: 32 -> output_size
            nn.Softmax(dim=1)  # Softmax activation for probabilities
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class probabilities.
        """
        return self.fn(x)


# Initialize the model, loss function, and optimizer
model = SimpleModel(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
    """
    Trains the neural network for 10 epochs and evaluates on the validation set after each epoch.
    """
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, torch.argmax(y_batch, axis=1))  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # Evaluate on the validation set
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    for X_val, y_val in val_loader:
        val_output = model(X_val)  # Forward pass for validation data
        val_loss += criterion(val_output, torch.argmax(y_val, axis=1)).item()  # Compute validation loss
        _, val_preds = torch.max(val_output, 1)  # Predicted classes
        val_correct += (val_preds == torch.argmax(y_val, axis=1)).sum().item()  # Count correct predictions
        val_total += y_val.size(0)  # Total validation samples

    val_acc = val_correct / val_total  # Validation accuracy
    val_loss /= len(val_loader)  # Average validation loss

    print(f"Epoch {epoch + 1}, train_loss: {loss.item():.4f}, val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")

# Evaluate on the test set
model.eval()
y_pred_probs = model(X_test_tensor)  # Forward pass for test data
y_pred = torch.argmax(y_pred_probs, axis=1).numpy()  # Predicted classes
y_test_labels = torch.argmax(y_test_tensor, axis=1).numpy()  # True classes

# Calculate accuracy on the test set
acc = accuracy_score(y_test_labels, y_pred)
print('Accuracy: ', acc)
