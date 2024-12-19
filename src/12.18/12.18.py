import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import re

# Load the 20 Newsgroups dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

x = data.data
y = data.target

# Label encoding for the target variable
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, encoded_labels, test_size=0.2, random_state=42)


# Text cleaning function using regex
def clean_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\b(?:https?|ftp)://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text



# Clean the texts
x_train = [clean_text(doc) for doc in x_train]
x_test = [clean_text(doc) for doc in x_test]

# Tokenization and padding
max_vocab_size = 6000  # Increased vocabulary size
max_sequence_length = 200  # Increased sequence length

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(x_train)

train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = pad_sequences(test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

# One-hot encoding for labels
num_classes = np.max(encoded_labels) + 1
y_test = to_categorical(y_test, num_classes)
y_train = to_categorical(y_train, num_classes)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Define a function to build, compile, and train a model
def build_and_train_model(model_type, x_train, y_train, x_test, y_test, max_vocab_size, max_sequence_length,
                          num_classes):
    print(f"Training {model_type} model...")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_sequence_length))

    if model_type == 'GRU':
        model.add(tf.keras.layers.GRU(128))
    elif model_type == 'LSTM':
        model.add(tf.keras.layers.LSTM(128))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size=32, epochs=10, verbose=1, callbacks=[early_stopping])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"{model_type} Test Accuracy: {test_accuracy:.4f}")

    # Plot training & validation accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_type} Model Accuracy')
    plt.show()

    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_type} Model Loss')
    plt.show()

    # Confusion Matrix
    predictions = model.predict(x_test)
    predicted_class = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(true_labels, predicted_class)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(map(str, label_encoder.classes_)))
    disp.plot()
    plt.title(f"{model_type} Confusion Matrix")
    plt.show()

    # Classification report
    print(f"{model_type} Classification Report:")
    print(classification_report(true_labels, predicted_class, target_names=list(map(str, label_encoder.classes_))))


# Train GRU model
build_and_train_model('GRU', x_train, y_train, x_test, y_test, max_vocab_size, max_sequence_length, num_classes)

# Train LSTM model
build_and_train_model('LSTM', x_train, y_train, x_test, y_test, max_vocab_size, max_sequence_length, num_classes)
