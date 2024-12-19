# Import necessary libraries
import numpy as np
import re
import nltk
import spacy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# Preload stopwords and spaCy model
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))  # Load stopwords as a set
nlp = spacy.load("en_core_web_sm")  # Load small spaCy model for lemmatization

# Load the 20 Newsgroups dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
x = data.data  # The text data
y = data.target  # The corresponding labels (categories)

# Encode the target labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# Split the dataset into Train, Validation, and Test sets
x_train, x_test, y_train, y_test = train_test_split(x, encoded_labels, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Define a function to preprocess text data
def clean_text(text):
    text = text.lower()  # Convert all text to lowercase
    text = re.sub(r'\b(?:https?|ftp)://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers and special characters
    doc = nlp(text)  # Apply spaCy NLP processing
    text = ' '.join([token.lemma_ for token in doc])  # Lemmatize words
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Clean the training, validation, and test text datasets (do this once to avoid redundancy)
x_train = [clean_text(doc) for doc in x_train]
x_val = [clean_text(doc) for doc in x_val]
x_test = [clean_text(doc) for doc in x_test]

# Tokenize the text data
max_vocab_size = 7000  # Limit the vocabulary size
max_sequence_length = 200  # Maximum sequence length (truncate/pad longer sequences)

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")  # Initialize tokenizer
tokenizer.fit_on_texts(x_train)  # Fit tokenizer only on the training data

# Convert text data into padded sequences of integers
x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_sequence_length, padding='post')
x_val_seq = pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=max_sequence_length, padding='post')
x_test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_sequence_length, padding='post')

# Handle class imbalance by computing class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define a function to build a model with GRU or LSTM
def build_model(model_type):
    """
    Build and return a Sequential neural network model.

    Parameters:
        model_type (str): 'GRU' or 'LSTM' to specify the recurrent layer type.

    Returns:
        tf.keras.Sequential: Compiled model ready for training.
    """
    model = Sequential([
        Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_sequence_length),  # Embedding layer
        GRU(128, return_sequences=False) if model_type == 'GRU' else LSTM(128, return_sequences=False),
        Dropout(0.5),  # Dropout for regularization to prevent overfitting
        Dense(64, activation='relu'),  # Fully connected layer with ReLU activation
        Dropout(0.5),  # Another Dropout layer for better regularization
        Dense(len(np.unique(y_train)), activation='softmax')  # Output layer with softmax activation for multi-class
    ])
    return model

# Define a function to train, evaluate, and visualize the model
def train_and_evaluate(model_type):
    """
    Train the GRU or LSTM model, evaluate it, and plot performance metrics.
    """
    print(f"Training {model_type} model...")

    # Build and compile the model
    model = build_model(model_type)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback to stop training if validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(
        x_train_seq, y_train,
        validation_data=(x_val_seq, y_val),
        epochs=10,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=2
    )

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test_seq, y_test, verbose=2)
    print(f"{model_type} Test Accuracy: {test_accuracy:.4f}")

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Generate confusion matrix and classification report
    predictions = model.predict(x_test_seq)
    predicted_classes = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_type} Confusion Matrix")
    plt.show()

    print(f"{model_type} Classification Report:")
    print(classification_report(y_test, predicted_classes, target_names=label_encoder.classes_))

# Train and evaluate both GRU and LSTM models
train_and_evaluate('GRU')
train_and_evaluate('LSTM')
