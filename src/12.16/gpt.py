import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Ensure GPU is used
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Optionally set memory growth to prevent TensorFlow from consuming all the GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load and preprocess the 20 Newsgroups dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
x = data.data
y = data.target

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove everything except letters and spaces
    words = text.split()  # Split into words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return " ".join(words)

x = [clean_text(doc) for doc in x]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, encoded_labels, test_size=0.2, random_state=42)

# Tokenization and padding
max_vocab_size = 20000  # Increased vocabulary size
max_sequence_length = 300  # Increased sequence length

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the model with GRU layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=256, input_length=max_sequence_length),  # Increased embedding size
    tf.keras.layers.GRU(128),  # Increased GRU units
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model with Adam optimizer and adjusted learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Increased learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(x_train_pad, y_train,
                    validation_data=(x_test_pad, y_test),
                    epochs=20,  # Increased epochs
                    batch_size=64,  # Batch size increased
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix
predictions = model.predict(x_test_pad)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Optionally print detailed classification report
from sklearn.metrics import classification_report
print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))
