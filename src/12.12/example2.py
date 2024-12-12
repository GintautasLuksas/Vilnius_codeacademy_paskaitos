# Importuojame reikalingas bibliotekas iš Keras ir scikit-learn
from tensorflow.keras.preprocessing.text import Tokenizer  # Norint paversti tekstą į skaitmeninius indeksus
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Norint užtikrinti, kad sekos būtų vienodo ilgio
from tensorflow.keras.models import Sequential  # Norint sukurti modelį
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout  # Skirtingi neuroninių tinklų sluoksniai
from sklearn.datasets import fetch_20newsgroups  # Norint gauti 20 naujienų grupių duomenis
from sklearn.preprocessing import LabelEncoder  # Norint paversti etiketes į skaitmeninius indeksus
from sklearn.model_selection import train_test_split  # Norint padalinti duomenis į treniravimo ir testavimo rinkinius

# Funkcija n-gramų kūrimui
def generate_ngrams(text, n):
    tokens = text.split()  # Paverčiame tekstą į žodžių sąrašą pagal tarpus
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]  # Sukuriame n-gramus
    return ngrams

# Atsisiunčiame duomenis iš 20 newsgroups dataset (straipsnių iš įvairių temų)
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))  
X, y = newsgroups.data, newsgroups.target  # X - tekstai, y - kategorijos (etiketės)

# Nustatome n-gramo dydį. Čia naudosime 2-gramus (pavyzdžiui, žodžių poras)
n = 2
X_ngrams = [" ".join(generate_ngrams(text, n)) for text in X]  # Sukuriame n-gramus kiekvienam tekstui

# Paverčiame etiketes į skaitmeninius indeksus
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Paverčiame kategorijas į skaitmenines reikšmes

# Padaliname duomenis į treniravimo ir testavimo rinkinius (80% - treniravimui, 20% - testavimui)
X_train, X_test, y_train, y_test = train_test_split(X_ngrams, y_encoded, test_size=0.2, random_state=42)

# Nustatome parametrus, kaip mūsų modelis apdoros tekstus
max_words = 200  # Nustatome maksimalų žodžių kiekį, kuris bus naudojamas analizei
tokenizer = Tokenizer(num_words=max_words)  # Sukuriame Tokenizer objektą su maksimaliu žodžių skaičiumi
tokenizer.fit_on_texts(X_train)  # Suformuojame žodyną, kuris žino, kurie žodžiai yra labiausiai dažni treniravimo duomenyse

# Paverčiame tekstus į skaitmenines sekas, kur kiekvienas žodis atitinka jo žodyno indeksą
X_train_seq = tokenizer.texts_to_sequences(X_train)  # Treniruotės duomenys
X_test_seq = tokenizer.texts_to_sequences(X_test)  # Testavimo duomenys

# Užtikriname, kad visos sekos turėtų tą patį ilgį, papildydami jas nuliniais reikšmėmis
max_sequence_length = 100  # Nustatome, kad kiekviena sekos ilgis bus 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')  # Padaliname sekas pagal nustatytą ilgį
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')  # Padaliname testavimo sekas

# Sukuriame modelį
model = Sequential([  # Naudojame "Sequential" modelį, kad sujungtume sluoksnius
    Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length),  # Sukuriame Embedding sluoksnį (žodžių įkrovimas)
    SimpleRNN(64, activation='relu', return_sequences=False),  # RNN sluoksnis (atpažįsta sekas)
    Dense(64, activation='relu'),  # Pilno susijungimo sluoksnis su ReLU aktyvacija
    Dense(len(label_encoder.classes_), activation='softmax')  # Galutinis sluoksnis su softmax, kad gautume tikėtinas kategorijas
])

# Kompiliuojame modelį, nustatydami optimizavimo metodą ir praradimo funkciją
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Mokome modelį, naudojant treniravimo duomenis
model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2)  # Treniravimas su validacijos rinkiniu

# Įvertiname modelio tikslumą naudojant testavimo duomenis
loss, accuracy = model.evaluate(X_test_pad, y_test)  # Įvertiname modelį su testavimo duomenimis

# Spausdiname modelio testavimo praradimo (loss) ir tikslumo (accuracy) reikšmes
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
