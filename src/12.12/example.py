# Importuojame reikalingas bibliotekas
from sklearn.datasets import fetch_20newsgroups  # Norint gauti 20 newsgroups duomenų rinkinį
from sklearn.feature_extraction.text import CountVectorizer  # Norint paversti tekstą į skaitmeninius dažnių matricas
from sklearn.model_selection import train_test_split  # Norint padalinti duomenis į treniravimo ir testavimo rinkinius
from sklearn.preprocessing import LabelEncoder  # Norint paversti tekstines etiketes į skaitmeninius indeksus
from tensorflow.keras.models import Sequential  # Norint sukurti Keras sekvencinį modelį
from tensorflow.keras.layers import Dense, SimpleRNN  # Norint sukurti sluoksnius, pvz., RNN ir pilno susijungimo sluoksnius

# Įkeliame 20 newsgroups duomenis ir pašaliname antraštes, pėdsakus ir citatas
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))  # Duomenų rinkinio nuskaitymas

# Padaliname duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)  # Duomenų padalinimas į 80% treniravimui ir 20% testavimui

# Sukuriame CountVectorizer, kad paverstume tekstus į n-gramų dažnių matricas
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500, stop_words='english')  
# ngram_range=(1, 2) reiškia, kad naudojame tiek 1-gramus (atskiri žodžiai), tiek 2-gramus (žodžių poras)
# max_features=500 nustato, kad paimame tik 500 dažniausių žodžių
# stop_words='english' pašalina angliškus stop žodžius, tokius kaip "the", "and" ir kt.

# Paverčiame tekstus į dažnių matricas
X_train = vectorizer.fit_transform(X_train).toarray()  # Treniruojami duomenys paverčiami į dažnių matricas
X_test = vectorizer.transform(X_test).toarray()  # Testavimo duomenys paverčiami į dažnių matricas

# Paverčiame kategorijas į skaitmeninius indeksus (etiketės į skaitmeninius kodus)
encoder = LabelEncoder()  # Sukuriame LabelEncoder objektą
y_train_encoded = encoder.fit_transform(y_train)  # Treniruojami duomenys paverčiami į skaitmeninius indeksus
y_test_encoded = encoder.transform(y_test)  # Testavimo duomenys paverčiami į skaitmeninius indeksus

# Surašome duomenis, kad jie atitiktų RNN modelio įėjimo formą (kiekvienas žodis kaip atskiras laipsnis)
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Duomenys turi būti 3D, kad atitiktų RNN formą (pavyzdžiui, [batch_size, timesteps, features])
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # Testavimo duomenys paverčiami į tą pačią formą

# Sukuriame RNN modelį
model = Sequential([  # Sekvencinis modelis, kuriame sluoksniai bus sujungti vienas po kito
    SimpleRNN(64, input_shape=(X_train_rnn.shape[1], 1), activation='relu'),  # RNN sluoksnis su 64 neuronais ir ReLU aktyvacija
    Dense(32, activation='relu'),  # Pilno susijungimo sluoksnis su 32 neuronais ir ReLU aktyvacija
    Dense(len(encoder.classes_), activation='softmax')  # Išėjimo sluoksnis, kuriame kiekvienas neuronų skaičius atitinka kategorijų skaičių
])

# Kompiliuojame modelį su Adam optimizatoriumi ir sparse_categorical_crossentropy praradimo funkcija
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Kompiliacija su optimizatoriumi ir tikslumo matavimu

# Mokome modelį naudodami treniravimo duomenis
model.fit(X_train_rnn, y_train_encoded, epochs=10, batch_size=16, validation_data=(X_test_rnn, y_test_encoded))
# Treniravimas: naudosime 5 epohas, 16 dydžio partiją ir validaciją su testavimo duomenimis

# Įvertiname modelio našumą naudojant testavimo duomenis
test_loss, test_accuracy = model.evaluate(X_test_rnn, y_test_encoded)  # Įvertiname testavimo praradimo ir tikslumo reikšmes

# Spausdiname modelio testavimo praradimo ir tikslumo rezultatus
print('Loss: ', test_loss, 'Accuracy: ', test_accuracy)  # Išvedame galutinius rezultatus
