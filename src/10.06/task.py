from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Pakraunamas duomenų rinkinys
data = fetch_20newsgroups()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Sukuriame TF-IDF ir Naive Bayes klasifikatorių su pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Treniruojame modelį
model.fit(X_train, y_train)

# Atliekame prognozes
predictions = model.predict(X_test)

# Skaičiuojame tikslumą
accuracy = accuracy_score(y_test, predictions)
print("Modelio tikslumas:", accuracy)
