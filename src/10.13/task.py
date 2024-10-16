from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Duomenų užkrovimas
data = load_breast_cancer()
X = data.data
y = data.target

# Mokymo ir testavimo rinkinių atskyrimas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sąrašai tikslumui išsaugoti
train_accuracies = []
test_accuracies = []
k_values = range(1, 21)  # Išbandome k reikšmes nuo 1 iki 20

# Išbandome įvairias k reikšmes
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Tikslumas treniravimo rinkinyje
    train_accuracy = accuracy_score(y_train, knn.predict(X_train))
    train_accuracies.append(train_accuracy)

    # Tikslumas testavimo rinkinyje
    test_accuracy = accuracy_score(y_test, knn.predict(X_test))
    test_accuracies.append(test_accuracy)

# Braižome linijinę diagramą
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Train Accuracy', marker='o', linestyle='-')
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='o', linestyle='--')
plt.xlabel('K reikšmė (kaimynų skaičius)')
plt.ylabel('Tikslumas')
plt.title('Tikslumo kaita treniravimo ir testavimo rinkiniuose, priklausomai nuo K')
plt.legend()
plt.xticks(k_values)
plt.grid(True)
plt.show()
