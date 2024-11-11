from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from scipy.stats import mode

# Įkeliamas ir standartizuojamas duomenų rinkinys
breast_cancer = load_breast_cancer()
data = breast_cancer.data
true_labels = breast_cancer.target  # Tikrosios etiketės: 0 - piktybinis, 1 - gerybinis
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Skirtingi sujungimo metodai
linkage_methods = ['single', 'complete', 'average', 'ward']
optimal_k = 2  # Kadangi turime 2 klases: gerybinį ir piktybinį

# Laikome tikslumus kiekvienam metodui
accuracies = {}

# Aglomeracinis klasterizavimas su kiekvienu sujungimo metodu
for method in linkage_methods:
    # Sukuriame klasterių modelį su pasirinktu sujungimo metodu
    agglomerative = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
    cluster_labels = agglomerative.fit_predict(data_standardized)

    # Pakeičiame klasterių etiketes, kad jos labiau atitiktų tikrąsias etiketes
    new_labels = np.zeros_like(cluster_labels)
    for i in range(optimal_k):
        mask = (cluster_labels == i)
        new_labels[mask] = mode(true_labels[mask])[0]

    # Apskaičiuojame tikslumą
    accuracy = accuracy_score(true_labels, new_labels)
    accuracies[method] = accuracy
    print(f"Method '{method}' accuracy: {accuracy:.2f}")

# Rezultatai
print("\nTikslumai kiekvienam sujungimo metodui:")
for method, accuracy in accuracies.items():
    print(f"{method.capitalize()} sujungimas: {accuracy:.2%}")
