from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Įkeliamas ir standartizuojamas duomenų rinkinys
breast_cancer = load_breast_cancer()
data = breast_cancer.data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Nustatomas optimalus klasterių skaičius (pvz., iš ankstesnio alkūnės metodo)
optimal_k = 4  # Tarkime, kad optimalus klasterių skaičius yra 4

# Naudojami skirtingi sujungimo metodai
linkage_methods = ['ward', 'complete', 'average', 'single']

# Braižome klasterizacijos rezultatus kiekvienam sujungimo metodui
plt.figure(figsize=(12, 8))
for i, method in enumerate(linkage_methods, 1):
    # Aglomeracinis klasterizavimas su konkrečiu sujungimo metodu
    agglomerative = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
    labels = agglomerative.fit_predict(data_standardized)

    # Braižome klasterių narių priklausomybės vaizdą
    plt.subplot(2, 2, i)
    plt.scatter(data_standardized[:, 0], data_standardized[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(f"Aglomeracinis klasterizavimas su '{method}' sujungimu")
    plt.xlabel("Pirmoji komponentė")
    plt.ylabel("Antroji komponentė")

plt.tight_layout()
plt.show()
