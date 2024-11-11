import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 1. Duomenų paruošimas
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Duomenų standartizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Alkūnės metodo taikymas WCSS skaičiavimui
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 4. Alkūnės diagramos braižymas
plt.plot(K_range, wcss, marker='o')
plt.xlabel('Klasterių skaičius (K)')
plt.ylabel('WCSS')
plt.title('Alkūnės metodas optimaliai K reikšmei nustatyti')
plt.show()

# 5. Optimalus klasterių skaičius (pavyzdžiui, K=3)
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# 6. Klasterizacijos rezultatų įvertinimas su silueto koeficientu
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silueto koeficientas (K={optimal_k}): {silhouette_avg:.3f}')

# 7. Rezultatų atvaizdavimas
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.xlabel('Pirmas požymis (scaled)')
plt.ylabel('Antras požymis (scaled)')
plt.title('K-Means klasterizacijos rezultatai')
plt.show()
