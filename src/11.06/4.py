from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Įkeliamas ir standartizuojamas duomenų rinkinys
breast_cancer = load_breast_cancer()
data = breast_cancer.data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Skaičiuojame WCSS su skirtingu klasterių skaičiumi
wcss = []
max_clusters = 10  # Galima keisti maksimalų klasterių skaičių
for k in range(1, max_clusters + 1):
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglomerative.fit_predict(data_standardized)

    # Skaičiuojame kiekvieno klasterio centroidus
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_points = data_standardized[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Skaičiuojame WCSS
    _, dists = pairwise_distances_argmin_min(data_standardized, centroids)
    wcss.append(np.sum(dists ** 2))

# Alkūnės metodo braižymas
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title("Alkūnės metodas optimaliam klasterių skaičiui nustatyti")
plt.xlabel("Klasterių skaičius")
plt.ylabel("WCSS")
plt.show()
