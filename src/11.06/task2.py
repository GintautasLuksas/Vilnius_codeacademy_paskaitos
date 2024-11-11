from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt

# Įkeliamas Breast Cancer duomenų rinkinys
breast_cancer = load_breast_cancer()
data = breast_cancer.data

# Standartizuojame duomenų rinkinį
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Step 1: Dendrogramos braižymas
plt.figure(figsize=(10, 7))
linked = linkage(data_standardized, method='ward')
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering on Standardized Breast Cancer Dataset")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 2: Skaičiuojame WCSS su skirtingu klasterių skaičiumi
wcss = []
max_clusters = 10
for k in range(1, max_clusters + 1):
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglomerative.fit_predict(data_standardized)

    # Skaičiuojame WCSS klasteriams, suformuotiems naudojant Agglomerative Clustering
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_points = data_standardized[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Skaičiuojame WCSS
    _, dists = pairwise_distances_argmin_min(data_standardized, centroids)
    wcss.append(np.sum(dists ** 2))

# Step 3: Alkūnės metodo braižymas
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K in Agglomerative Clustering on Standardized Breast Cancer Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
