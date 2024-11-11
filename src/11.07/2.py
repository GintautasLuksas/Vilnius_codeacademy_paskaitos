import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# Load dataset
data = load_diabetes()
X = data.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Elbow method for KMeans
inertia = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for KMeans')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Elbow method for Agglomerative Clustering
agg_inertia = []
for k in k_range:
    agglomerative = AgglomerativeClustering(n_clusters=k)
    agglomerative.fit(X_scaled)
    # Calculate inertia as the sum of squared distances between points and their centroids (approximate)
    labels = agglomerative.labels_
    inertia_value = 0
    for cluster in range(k):
        cluster_points = X_scaled[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        inertia_value += np.sum((cluster_points - centroid) ** 2)
    agg_inertia.append(inertia_value)

plt.figure(figsize=(8, 6))
plt.plot(k_range, agg_inertia, marker='o', color='red')
plt.title('Elbow Method for Agglomerative Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# KMeans and Agglomerative Clustering with Silhouette Scores
for k in k_range[1:]:
    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    print(f"KMeans (k={k}) Silhouette Score: {kmeans_silhouette:.2f}")

    # Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=k)
    agglomerative_labels = agglomerative.fit_predict(X_scaled)
    agglomerative_silhouette = silhouette_score(X_scaled, agglomerative_labels)
    print(f"Agglomerative Clustering (k={k}) Silhouette Score: {agglomerative_silhouette:.2f}")
