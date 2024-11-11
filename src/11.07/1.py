from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
data = load_digits()
X = data.data

# Step 1: Standardize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Reduce dimensionality using PCA (optional, can also tune the number of components)
pca = PCA(n_components=50)  # Adjust the number of components based on the data
X_pca = pca.fit_transform(X_scaled)

# KMeans Clustering (does not use linkage)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)
kmeans_silhouette = silhouette_score(X_pca, kmeans_labels)
print(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}")

# Agglomerative Clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
for linkage in linkage_methods:
    agglomerative = AgglomerativeClustering(n_clusters=10, linkage=linkage)
    agglomerative_labels = agglomerative.fit_predict(X_pca)
    agglomerative_silhouette = silhouette_score(X_pca, agglomerative_labels)
    print(f"Agglomerative Clustering (linkage={linkage}) Silhouette Score: {agglomerative_silhouette:.2f}")
