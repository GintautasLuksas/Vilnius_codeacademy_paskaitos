import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial import distance_matrix
import numpy as np
from itertools import product

# Load data
df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Vilnius_codeacademy_paskaitos/src/11.11/Mall_Customers.csv')

# Drop the CustomerID and Gender columns as they're not needed for clustering
df = df.drop(['CustomerID', 'Gender'], axis=1)

# Select the features that need to be standardized
features = ['Annual Income (k$)', 'Spending Score (1-100)']

# Standardize the data using StandardScaler
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Apply PCA for dimensionality reduction (reduce to 2 components for visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[features])

# --- Parameter Search for DBSCAN ---
best_dbscan_params = {'eps': None, 'min_samples': None}
best_dbscan_score = -1  # Initialize with a low silhouette score

# Define parameter ranges for DBSCAN
eps_range = np.arange(0.2, 1.0, 0.1)  # Example range for eps
min_samples_range = range(3, 10)  # Example range for min_samples

for eps, min_samples in product(eps_range, min_samples_range):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_clusters = dbscan.fit_predict(pca_components)

    # Check if there are enough clusters (exclude noise)
    if len(set(dbscan_clusters) - {-1}) > 1:
        try:
            silhouette_avg = silhouette_score(pca_components, dbscan_clusters)
            if silhouette_avg > best_dbscan_score:
                best_dbscan_score = silhouette_avg
                best_dbscan_params['eps'] = eps
                best_dbscan_params['min_samples'] = min_samples
        except ValueError:
            continue

print(f"Best DBSCAN Parameters: {best_dbscan_params}, Silhouette Score: {best_dbscan_score:.3f}")

# --- DBSCAN Clustering with Best Parameters ---
dbscan = DBSCAN(eps=best_dbscan_params['eps'], min_samples=best_dbscan_params['min_samples'])
dbscan_clusters = dbscan.fit_predict(pca_components)

# Check unique clusters, including noise (-1)
unique_clusters = set(dbscan_clusters)
print("Unique DBSCAN clusters:", unique_clusters)

# Calculate silhouette score only if there are valid clusters (excluding noise)
if len(unique_clusters - {-1}) > 1:  # More than one valid cluster
    try:
        dbscan_silhouette_avg = silhouette_score(pca_components, dbscan_clusters)
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette_avg:.3f}")
        dbscan_davies_bouldin = davies_bouldin_score(pca_components, dbscan_clusters)
        print(f"DBSCAN Davies-Bouldin Index: {dbscan_davies_bouldin:.3f}")
    except ValueError:
        print("Error in calculating silhouette score for DBSCAN.")
else:
    print("DBSCAN did not form enough clusters for silhouette score calculation.")

# --- Parameter Search for K-Means ---
best_kmeans_params = {'n_clusters': None}
best_kmeans_score = -1  # Initialize with a low silhouette score

# Define parameter range for K-Means
n_clusters_range = range(2, 11)  # Example range for n_clusters

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(pca_components)

    silhouette_avg = silhouette_score(pca_components, kmeans_clusters)
    if silhouette_avg > best_kmeans_score:
        best_kmeans_score = silhouette_avg
        best_kmeans_params['n_clusters'] = n_clusters

print(f"Best K-Means Parameters: {best_kmeans_params}, Silhouette Score: {best_kmeans_score:.3f}")

# --- K-Means Clustering with Best Parameters ---
kmeans = KMeans(n_clusters=best_kmeans_params['n_clusters'], random_state=42)
kmeans_clusters = kmeans.fit_predict(pca_components)

# Calculate silhouette score for K-Means
kmeans_silhouette_avg = silhouette_score(pca_components, kmeans_clusters)
print(f"K-Means Silhouette Score: {kmeans_silhouette_avg:.3f}")

# Calculate Davies-Bouldin Index for K-Means
kmeans_davies_bouldin = davies_bouldin_score(pca_components, kmeans_clusters)
print(f"K-Means Davies-Bouldin Index: {kmeans_davies_bouldin:.3f}")


# --- Dunn Index Calculation Function ---
def dunn_index(points, labels):
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise for DBSCAN
    if len(unique_labels) < 2:
        return np.nan  # Cannot compute if less than 2 clusters

    inter_cluster_distances = []
    intra_cluster_distances = []

    for label in unique_labels:
        cluster_points = points[labels == label]
        intra_cluster_distances.append(np.max(distance_matrix(cluster_points, cluster_points)))

        for other_label in unique_labels:
            if label != other_label:
                other_points = points[labels == other_label]
                inter_cluster_distances.append(np.min(distance_matrix(cluster_points, other_points)))

    min_inter = np.min(inter_cluster_distances)
    max_intra = np.max(intra_cluster_distances)

    return min_inter / max_intra if max_intra != 0 else np.nan


# Dunn Index for DBSCAN
dbscan_dunn = dunn_index(pca_components, dbscan_clusters)
print(f"DBSCAN Dunn Index: {dbscan_dunn:.3f}")

# Dunn Index for K-Means
kmeans_dunn = dunn_index(pca_components, kmeans_clusters)
print(f"K-Means Dunn Index: {kmeans_dunn:.3f}")

# --- Visualizing the Clusters ---
plt.figure(figsize=(14, 6))

# DBSCAN Clusters Visualization
plt.subplot(1, 2, 1)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_clusters, cmap='viridis', marker='o')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('DBSCAN Clustering after PCA')
plt.colorbar(label='Cluster')

# K-Means Clusters Visualization
plt.subplot(1, 2, 2)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering after PCA')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
