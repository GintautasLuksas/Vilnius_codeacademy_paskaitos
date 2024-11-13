import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial import distance

# Load the diabetes dataset from sklearn
from sklearn.datasets import load_diabetes

# Load and create a DataFrame
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA for dimensionality reduction (reduce to 2 components for visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(pca_components)

# Check unique clusters, including noise (-1)
unique_clusters = set(dbscan_clusters)
print("Unique DBSCAN clusters:", unique_clusters)

# Calculate silhouette score and Davies-Bouldin index for DBSCAN if valid clusters exist
if len(unique_clusters - {-1}) > 1:  # More than one cluster excluding noise
    dbscan_silhouette_avg = silhouette_score(pca_components, dbscan_clusters)
    dbscan_davies_bouldin = davies_bouldin_score(pca_components, dbscan_clusters)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette_avg:.3f}")
    print(f"DBSCAN Davies-Bouldin Index: {dbscan_davies_bouldin:.3f}")
else:
    print("DBSCAN did not form enough clusters for silhouette score or Davies-Bouldin index calculation.")

# --- K-Means Clustering ---
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_clusters = kmeans.fit_predict(pca_components)

# Calculate silhouette score and Davies-Bouldin index for K-Means
kmeans_silhouette_avg = silhouette_score(pca_components, kmeans_clusters)
kmeans_davies_bouldin = davies_bouldin_score(pca_components, kmeans_clusters)
print(f"K-Means Silhouette Score: {kmeans_silhouette_avg:.3f}")
print(f"K-Means Davies-Bouldin Index: {kmeans_davies_bouldin:.3f}")

# --- Dunn Index Calculation ---
def calculate_dunn_index(points, labels):
    unique_labels = set(labels)
    clusters = [points[labels == label] for label in unique_labels if label != -1]  # Exclude noise

    if len(clusters) < 2:
        return float('inf')  # Dunn index is undefined for a single cluster

    intra_distances = [max(distance.pdist(cluster)) for cluster in clusters]
    inter_distances = [distance.euclidean(c1.mean(axis=0), c2.mean(axis=0)) for i, c1 in enumerate(clusters) for c2 in clusters[i + 1:]]

    return min(inter_distances) / max(intra_distances)

# Calculate Dunn index for DBSCAN
dbscan_dunn_index = calculate_dunn_index(pca_components, dbscan_clusters)
print(f"DBSCAN Dunn Index: {dbscan_dunn_index:.3f}")

# Calculate Dunn index for K-Means
kmeans_dunn_index = calculate_dunn_index(pca_components, kmeans_clusters)
print(f"K-Means Dunn Index: {kmeans_dunn_index:.3f}")

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
