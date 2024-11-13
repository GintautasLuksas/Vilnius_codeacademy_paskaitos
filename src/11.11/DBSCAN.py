import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

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

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_clusters = dbscan.fit_predict(pca_components)

# Check if DBSCAN has formed more than one cluster or noise (-1)
unique_clusters = set(dbscan_clusters)
if len(unique_clusters) > 1 and -1 not in unique_clusters:
    try:
        dbscan_silhouette_avg = silhouette_score(pca_components, dbscan_clusters)
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette_avg:.3f}")
    except ValueError:
        print("Error in calculating silhouette score for DBSCAN.")
else:
    print("DBSCAN formed insufficient clusters or only noise points (-1). Adjusting for silhouette score calculation.")
    # If DBSCAN has only one cluster or noise, treat noise as part of the cluster
    dbscan_clusters[dbscan_clusters == -1] = 0
    try:
        dbscan_silhouette_avg = silhouette_score(pca_components, dbscan_clusters)
        print(f"Adjusted DBSCAN Silhouette Score: {dbscan_silhouette_avg:.3f}")
    except ValueError:
        print("Unable to calculate silhouette score for DBSCAN.")

# --- K-Means Clustering ---
kmeans = KMeans(n_clusters=5, random_state=42)  # Set the number of clusters, e.g., 3
kmeans_clusters = kmeans.fit_predict(pca_components)

# Print the K-Means cluster assignments
print(f"K-Means Clusters: {kmeans_clusters}")

# Calculate silhouette score for K-Means
kmeans_silhouette_avg = silhouette_score(pca_components, kmeans_clusters)
print(f"K-Means Silhouette Score: {kmeans_silhouette_avg:.3f}")

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
