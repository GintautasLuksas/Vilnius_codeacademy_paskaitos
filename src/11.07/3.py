from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
X = data.data

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

overall_silhouette_score = silhouette_score(X, labels)
print(f"Overall Silhouette Score: {overall_silhouette_score}")

silhouette_vals = silhouette_samples(X, labels)
silhouette_df = pd.DataFrame({'Cluster': labels, 'Silhouette Score': silhouette_vals})
cluster_silhouette_scores = silhouette_df.groupby('Cluster')['Silhouette Score'].mean()

print("Silhouette Scores for Each Cluster:")
print(cluster_silhouette_scores)

#