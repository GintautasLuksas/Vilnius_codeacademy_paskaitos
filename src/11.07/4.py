from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Load the text data
data = fetch_20newsgroups(subset='all')
documents = data.data

# Step 1: Feature Extraction using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 2: Dimensionality Reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())


# Function to run KMeans for different cluster sizes and print silhouette scores
def evaluate_kmeans(X, max_clusters=10):
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        overall_score = silhouette_score(X, labels)
        print(f"Clusters: {k}, Overall Silhouette Score: {overall_score:.2f}")

        # Visualize the results for each k
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
        plt.title(f"Cluster Visualization for k={k}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()


# Run evaluation
evaluate_kmeans(X, max_clusters=10)
