from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

def load_and_standardize_data():
    """
    Loads the wine dataset and applies standardization to ensure each feature has zero mean and unit variance.

    Returns:
        X (ndarray): Standardized feature matrix.
    """
    data = load_wine()
    X = data.data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized

def calculate_wcss(X):
    """
    Calculates the within-cluster sum of squares (WCSS) for different numbers of clusters (k) from 2 to 10.

    Args:
        X (ndarray): Standardized feature matrix.

    Returns:
        list: WCSS values for each k.
    """
    wcss = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow(wcss):
    """
    Plots the elbow curve using WCSS values to help determine the optimal number of clusters.

    Args:
        wcss (list): WCSS values for each k.
    """
    plt.plot(range(2, 11), wcss, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal k")
    plt.show()

def plot_clusters(X, y_kmeans, centers):
    """
    Plots the clustered data along with the cluster centers.

    Args:
        X (ndarray): Standardized feature matrix.
        y_kmeans (ndarray): Predicted cluster labels.
        centers (ndarray): Coordinates of cluster centers.
    """
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label="Data Points")
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label="Cluster Centers")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()

# Main code
X = load_and_standardize_data()
wcss = calculate_wcss(X)
plot_elbow(wcss)

# Optimal number of clusters
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Plotting clusters
plot_clusters(X, y_kmeans, centers)
