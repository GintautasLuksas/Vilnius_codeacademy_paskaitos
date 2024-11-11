import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load wine dataset
data = load_wine()
X = data.data

# Standardize the dataset
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Calculate WCSS for different values of k (1 to 10)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_standardized)
    wcss.append(kmeans.inertia_)

# Plot WCSS vs. number of clusters (k)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.show()
