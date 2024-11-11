from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Įkeliam vyno duomenis
data = load_wine()
X = data.data

# Standartizuojam duomenis
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Elbow metodas optimaliam klasterių skaičiui nustatyti
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_standardized)
    wcss.append(kmeans.inertia_)

# Piešiam Elbow grafiką
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel("Klasterių skaičius")
plt.ylabel("Nuostolis (WCSS)")
plt.title("Elbow metodas")
plt.show()

# Optimalus klasterių skaičius (pvz., 3) ir klasterizacija
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X_standardized)

# Klasterių vizualizacija
plt.scatter(X_standardized[:, 0], X_standardized[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.xlabel('Bruožas 1 (standartizuotas)')
plt.ylabel('Bruožas 2 (standartizuotas)')
plt.title("K-Means klasterizacija su standartizuotais duomenimis")
plt.show()
