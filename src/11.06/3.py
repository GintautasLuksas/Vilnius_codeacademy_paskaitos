from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Įkeliamas ir standartizuojamas duomenų rinkinys
breast_cancer = load_breast_cancer()
data = breast_cancer.data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Sugeneruojamas ryšių (linkage) masyvas
linked = linkage(data_standardized, method='ward')

# Dendrogramos braižymas
plt.figure(figsize=(10, 7))
dendrogram(linked, color_threshold=150)  # Naudokite `color_threshold` (pvz., 150)
plt.title("Dendrogram for Hierarchical Clustering on Standardized Breast Cancer Dataset")
plt.xlabel("Data Points")
plt.ylabel("Distance")

# Parodymas
plt.show()
