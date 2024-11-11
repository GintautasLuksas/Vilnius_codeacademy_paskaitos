import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 1. Duomenų paruošimas
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Duomenų standartizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aglomeratyvioji klasterizacija (pavyzdžiui, 3 klasteriai)
agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels = agg_clustering.fit_predict(X_scaled)

# 4. Apskaičiuojame linkavimą
linked = linkage(X_scaled, 'ward')

# 5. Dendrogramos braižymas su žymėjimais
plt.figure(figsize=(10, 7))
dendrogram(linked)

# Žymėjimas, kur galima "pjauti" dendrogramą
plt.axhline(y=7, color='r', linestyle='--')  # Pasirinktas atstumas (tarkime 7)
plt.text(5, 7 + 0.5, 'Atstumas = 7', color='r', horizontalalignment='center')

plt.title('Dendrograma: Aglomeratyvioji klasterizacija')
plt.xlabel('Indeksai')
plt.ylabel('Atstumas')
plt.show()

# 6. Klasterių paskirstymas pagal pasirinkto atstumo ribą
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(linked, 7, criterion='distance')

# 7. Klasterių atvaizdavimas
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Pirmas požymis (scaled)')
plt.ylabel('Antras požymis (scaled)')
plt.title('Aglomeratyvios klasterizacijos rezultatai pagal 3 klasterius')
plt.show()

# Silhouette koeficiento apskaičiavimas
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f'Silhouette koeficientas: {silhouette_avg:.3f}')