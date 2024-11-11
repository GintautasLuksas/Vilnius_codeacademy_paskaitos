from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data, _ = make_moons(n_samples=600, noise=0.1, random_state=42)

dbscan = DBSCAN(eps=0.15, min_samples=10)
# kmeans = KMeans(n_clusters=2)
clusters = dbscan.fit_predict(data)
# clusters = kmeans.fit_predict(data)

print(clusters)

plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], c=clusters, marker='o')
plt.xlabel('1 stulpelis')
plt.ylabel('2 stulpelis')
plt.show()

# data, _ = make_blobs(n_samples = ?, center = ?, cluster_str=0.6)
# https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python