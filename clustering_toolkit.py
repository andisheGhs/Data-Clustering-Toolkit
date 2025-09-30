import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from datasets import get_datasets

sns.set(style="whitegrid")

def run_clustering(X, name):
    algorithms = {
        "KMeans": KMeans(n_clusters=4, random_state=42),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=4)
    }

    plt.figure(figsize=(12, 3))
    plt.suptitle(f"{name.capitalize()} Dataset - Clustering Comparison", fontsize=14)

    for i, (alg_name, alg) in enumerate(algorithms.items()):
        labels = alg.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan

        plt.subplot(1, 3, i+1)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
        plt.title(f"{alg_name}\nSilhouette: {score:.2f}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


if __name__ == "__main__":
    datasets = get_datasets()
    for name, (X, y) in datasets.items():
        run_clustering(X, name)
