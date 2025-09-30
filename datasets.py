from sklearn.datasets import make_blobs, make_moons

def get_datasets():
    datasets = {
        "blobs": make_blobs(n_samples=500, centers=4, cluster_std=0.7, random_state=42),
        "moons": make_moons(n_samples=500, noise=0.07, random_state=42)
    }
    return datasets
