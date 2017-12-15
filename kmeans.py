import numpy as np
import matplotlib.pyplot as plt
from utility import euclidean_distance
from utility import show_cluster


# initial centroids with random samples
def init_centroids(data, k):
    num_samples, dim = data.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, num_samples))
        centroids[i] = data[index]
    return centroids


# k-means clusters
def k_means(data, k=2):
    print("Data shape for K-means:", data.shape)
    if data.ndim == 1:
        raise Exception("Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
                        "or array.reshape(1, -1) if it contains a single sample.")

    num_samples = data.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assignment = np.zeros((num_samples, 2))
    cluster_changed = True

    # step 1: init centroids
    centroids = init_centroids(data, k)
    # show_cluster(data, k, cluster_assignment[:, 0], centroids)

    while cluster_changed:
        cluster_changed = False
        # for each sample
        for j in range(num_samples):
            min_distance = 100000.0
            min_index = 0
            # for each centroid
            # step 2: find the centroid who is closest
            for i in range(k):
                distance = euclidean_distance(data[j], centroids[i])
                if distance < min_distance:
                    min_distance = distance
                    min_index = i

            # step 3: update its cluster
            if cluster_assignment[j, 0] != min_index:
                cluster_changed = True
                cluster_assignment[j] = min_index, np.power(min_distance, 2)

        # step 4: update centroids
        for i in range(k):
            points_in_cluster = data[np.nonzero(cluster_assignment[:, 0] == i)[0]]
            centroids[i] = np.mean(points_in_cluster, axis=0)

        # show_cluster(data, k, cluster_assignment[:, 0], centroids)

    return centroids, cluster_assignment[:, 0]
