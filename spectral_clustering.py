import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
import scipy.linalg
from kmeans import k_means
import matplotlib.pyplot as plt
from utility import rbf_kernel, rbf_kernel_gram_matrix, reorder, show_cluster


# Compute D
def compute_degrees(W):
    num_samples = W.shape[0]
    D = np.zeros_like(W)
    for i in range(num_samples):
        D[i, i] = np.sum(W[i])

    return D


# Normalize Cut
# Spectral clustering by rbf kernel
def spectral_clustering(data, k, gamma=0.1):
    print("Spectral clustering...")
    # Number of samples
    n = data.shape[0]

    # Adjacency matrix
    W = rbf_kernel_gram_matrix(data, gamma)

    # Build Degree matrix
    D = compute_degrees(W)

    # Graph Laplacian
    L = D - W

    # Normalized Cut
    D_inv_sqrt = scipy.linalg.fractional_matrix_power(D, -1/2)
    L_sym = D_inv_sqrt.dot(L).dot(D_inv_sqrt)

    # Compute the first k eigenvectors of L_sym
    eig_values, eig_vectors = LA.eig(L_sym)

    # T contains first k eigenvectors of normalized Laplacian
    T = np.zeros((n, k))
    for i in range(k):
        T[:, i] = eig_vectors[:, i]

    # Resubstitude matrix H and normalize by its rows.
    H = D_inv_sqrt.dot(T)
    H /= LA.norm(H, axis=1, ord=2)[:, np.newaxis]

    # Cluster data points in eigenspace
    centroids, cluster_assignment = k_means(H, k)

    # Show the data points at the same point in eigenspace.
    # discrete_h = np.zeros_like(H)
    # c, discrete_h[:, 0] = k_means(H[:, 0].reshape(-1, 1), k)
    # c, discrete_h[:, 1] = k_means(H[:, 1].reshape(-1, 1), k)
    # centroids, cluster_assignment = k_means(discrete_h, k)

    return cluster_assignment
