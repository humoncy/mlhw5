import numpy as np
import matplotlib.pyplot as plt
from utility import rbf_kernel, reorder, show_cluster


# RBF kernel Gram matrix
def rbf_kernel_gram_matrix(data, gamma):
    num_samples = data.shape[0]
    gram = np.identity(num_samples)

    for i in range(num_samples):
        for j in range(num_samples):
            if i > j:
                gram[i, j] = rbf_kernel(data[i], data[j], gamma)
                gram[j, i] = gram[i, j]

    return gram


def kkm_second_term(j, kernel_gram_mat, alpha):
    num_samples = kernel_gram_mat.shape[0]
    sum = 0
    for n in range(num_samples):
        sum += alpha[n] * kernel_gram_mat[j, n]

    return sum


def kkm_third_term(kernel_gram_mat, alpha):
    num_samples = kernel_gram_mat.shape[0]
    sum = 0
    for p in range(num_samples):
        for q in range(num_samples):
            sum += alpha[p] * alpha[q] * kernel_gram_mat[p, q]

    return sum


# rbf kernel k-means clusters
def kernel_k_means(data, k, gamma=0.03125):
    num_samples = data.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assignment = np.zeros((num_samples, 2))
    cluster_changed = True

    # initial clusters
    cluster_assignment[:, 0] = np.random.randint(2, size=num_samples)
    # show_cluster(data, k, cluster_assignment[:, 0])

    # To see some diagonal structure
    # reordered_data = reorder(data, cluster_assignment[:, 0])
    # data = reordered_data

    # Gram matrix built by kernel
    kernel_gram_mat = rbf_kernel_gram_matrix(data, gamma)
    # plt.imshow(kernel_gram_mat)

    # where if the data point is assigned to the k-th cluster
    alpha = np.zeros((k, num_samples))

    num_iterations = 0
    while cluster_changed:
        print("Number of iterations:", num_iterations)
        num_iterations += 1
        cluster_changed = False

        # This term is the same among each iteration
        kkm_third_terms = np.zeros(k)
        for i in range(k):
            alpha[i, np.nonzero(cluster_assignment[:, 0] == i)[0]] = 1
            kkm_third_terms[i] = kkm_third_term(kernel_gram_mat, alpha[i])

        # for each sample
        for j in range(num_samples):
            min_distance = 100000.0
            min_index = 0
            for i in range(k):
                num_points_in_cluster = np.sum(alpha[i])
                b = kkm_second_term(j, kernel_gram_mat, alpha[i]) / num_points_in_cluster
                c = kkm_third_terms[i] / np.power(num_points_in_cluster, 2)
                distance = 1 - 2 * b + c
                if distance < min_distance:
                    min_distance = distance
                    min_index = i

            # step 3: update its cluster
            if cluster_assignment[j, 0] != min_index:
                cluster_changed = True
                cluster_assignment[j] = min_index, np.power(min_distance, 2)

        # show_cluster(data, k, cluster_assignment[:, 0])

    return cluster_assignment[:, 0]
