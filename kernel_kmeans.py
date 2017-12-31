import numpy as np
import matplotlib.pyplot as plt
from utility import rbf_kernel, rbf_kernel_gram_matrix, reorder, show_cluster


# Second term of kernel k-means
def kkm_second_term(j, kernel_gram_mat, alpha):
    num_samples = kernel_gram_mat.shape[0]
    sum = 0
    for n in range(num_samples):
        sum += alpha[n] * kernel_gram_mat[j, n]

    return sum


# Third term of kernel k-means
def kkm_third_term(kernel_gram_mat, alpha):
    num_samples = kernel_gram_mat.shape[0]
    sum = 0
    for p in range(num_samples):
        for q in range(num_samples):
            sum += alpha[p] * alpha[q] * kernel_gram_mat[p, q]

    return sum


# RBF kernel k-means clusters
def kernel_k_means(data, k, gamma=0.03125):
    print("Kernel K-means clustering...")
    num_samples = data.shape[0]
    # First column stores which cluster this sample belongs to,
    # Second column stores the error between this sample and its centroid
    cluster_assignment = np.zeros((num_samples, 2))
    cluster_changed = True

    # Initial clusters
    cluster_assignment[:, 0] = np.random.randint(k, size=num_samples)
    # show_cluster(data, k, cluster_assignment[:, 0], title="Kernel K-means, initial clusters")

    # Reorder the data to see if the gram matrix has some diagonal structure
    # reordered_data = reorder(data, cluster_assignment[:, 0])
    # data = reordered_data

    # Gram matrix built by kernel
    kernel_gram_mat = rbf_kernel_gram_matrix(data, gamma)
    # plt.imshow(kernel_gram_mat)

    # Store if a data point is assigned to the k-th cluster
    alpha = np.zeros((k, num_samples))

    num_iterations = 0
    while cluster_changed:
        print("Number of iterations:", num_iterations)
        cluster_changed = False

        # Third term is the same among each iteration
        kkm_third_terms = np.zeros(k)
        for i in range(k):
            alpha[i, np.nonzero(cluster_assignment[:, 0] == i)[0]] = 1
            kkm_third_terms[i] = kkm_third_term(kernel_gram_mat, alpha[i])

        # For each sample
        for j in range(num_samples):
            min_distance = 100000.0
            min_index = 0
            for i in range(k):
                num_points_in_cluster = np.sum(alpha[i])
                b = kkm_second_term(j, kernel_gram_mat, alpha[i]) / num_points_in_cluster
                c = kkm_third_terms[i] / np.power(num_points_in_cluster, 2)
                # Distance in kernel space
                distance = 1 - 2 * b + c
                if distance < min_distance:
                    min_distance = distance
                    min_index = i

            # Update its cluster
            if cluster_assignment[j, 0] != min_index:
                cluster_changed = True
                cluster_assignment[j] = min_index, np.power(min_distance, 2)

        # title = "Kernel K-means, #iter:" + num_iterations.__str__()
        # show_cluster(data, k, cluster_assignment[:, 0], title=title)

        num_iterations += 1

    return cluster_assignment[:, 0]
