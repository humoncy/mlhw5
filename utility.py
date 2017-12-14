import numpy as np
import matplotlib.pyplot as plt


# calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))


# RBF kernel
def rbf_kernel(x1, x2, gamma):
    return np.exp(- gamma * np.power(euclidean_distance(x1, x2), 2))


# reorder data by their cluster_assignment
def reorder(data, cluster_assignment):
    return data[cluster_assignment.argsort()]


# show your cluster only available with 2-D data
def show_cluster(data, k, cluster_assignment, centroids=None):
    num_samples, dim = data.shape
    if dim != 2:
        raise Exception("Sorry, I can not draw because the dimension of your data is not 2!")

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        raise Exception("Sorry! Your k is too large to draw.")

    # draw all samples
    for i in range(num_samples):
        mark_index = int(cluster_assignment[i])
        plt.plot(data[i, 0], data[i, 1], mark[mark_index])

    if centroids is not None:
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # draw the centroids
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=10)

    plt.show()
