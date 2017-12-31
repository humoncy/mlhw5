import numpy as np
import matplotlib.pyplot as plt


# calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))


# RBF kernel
def rbf_kernel(x1, x2, gamma):
    return np.exp(- gamma * np.power(euclidean_distance(x1, x2), 2))


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


# reorder data by their cluster_assignment
def reorder(data, cluster_assignment):
    return data[cluster_assignment.argsort()]


# show your cluster only available with 2-D data
def show_cluster(data, k, cluster_assignment, centroids=None, title='clustering', data_name="data2"):
    num_samples, dim = data.shape
    if dim != 2:
        print("Sorry, I can not draw because the dimension of your data is not 2!")
        return

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        raise Exception("Sorry! Your k is too large to draw.")

    # draw all samples
    for i in range(num_samples):
        mark_index = int(cluster_assignment[i])
        plt.plot(data[i, 0], data[i, 1], mark[mark_index])

    if centroids is None:
        plt.title(title + " , " + k.__str__() + " clusters")
        plt.savefig("result/" + data_name + "_" + title + "_" + k.__str__() + ".png")
        plt.show()
    else:
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # draw the centroids
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=10)
        plt.title(title + " , " + k.__str__() + " clusters")
        plt.savefig("result/" + data_name + "_" + title + "_" + k.__str__() + ".png")
        plt.show()

