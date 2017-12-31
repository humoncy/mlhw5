import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from kernel_kmeans_ref import KernelKMeans
from kmeans import k_means
from kernel_kmeans import kernel_k_means
from utility import show_cluster
from spectral_clustering import spectral_clustering


if __name__ == "__main__":
    data1 = np.loadtxt('data/test1_data.txt')
    gt1 = np.loadtxt('data/test1_ground.txt')

    data2 = np.loadtxt('data/test2_data.txt')
    gt2 = np.loadtxt('data/test2_ground.txt')

    data = data2
    gt = gt2

    data_name = "data2"

    # Show data with ground truth
    plt.scatter(data[:, 0], data[:, 1], c=gt)
    plt.title("Ground Truth of " + data_name)
    plt.show()

    k = 2

    # sklearn kmeans
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    # plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    # plt.show()
    #

    # K-means on my own
    centroids, cluster_assignment = k_means(data, k)
    show_cluster(data, k, cluster_assignment, centroids, title="K-means clusters", data_name=data_name)

    # sklearn kernel k-means
    # kernel_kmeans = KernelKMeans(n_clusters=2, kernel='rbf', gamma=0.03125)
    # plt.scatter(data[:, 0], data[:, 1], c=kernel_kmeans.fit_predict(data))
    # plt.show()
    #

    # Kernel k-means on my own
    cluster_assignment = kernel_k_means(data, k, gamma=0.03125)
    show_cluster(data, k, cluster_assignment, title="Kernel K-means clusters", data_name=data_name)

    # sklearn spectral clustering
    # spectral_clusters = SpectralClustering(n_clusters=2, gamma=0.1)
    # plt.scatter(data[:, 0], data[:, 1], c=spectral_clusters.fit_predict(data))
    # plt.show()
    #

    # Spectral clustering on my own
    cluster_assignment = spectral_clustering(data, k, gamma=0.1)
    show_cluster(data, k, cluster_assignment, title="Spectral clustering", data_name=data_name)

'''
    The following code is used to run different number of clusters at the same time
'''
    # ks = [2, 3, 5, 10]
    # datas = [data1, data2]
    # # print(datas[1].shape)
    # # data = datas[1]
    # # plt.scatter(data[:, 0], data[:, 1], c=gt2)
    # # plt.title("Ground Truth of " + data_name)
    # # plt.show()
    # for data_index in range(2):
    #     for index, val in enumerate(ks):
    #         data = datas[data_index]
    #         data_name = "data" + (data_index+1).__str__()
    #
    #         k = val
    #
    #         centroids, cluster_assignment = k_means(data, k)
    #         show_cluster(data, k, cluster_assignment, centroids, title="K-means clusters", data_name=data_name)
    #
    #         cluster_assignment = kernel_k_means(data, k, gamma=0.03125)
    #         show_cluster(data, k, cluster_assignment, title="Kernel K-means clusters", data_name=data_name)
    #
    #         cluster_assignment = spectral_clustering(data, k, gamma=0.1)
    #         show_cluster(data, k, cluster_assignment, title="Spectral clustering", data_name=data_name)
