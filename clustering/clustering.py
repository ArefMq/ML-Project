from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as HierarchicalClustering
from sklearn.metrics import silhouette_score
from sklearn import decomposition
import matplotlib.pyplot as plt

from dataset.dataset_buddy import get_clustering_dataset

colors = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.',
          'b*', 'g*', 'r*', 'c*', 'm*', 'y*', 'k*']


def plot_flatten_data(x, y, figure=1, title=None):
    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    x_2d = pca.transform(x)

    plt.subplot(3, 3, figure)
    if title:
        plt.title(title)
    for _x, _y in zip(x_2d, y):
        plt.plot(_x[0], _x[1], colors[_y])
    # plt.ion()
    # plt.pause(0.001)


def run_k_means_clustering():
    x_train, y_train = get_clustering_dataset()
    best_score = 0
    best_k = 1
    for n_cluster in range(2, 10):
        k_means = KMeans(n_clusters=n_cluster).fit(x_train)
        score = silhouette_score(x_train, k_means.labels_)
        if best_score < score:
            best_score = score
            best_k = n_cluster
        print '  %d) %.2f' % (n_cluster, score)

        plot_flatten_data(x_train, k_means.labels_, figure=n_cluster - 1, title='n=%d' % n_cluster)
    plt.show()
    print 'best k = %.2f' % best_k


def run_hierarchical_clustering():
    x_train, y_train = get_clustering_dataset()
    best_score = 0
    best_k = 1
    for n_cluster in range(2, 10):
        hc = HierarchicalClustering(n_clusters=n_cluster).fit(x_train)
        score = silhouette_score(x_train, hc.labels_)
        if best_score < score:
            best_score = score
            best_k = n_cluster
        print '  %d) %.2f' % (n_cluster, score)

        plot_flatten_data(x_train, hc.labels_, figure=n_cluster - 1, title='n=%d' % n_cluster)
    plt.show()
    print 'best k = ', best_k
