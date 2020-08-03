import numpy as np
import torch as t
import argparse
import os
import Util as util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering

np.random.seed(0)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', help='embedding_file', type=str, required=True)
    parser.add_argument('-c', '--cluster', help='number of clusters', type=int, required=True)

    args = parser.parse_args()
    f = args.file
    nc = args.cluster
    embedding = np.genfromtxt(f)
    color = ['#ca3537', '#0069aa','#298346']
    plt.scatter(embedding[0:10, 0], embedding[0:10, 1], color=color[0], label='Cluster {0}'.format(1))
    plt.scatter(embedding[10:, 0], embedding[10:, 1], color=color[1], label='Cluster {0}'.format(2))

    # # kpca = KernelPCA(n_components=nc, kernel='rbf', gamma=3).fit(embedding)
    # # embedding = kpca.transform(embedding)
    # n_clusters = nc
    
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)


    # np.savetxt(f+'_cluster', kmeans.labels_)
    # 
    # for j in range(n_clusters):
    #     idx = kmeans.labels_ == j
    #     cur_x = embedding[idx]
    #     plt.scatter(cur_x[:, 0], cur_x[:, 1], color=color[j], label='Cluster {0}'.format(j+1))
    plt.xticks(np.linspace(-0.4, 0.8, 4))
    plt.yticks(np.linspace(-0.4, 0.8, 4))
    plt.legend()
    plt.show()
    # plt.savefig(f+'.png')


