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

class Effect:
    def __init__(self, params, entity_dim, dataset):
        self.dataset = dataset
        self.entity_typenb = len(entity_dim)
        self.jitter = 1e-5
        self.embedding = []
        for _ in range(self.entity_typenb):
            self.embedding.append(params.pop(0))
        # kernel
        self.kernel_rbf = util.Kernel_RBF(self.jitter)
        # f
        self.f_log_ls = params.pop(0)
        # g
        self.g_log_ls = params.pop(0)
        # k
        self.k_log_ls = params.pop(0)

        # log_dr = register_param(t.tensor([0.]))
        self.log_dr = params.pop(0)
        self.sp_log_ls = params.pop(0)

        # posterior
        # f
        self.f_b = params.pop(0)
        self.f_L = params.pop(0)
        self.f_B = params.pop(0)
        # g
        self.g_b = params.pop(0)
        self.g_L = params.pop(0)
        self.g_B = params.pop(0)

    def construct_x(self, ind):
        x = []
        # print(ind.shape)
        for i in range(self.entity_typenb):
            ind_i = ind[:,i].view(-1)
            x.append(self.embedding[i].index_select(0, ind_i))
        x = t.cat(x, 1)
        return x
    
    def X(self, ind1, ind2, sample=True, ratio=1.0):
        ind1_nb = ind1.shape[0]
        ind2_nb = ind2.shape[0]
        
        x1 = self.construct_x(ind1)
        x2 = self.construct_x(ind2)
        dim = x1.shape[1]
        # f
        # fkXB = self.kernel_rbf.cross(x1, self.f_B, t.exp(self.f_log_ls))
        # fkBB = self.kernel_rbf.matrix(self.f_B, t.exp(self.f_log_ls))
        # if sample:
        #     # sample f_b
        #     fLtril = t.tril(self.f_L)
        #     epsilon = self.f_b.new(self.f_b.shape).normal_(mean=0, std=1)
        #     f_b = self.f_b + fLtril @ epsilon
        #     # sample f
        #     mean = fkXB @ t.solve(f_b, fkBB)[0]
        #     var  = 1.0 + self.jitter - (fkXB.transpose(0, 1) * t.solve(fkXB.transpose(0, 1), fkBB)[0]).sum(0) 
        #     std = t.sqrt(var).view(mean.shape)
        #     epsilon = mean.new(mean.shape).normal_(mean=0, std=1)
        #     f = mean + std * epsilon
        # else:
        #     f_b = self.f_b
        #     mean = fkXB @ t.solve(f_b, fkBB)[0]
        #     f = mean

        # k
        K = self.kernel_rbf.cross(x1, x2, t.exp(self.k_log_ls)) # n1 x n2

        # g
        x = x2.view((1, ind2_nb, dim)) - x1.view((ind1_nb, 1, dim))
        x = x.view((-1, dim))
        gkXB = self.kernel_rbf.cross(x, self.g_B, t.exp(self.g_log_ls))
        gkBB = self.kernel_rbf.matrix(self.g_B, t.exp(self.g_log_ls))
        if sample:
            # sample g_b
            gLtril = t.tril(self.g_L)
            epsilon = self.g_b.new(self.g_b.shape).normal_(mean=0, std=1)
            g_b = self.g_b + gLtril @ epsilon
            # sample g
            mean = gkXB @ t.solve(g_b, gkBB)[0]
            var = 1.0 + self.jitter - (gkXB.transpose(0, 1) * t.solve(gkXB.transpose(0, 1), gkBB)[0]).sum(0)
            std = t.sqrt(var).view(mean.shape)
            epsilon = mean.new(mean.shape).normal_(mean=0, std=1)
            g = mean + std * epsilon
        else:
            g_b = self.g_b
            mean = gkXB @ t.solve(g_b, gkBB)[0]
            g = mean
            
        # mask
        # mask = (time1 > time2.transpose(0, 1)).type(t.float32) # n1 x n2

        # reshape
        g = g.view((ind1_nb, ind2_nb))
        # t_diff = (time1 - time2.transpose(0,1)) * mask
        # effect = t.tanh(g) * K * t.exp(-1.0 * t_diff / t.exp(self.log_dr)) * mask
        # effect = effect.sum(1).view(f.shape) * ratio
        # X = f + effect
        return g
    def effect(self, ind):
        # ind = t.from_numpy(ind).cuda()
        g = self.X(ind, ind, False)
        return g.detach().cpu().numpy()
    
    def effect1(self, ind1, ind2):
        # ind = t.from_numpy(ind).cuda()
        g = self.X(ind1, ind2, False)
        return g.detach().cpu().numpy()
    
    def each_mode(self):
       for i in range(self.entity_typenb):
            embedding = self.embedding[i].detach().cpu().numpy()
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
            kpca = KernelPCA(n_components=2, kernel='rbf', gamma=3).fit(embedding)
            embedding = kpca.transform(embedding)
            plt.figure()
            for j in range(n_clusters):
                idx = kmeans.labels_ == j
                cur_x = embedding[idx]
                plt.scatter(cur_x[:, 0], cur_x[:, 1])
            plt.xticks(np.linspace(-0.6, 0.6, 5))
            plt.yticks(np.linspace(-0.6, 0.6, 5))
            plt.savefig('{0}_emb{1}.png'.format(self.dataset, i))

    def each_modev1(self):
       for i in range(self.entity_typenb):
            embedding = self.embedding[i].detach().cpu().numpy()
            kpca = KernelPCA(n_components=2, kernel='rbf', gamma=3).fit(embedding)
            embedding = kpca.transform(embedding)
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
            plt.figure()
            for j in range(n_clusters):
                idx = kmeans.labels_ == j
                cur_x = embedding[idx]
                plt.scatter(cur_x[:, 0], cur_x[:, 1])
            plt.savefig('{0}_emb{1}.png'.format(self.dataset, i))

    def each_modev2(self):
       for i in range(self.entity_typenb):
            embedding = self.embedding[i].detach().cpu().numpy()
            n_clusters = 4
            clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(embedding)
            kpca = KernelPCA(n_components=2, kernel='rbf', gamma=3).fit(embedding)
            embedding = kpca.transform(embedding)
            plt.figure()
            for j in range(n_clusters):
                idx = clustering.labels_ == j
                cur_x = embedding[idx]
                plt.scatter(cur_x[:, 0], cur_x[:, 1])
            plt.xticks(np.linspace(-0.6, 0.6, 5))
            plt.yticks(np.linspace(-0.6, 0.6, 5))
            plt.savefig('{0}_emb{1}.png'.format(self.dataset, i))
    
    


    def cluster(self, ind):
        ind = t.from_numpy(ind).cuda()
        x0 = self.construct_x(ind).detach().cpu().numpy()
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x0)
        # pca = PCA(n_components=2).fit(x)
        # svd = TruncatedSVD(n_components=2).fit(x0)
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=3).fit(x0)
        # x = pca.transform(x)
        # x = svd.transform(x0)
        x = kpca.transform(x0)
        for i in range(n_clusters):
            idx = kmeans.labels_ == i
            cur_x = x[idx]
            plt.scatter(cur_x[:, 0], cur_x[:, 1])
        num_ind = 20
        idx = np.random.choice(x.shape[0], num_ind)
        # idx2 = np.random.choice(x.shape[0], num_ind)
        g = self.effect(ind[idx])
        # g = g / np.max(np.abs(g))
        for i in range(num_ind):
            p1 = x[idx[i]]
            for j in range(num_ind):
                p2 = x[idx[j]]
                px = [p1[0], p2[0]]
                py = [p1[1], p2[1]]
                thick = 6
                if g[i, j] > 0:
                    color='r'
                else:
                    color='g'
                plt.plot(px, py, color=color, alpha=0.5)

        plt.savefig('{0}.png'.format(self.dataset))

    def inner_cluster(self, ind):
        ind = t.from_numpy(ind).cuda()
        x0 = self.construct_x(ind).detach().cpu().numpy()
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x0)
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=5).fit(x0)
        x = kpca.transform(x0)
        plt.figure()
        for i in range(n_clusters):
            cur_x = x[kmeans.labels_ == i]
            cur_ind = ind[kmeans.labels_ == i]
            plt.scatter(cur_x[:, 0], cur_x[:, 1])

            num_ind = 10

            idx = np.random.choice(cur_ind.shape[0], num_ind)
            g = self.effect(cur_ind[idx])
            for i in range(num_ind):
                p1 = cur_x[i]
                for j in range(num_ind):
                    p2 = cur_x[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)
        plt.xticks(np.linspace(-0.4, 0.4, 3))
        plt.yticks(np.linspace(-0.4, 0.4, 3))

        plt.savefig('{0}_inner.png'.format(self.dataset))

    def cross_cluster(self, ind):
            ind = t.from_numpy(ind).cuda()
            x0 = self.construct_x(ind).detach().cpu().numpy()
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x0)
            kpca = KernelPCA(n_components=2, kernel='rbf', gamma=5).fit(x0)
            x = kpca.transform(x0)

            ind_list = []
            x_list = []
            num_ind = 10
            plt.figure()
            for i in range(n_clusters):
                cur_x = x[kmeans.labels_ == i]
                cur_ind = ind[kmeans.labels_ == i]
                plt.scatter(cur_x[:, 0], cur_x[:, 1])

                

                idx = np.random.choice(cur_ind.shape[0], num_ind)
                ind_list.append(cur_ind[idx])
                x_list.append(cur_x)

            ind1 = ind_list[0]
            ind2 = ind_list[1]
            x1 = x_list[0]
            x2 = x_list[1]
            g = self.effect1(ind1, ind2)
            for i in range(num_ind):
                p1 = x1[i]
                for j in range(num_ind):
                    p2 = x2[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)
            
            g = self.effect1(ind2, ind1)
            for i in range(num_ind):
                p1 = x2[i]
                for j in range(num_ind):
                    p2 = x1[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)
            
            ind1 = ind_list[0]
            ind2 = ind_list[2]
            x1 = x_list[0]
            x2 = x_list[2]
            g = self.effect1(ind1, ind2)
            for i in range(num_ind):
                p1 = x1[i]
                for j in range(num_ind):
                    p2 = x2[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)
            
            g = self.effect1(ind2, ind1)
            for i in range(num_ind):
                p1 = x2[i]
                for j in range(num_ind):
                    p2 = x1[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)
            
            ind1 = ind_list[1]
            ind2 = ind_list[2]
            x1 = x_list[1]
            x2 = x_list[2]
            g = self.effect1(ind1, ind2)
            for i in range(num_ind):
                p1 = x1[i]
                for j in range(num_ind):
                    p2 = x2[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)
            
            g = self.effect1(ind2, ind1)
            for i in range(num_ind):
                p1 = x2[i]
                for j in range(num_ind):
                    p2 = x1[j]
                    px = [p1[0], p2[0]]
                    py = [p1[1], p2[1]]
                    if g[i, j] > 0:
                        color='r'
                    else:
                        color='y'
                    plt.plot(px, py, color=color, alpha=0.5)

            

            plt.xticks(np.linspace(-0.4, 0.4, 3))
            plt.yticks(np.linspace(-0.4, 0.4, 3))

            plt.savefig('{0}_cross.png'.format(self.dataset))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help='Dataset name: 911, article, ufo, slc', type=str, required=True)
    parser.add_argument('-r', '--rank', help='Embedding rank', type=int, required=True)
    parser.add_argument('-g', '--granularity', help='Time Granularity', type=int, default=0)
    parser.add_argument('-t', '--tauratio', help='Initial decay rate ratio', type=int, default=0)
    parser.add_argument('-s', '--spscale', help='Initial softplus scale', type=int, default=100)

    args = parser.parse_args()

    (ind, y),( train_ind, train_y), ( test_ind, test_y) = util.load_dataSet(args.dataset, './Data/')
    y_max = np.max(train_y)

    entity_dim = np.max(np.concatenate((train_ind, test_ind), axis=0), axis=0) + 1

    if args.granularity > 0:
        y_max = y_max / args.granularity
    else:
        y_max = 1
    
    if args.tauratio > 0:
        tau_ratio = args.tauratio
    else:
        tau_ratio = 1

    embedding_dim = args.rank

    log_file = 'SMIE_{0}_r{1}_g{2}_t{3}_s{4}.txt'.format(args.dataset, embedding_dim, args.granularity, tau_ratio, args.spscale)
    
    params = t.load(log_file + 'params')
    
    ind = np.unique(train_ind, axis=0)
    idx = np.random.choice(ind.shape[0], 1000, replace=False)
    # ind = ind[idx]

    effect = Effect(params, entity_dim, args.dataset)
    # effect.effect(ind)    
    effect.inner_cluster(ind)
    effect.cross_cluster(ind)
    effect.each_mode()
    
