import numpy as np
import random
import math
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcl
import matplotlib.pyplot as plt

import utils

# Cluster initialization methods

def rand(x: np.array, k: int, seed: int) -> np.array:
    r = random.Random(seed)
    min = x.min(axis=0)
    max = x.max(axis=0)
    return [np.array([(r.random()*(max[j] - min[j]) + min[j]) for j in range(x.shape[1])]) for i in range(k)]

def samples(x: np.array, k: int, seed: int) -> np.array:
    r = random.Random(seed)
    shuffled_idxs = [i for i in range(x.shape[0])]
    r.shuffle(shuffled_idxs)
    return np.array([x[i] for i in shuffled_idxs])[:k]

def _dist(x1: np.array, x2: np.array) -> float:
    return np.linalg.norm(x1-x2) # norm2 default

def _most_distant(x: np.array, c: list):
    most_distant = (c[0], 0)
    for e in c:
        d = 0
        for i in range(x.shape[0]):
            xi = x[i]
            d += _dist(e, xi)
        d /= x.shape[0]
        if d > most_distant[1]:
            most_distant = (e, d)
    return most_distant[0]

def _most_distant_pair(x: np.array):
    most_distant_pair = ((x[0], x[0]), 0)
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
            d = _dist(x[i], x[j])
            if d > most_distant_pair[1]:
                most_distant_pair = ((x[i], x[j]), d)
    return list(most_distant_pair[0])

def distant(x: np.array, k: int, seed: int):
    c = _most_distant_pair(x)
    for i in range(0, k-2):
        e = _most_distant(x, c)
        c.append(e)
    return c[:k]

cluster_init_algs = {
    "random":  rand,
    "samples": samples,
    "distant": distant
}

class KMeans:
    def __init__(self, k: int, init_alg: str='random', seed: int=0) -> None:
        self.k         = k
        self._init_alg = self._get_init_alg(init_alg)
        self.seed      = seed

    # Cluster initialization
    def _get_init_alg(self, init_alg: str):
        l_init_alg = init_alg.lower()
        return cluster_init_algs.get(l_init_alg, cluster_init_algs['random']) # default is rand

    def _initialize_clusters(self, x: np.array) -> None:
        self.clusters       = self._init_alg(x, self.k, self.seed)
        self._last_clusters = None

    def _nearest_cluster_idx(self, x: np.array):
        nearest = (0, math.inf)
        for i, c in enumerate(self.clusters):
            d = _dist(x, c)
            if d < nearest[1]:
                nearest = (i, d)
        return nearest[0]

    def _clusterize(self, x: np.array) -> list:
        clustered = [[] for i in range(len(self.clusters))]
        for i in range(x.shape[0]):
            c_idx = self._nearest_cluster_idx(x[i])
            clustered[c_idx].append(i)
        return clustered

    def clusterize(self, x: np.array) -> list:
        std_x = self.std_scaler.transform(x)
        clustered_idxs = self._clusterize(std_x)
        return [x[idxs] for idxs in clustered_idxs]

    def _build_new_clusters(self, x: np.array, clustered: list) -> list:
        new_clusters = []
        for i, cl in enumerate(self.clusters):
            if len(clustered[i]) == 0:
                new_clusters.append(cl)
            else:
                xs = np.array([x[idx] for idx in clustered[i]])
                new_clusters.append(np.mean(xs, axis=0))
        return new_clusters

    # Stop condition
    def _stop(self) -> bool:
        last = set(list(map(lambda e: tuple(e), self._last_clusters))) if self._last_clusters is not None else set()
        curr = set(list(map(lambda e: tuple(e), self.clusters)))
        return curr == last

    def train(self, x: np.array, iterations: int, show_loading_bar: bool=False) -> None:
        self.std_scaler  = StandardScaler()
        std_x            = self.std_scaler.fit_transform(X=x)

        self._initialize_clusters(std_x)
        # print(f"Initial Clusters: {self.clusters}")
        
        if show_loading_bar:
            loading_bar = utils.LoadingBar()
            loading_bar.init()

        it = 0
        while not self._stop() and it < iterations:
            if show_loading_bar:
                loading_bar = loading_bar.update(1.0*it/iterations)

            self._last_clusters = self.clusters

            clustered     = self._clusterize(std_x)
            self.clusters = self._build_new_clusters(std_x, clustered)

            it += 1

        if show_loading_bar:
            loading_bar = loading_bar.end()

        return it

    # Plotting

    def plot2d(self, x: np.array, labels: list, y: np.array=None):    
        denormalized_cluster_centers = self.std_scaler.inverse_transform(np.array(self.clusters))

        colors = list(mcl.TABLEAU_COLORS.values())
        print(colors)

        # Clustered points
        clustered = self.clusterize(x)
        for idx, c in enumerate(clustered):
            ci = colors[idx % len(colors)]
            plt.scatter(c[:,0], c[:,1], c=[ci for i in range(len(c))], alpha=0.4)

        # Clusters centers
        plt.scatter(denormalized_cluster_centers[:,0], denormalized_cluster_centers[:,1], c=['b' for i in range(denormalized_cluster_centers.shape[0])])

        if y is not None:
            std_x = self.std_scaler.transform(x)
            idxs = self._clusterize(std_x)
            
            for idx, c in enumerate(denormalized_cluster_centers):
                count = sum([1 if (y[clustered_idx] == 1) else 0 for clustered_idx in idxs[idx]])
                pctg = count / len(idxs[idx]) if len(idxs[idx]) > 0 else 0

                plt.text(x=c[0]+0.5, y=c[1]+0.5, s=f'{pctg*100:.0f}%', 
                    fontdict=dict(color='red',size=10),
                    bbox=dict(facecolor='yellow',alpha=0.5)
                )


        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        pass

    def plot3d(self, x: np.array, labels: list):
        denormalized_cluster_centers = self.std_scaler.inverse_transform(np.array(self.clusters))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        colors = ['r', 'g', 'b']

        plt.scatter(denormalized_cluster_centers[:,0], denormalized_cluster_centers[:,1], denormalized_cluster_centers[:,2], c = [colors[i % len(colors)] for i in range(denormalized_cluster_centers.shape[0])])

        clustered = self.clusterize(x)

        for idx, c in enumerate(clustered):
            ci = colors[idx % len(colors)]
            plt.scatter(c[:,0], c[:,1], c[:,2], c = [ci for i in range(len(c))])

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        pass