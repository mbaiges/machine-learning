import numpy as np
import random

import utils

# Cluster initialization methods

def rand(x: np.array, k: int, seed: int) -> np.array:
    r = random.Random(seed)
    min = x.min(axis=0)
    max = x.max(axis=0)
    return [np.array([(random.random()*(max[j] - min[j]) + min[j]) for j in range(x.shape[1])]) for i in range(k)]

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

    # Stop condition
    def _stop(self) -> bool:
        return self._last_clusters is not None and set(self.clusters) == set(self._last_clusters)

    def train(self, x: np.array, iterations: int, show_loading_bar: bool=False) -> None:
        self._initialize_clusters(x)
        print(f"Initial Clusters: {self.clusters}")
        
        if show_loading_bar:
            loading_bar = utils.LoadingBar()
            loading_bar = loading_bar.init()

        it = 0
        # while not self._stop() and it < iterations:
        #     if show_loading_bar:
        #         loading_bar = loading_bar.update(1.0*it/iterations)

        #     self._last_clusters = self.clusters

        #     self.clusters = self.clusters

        #     it += 1

        if show_loading_bar:
            loading_bar = loading_bar.end()
        print(f"KMeans iterations: {it}")