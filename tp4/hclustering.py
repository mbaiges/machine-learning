import utils
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram

class HClustering:

    class Cluster:
        def __init__(self, idx:int, dist: float, elems: np.ndarray, contained: set, center: np.ndarray) -> None:
            self.idx = idx
            self.dist = dist
            self.elems = elems
            self.contained = contained
            self.center = center
            self.original_observations = self.elems.shape[0] 

        def __eq__(self, other: object) -> bool:
            return isinstance(other, HClustering.Cluster) and self.dist == other.dist and np.array_equal(self.elems, other.elems)

        def __hash__(self):
            return hash(self.dist, self.elems)

        def __str__(self) -> str:
            return self.__repr__()

        def __repr__(self) -> str:
            return f"Cluster: {{dist {self.dist} - elems: {self.elems}}}" 
        
    def __init__(self, criteria: str='max') -> None:
        def norm(t1,t2):
            return np.linalg.norm(np.array(t1) - np.array(t2))
        self.criteria = HClustering.linkage_criteria(criteria)
        self.distance_method = norm

    def _distance_matrix(self, x: np.ndarray, clusters: list) -> np.ndarray:
        total = len(clusters)
        distance_matrix = np.zeros((total, total))
        for i in range(total):
            for j in range(total):
                if i == j:
                    continue
                distance_matrix[i,j] = self.criteria(x, clusters[i], clusters[j], distance_method=self.distance_method)
        return distance_matrix
    
    def train(self, x, show_loading_bar: bool=True) -> None:
        start = time.time()
        def first_elem(e):
            return e[0]
        
        # x = np.array([[1,2,3],[4,5,6],[6,7,8]])
        self.scaler = StandardScaler()
        std_x       = self.scaler.fit_transform(X=x)
        total = std_x.shape[0]

        clusters    = [HClustering.Cluster(i, 0, np.array([std_x[i]]), set([i]), std_x[i]) for i in range(total)]
        distances   = []
        clusters_idx = total

        ## Show distance matrix
        distance_matrix = self._distance_matrix(std_x, clusters)
        # print(distance_matrix)
        
        ## Initially fill distances 
        for i in range(len(clusters)):
                c1 = clusters[i]
                for j in range(i+1, len(clusters)):
                    c2 = clusters[j]
                    d = self.criteria(std_x, c1, c2, distance_method=self.distance_method)
                    distances.append((d, (c1,c2)))
        distances = sorted(distances, key=first_elem)
        linkage_resp = []
        
        if show_loading_bar:
            loading_bar = utils.LoadingBar()
            loading_bar.init()
        it = 0
        while not HClustering.end(clusters):
            if show_loading_bar:
                loading_bar = loading_bar.update(1.0*it/total)
            
            # Get closest clusters
            closest = distances.pop(0)
            dist, c1, c2 = closest[0], closest[1][0], closest[1][1]
            clusters.remove(c1)
            clusters.remove(c2)
            
            # Add linkage to response
            linkage_resp.append([c1.idx, c2.idx, dist, c1.original_observations + c2.original_observations])

            # Create new cluster
            # print("--------------------")
            # print(f"c1: {c1.elems}")
            # print(f"c1.center: {c1.center}")
            # print(f"c2: {c2.elems}")
            # print(f"c2.center: {c2.center}")
            # print(f"new_center: {(1.0*c1.center*len(c1.contained)+c2.center*len(c2.contained))/(len(c1.contained)+len(c2.contained))}")
            # print("--------------------")

            new_cluster     = HClustering.Cluster(clusters_idx, dist, np.concatenate((c1.elems, c2.elems), axis=0), set.union(c1.contained, c2.contained), (1.0*c1.center*len(c1.contained)+c2.center*len(c2.contained))/(len(c1.contained)+len(c2.contained)))
            clusters_idx += 1 
            new_distances   = [(self.criteria(std_x, c, new_cluster, distance_method=self.distance_method, cache=distance_matrix), (c, new_cluster)) for c in clusters]
            clusters.append(new_cluster)

            # Add new distances
            distances = [d for d in distances if d[1][0] != c1 and d[1][0] != c2 and d[1][1] != c1 and d[1][1] != c2]
            distances.extend(new_distances)
            distances = sorted(distances, key=first_elem)
            it += 1
        
        if show_loading_bar:
            loading_bar.end()
            
        self.linkage_resp = np.array(linkage_resp)
        end = time.time()
        print(f"HClustering train duration: {start - end} s")

    def dendrogram(self) -> None:
        dendrogram(self.linkage_resp)

    @staticmethod
    def end(clusters: list) -> bool:
        return len(clusters) <= 1

    @staticmethod
    def linkage_criteria(criteria: str):
        if criteria == "max":
            return HClustering._max
        elif criteria == "min":
            return HClustering._min
        elif criteria == "mean":
            return HClustering._mean
        elif criteria == "center":
            return HClustering._center
        
    @staticmethod
    def _max(dataset, c1, c2, distance_method, cache=None):
        return max([distance_method(dataset[i1], dataset[i2]) if cache is None else cache[i1][i2] for i1 in c1.contained for i2 in c2.contained])

    @staticmethod
    def _min(dataset, c1, c2, distance_method, cache=None):
        return min([distance_method(dataset[i1], dataset[i2]) if cache is None else cache[i1][i2] for i1 in c1.contained for i2 in c2.contained])

    @staticmethod
    def _mean(dataset, c1, c2, distance_method, cache=None):
        dists = [distance_method(dataset[i1], dataset[i2]) if cache is None else cache[i1][i2] for i1 in c1.contained for i2 in c2.contained]
        return sum(dists) / len(dists)

    @staticmethod
    def _center(dataset, c1, c2, distance_method, cache=None):
        return distance_method(c1.center, c2.center)