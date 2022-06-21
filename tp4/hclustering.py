import utils
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram

class HClustering:

    class Cluster:
        def __init__(self, idx:int, dist: float, elems: np.ndarray) -> None:
            self.idx = idx
            self.dist = dist
            self.elems = elems
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

    def _distance_matrix(self, x: np.ndarray) -> np.ndarray:
        total = x.shape[0]
        distance_matrix = np.zeros((x.shape[0], x.shape[0]))
        for i in range(total):
            for j in range(total):
                if i == j:
                    continue
                distance_matrix[i,j] = self.criteria(x[i], x[j], distance_method=self.distance_method)
        return distance_matrix
    
    def train(self, x, show_loading_bar: bool=True) -> None:
        start = time.time()
        def first_elem(e):
            return e[0]
        
        # x = np.array([[1,2,3],[4,5,6],[6,7,8]])
        self.scaler = StandardScaler()
        std_x       = self.scaler.fit_transform(X=x)
        total = std_x.shape[0]

        ## Show distance matrix
        # distance_matrix = self._distance_matrix(std_x)
        # print(distance_matrix)

        clusters    = [HClustering.Cluster(i, 0, np.array([std_x[i]])) for i in range(total)]
        distances   = []
        clusters_idx = total
        
        ## Initially fill distances 
        for i in range(len(clusters)):
                c1 = clusters[i]
                for j in range(i+1, len(clusters)):
                    c2 = clusters[j]
                    d = self.criteria(c1.elems, c2.elems, distance_method=self.distance_method)
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
            new_cluster     = HClustering.Cluster(clusters_idx, dist, np.concatenate((c1.elems, c2.elems), axis=0))
            clusters_idx += 1 
            new_distances   = [(self.criteria(c.elems, new_cluster.elems, distance_method=self.distance_method), (c, new_cluster)) for c in clusters]
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
    def _max(c1, c2, distance_method):
        return max([distance_method(t1, t2) for t1 in c1 for t2 in c2])

    @staticmethod
    def _min(c1, c2, distance_method):
        return min([distance_method(t1, t2) for t1 in c1 for t2 in c2])

    @staticmethod
    def _mean(c1, c2, distance_method):
        dists = [distance_method(e1,e2) for e1 in c1 for e2 in c2]
        return sum(dists) / len(dists)

    @staticmethod
    def _center(c1, c2, distance_method):
        return distance_method(np.mean(c1, axis=0), np.mean(c2, axis=0))