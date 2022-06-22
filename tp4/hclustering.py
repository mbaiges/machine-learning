import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

class HClustering:
        
    def __init__(self, criteria: str='max') -> None:
        self.criteria = HClustering._criteria(criteria)
        pass

    @staticmethod
    def _criteria(s: str):
        if s == 'max':
            return 'complete'
        elif s == 'min':
            return 'single'
        elif s == 'mean':
            return 'average'
        else: # s == 'centroid' # default
            return 'centroid'

    def train(self, x, show_loading_bar: bool=True) -> None:
        self.z = linkage(x, self.criteria)
        pass

    def dendrogram(self, p: int=None, truncate_mode: str=None) -> None:
        dn = dendrogram(self.z, p=p, truncate_mode=truncate_mode)