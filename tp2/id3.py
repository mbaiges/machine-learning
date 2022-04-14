import numpy as np

def shannon_entropy(x) -> float:
    pass

class Node:

    def __init__(self, name: str, childs: dict={}, value: int=None):
        self.name = name
        self.value = None
        self.childs = childs

    def is_leaf(self) -> bool:
        return len(self.childs.keys()) and self.value is None

class ID3:
            
    def __init__(self) -> None:
        self.gain = shannon_entropy

    def _generate_tree(self) -> None:
        pass

    def load(self, x: np.ndarray, t: np.ndarray) -> None:
        self.x = x
        self.t = t
        self._generate_tree()