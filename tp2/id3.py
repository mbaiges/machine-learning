import numpy as np

def shannon_entropy(x) -> float:
    pass

class Node:

    def __init__(self, name: str, childs: dict={}, value: int=None) -> None:
        self.name = name
        self.value = None
        self.childs = childs

    def is_leaf(self) -> bool:
        return len(self.childs.keys()) and self.value is None

class ID3:
            
    def __init__(self, gain_f: str='shannon') -> None:
        self._choose_gain(gain_f)

    def _choose_gain(self, gain_f: str='shannon'):
        # if gain_f == 'name_a_gain_function':
        #     self.gain = function
        # else: # if gain_f == 'shannon'
        #     self.gain = shannon_entropy
        self.gain = shannon_entropy

    def _generate_tree(self) -> None:
        print("Generating tree")
        pass

    def load(self, x: np.ndarray, t: np.ndarray) -> None:
        self.x = x
        self.t = t
        self._generate_tree()