from matplotlib.pyplot import plot
import pandas as pd
import copy
import math
from typing import Iterable, Union

from utils import mode, replace_at

def shannon_entropy(df: pd.DataFrame, atr: str) -> float:
    atr_values = sorted(df[atr].unique())
    total = df.shape[0]
    entropy = 0
    for v in atr_values:
        rel_freq = df.loc[df[atr] == v].shape[0] / total
        entropy -=  rel_freq * math.log(rel_freq, 2)
    return entropy

class Stack:
    class ValuedAttribute:
        def __init__(self, name: str, value: int) -> None:
            self.name = name
            self.value = value

        def __repr__(self):
            return "(" + f"{self.name.upper()}={self.value}" + ")"
    
        def __str__(self):
            return self.__repr__()
    
    def __init__(self):
        self.l = []

    def push(self, e: ValuedAttribute) -> None:
        self.l.append(e)
    
    def pop(self) -> ValuedAttribute:
        return self.l.pop()

    def path(self) -> list:
        return copy.deepcopy(self.l)

    def __iter__(self):
        return self.path()

    def __repr__(self):
        return self.l.__repr__()
    
    def __str__(self):
        return self.l.__str__()

class Node:

    def __init__(self, name: str, childs: dict={}, value: int=None, depth: int=None) -> None:
        self.name = name
        self.childs = childs
        self.value = value
        self.depth = depth

    # TODO(matías): dont know the type of e (pandas row)
    def traverse(self, e, s: Stack=Stack()):
        if self.is_leaf():
            return s, self.value
        else:
            att_val = e[self.name]
            if att_val in self.childs:
                va = Stack.ValuedAttribute(name=self.name, value=att_val)
                s = copy.deepcopy(s)
                s.push(va)
                return self.childs[att_val].traverse(e, s)
            else:
                return s, None

    def is_leaf(self) -> bool:
        return len(self.childs.keys()) == 0 and self.value is not None

    def __repr__(self):
        s = ""
        if self.is_leaf():
            s = "<" + f"{self.value}" + ">"
        else:
            s = ":" + f"{self.name.upper()}" + ":"
        return s

    def __str__(self):
        return self.__repr__()

    def text_repr(self, att_value: str, prefix: str, remove_at_pos: int=None):
        b = prefix + f"({att_value if att_value is not None else ''})" + self.__str__() + "\n"
        if remove_at_pos is not None and remove_at_pos < len(prefix)-1:
            prefix = replace_at(prefix, remove_at_pos, ' ')
            remove_at_pos = None
        # print(f"Visiting node: {self}")
        if self.is_leaf():
            # print(f"{self} is leaf!")
            return b
        for idx, (att_value, child_node) in enumerate(self.childs.items()):
            pref = prefix
            rep_pos = None
            if len(pref) > 0:
                pref = pref.replace("-", " ").replace(">", " ")
                pref += " "
            if idx == (len(self.childs.keys())-1):
                rep_pos = self.depth*5          
            pref += "|-->"
            b += child_node.text_repr(att_value, pref, remove_at_pos=rep_pos)
        return b

    def count_childs(self) -> int:
        return 0 if self.is_leaf() else len(self.childs)*2 + sum([child.count_childs() for child in self.childs.values()])
class ID3:
            
    def __init__(self, gain_f: str='shannon', max_depth: int=None) -> None:
        self._choose_gain(gain_f)
        self.max_depth = max_depth

    def _choose_gain(self, gain_f: str='shannon') -> None:
        # if gain_f == 'name_a_gain_function':
        #     self.gain = function
        # else: # if gain_f == 'shannon'
        #     self.gain = shannon_entropy
        self.gain_method = shannon_entropy

    def gain(self, df: pd.DataFrame, atr: str) -> float:
        gain = self.gain_method(df, self.target_atr)
        total = df.shape[0]
        atr_values = sorted(df[atr].unique())

        for value in atr_values:
            gain -= (df.loc[df[atr] == value].shape[0] / total) * self.gain_method(df.loc[df[atr] == value], self.target_atr)
        return gain
        
    def _get_max_gain_att(self, df: pd.DataFrame, attrs: set):
        max_gain = None
        for atr in attrs:
            gain = self.gain(df, atr)
            if max_gain is None or (gain > max_gain[1] or (gain == max_gain[1] and atr < max_gain[0])):
                max_gain = (atr, gain)
        return max_gain

    def _generate_node(self, stack: Stack, pending: set, depth: int) -> tuple:
        if pending and (self.max_depth is None or depth < self.max_depth):
            df = self._get_filtered_dataframe(stack) 
            obj_values = sorted(df[self.target_atr].unique())
            # print(obj_values)
            if len(obj_values) == 1: # TODO: Esto no hace que si después te paso de testing un conjunto que tiene otra clase, crashee porque no sepa que decir?
                # print("Found 1 D:")
                return depth, Node(None, value=obj_values[0], depth=depth)
            max_gain = self._get_max_gain_att(df, pending)
            max_gain_attr = max_gain[0]
            atr_values = sorted(df[max_gain_attr].unique())
            childs = {}
            new_pending = copy.deepcopy(pending)
            # new_pending.discard(max_gain_attr)
            # print(new_pending)
            # print(f"discarding attribute {max_gain[0]}:")
            new_pending.discard(max_gain_attr)
            # print(new_pending)
            # print(len(new_pending))
            # print("--------------------------------")
            max_child_depth = 0
            for value in atr_values:
                stack.push(Stack.ValuedAttribute(max_gain_attr, value))
                child_depth, childs[value] = self._generate_node(stack, new_pending, depth=depth+1)
                stack.pop()
                if child_depth > max_child_depth:
                    max_child_depth = child_depth
            return max_child_depth, Node(max_gain_attr, childs=childs, depth=depth)
        else:
            return depth, Node(None, value=self.s_mode(stack), depth=depth)

    def s_mode(self, stack: Stack) -> float:
        df = self._get_filtered_dataframe(stack)
        return mode(df[self.target_atr].to_numpy())
    
    def _get_filtered_dataframe(self, stack: Stack) -> pd.DataFrame:
        p = stack.path()
        recorte = self.examples
        # print("------------------------------")
        # print(p)
        # print(recorte)
        for v_atr in p:
            recorte = recorte.loc[recorte[v_atr.name] == v_atr.value]
        #     print(recorte)
        # print(recorte)
        # print("------------------------------")
        return recorte
    
    def _generate_tree(self) -> None:
        # print("Generating tree")
        self.depth, self.tree = self._generate_node(Stack(), set(self.x.columns), depth=0)
        # print(f"max depth: {self.depth}")

    def load(self, x: pd.DataFrame, t: pd.DataFrame) -> None:
        self.examples = pd.concat([x, t], axis=1)
        self.x = x
        self.t = t
        self.target_atr = self.t.columns[0]
        self._generate_tree()

    def predict(self, x: pd.DataFrame) -> Iterable[Union[int, float]]:
        ret = []
        for idx, e in x.iterrows():
            s, value = self.tree.traverse(e)
            if value is None:
                value = mode(self._get_filtered_dataframe(s)[self.target_atr])
            ret.append(value)
        return ret

    def eval(self, x: pd.DataFrame, t: pd.DataFrame) -> float:
        results = self.predict(x)
        err = 0
        for idx, pred_t in t.iterrows():
            if pred_t[self.target_atr] != results[idx]:
                err += 1
        err /= x.shape[0]
        return results, err

    def repr_tree(self):
        return self.tree.text_repr(None, "", remove_at_pos=None)
    
    def print_tree(self):
        print(f"Tree Representation n={self.count_nodes()}")
        print(self.repr_tree())
        print(f"Final Depth {self.count_real_depth()}")

    def count_nodes(self):
        return 1 + self.tree.count_childs()

    def count_real_depth(self):
        return self.depth*2 + 1