import pandas as pd
import copy
import math

from utils import mode

def shannon_entropy(df: pd.DataFrame, atr: str) -> float:
    atr_values = df[atr].unique()
    total = df.size
    entropy = 0
    for v in atr_values:
        rel_freq = df[df[atr] == v].size / total
        entropy -=  rel_freq * math.log(rel_freq, 2)
    return entropy

class Stack:
    class ValuedAttribute:
        def __init__(self, name: str, value: int) -> None:
            self.name = name
            self.value = value
    
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

class Node:

    def __init__(self, name: str, childs: dict={}, value: int=None, depth: int=None) -> None:
        self.name = name
        self.childs = childs
        self.value = value
        self.depth = depth

    def is_leaf(self) -> bool:
        return len(self.childs.keys()) and self.value is None

class ID3:
            
    def __init__(self, gain_f: str='shannon') -> None:
        self._choose_gain(gain_f)

    def _choose_gain(self, gain_f: str='shannon') -> None:
        # if gain_f == 'name_a_gain_function':
        #     self.gain = function
        # else: # if gain_f == 'shannon'
        #     self.gain = shannon_entropy
        self.gain_method = shannon_entropy

    def gain(self, df: pd.DataFrame, atr: str) -> float:
        gain = self.gain_method(df, self.target_atr)
        total = df.size
        atr_values = df[atr].unique()

        for value in atr_values:
            gain -= (df[df[atr] == value].size / total) * self.gain_method(df[df[atr] == value], self.target_atr)
        return gain
        
    def _generate_node(self, atr: str, stack: Stack, pending: set, depth: int) -> None:
        if pending:
            df = self._get_filtered_dataframe(stack) 
            obj_values = df[self.target_atr].unique()
            if len(obj_values) == 1:
                return Node(atr, value=obj_values[0], depth=depth)
            max_gain = None
            for atr in pending:
                gain = self.gain(df, atr)
                if max_gain is None or gain > max_gain[1]:
                    max_gain = (atr, gain)
            atr_values = df[atr].unique()
            childs = {}
            new_pending = copy.deepcopy(pending)
            # new_pending.discard(max_gain[0])
            print(f"discarding attribute {max_gain[0]}: {new_pending.discard(max_gain[0])}")
            print(new_pending)
            for value in atr_values:
                stack.push(Stack.ValuedAttribute(max_gain[0], value))
                print("Hola")
                childs[value] = self._generate_node(max_gain[0], stack, new_pending, depth+1)
                stack.pop()
            return Node(max_gain[0], childs=childs, depth=depth)
        else:
            return Node(atr, value=self.s_mode(stack), depth=depth)

    def s_mode(self, stack: Stack) -> float:
        df = self._get_filtered_dataframe(stack)
        return mode(df[self.target_atr])
    
    def _get_filtered_dataframe(self, stack: Stack) -> pd.DataFrame:
        p = stack.path()
        recorte = self.examples
        for v_atr in p:
            print(f"{v_atr.name} - {v_atr.value}")
            recorte = recorte[recorte[v_atr.name] == v_atr.value]
        return recorte
    
    def _generate_tree(self) -> None:
        print("Generating tree")
        self.tree = self._generate_node(self.target_atr, Stack(), set(self.x.columns), depth=0)

    def _find_most_gain_att(self, x: pd.DataFrame, t: pd.DataFrame) -> str:
        pass

    def load(self, x: pd.DataFrame, t: pd.DataFrame) -> None:
        self.examples = pd.concat([x, t], axis=1)
        self.x = x
        self.t = t
        self.target_atr = self.t.columns[0]
        self._generate_tree()