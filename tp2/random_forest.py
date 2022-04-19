import pandas as pd
import numpy as np
from typing import Iterable, Union

from id3 import ID3
from utils import mode, bootstrap_df_build_sample

class RandomForest:

    def __init__(self, gain_f: str='shannon', max_depth: int=None, sample_size: int=500, n: int=3) -> None:
        self.models = [ID3(gain_f=gain_f, max_depth=max_depth) for i in range(0, n)]
        self.sample_size = sample_size

    def load(self, x: pd.DataFrame, t: pd.DataFrame) -> None:
        for m in self.models:
            x_train, t_train = bootstrap_df_build_sample(x, t, k=self.sample_size)
            m.load(x_train, t_train)
        self.target_atr = t.columns[0]

    def predict(self, x: pd.DataFrame) -> Iterable[Union[int, float]]:
        results = []
        for m in self.models:
            results.append(m.predict(x))
        results = np.array(results)
        ret = []
        for i in range(0, results.shape[1]):
            ret.append(mode(results[:,i]))
        return ret

    def eval(self, x: pd.DataFrame, t: pd.DataFrame) -> float:
        results = self.predict(x)
        err = 0
        for idx, pred_t in t.iterrows():
            if pred_t[self.target_atr] != results[idx]:
                err += 1
        err /= x.shape[0]
        return results, err

    def count_nodes(self):
        return sum([m.count_nodes() for m in self.models]) / len(self.models)