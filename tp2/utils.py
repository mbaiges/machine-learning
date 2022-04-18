import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Union

def df_to_np(df: pd.DataFrame, x_columns: str, t_column: str) -> tuple:
    x = []
    t = []
    for idx, row in df.iterrows():
        xi = []
        for col_name in x_columns:
            xi.append(row[col_name])
        x.append(xi)
        t.append(row[t_column])
    return np.array(x), np.array(t)

def normalize(x: np.array) -> tuple:
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    
    def nx(xi):
        return (xi - x_mean)/x_std
    def dx(norm_x):
        return norm_x * x_std + x_mean

    return nx(x), (nx, dx)

def mode(x: np.array) -> np.array:
    if x.shape[0] == 0:
        return None
    counts = {}
    for e in x:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)

# def mode(x: np.array, axis: int=None) -> np.array:
#     print(x)
#     return stats.mode(x, axis=axis).mode

def bootstrap_df_build_sample(x: pd.DataFrame, t: pd.DataFrame, k: int=None) -> tuple:# Como se define que retorna tipo (pd.Dataframe, pd.Dataframe)
    k = x.shape[0] if k is None else k
    indexes = np.random.randint(0, x.shape[0], size=k)
    x_columns = x.columns
    t_columns = t.columns
    x_arr, t_arr = x.to_numpy(), t.to_numpy()
    x_ret, t_ret = np.array([x_arr[i] for i in indexes]), np.array([t_arr[i] for i in indexes])
    return pd.DataFrame(x_ret, columns=x_columns), pd.DataFrame(t_ret, columns=t_columns)

def bootstrap_df(x: pd.DataFrame, t: pd.DataFrame, train_size: int=None, test_size: int=None) -> tuple:
    train_size = x.shape[0] if train_size is None else train_size
    test_size  = x.shape[0] if test_size is None else test_size
    return bootstrap_df_build_sample(x, t, train_size), bootstrap_df_build_sample(x, t, test_size)

def iqr(x: np.array) -> Union[int, float]:
    q75, q25 = np.percentile(x, [75 ,25])
    return q75 - q25

def hist(x: np.array, title: str) -> None:
    plt.rcParams.update({'font.size': 22})
    plt.title(label=title)
    plt.hist(x, bins=bins(x, alg='diac'))
    plt.show()

# class BinningRule:
#     class BinningRuleType(Enum):
#         WIDTH  = "width"
#         RANGES = "ranges"

#     def __init__(self, type: BinningRuleType, data: object=None) -> None:
#         self.type = type
#         self.data = data

def _bins_dist(x: np.array, options={}) -> Iterable[Union[int, float]]:
    eps = 1e-4
    step = options.get('dist', 1)
    min = options.get('min', np.min(x))
    max = options.get('max', np.max(x))
    return np.arange(min, max + eps, step=step)

def _bins_perc(x: np.array, options={}) -> Iterable[Union[int, float]]:
    x = np.sort(x)
    n = options.get('n', 1)
    size = int(x.shape[0]/n)
    bins = set()
    for idx, e in enumerate(x):
        if idx % size == 0:
            bins.add(e)
    if idx % size != 0:
        bins.add(x[-1])
    return sorted(bins)

## https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
def _bins_diac(x: np.array, options={}) -> Iterable[Union[int, float]]:
    bins_width = int(2 * iqr(x) / x.shape[0]**(1/3))
    return range(np.min(x), np.max(x) + bins_width, bins_width)

def bins(x: np.array, alg: str='diac', options: object={}) -> Iterable[Union[int, float]]:
    ret = None
    if alg == 'perc':
        ret = _bins_perc(x, options)
    elif alg == 'dist':
        ret = _bins_dist(x, options)
    else: # alg == 'diac'
        ret = _bins_diac(x, options)
    return ret

# Strings

def lreplace(s, old, new, occurrence):
    li = s.lsplit(old, occurrence)
    return new.join(li)

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def replace_at(s, pos, new_char):
    temp = list(s)
    temp[pos] = new_char
    return "".join(temp)

# Loading bar

class LoadingBar:

    def __init__(self, width: int=20):
        self.width = 20

    def init(self):
        print("")
        self.update(0.0)

    def end(self):
        self.update(1.0)
        print("")
    
    def update(self, percentage: float):
        bar = "["
        for b in range(0, self.width):
            p = b/self.width
            p_next = (b+1)/self.width
            p_mid = (p+p_next)/2
            char = ' '
            if percentage > p:
                if percentage < p_mid:
                    char = '▄'
                else:
                    char = '█'
                    
            bar += char
        bar += f"] ({percentage*100:05.2f}%)"
        print(f"\r{bar}", end = '')