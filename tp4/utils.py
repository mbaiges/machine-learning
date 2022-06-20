import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Union

#### Logging ####

def log(s: str, sep: str=None):
    if sep is not None:
        print(sep)
    print(s)
    if sep is not None:
        print(sep)

#### Statistics ####

def iqr(x: np.array) -> Union[int, float]:
    q75, q25 = np.percentile(x, [75 ,25])
    return q75 - q25

#### Binning ####

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
    bins_width = 2 * iqr(x) / x.shape[0]**(1/3)
    return np.arange(np.min(x), np.max(x) + bins_width, bins_width)

def bins(x: np.array, alg: str='diac', options: object={}) -> Iterable[Union[int, float]]:
    ret = None
    if alg == 'perc':
        ret = _bins_perc(x, options)
    elif alg == 'dist':
        ret = _bins_dist(x, options)
    else: # alg == 'diac'
        ret = _bins_diac(x, options)
    return ret

#### Plotting ####

def hist(x: np.array, title: str, bins_alg: str='diac', bins_options: object={}) -> None:
    weights = np.ones_like(x) / x.shape[0]
    plt.rcParams.update({'font.size': 22})
    plt.title(label=title)
    plt.hist(x, bins=bins(x, alg=bins_alg, options=bins_options), weights=weights)
    plt.show()

def bars(x: np.array, title: str) -> None:
    labels = sorted(set(x.tolist()))
    c = {}
    for e in x:
        c[e] = c.get(e, 0) + 1
    print(c)
    x, y = [], []
    for key, value in sorted(c.items()):
        x.append(key)
        y.append(value)

    plt.rcParams.update({'font.size': 22})
    plt.title(label=title)
    plt.bar(x, y)
    plt.show()

#### Loading Bar ####

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