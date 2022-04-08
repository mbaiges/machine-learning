import copy
import numpy as np

from utils import normalize
from distances import man, euc

class KNN:

    def __init__(self, k=3, d='euc', weighted=False):
        self.k = k
        self.d = self._choose_d(d)
        self.weighted = weighted
   
    def _choose_d(self, d_n):
        if d_n == 'man':
            return man
        else: # default is euc
            return euc

    def load(self, x, t):
        self.x, (self.nx, self.dx) = normalize(x)
        self.t = t
        self.w = []
        for i in range(0, self.x.shape[0]):
            wr = {
                'x': self.x[i],
                't': self.t[i]
            }
            self.w.append(wr)

    def _closest(self, e):
        w = sorted(self.w, key=lambda wr: self.d(e, wr['x']))
        return list(map(lambda wr: wr, w))

    # denormaliza w y lo deja en el mismo formato
    def _denorm_w(self, w):
        return list(map(lambda e: {
            'x': self.dx(e['x']),
            't': e['t']
        }, w))

    def find(self, x):
        res = []
        for e in x:
            closest = self._closest(self.nx(np.array(e)))
            # print(self._denorm_w(closest))
            res.append(self._denorm_w(closest[:self.k]))
        return res