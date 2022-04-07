import copy

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

    def _closest(self, e):
        w = []
        for i in range(0, self.x.shape[0]):
            wr = {
                'x': self.x[i],
                't': self.t[i]
            }
            w.append(wr)
        
        w = sorted(w, key=lambda wr: self.d(e, wr['x']))
        return list(map(lambda wr: wr, w))

    def find(self, x):
        res = []
        
        for e in x.shape[0]:
            closest = self._closest(self.nx(e))
            print(closest[0:self.k])