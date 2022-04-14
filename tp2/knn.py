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

    def _most_frequents_for_k(self, target, w):
        def update_method(curr, dist, only_zeros=False):
            if only_zeros and dist == 0:
                return curr + 1
            elif only_zeros and dist != 0:
                return curr
            return curr + dist
        most_frequent_found = False
        k = self.k
        only_zeros = False
        freq = {}
        max = None
        max_n = 0
        while not most_frequent_found and k <= len(w):
            # Found most freq
            freq.clear()
            max = None
            max_n = 0
            for e in w[:k]:
                # if e['t'] not in freq:
                #     freq[e['t']] = 0
                dist = 1
                if self.weighted:
                    dist = euc(target, e['x'])**2
                    if dist == 0 and not only_zeros:
                        freq.clear()
                        max = None
                        max_n = 0
                        only_zeros = True
                f = update_method(freq.get(e['t'], 0), dist, only_zeros=only_zeros)
                if f > 0 or not only_zeros:
                    freq[e['t']] = f
                if max == None or (max != e['t'] and freq[e['t']] >= freq[max]):
                    if max != None and freq[e['t']] == freq[max]:
                        max_n += 1
                    else:
                        max_n = 1
                        max = e['t']

            # Check if there is a most_freq
            if max_n == 1:
                most_frequent_found = True
        
            k += 1
        k -= 1

        probs = {}
        total = sum(list(freq.values()))
        for cat, freq in freq.items():
            probs[cat] = freq/total

        return max, probs, k
    
    ### Returns as 
    #(
    #   k,
    #   found_t,
    #   probabilities
    #   w   
    #)
    def find(self, x):
        res = []
        for e in x:
            e = np.array(e)
            closest = self._closest(self.nx(e))
            # print(self._denorm_w(closest))
            max, probs, k = self._most_frequents_for_k(e, closest)
            res.append((
                k,
                max, 
                probs, 
                self._denorm_w(closest[:self.k])    
            ))
        return res