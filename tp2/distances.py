import math

def man(x, y):
    d = 0
    for i in range(0, len(x)):
        d += math.abs(x[i] - y[i], 2)
    return d
    
def euc(x, y):
    d = 0
    for i in range(0, len(x)):
        d += math.pow(x[i] - y[i], 2)
    return math.sqrt(d)