import math
import numpy as np

## Norma orden 1
def man(x, y):
    return np.linalg.norm(x - y, 1) 
    
## Norma orden 2
def euc(x, y):
    return np.linalg.norm(x - y) 
