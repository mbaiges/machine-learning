import math
import numpy as np

## TODO criterio de corte

def loss_function(x: np.array, y: np.array, w, b, c):
    def l(xi, yi, w, b):
        t = yi * np.inner(w, xi) + b
        return 0 if t >= 1 else 1 - t
    
    s = 0
    for i in range(x.shape[0]):
        s += l(x[i], y[i], w, b)

    return (1/2) * np.inner(w, np.transpose(w)) + c * s


def loss(dataset: np.array, max_iter: int=1000, c: float=0.5, debug: bool=False):
    # Init variables
    x, y = dataset[:,:2], dataset[:,2]
    d = x.shape[1]
    w = np.random.uniform(-1,1,size=(d))
    w /= np.linalg.norm(w)

    b = np.random.uniform(-1,1)

    # Algorithm
    k_w = 0.0001
    k_b = 0.0001

    min_loss = math.inf

    iter = 1
    while iter < max_iter:
        ## TODO mejorar cuanto decremento
        for i, x_i in enumerate(x):
            t =  y[i] * (np.inner(w, x_i) + b)
            if t < 1:
                aux = np.zeros(d)
                for j in range(x.shape[0]):
                    aux += -1 * y[j] * x[j]
                w = w - k_w * (w + c * aux)
                b = b - k_b * c * (-1) * np.sum(y)
            else:
                w = w - k_w * w
            w = w / np.linalg.norm(w)
        
        # Loss
        loss = loss_function(x, y, w, b, c)
        if loss < min_loss:
            min_loss = loss

        # Debugging
        if debug:
            print(f'w: {w}, b: {b}, loss: {loss}')        

        # Learning rate decrement
        k_w = k_w - 0.000001 if k_w > 0.000001 else k_w
        k_b = k_b - 0.000001 if k_b > 0.000001 else k_b

        iter += 1
    return w, b, min_loss