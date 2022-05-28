import numpy as np

## TODO criterio de corte

def loss(x, y, max_iter, w, b, c):
    iter = 1
    k_w = 0.0001
    k_b = 0.0001
    while iter < max_iter:
        ## TODO mejorar cuanto decremento
        for i, x_i in enumerate(x):
            t =  y[i] * (np.dot(w, x_i) + b)
            if t < 1:
                aux = np.zeros(x.shape[1])
                for j in range(x.shape[0]):
                    aux += -1 * y[j] * x[j]
                w = w - k_w * (w + c * aux)
                
                b = b - k_b * c * (-1) * np.sum(y)
            else:
                w = w - k_w * w
            w = w / np.linalg.norm(w)
        k_w = k_w - 0.000001 if k_w > 0.000001 else k_w
        k_b = k_b - 0.000001 if k_b > 0.000001 else k_b
        iter += 1
    return w, b