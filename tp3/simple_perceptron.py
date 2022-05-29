import math
import numpy as np

import utils

def sign_activation(value: float):
    return 1 if value >= 0 else -1
    
def error_function(x, y, w):
    err = 0
    for i in range(0, x.shape[0]):
        x_i = x[i]
        y_i = y[i]

        excitement = np.inner(x_i, w)
        activation = sign_activation(excitement)

        err += abs(y_i - activation)
    return err

def fit(dataset: np.array, epochs: int=1000, batch_size: int=200, debug: bool=False, show_loading_bar: bool=False):
    # Init variables
    x, y = dataset[:,:2], dataset[:,2]
    d = x.shape[1]

    x = np.append(x, np.ones((x.shape[0],1)), axis=1) # add bias weight=1
    w = np.random.uniform(-1, 1, (d+1))
    w /= np.linalg.norm(w)
    min_w = w

    # Algorithm
    learning_rate = 0.001

    min_error = x.shape[0] * 2

    if show_loading_bar:
        loading_bar = utils.LoadingBar()
        loading_bar.init()

    iter = 1

    queue = []
    while iter < epochs and min_error > 0:
        if show_loading_bar:
            loading_bar.update(1.0 * iter / epochs)

        queue = np.array([i for i in range(0, x.shape[0])])
        np.random.shuffle(queue)
        if queue.shape[0] >= batch_size:
            queue = queue[:batch_size]
        else:
            queue = np.append(queue, np.random.randint(0, x.shape[0], (batch_size - queue.shape[0])), axis=0)
        queue = queue.tolist()

        delta_w = 0
        for idx in queue:
            xp = x[idx]
            yp = y[idx]

            excitement = np.inner(xp, w)
            activation = sign_activation(excitement)

            delta_w += learning_rate * (yp - activation) * xp

            error = error_function(x, y, w)
            if error < min_error:
                min_error = error
                min_w = w

                # Debugging
                if debug:
                    print(f'w: {w[:-1]}, b: {w[-1]}, error: {error}')     

        w = w + delta_w
        w = w / np.linalg.norm(w)   

        # Learning rate decrement
        ## eta adaptativo
        # learning_rate = learning_rate - 0.000001 if learning_rate > 0.000001 else learning_rate

        iter += 1

    if show_loading_bar:
        loading_bar.end()

    return min_w[:-1], min_w[-1], min_error