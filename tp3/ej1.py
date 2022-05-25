import numpy as np

# seed = 59076
# np.random.seed(seed)

import utils

if __name__ == '__main__':
    print("Ej 1")
    n: int = 20
    x_boundaries = [0, 5]
    y_boundaries = x_boundaries
    random_dataset, discriminator, line_points = utils.build_linear_separable_dataset(n, x_boundaries[0], x_boundaries[1], y_boundaries[0], y_boundaries[1])
    print("Random Generated Dataset:")
    print(random_dataset)
    utils.plot_points(random_dataset, line_points)