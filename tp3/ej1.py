import numpy as np

seed = 59076
np.random.seed(seed)

import utils

if __name__ == '__main__':
    print("Exercise 1")

    # Parameters
    n: int = 20
    x_boundaries = (0, 5)
    y_boundaries = x_boundaries
    err_dist_pctg = 0.08
    err_prob = 0.5

    # Build random linear separable dataset
    random_dataset, discriminator, line_points = utils.build_linear_separable_dataset(
        n, 
        x_boundaries[0], 
        x_boundaries[1], 
        y_boundaries[0], 
        y_boundaries[1],
        border_error_tolerance_dist_pctg=err_dist_pctg,
        border_error_prob=err_prob
    )

    print("Random Generated Dataset:")
    print(random_dataset)

    # Plot points
    # utils.plot_points(random_dataset, line_points, limits=(x_boundaries, y_boundaries))

    # Optimal hyperplane
    line_points = utils.optimal_hyperplane(random_dataset)