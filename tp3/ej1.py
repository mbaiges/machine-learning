import numpy as np

seed = 59076
np.random.seed(seed)

import utils
import loss

if __name__ == '__main__':
    print("Exercise 1")

    # Parameters
    n: int = 50
    x_boundaries = (0, 5)
    y_boundaries = x_boundaries
    err_dist_pctg = 0
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

    print(f'Line Formula (Simple format) --> {utils.get_line_formula(line_points, fmt="simple")}')
    print(f'Line Formula (Full format) --> {utils.get_line_formula(line_points, fmt="full")}')

    # Plot points
    utils.plot_points(random_dataset, line_points, limits=(x_boundaries, y_boundaries))

    # d = random_dataset[:,:2].shape[1]
    # w = np.random.uniform(-1,1,size=(d))
    # w = w / np.linalg.norm(w)

    # b = np.random.uniform(-1,1)


    # w_loss, b_loss = loss.loss(random_dataset[:,:2], random_dataset[:,2], 1000, w, b, 10)

    # print("Loss")


    # print("w loss")
    # print(w_loss)

    # print("b loss")
    # print(b_loss)

    # Optimal hyperplane
    optimal_line_points, dist = utils.optimal_hyperplane(random_dataset, line_points, show_loading_bar=True)
    print(f'Optimal Line Formula (Simple format) --> {utils.get_line_formula(line_points, fmt="simple")}')
    print(f'Optimal Line Formula (Full format) --> {utils.get_line_formula(line_points, fmt="full")}')

    utils.plot_points(random_dataset, optimal_line_points, limits=(x_boundaries, y_boundaries))