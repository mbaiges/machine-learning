from re import L
import numpy as np
from perceptron import SimplePerceptron
from sklearn import svm

seed = 59077

import utils
import loss


def punto_c(only_dataset:bool = True):
    np.random.seed(seed)
    # Build random linear separable dataset
    # Parameters
    n: int = 60
    x_boundaries = (0, 5)
    y_boundaries = x_boundaries
    err_dist_pctg = 0.2
    err_prob = 0.5
    margin = 2
    iterations = 10000
    
    random_dataset, discriminator, line_points = utils.build_linear_separable_dataset(
        n, 
        x_boundaries[0], 
        x_boundaries[1], 
        y_boundaries[0], 
        y_boundaries[1],
        border_error_tolerance_dist_pctg=err_dist_pctg,
        border_error_prob=err_prob,
        margin=margin
    )
    if only_dataset:
        return random_dataset, x_boundaries, y_boundaries
    
    print(f'Line Formula (Simple format) --> {utils.get_line_formula(line_points, fmt="simple")}')
    print(f'Line Formula (Full format) --> {utils.get_line_formula(line_points, fmt="full")}')
    utils.plot_points(random_dataset, line_points, limits=(x_boundaries, y_boundaries))

    perceptron = SimplePerceptron(random_dataset[:,:2], random_dataset[:,2], 0.01)
    error = perceptron.train(iterations)
    w_min = perceptron.w_min
    b_loss = w_min[0]
    w_loss = w_min[1:]
    
    predicted_line_points = utils.full_formula_to_line_points(w_loss, b_loss)
    print(f'Predicted Line Formula (Simple format) --> {utils.get_line_formula(predicted_line_points, fmt="simple")}')
    print(f'Predicted Line Formula (Full format) --> {utils.get_line_formula(predicted_line_points, fmt="full")}')
    utils.plot_points(random_dataset, predicted_line_points, limits=(x_boundaries, y_boundaries))

    return random_dataset, x_boundaries, y_boundaries

def punto_a_b(only_dataset:bool = True):
    np.random.seed(seed)
    # Parameters
    n: int = 60
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
    if only_dataset:
        return random_dataset, x_boundaries, y_boundaries
    print(f'Line Formula (Simple format) --> {utils.get_line_formula(line_points, fmt="simple")}')
    print(f'Line Formula (Full format) --> {utils.get_line_formula(line_points, fmt="full")}')
    utils.plot_points(random_dataset, line_points, limits=(x_boundaries, y_boundaries), margin='find')

    # True Optimal hyperplane
    optimal_line_points, dist = utils.optimal_hyperplane(random_dataset, line_points, show_loading_bar=False)
    print(f'Optimal Margin: {dist}')
    print(f'True Optimal Line Formula (Simple format) --> {utils.get_line_formula(optimal_line_points, fmt="simple")}')
    print(f'True Optimal Line Formula (Full format) --> {utils.get_line_formula(optimal_line_points, fmt="full")}')
    utils.plot_points(random_dataset, optimal_line_points, limits=(x_boundaries, y_boundaries), margin=dist)

    # Loss Function based Algorithm 

    ## Parameters
    iterations = 10000
    # c = 10
    batch_size = 10

    # w_loss, b_loss, err = simple_perceptron.fit(random_dataset, iterations, batch_size, debug=True, show_loading_bar=True)
    perceptron = SimplePerceptron(random_dataset[:,:2], random_dataset[:,2], 0.001)
    error = perceptron.train(iterations)
    w_min = perceptron.w_min
    b_loss = w_min[0]
    w_loss = w_min[1:]
    
    predicted_line_points = utils.full_formula_to_line_points(w_loss, b_loss)
    print(f'Predicted Line Formula (Simple format) --> {utils.get_line_formula(predicted_line_points, fmt="simple")}')
    print(f'Predicted Line Formula (Full format) --> {utils.get_line_formula(predicted_line_points, fmt="full")}')
    utils.plot_points(random_dataset, predicted_line_points, limits=(x_boundaries, y_boundaries), margin='find')

    # Optimal hyperplane
    optimal_line_points, dist = utils.optimal_hyperplane(random_dataset, predicted_line_points, show_loading_bar=False)
    print(f'Optimal Margin: {dist}')
    print(f'Optimal Line Formula (Simple format) --> {utils.get_line_formula(optimal_line_points, fmt="simple")}')
    print(f'Optimal Line Formula (Full format) --> {utils.get_line_formula(optimal_line_points, fmt="full")}')
    utils.plot_points(random_dataset, optimal_line_points, limits=(x_boundaries, y_boundaries), margin=dist)
    
    return random_dataset, x_boundaries, y_boundaries

def punto_d(separable_dataset_and_boundaries, non_separable_dataset_and_boundaries):
    random_dataset_separable, x_boundaries, y_boundaries = separable_dataset_and_boundaries
    c = 1.0
    kernel = 'linear'
    clf = svm.SVC(C=c, kernel=kernel)
    clf.fit(random_dataset_separable[:,:2], random_dataset_separable[:,2])
    support_vectors = clf.support_vectors_
    print(support_vectors)
    print(clf.n_support_)
    w = clf.coef_[0]           # w consists of 2 elements
    b = clf.intercept_[0]      # b consists of 1 element
    predicted_line_points = utils.full_formula_to_line_points(w, b)
    print(f'Predicted Line Formula (Simple format) --> {utils.get_line_formula(predicted_line_points, fmt="simple")}')
    print(f'Predicted Line Formula (Full format) --> {utils.get_line_formula(predicted_line_points, fmt="full")}')
    utils.plot_points(random_dataset_separable, predicted_line_points, limits=(x_boundaries, y_boundaries))

    
    random_dataset_non_separable, x_boundaries, y_boundaries = non_separable_dataset_and_boundaries
    c = 1.0
    kernel = 'linear'
    clf = svm.SVC(C=c, kernel=kernel)
    clf.fit(random_dataset_non_separable[:,:2], random_dataset_non_separable[:,2])
    support_vectors = clf.support_vectors_
    print(support_vectors)
    print(clf.n_support_)
    w = clf.coef_[0]           # w consists of 2 elements
    b = clf.intercept_[0]      # b consists of 1 element
    predicted_line_points = utils.full_formula_to_line_points(w, b)
    print(f'Predicted Line Formula (Simple format) --> {utils.get_line_formula(predicted_line_points, fmt="simple")}')
    print(f'Predicted Line Formula (Full format) --> {utils.get_line_formula(predicted_line_points, fmt="full")}')
    utils.plot_points(random_dataset_non_separable, predicted_line_points, limits=(x_boundaries, y_boundaries))

if __name__ == '__main__':
    print("Exercise 1")

    separable_dataset_and_boundaries = punto_a_b(only_dataset=False)
    # non_separable_dataset_and_boundaries = punto_c(only_dataset=True)

    # punto_d(separable_dataset_and_boundaries, non_separable_dataset_and_boundaries)
