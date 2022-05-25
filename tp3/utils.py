import math
from turtle import up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 
# Calculates the distance between a point1 definded as
# (x1, y1) through parameter p1, and a point defined as
# (x2, y2) through parameter p2.
#
# Uses Norm 2 (Euler distance)
# 
def distance_between_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

# 
# Calculates the distance between a line defined as
# ((x1, y1), (x2, y2)) through parameter line_points
# and a point defined as (x, y) through parameter point.
# 
def distance_between_line_and_point(line_points, point):
    (x1, y1), (x2, y2) = line_points
    x, y = point

    dist: float = 0
    if x1 == x2:
        dist = abs(x - x1)
    elif y1 == y2:
        dist = abs(y - y1)
    else:
        # y' = slope * x' + b
        slope: float = (y2-y1)/(x2-x1)
        b: float = y1 - slope * x1
        perp_slope: float = -1/slope
        # y = perp_slope * x + perp_b
        perp_b = y - perp_slope * x

        # then we find interception with the original 
        # line that devides the dataset
        
        # slope * xe + b = ye = perp_slope * xe + perp_b
        # (slope - perp_slope) * xe + (b - perp_b) = 0
        # xe = - (b - perp_b)/(slope - perp_slope)
        # ye = slope * xe + b
        xe = - (b - perp_b)/(slope - perp_slope)
        ye = slope * xe + b

        # finally, we get the distance
        dist = distance_between_points((x, y), (xe, ye))
        
    return dist

#
# Build a dataset of n points separated by a random line.
# The dataset has points between (x_min, x_max) and (y_min, y_max). 
#
# Also, it supports adding error to near border points.
# For this, we have a border_error_tolerance_dist_pctg
# (let's say the size of the box is (0,5), (0,6) it will 
# take the min distance between borders: 5-0 = 5 and 
# use as error tolerance distance the value of: 
# 5 * border_error_tolerance_dist_pctg, with a probability
# of confusing the point of border_error_prob)
#
def build_linear_separable_dataset(n: int, x_min: float, x_max: float, y_min: float, y_max: float, border_error_tolerance_dist_pctg: float = 0, border_error_prob: float = 0.2) -> pd.DataFrame:
    # correction in case of an error
    (x_min, x_max) = (x_min, x_max) if x_max >= x_min else (x_max, x_min)
    (y_min, y_max) = (y_min, y_max) if y_max >= y_min else (y_max, y_min)
    
    # sample points
    x_points: np.array = np.random.uniform(x_min, x_max, size=n)
    y_points: np.array = np.random.uniform(y_min, y_max, size=n)

    # separation line
    ## point 1
    p1_x: float = np.random.uniform(x_min, x_max)
    p1_y: float = np.random.uniform(y_min, y_max)
    ## point 2
    p2_x: float = np.random.uniform(x_min, x_max)
    p2_y: float = np.random.uniform(y_min, y_max)

    # line points ((x1, y1), (x2, y2)).
    # This is a helper so we can build the line later
    line_points = ((p1_x, p1_y), (p2_x, p2_y))

    # discriminator function
    discriminator: function = lambda x, y: 1
    if (p1_x == p2_x):
        # x = p1_x
        discriminator = lambda x, y: 1 if x >= p1_x else -1
    else:
        # y = slope * x + b
        slope: float = (p2_y - p1_y)/(p2_x - p1_x)
        b: float = p1_y - slope * p1_x
        discriminator = lambda x, y: 1 if y >= (slope * x + b) else -1

    # support for error near line
    min_dist = (x_max - x_min) if (x_max - x_min) >= (y_max - y_min) else (y_max - y_min)
    err_dist = min_dist * border_error_tolerance_dist_pctg

    # tagged points dataset
    dataset_list: list = []
    for i in range(n):
        x, y = x_points[i], y_points[i]
        t = discriminator(x, y)
        dist_from_line = distance_between_line_and_point(line_points, (x, y))
        # if is between the error margin and it is probable that the error occurs
        if dist_from_line < err_dist and np.random.uniform(0, 1) <= border_error_prob:
            t = -t # we swap into the opposite
        dataset_list.append(list((x, y, t)))

    return np.array(dataset_list), discriminator, line_points

POINTS_COLORS = ["#81b29a", "#e07a5f"]
LINE_COLOR = "#3d405b"
# 
# Plots a scatter of points using Matplotlib, using different
# colors for the different tagged entries.
# 
# Eg. If we have an entry that says [0.4, 3, 1], and another
# that says [2.3, 4.1, -1], the tag (third column) will force
# the plotter to differentiate points with different colors
# 
# If a line is known, you can pass ((x1, y1), (x2, y2))
# as an argument (line_points), and draw the line.
# 
def plot_points(dataset: np.array, line_points=None, limits=None, title="Puntos"):
    # we make a copy of the original dataset
    d = np.copy(dataset)
    # we make a map defining a color for each possible tag value
    c_by_value = {value: POINTS_COLORS[idx] for idx, value in enumerate(sorted(set(d[:,2])))}
    # we make a colors list for each point
    c = list(map(lambda t: c_by_value[t], d[:,2]))

    fig, ax = plt.subplots()
    ax.scatter(d[:,0], d[:,1], c=c)

    # if a line should be drawn
    if line_points is not None:
        p1, p2 = line_points
        ax.axline(p1, p2, color=LINE_COLOR, linewidth=0.7, label="Hiperplano")

    # if limits are given
    if limits is not None:
        (x_min, x_max), (y_min, y_max) = limits
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)

    # set title
    ax.set_title(title)

    plt.show()