import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_linear_separable_dataset(n: int, x_min: float, x_max: float, y_min: float, y_max: float) -> pd.DataFrame:
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

    discriminator: function = lambda x, y: 1
    if (p1_x == p2_x):
        # x = p1_x
        discriminator = lambda x, y: 1 if x >= p1_x else -1
    else:
        # y = slope * x + b
        slope: float = (p2_y - p1_y)/(p2_x - p1_x)
        b: float = p1_y - slope * p1_x
        discriminator = lambda x, y: 1 if y >= (slope * x + b) else -1

    dataset_list: list = []

    for i in range(n):
        dataset_list.append(list((x_points[i], y_points[i], discriminator(x_points[i], y_points[i]))))

    return np.array(dataset_list), discriminator, ((p1_x, p1_y), (p2_x, p2_y))

POINTS_COLORS = ["#81b29a", "#e07a5f"]
LINE_COLOR = "#3d405b"
def plot_points(dataset: np.array, line_points=None, title="Puntos"):
    fig, ax = plt.subplots()
    d = np.copy(dataset)
    c_by_value = {value: POINTS_COLORS[idx] for idx, value in enumerate(sorted(set(d[:,2])))}
    c = list(map(lambda t: c_by_value[t], d[:,2]))
    ax.scatter(d[:,0], d[:,1], c=c)
    if line_points is not None:
        p1, p2 = line_points
        ax.axline(p1, p2, color=LINE_COLOR, linewidth=0.7, label="Hiperplano")
    ax.set_title(title)
    plt.show()