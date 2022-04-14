from itertools import count
import numpy as np

def df_to_np(df, x_columns, t_column):
    x = []
    t = []
    for idx, row in df.iterrows():
        xi = []
        for col_name in x_columns:
            xi.append(row[col_name])
        x.append(xi)
        t.append(row[t_column])
    return np.array(x), np.array(t)

def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    
    def nx(xi):
        return (xi - x_mean)/x_std
    def dx(norm_x):
        return norm_x * x_std + x_mean

    return nx(x), (nx, dx)

def mode(x):
    if not x:
        return None
    counts = {}
    for e in x:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)