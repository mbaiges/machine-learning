from itertools import count
import numpy as np
import pandas as pd

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

def bootstrap_df_build_sample(x: pd.DataFrame, t: pd.DataFrame, k: int=None):# Como se define que retorna tipo (pd.Dataframe, pd.Dataframe)
    k = x.shape[0] if k is None else k
    indexes = np.random.randint(0, x.shape[0], size=k)
    x_columns = x.columns
    x_arr, t_arr = x.to_numpy(), t.to_numpy()
    x_ret, t_ret = np.array([x_arr[i] for i in indexes]), np.array([t_arr[i] for i in indexes])
    return pd.DataFrame(x_ret, columns=x_columns), pd.DataFrame(t_ret)

def bootstrap_df(x: pd.DataFrame, t: pd.DataFrame, train_size: int=None, test_size: int=None):
    train_size = x.shape[0] if train_size is None else train_size
    test_size  = x.shape[0] if test_size is None else test_size
    return bootstrap_df_build_sample(x, t, train_size), bootstrap_df_build_sample(x, t, test_size)
