import pandas as pd
import numpy as np
import random
import math

import utils

seed = 59076

RANDOM = random.Random(seed)

FILEPATH = 'acath.csv'

ATT_SEX      = "sex"
ATT_AGE      = "age"
ATT_CAD_DUR  = "cad.dur"
ATT_CHOLESTE = "choleste"
ATT_SIGDZ    = "sigdz"
ATT_TVLM     = "tvlm"

CSV_HEADER = [ATT_AGE,ATT_CAD_DUR,ATT_CHOLESTE,ATT_SIGDZ,ATT_TVLM]

LOG_LONG_DELIMITER = "---------------------"
LOG_SHORT_DELIMITER = "---------"

###### Logs ######

def log_long(s: str) -> None:
    utils.log(s, LOG_LONG_DELIMITER)
def log_short(s: str) -> None:
    utils.log(s, LOG_SHORT_DELIMITER)
def log(s: str) -> None:
    utils.log(s)

###### Helpers ######

def shuffle(x: np.array, y: np.array, seed: int):
    r = random.Random(seed)
    shuffled_idxs = [i for i in range(x.shape[0])]
    r.shuffle(shuffled_idxs)
    return np.array([x[i] for i in shuffled_idxs]), np.array([y[i] for i in shuffled_idxs])

def split_combinations(x: np.array, y: np.array, test_percentage: float):
    size = math.ceil(x.shape[0]*test_percentage)
    return math.ceil(x.shape[0]/size)

def random_split(x: np.array, y: np.array, test_percentage: float, index: int):
    size = math.ceil(x.shape[0]*test_percentage)
    top = (index+1)*size if (index+1)*size < x.shape[0] else (x.shape[0]-1)
    return np.concatenate((x[0:index*size],x[top:x.shape[0]]), axis=0), np.concatenate((y[0:index*size],y[top:y.shape[0]]), axis=0), x[index*size:top], y[index*size:top]

def analysis(df: pd.DataFrame) -> None:

    # Sex
    sexs = df[ATT_SEX].to_numpy()
    utils.bars(sexs, title=ATT_SEX)

    # Ages
    ages = df[ATT_AGE].to_numpy()
    utils.hist(ages, title=ATT_AGE)

    # Symptoms Duration
    durations = df[ATT_CAD_DUR].to_numpy()
    utils.hist(durations, title=ATT_CAD_DUR) # Poisson? Pienso que tiene sentido

    # Cholesterol
    chols = df[ATT_CHOLESTE].to_numpy()
    filter(lambda c: c is not None, chols) # Filtering None
    utils.hist(chols, title=ATT_CHOLESTE)

    # SIGDZ
    sigdz = df[ATT_SIGDZ].to_numpy()
    utils.bars(sigdz, title=ATT_SIGDZ)


###### Main ######

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=';')
    df = df.astype(object).replace(np.nan, None)
    log_long(f"{df.size} rows loaded")

    # analysis(df)

    x = df[[ATT_AGE, ATT_CAD_DUR, ATT_CHOLESTE]].to_numpy()
    y = df[ATT_SIGDZ].to_numpy()
    
    sx, sy = shuffle(x, y, seed)

    ts_pctg = 0.4
    combinations = split_combinations(x, y, ts_pctg) # max posible index to call random_split with
    tr_x, tr_y, ts_x, ts_y = random_split(x, y, ts_pctg, 0) # purposely selecting index 0
