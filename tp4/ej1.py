import pandas as pd
import numpy as np
import random
import math

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

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

CSV_HEADER = [ATT_SEX,ATT_AGE,ATT_CAD_DUR,ATT_CHOLESTE,ATT_SIGDZ,ATT_TVLM]

NUM_VARS = [ATT_AGE,ATT_CAD_DUR,ATT_CHOLESTE]

SIGDZ = "sigdz"

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

# def random_split(df: pd.DataFrame, test_percentage: float, index: int):
#     size = 

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

def cross_validation(model, x, t, k=20):
    total = x.shape[0]
    amount = int(total/k)
    acum = 0
    new_total = total
    max_ = math.inf
    best_x_train, best_t_train, best_x_test, best_t_test  = None, None, None, None
    counter = 0
    if total % amount != 0:
        new_total -= amount ## in order to add extra cases to last test set
        k -= 1
    for i in range(0, new_total, amount):
        if(i + amount > new_total): ## in order to add extra cases to last test set
            x_test = x[i:]
            t_test = t[i:]
            x_train = x[0:i]
            t_train = t[0:i]
        else: 
            x_test = x[i:i+amount]
            t_test = t[i:i+amount]
            x_train = np.concatenate((x[0:i], x[i+amount:]), axis=0)
            t_train = np.concatenate((t[0:i], t[i+amount:]), axis=0)
        if not (len(set(t_test)) == len(set(t_train)) == len(CSV_HEADER)):
            # print(f"Skipping at K: {k}, with i: {i}")
            continue
        counter += 1
        model.train(x_train, t_train)
        err = 0
        res = model.predict(x_test)
        for r, cat in zip(res, t_test):
            t_found = r[1]
            err += 1 if cat != t_found else 0
        if(err < max_):
            max_ = err
            best_x_train, best_t_train, best_x_test, best_t_test  = x_train, t_train, x_test, t_test
        acum += err
    return (best_x_train, best_t_train, best_x_test, best_t_test), acum / counter, max_

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=';')
    df = df.astype(object).replace(np.nan, None)
    log_long(f"{df.size} rows loaded")

    # analysis(df)

    x = df[NUM_VARS].to_numpy()
    y = df[ATT_SIGDZ].to_numpy()

    log(x)
    log(y)
    
    log_long(f"{df.size} rows loaded")

    analysis(df)

    std_scaler = StandardScaler()
    std_x = std_scaler.fit_transform(x)
    print(std_x)
    logit_mod = sm.Logit(std_x, y)
    logit_res = logit_mod.fit()
    print(logit_res.summary())
