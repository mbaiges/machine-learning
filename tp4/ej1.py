import pandas as pd
import numpy as np
import random

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

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=';')
    df = df.astype(object).replace(np.nan, None)
    log_long(f"{df.size} rows loaded")

    # analysis(df)

    x = df[[ATT_SEX, ATT_AGE, ATT_CAD_DUR, ATT_CHOLESTE]].to_numpy()
    y = df[ATT_SIGDZ].to_numpy()

    log(x)
    log(y)