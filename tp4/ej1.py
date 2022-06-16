import pandas as pd
import numpy as np

import utils

seed = 59076

FILEPATH = 'acath.csv'

CSV_HEADER = ["sex","age","cad.dur","choleste","sigdz","tvdlm"]

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

def analysis(df: pd.DataFrame) -> None:

    # Credit amount
    credit_amounts = df[CREDIT_AMOUNT].to_numpy()
    utils.hist(credit_amounts, title=CREDIT_AMOUNT)

    # Ages
    ages = df[AGE].to_numpy()
    utils.hist(ages, title=AGE)

    # PCA para ver como se ubican en un diagrama de biplot???????? Si sale super rapido nomas

    # Creditability
    creditabilities = df[CREDITABILITY].to_numpy()
    utils.bars(creditabilities, title=CREDITABILITY)

###### Main ######

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=';')
    log_long(f"{df.size} rows loaded")

    analysis(df)
