import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(59076)

from utils import bootstrap_df, hist, bins
from id3 import ID3

FILEPATH                          = "german_credit.csv"

CREDITABILITY                     = "Creditability"
ACCOUNT_BALANCE                   = "Account Balance"
DURATION_OF_CREDIT_MONTH          = "Duration of Credit (month)"
PAYMENT_STATUS_OF_PREVIOUS_CREDIT = "Payment Status of Previous Credit"
PURPOSE                           = "Purpose"
CREDIT_AMOUNT                     = "Credit Amount"
VALUE_SAVINGS_STOCKS              = "Value Savings/Stocks"
LENGTH_OF_CURRENT_EMPLOYMENT      = "Length of current employment"
INSTALMENT_PER_CENT               = "Instalment per cent"
SEX_AND_MARITAL_STATUS            = "Sex & Marital Status"
GUARANTORS                        = "Guarantors"
DURATION_IN_CURRENT_ADDRESS       = "Duration in Current address"
MOST_VALUABLE_AVAILABLE_ASSET     = "Most valuable available asset"
AGE                               = "Age (years)"
CONCURRENT_CREDITS                = "Concurrent Credits"
TYPE_OF_APARTMENT                 = "Type of apartment"
NO_OF_CREDITS_AT_THIS_BANK        = "No of Credits at this Bank"
OCCUPATION                        = "Occupation"
NO_OF_DEPENDENTS                  = "No of dependents"
TELEPHONE                         = "Telephone"
FOREIGN_WORKER                    = "Foreign Worker"

T_NAME = CREDITABILITY
ATTRIBUTES_NAMES = [
    ACCOUNT_BALANCE,
    DURATION_OF_CREDIT_MONTH,
    PAYMENT_STATUS_OF_PREVIOUS_CREDIT,
    PURPOSE,
    CREDIT_AMOUNT,
    VALUE_SAVINGS_STOCKS,
    LENGTH_OF_CURRENT_EMPLOYMENT,
    INSTALMENT_PER_CENT,
    SEX_AND_MARITAL_STATUS,
    GUARANTORS,
    DURATION_IN_CURRENT_ADDRESS,
    MOST_VALUABLE_AVAILABLE_ASSET,
    AGE,
    CONCURRENT_CREDITS,
    TYPE_OF_APARTMENT,
    NO_OF_CREDITS_AT_THIS_BANK,
    OCCUPATION,
    NO_OF_DEPENDENTS,
    TELEPHONE,
    FOREIGN_WORKER
]

def analysis(df: pd.DataFrame) -> None:

    # Credit amount
    credit_amounts = df[CREDIT_AMOUNT].to_numpy()
    hist(credit_amounts, title=CREDIT_AMOUNT)

    # Ages
    ages = df[AGE].to_numpy()
    hist(ages, title=AGE)

    # PCA para ver como se ubican en un diagrama de biplot???????? Si sale super rapido nomas

    # Creditability

def _discretize_with_bins(b, v):
    for i, bv in enumerate(b[1:]):
        if v < bv:
            return i
    return i

def discretize(df) -> None:
    # Duration of Credit Month
    x = df[DURATION_OF_CREDIT_MONTH]
    b = bins(x, alg='dist', options={'min': 0, 'dist': 6})
    print(f"{DURATION_OF_CREDIT_MONTH} - Bins: {b}")
    df[DURATION_OF_CREDIT_MONTH].apply(lambda v: _discretize_with_bins(b, v))

    # Credit Amount
    x = df[CREDIT_AMOUNT]
    b = bins(x, alg='perc', options={'n': 10})
    print(f"{CREDIT_AMOUNT} - Bins: {b}")
    df[CREDIT_AMOUNT].apply(lambda v: _discretize_with_bins(b, v))

    # Age
    x = df[AGE]
    b = bins(x, alg='perc', options={'n': 10})
    print(f"{AGE} - Bins: {b}")
    df[AGE].apply(lambda v: _discretize_with_bins(b, v))

if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv(FILEPATH, sep=',')
    x = df[ATTRIBUTES_NAMES]
    t = df[T_NAME].to_frame()
    print(f"Loaded {x.shape[0]} rows")

    # Analyze
    analysis(df)

    # Discretize
    discretize(df)
    
    # Train and test with bootstrap
    list_size = 500
    (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=list_size, test_size=list_size)
    print(x_train)

    id3 = ID3()
    id3.load(x_train, t_train)