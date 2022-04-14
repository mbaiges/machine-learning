import pandas as pd
import numpy as np

from id3 import ID3

FILEPATH                          = "german_credit.csv"

CREDITABILITY                     = "Creditability"
ACCOUNT_BALANCE                   = "Account Balance"
DURATION_BALANCE                  = "Duration of Credit (month)"
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
    DURATION_BALANCE,
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

def b_build_array(x: pd.DataFrame, t: pd.DataFrame, k: int=None):
    k = x.shape[0] if k is None else k
    indexes = np.random.randint(0, x.shape[0], size=k)
    return np.array([x[i] for i in indexes]), np.array([t[i] for i in indexes])

def bootstrap(x: pd.DataFrame, t: pd.DataFrame, train_size: int=None, test_size: int=None):
    train_size = x.shape[0] if train_size is None else train_size
    test_size  = x.shape[0] if test_size is None else test_size
    return b_build_array(x, t, train_size), b_build_array(x, t, test_size)

if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv(FILEPATH, sep=',')
    x = df[ATTRIBUTES_NAMES]
    t = df[T_NAME]
    print(f"Loaded {x.shape[0]} rows")
    
    # Train and test with bootstrap
    list_size = 500
    (x_train, t_train), (x_test, t_test) = bootstrap(x, t, train_size=list_size, test_size=list_size)
    print(x_train)

    id3 = ID3()