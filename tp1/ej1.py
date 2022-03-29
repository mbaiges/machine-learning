import pandas as pd

from bayes_1 import Naive

FILEPATH = 'PreferenciasBritanicos.csv'

SCONES       = 'scones'
CERVEZA      = 'cerveza'
WHISKY       = 'whisky'
AVENA        = 'avena'
FULBO        = 'futbol'
NACIONALIDAD = "Nacionalidad"

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=',')
    naive = Naive()
    x_train = df.loc[:, df.columns != NACIONALIDAD]
    naive.train(df.loc[:, df.columns != NACIONALIDAD], df.loc[:,NACIONALIDAD], None, None)
    
    data = [
        (1, 0, 1, 1, 0),
        (0, 1, 1, 0, 1)
    ]
    
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns = x_train.columns)
    print(df)
    results = naive.eval(df)
    print(results)

