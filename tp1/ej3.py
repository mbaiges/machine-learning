import pandas as pd

from bayes_network_3 import *

FILEPATH = 'binary.csv'

def discretize(df):
    df.loc[df[GRE] < 500, GRE] = 0
    df.loc[df[GRE] >= 500, GRE] = 1
    df.loc[df[GPA] < 3, GPA] = 0
    df.loc[df[GPA] >= 3, GPA] = 1
    return df.astype(int)

## tabla:

# { Rank
#     rangoValor1: cantidad,
# }

# { GPA
#     (rangoValor1, gpaValor1): cantidad,
#     (rangoValor1, gpaValor2): cantidad
#     ...
#     (rangoValor1, gpaValorN): cantidad
#     ...
#     (rangoValorN, gpaValor1): cantidad
#     ...
#     (rangoValorN, gpaValorN): cantidad
# }




if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=',')

    df = discretize(df)
    print(df)

    bn = BayesNetwork()
    bn.train(df)
    
    res_a = bn.a()
    print(f"Ejercicio a: {res_a}")

    res_b = bn.b()
    print(f"Ejercicio b: {res_b}")

    print(f"Ejercicio c:")

    all = bn.all()
    print(f"Todas las conjuntas suman... {all}")
    