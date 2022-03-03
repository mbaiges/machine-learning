import pandas as pd
from matplotlib import pyplot as plt

FILEPATH = 'DatosTrabajo1.tsv'

GRASAS_SATURADAS = 'Grasas_sat'
ALCOHOL          = 'Alcohol'
CALORIAS         = 'Calorias'
SEXO             = 'Sexo'

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep='\t')

    # 1. Reemplazamos los 999.99
    for index, row in df.iterrows():
        if row[ALCOHOL] == 999.99:
            print(row)
        if row[GRASAS_SATURADAS] == 999.99:
            print(row)

    # 2. Boxplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    ## Grasas saturadas
    ax1.boxplot(df[GRASAS_SATURADAS].drop_values(999.99))

    ## Alcohol
    ax2.boxplot(df[ALCOHOL].drop_values(999.99))

    ## Calorias
    ax3.boxplot(df[CALORIAS])

    plt.show()