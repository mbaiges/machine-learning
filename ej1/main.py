import pandas as pd
from matplotlib import pyplot as plt

FILEPATH = 'DatosTrabajo1.tsv'

GRASAS_SATURADAS = 'Grasas_sat'
ALCOHOL          = 'Alcohol'
CALORIAS         = 'Calorias'
SEXO             = 'Sexo'

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep='\t')

    #todo: no ignorarlos
    # 1. Reemplazamos los 999.99
    for index, row in df.iterrows():
        if row[ALCOHOL] == 999.99:
            print(row)
        if row[GRASAS_SATURADAS] == 999.99:
            print(row)

    # 2. Boxplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    
    ## Grasas saturadas
    mask = df[GRASAS_SATURADAS].isin([999.99])
    ax1.boxplot(df[~mask][GRASAS_SATURADAS])
    ax1.set_title('Grasas Saturadas')

    ## Alcohol
    mask = df[ALCOHOL].isin([999.99])
    ax2.boxplot(df[~mask][ALCOHOL])
    ax2.set_title('Alcohol')

    ## Calorias
    ax3.boxplot(df[CALORIAS])
    ax3.set_title('Calorias')
    
    plt.show()

    # 3. Análisis de acuerdo al Sexo

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    male   = df.loc[df[SEXO] == 'M']
    female = df.loc[df[SEXO] == 'F']
    ## Grasas saturadas
    mask = df[GRASAS_SATURADAS].isin([999.99])
    ax1.boxplot([male[~mask][GRASAS_SATURADAS], female[~mask][GRASAS_SATURADAS]])
    ax1.set_xticklabels(['M', 'F'], fontsize=12)
    ax1.set_title('Grasas Saturadas')

    ## Alcohol
    mask = df[ALCOHOL].isin([999.99])
    ax2.boxplot([male[~mask][ALCOHOL], female[~mask][ALCOHOL]])
    ax2.set_xticklabels(['M', 'F'], fontsize=12)
    ax2.set_title('Alcohol')

    ## Calorias
    ax3.boxplot([male[~mask][CALORIAS], female[~mask][CALORIAS]])
    ax3.set_xticklabels(['M', 'F'], fontsize=12)
    ax3.set_title('Calorias')

    plt.show()
    
    # 4. Análisis de Alcohol
    
    CATE1 = df.loc[df[CALORIAS] <= 1100]
    CATE2 = df.loc[(df[CALORIAS] > 1100) & (df[CALORIAS] <= 1700)]
    CATE3 = df.loc[df[CALORIAS] > 1700]

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    ## Alcohol
    mask = df[ALCOHOL].isin([999.99])
    ax1.boxplot([CATE1[~mask][ALCOHOL], CATE2[~mask][ALCOHOL], CATE3[~mask][ALCOHOL]])
    ax1.set_xticklabels(['CATE1', 'CATE2', 'CATE3'], fontsize=12)
    ax1.set_title('Alcohol')

    plt.show()