import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

FILEPATH = 'DatosTrabajo1.tsv'

GRASAS_SATURADAS = 'Grasas_sat'
ALCOHOL          = 'Alcohol'
CALORIAS         = 'Calorias'
SEXO             = 'Sexo'

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep='\t')

    #todo: no ignorarlos
    # 1. Reemplazamos los 999.99

    filtered_df = df.loc[(df[ALCOHOL] != 999.99) & (df[GRASAS_SATURADAS] != 999.99)]

    grasas_data = filtered_df.to_numpy()
    grasas_data[grasas_data == 'F'] = 0
    grasas_data[grasas_data == 'M'] = 1
    grasas_data = np.array(grasas_data, dtype=np.float64)

    y = grasas_data[:,0]
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    x = grasas_data[:,1:]
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)

    x_final = (x-x_min)/(x_max-x_min)
    y_final = (y-y_min)/(y_max-y_min)

    grasas_model = Sequential()
    grasas_model.add(Dense(300, input_dim=x_final.shape[1], activation='tanh'))
    grasas_model.add(Dense(10, activation='sigmoid'))
    grasas_model.add(Dense(1, activation='relu'))

    grasas_model.compile(loss='mean_squared_error', optimizer='adam')
    grasas_model.fit(x_final, y_final, batch_size=100, epochs=100000)
    print(x_final[0])

    for i in range(x_final.shape[0]):
        x_to_eval = (np.array([x[i]])-x_min)/(x_max-x_min)
        pred = grasas_model.predict(x_to_eval)
        pred = pred*(y_max-y_min)+y_min
        print(f"{x[i]}\t{y[i]}\t{pred[0,0]}\t{abs(y[i] - pred[0,0])}")

    x_to_eval = np.array([[37.90, 2334, 1]])
    x_to_eval = (x_to_eval-x_min)/(x_max-x_min)
    pred = grasas_model.predict(x_to_eval)

    print(x_to_eval*(x_max-x_min)+x_min)
    print(pred*(y_max-y_min)+y_min)

    # # 2. Boxplots
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    
    # ## Grasas saturadas
    # mask = df[GRASAS_SATURADAS].isin([999.99])
    # ax1.boxplot(df[~mask][GRASAS_SATURADAS])
    # ax1.set_title('Grasas Saturadas')

    # ## Alcohol
    # mask = df[ALCOHOL].isin([999.99])
    # ax2.boxplot(df[~mask][ALCOHOL])
    # ax2.set_title('Alcohol')

    # ## Calorias
    # ax3.boxplot(df[CALORIAS])
    # ax3.set_title('Calorias')
    
    # plt.show()

    # # 3. Análisis de acuerdo al Sexo

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    # male   = df.loc[df[SEXO] == 'M']
    # female = df.loc[df[SEXO] == 'F']
    # ## Grasas saturadas
    # mask = df[GRASAS_SATURADAS].isin([999.99])
    # ax1.boxplot([male[~mask][GRASAS_SATURADAS], female[~mask][GRASAS_SATURADAS]])
    # ax1.set_xticklabels(['M', 'F'], fontsize=12)
    # ax1.set_title('Grasas Saturadas')

    # ## Alcohol
    # mask = df[ALCOHOL].isin([999.99])
    # ax2.boxplot([male[~mask][ALCOHOL], female[~mask][ALCOHOL]])
    # ax2.set_xticklabels(['M', 'F'], fontsize=12)
    # ax2.set_title('Alcohol')

    # ## Calorias
    # ax3.boxplot([male[~mask][CALORIAS], female[~mask][CALORIAS]])
    # ax3.set_xticklabels(['M', 'F'], fontsize=12)
    # ax3.set_title('Calorias')

    # plt.show()
    
    # # 4. Análisis de Alcohol
    
    # CATE1 = df.loc[df[CALORIAS] <= 1100]
    # CATE2 = df.loc[(df[CALORIAS] > 1100) & (df[CALORIAS] <= 1700)]
    # CATE3 = df.loc[df[CALORIAS] > 1700]

    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    # ## Alcohol
    # mask = df[ALCOHOL].isin([999.99])
    # ax1.boxplot([CATE1[~mask][ALCOHOL], CATE2[~mask][ALCOHOL], CATE3[~mask][ALCOHOL]])
    # ax1.set_xticklabels(['CATE1', 'CATE2', 'CATE3'], fontsize=12)
    # ax1.set_title('Alcohol')

    # plt.show()