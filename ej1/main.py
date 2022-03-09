import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

FILEPATH = 'DatosTrabajo1.tsv'
FIXED_FILEPATH = 'DatosTrabajo1_fixed.tsv'

GRASAS_SATURADAS = 'Grasas_sat'
ALCOHOL          = 'Alcohol'
CALORIAS         = 'Calorias'
SEXO             = 'Sexo'

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep='\t')
    fixed_df = pd.read_csv(FILEPATH, sep='\t')

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

    # Now let's use a new stop condition on the network.  This will check if the 'loss' value is less than 0.1
    class new_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}): 
            if (logs.get('loss')<0.0001):
                print("Loss bellow threshold ! ---- Stopping !")
                self.model.stop_training = True

    callback = new_callback()

    grasas_model = Sequential()
    grasas_model.add(Dense(250, input_dim=x_final.shape[1], activation='tanh'))
    grasas_model.add(Dense(120, activation='tanh'))
    grasas_model.add(Dense(50, activation='tanh'))
    grasas_model.add(Dense(25, activation='tanh'))
    grasas_model.add(Dense(5, activation='tanh'))
    grasas_model.add(Dense(1, activation='tanh'))

    grasas_model.compile(loss='mean_squared_error', optimizer='adam')
    grasas_model.fit(x_final, y_final, batch_size=50, epochs=200000, callbacks=[callback])
    print(x_final[0])

    for i in range(x_final.shape[0]):
        x_to_eval = (np.array([x[i]])-x_min)/(x_max-x_min)
        pred = grasas_model.predict(x_to_eval)
        pred = pred*(y_max-y_min)+y_min
        print(f"{x[i]}\t{y[i]}\t{pred[0,0]}\t{abs(y[i] - pred[0,0])}")

    x_to_eval_raw = np.array([[37.90, 2334, 1]])
    x_to_eval = (x_to_eval_raw-x_min)/(x_max-x_min)
    pred = grasas_model.predict(x_to_eval)
    print(f"{x_to_eval_raw} -> {pred*(y_max-y_min)+y_min}")

    alcohol_data = filtered_df.to_numpy()
    alcohol_data[alcohol_data == 'F'] = 0
    alcohol_data[alcohol_data == 'M'] = 1
    alcohol_data = np.array(alcohol_data, dtype=np.float64)

    y = alcohol_data[:,1]
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    x = alcohol_data[:,[0,2,3]]
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)

    x_final = (x-x_min)/(x_max-x_min)
    y_final = (y-y_min)/(y_max-y_min)

    alcohol_model = Sequential()
    alcohol_model.add(Dense(100, input_dim=x_final.shape[1], activation='tanh'))
    alcohol_model.add(Dense(50, activation='tanh'))
    alcohol_model.add(Dense(25, activation='tanh'))
    alcohol_model.add(Dense(5, activation='tanh'))
    alcohol_model.add(Dense(1, activation='tanh'))

    alcohol_model.compile(loss='mean_squared_error', optimizer='adam')
    alcohol_model.fit(x_final, y_final, batch_size=100, epochs=50000)
    # print(x_final[0])

    for i in range(x_final.shape[0]):
        x_to_eval = (np.array([x[i]])-x_min)/(x_max-x_min)
        pred = alcohol_model.predict(x_to_eval)
        pred = pred*(y_max-y_min)+y_min
        print(f"{x[i]}\t{y[i]}\t{pred[0,0]}\t{abs(y[i] - pred[0,0])}")

    x_to_eval_raw = np.array([[36.31, 1000, 0]])
    x_to_eval = (x_to_eval_raw-x_min)/(x_max-x_min)
    pred = alcohol_model.predict(x_to_eval)
    print(f"{x_to_eval_raw} -> {pred*(y_max-y_min)+y_min}")

    x_to_eval_raw = np.array([[41.01, 800, 0]])
    x_to_eval = (x_to_eval_raw-x_min)/(x_max-x_min)
    pred = alcohol_model.predict(x_to_eval)
    print(f"{x_to_eval_raw} -> {pred*(y_max-y_min)+y_min}")

    x_to_eval_raw = np.array([[27.08, 2054, 1]])
    x_to_eval = (x_to_eval_raw-x_min)/(x_max-x_min)
    pred = alcohol_model.predict(x_to_eval)
    print(f"{x_to_eval_raw} -> {pred*(y_max-y_min)+y_min}")

    # 2. Boxplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    
    ## Grasas saturadas
    ax1.boxplot(fixed_df[GRASAS_SATURADAS])
    ax1.set_title('Grasas Saturadas')

    ## Alcohol
    ax2.boxplot(fixed_df[ALCOHOL])
    ax2.set_title('Alcohol')

    ## Calorias
    ax3.boxplot(fixed_df[CALORIAS])
    ax3.set_title('Calorias')
    
    plt.show()

    # 3. Análisis de acuerdo al Sexo

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    male   = fixed_df.loc[fixed_df[SEXO] == 'M']
    female = fixed_df.loc[fixed_df[SEXO] == 'F']
    ## Grasas saturadas
    ax1.boxplot([male[GRASAS_SATURADAS], female[GRASAS_SATURADAS]])
    ax1.set_xticklabels(['M', 'F'], fontsize=12)
    ax1.set_title('Grasas Saturadas')

    ## Alcohol
    ax2.boxplot([male[ALCOHOL], female[ALCOHOL]])
    ax2.set_xticklabels(['M', 'F'], fontsize=12)
    ax2.set_title('Alcohol')

    ## Calorias
    ax3.boxplot([male[CALORIAS], female[CALORIAS]])
    ax3.set_xticklabels(['M', 'F'], fontsize=12)
    ax3.set_title('Calorias')

    plt.show()
    
    # 4. Análisis de Alcohol
    
    CATE1 = fixed_df.loc[fixed_df[CALORIAS] <= 1100]
    CATE2 = fixed_df.loc[(fixed_df[CALORIAS] > 1100) & (fixed_df[CALORIAS] <= 1700)]
    CATE3 = fixed_df.loc[fixed_df[CALORIAS] > 1700]

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    ## Alcohol
    ax1.boxplot([CATE1[ALCOHOL], CATE2[ALCOHOL], CATE3[ALCOHOL]])
    ax1.set_xticklabels(['CATE1', 'CATE2', 'CATE3'], fontsize=12)
    ax1.set_title('Alcohol')

    plt.show()