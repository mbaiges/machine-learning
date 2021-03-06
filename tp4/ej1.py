import pandas as pd
import numpy as np
import random
import math
from sklearn.cluster import k_means

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, 
                           accuracy_score, ConfusionMatrixDisplay)
import matplotlib.colors as mcl
import matplotlib.pyplot as plt

import utils
import models
from kmeans import KMeans
from hclustering import HClustering
from kohonen import Kohonen

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

NUM_VARS = [ATT_AGE,ATT_CAD_DUR,ATT_CHOLESTE]
NUM_VARS_MINUS_CAD_DUR = [ATT_AGE,ATT_CHOLESTE]
ALL_VARS = [ATT_SEX,ATT_AGE,ATT_CAD_DUR,ATT_CHOLESTE]

SIGDZ = "sigdz"

LOG_LONG_DELIMITER   = "------------------------------------"
LOG_MEDIUM_DELIMITER = "------------------------"
LOG_SHORT_DELIMITER  = "----------------"

###### Logs ######

def log_long(s: str) -> None:
    utils.log(s, LOG_LONG_DELIMITER)
def log_medium(s: str) -> None:
    utils.log(s, LOG_MEDIUM_DELIMITER)
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

def split_combinations(x: np.array, y: np.array, test_percentage: float):
    size = math.ceil(x.shape[0]*test_percentage)
    return math.ceil(x.shape[0]/size)

def random_split(x: np.array, y: np.array, test_percentage: float, index: int):
    size = math.ceil(x.shape[0]*test_percentage)
    top = (index+1)*size if (index+1)*size < x.shape[0] else (x.shape[0]-1)
    return (np.concatenate((x[0:index*size],x[top:x.shape[0]]), axis=0), np.concatenate((y[0:index*size],y[top:y.shape[0]]), axis=0)), (x[index*size:top], y[index*size:top])

def random_pick(x: np.array, y: np.array, size: int, seed: int):
    r = random.Random(seed)
    shuffled_idxs = [i for i in range(x.shape[0])]
    r.shuffle(shuffled_idxs)
    shuffled_idxs = shuffled_idxs[:size]
    return np.array([x[i] for i in shuffled_idxs]), np.array([y[i] for i in shuffled_idxs])

def analysis(df: pd.DataFrame) -> None:

    # Sex
    sexs = df[ATT_SEX].to_numpy()
    utils.bars(sexs, title=f"Variable '{ATT_SEX}'")

    # Ages
    ages = df[ATT_AGE].to_numpy()
    utils.hist(ages, title=f"Variable '{ATT_AGE}'")

    # Symptoms Duration
    durations = df[ATT_CAD_DUR].to_numpy()
    utils.hist(durations, title=f"Variable '{ATT_CAD_DUR}'") # Poisson? Pienso que tiene sentido

    # Cholesterol
    chols = df[ATT_CHOLESTE].to_numpy()
    filter(lambda c: c is not None, chols) # Filtering None
    utils.hist(chols, title=f"Variable '{ATT_CHOLESTE}'")

    # SIGDZ
    sigdz = df[ATT_SIGDZ].to_numpy()
    utils.bars(sigdz, title=f"Variable '{ATT_SIGDZ}'")

    # Arteries narrowing per sex
    ## Male
    sigdz = df[df[ATT_SEX] == 0][ATT_SIGDZ].to_numpy()
    utils.bars(sigdz, title=f"Variable '{ATT_SIGDZ}', para sexo Masculino")

    ## Female
    sigdz = df[df[ATT_SEX] == 1][ATT_SIGDZ].to_numpy()
    utils.bars(sigdz, title=f"Variable '{ATT_SIGDZ}', para sexo Femenino")

def _compare_kmeans_initializations(x: np.array, y: np.array, kmeans: dict):
    init_algs = ["random", "samples", "distant"]
    ks = [3, 4, 5]
    for k in ks:
        for init_alg in init_algs:
            km = KMeans(k=k, init_alg=init_alg, seed=seed)
            it = km.train(x, iterations=kmeans["iterations"], show_loading_bar=False)
            log(f"Iterations: {it}")
            log(f"Clusters: {km.clusters}")
            clustered = km.clusterize(x)
            print(f"Clustered: {list(map(lambda c: len(c), clustered))}")
            plt.figure()
            plt.title(f"K Means ({k}) con inicializaci??n '{init_alg}'")
            km.plot2d(x, labels=['Edad', 'Colesterol'])

            plt.figure()
            plt.title(f"K Means ({k}) con inicializaci??n '{init_alg}' y porcentajes")
            km.plot2d(x, labels=['Edad', 'Colesterol'], y=y)
            # km.plot3d(x, labels=['Age', 'Symptoms Duration', 'Cholesterol'])
        plt.show()
    

def cross_validation(x, t, k=20, logit_max_iter=100):
    std_scaler  = StandardScaler()
    std_x       = std_scaler.fit_transform(X=x)
    _std_x       = sm.add_constant(std_x)
    total = x.shape[0]
    amount = int(total/k)
    acum = 0
    new_total = total
    max_ = math.inf
    best_x_train, best_t_train, best_x_test, best_t_test  = None, None, None, None
    counter = 0
    if total % amount != 0:
        new_total -= amount ## in order to add extra cases to last test set
        k -= 1
    for i in range(0, new_total, amount):
        if(i + amount > new_total): ## in order to add extra cases to last test set
            std_x_test = _std_x[i:]
            std_x_train = _std_x[0:i]
            x_test = x[i:]
            t_test = t[i:]
            x_train = x[0:i]
            t_train = t[0:i]
            
        else: 
            std_x_test = _std_x[i:i+amount]
            std_x_train = np.concatenate((_std_x[0:i], _std_x[i+amount:]), axis=0)
            x_test = x[i:i+amount]
            t_test = t[i:i+amount]
            x_train = np.concatenate((x[0:i], x[i+amount:]), axis=0)
            t_train = np.concatenate((t[0:i], t[i+amount:]), axis=0)
        # if not (len(set(t_test)) == len(set(t_train)) == len(CATEGORIES)):
        #     # print(f"Skipping at K: {k}, with i: {i}")
        #     continue
        counter += 1
        logit_mod   = sm.Logit(endog = t_train, exog=std_x_train)
        logit_res   = logit_mod.fit(method='newton', maxiter=logit_max_iter, disp=0)
        err = 0
        res = logit_res.predict(std_x_test)
        for r, cat in zip(res, t_test):
            r = round(r)
            err += 1 if cat != r else 0
        if(err < max_):
            max_ = err / x_test.shape[0]
            best_x_train, best_t_train, best_x_test, best_t_test  = x_train, t_train, x_test, t_test
        acum += err
    return (best_x_train, best_t_train, best_x_test, best_t_test), acum / counter, max_ 

def _compare_hclustering(x: np.array, hclustering: dict):
    criterias = ['max','min','mean','center']
    max_clusters = hclustering['max_clusters']
    for criteria in criterias:
        hc = HClustering(criteria=criteria)
        hc.train(x)
        for tr in ['none','lastp']:
            plt.figure()
            plt.title(f"Agrupamiento jer??rquico con criterio '{criteria}'{(' truncado a ' + str(max_clusters) + ' clusters') if tr == 'lastp' else ''}")
            hc.dendrogram(p=max_clusters, truncate_mode=tr)
    plt.show()

def _e_analysis_plot2d_points(x: np.array, y: np.array, labels: list):
    colors = list(mcl.TABLEAU_COLORS.values())
    print(set(y))
    for val in set(y):
        l = ">= 75% Estr. Arterial" if val == 1 else "< 75% Estr. Arterial"
        c = colors[val % len(colors)]
        idxs = np.argwhere(y[:] == val)[:,0]
        rx = x[idxs]
        plt.scatter(rx[:,0], rx[:,1], c=c, label=l, alpha=0.4)
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

def e_analysis(x: np.array, y: np.array, title):
    plt.figure()
    plt.title(f"[{title}] Ejemplos de cada clase")
    _e_analysis_plot2d_points(x, y, labels=["Edad", "Colesterol"])
    plt.show()
    
def unsupervised(x: np.array, y: np.array, kmeans: dict, hclustering: dict, kohonen: dict):
    ## KMeans
    log_short("K-Means")
    km = None # to avoid defining yet
    # _compare_kmeans_initializations(x, y, kmeans)

    ## Hierarchical Clustering
    log_short("Hierarchical Clustering")
    hc = None # to avoid defining yet
    # _compare_hclustering(x, hclustering)

    ## Kohonen
    log_short("Kohonen")
    ko = Kohonen(x, grid_dimension=kohonen["grid_dimension"], radius=kohonen["radius"], input_weights=kohonen["input_weights"], learning_rate=kohonen["learning_rate"])
    # ko.train(kohonen["epochs"])
    # plt.show()

    return km, hc, ko

###### Exercise Points ######

def a(x: np.array, y: np.array):
    
    iterations = 100
    k_limits = (55, 65)
    k_step = 5
    logit_max_iter = 100

    max__ = math.inf
    best_k = None
    best_x_train, best_t_train, best_x_test, best_t_test = None, None, None, None

    loading_bar = utils.LoadingBar()
    loading_bar.init()
    it = 0
    total_iterations = int(iterations*((k_limits[1]-k_limits[0])/k_step))
    for i in range(iterations):
        sx, sy = shuffle(x, y, seed+i)
        for k in range(k_limits[0], k_limits[1]+k_step, k_step):
            loading_bar.update(1.0*it/total_iterations)
            (_best_x_train, _best_t_train, _best_x_test, _best_t_test), error_pctg, max_ = cross_validation(x=sx, t=sy, k=k, logit_max_iter=logit_max_iter)
            if max_ < max__:
                best_x_train, best_t_train, best_x_test, best_t_test = _best_x_train, _best_t_train, _best_x_test, _best_t_test
                best_k = k
                max__ = max_
            it += 1
    loading_bar.end()

    log_medium(f"Cross Validation")
    log_short(f"Best x test shape = {best_x_test.shape}")
    log_short(f"Best k = {best_k}")
    log_short(f"Best error: {max__}")
    return (best_x_train, best_t_train), (best_x_test, best_t_test)


def b(tr_x: np.array, tr_y: np.array, ts_x: np.array, ts_y: np.array):
    s_tr_x      = np.delete(tr_x, 0, 1)
    s_ts_x      = np.delete(ts_x, 0, 1)
    
    std_scaler  = StandardScaler()
    std_x       = std_scaler.fit_transform(X=s_tr_x)
    std_x       = sm.add_constant(std_x)
    logit_mod   = sm.Logit(endog=tr_y, exog=std_x)
    logit_res   = logit_mod.fit(method='newton')
    print(logit_res.summary())  ##TODO: x2 (cad dur)no sirve, por coef chico y P > |z|

    # Borramos la columna de la duracion 
    std_x       = np.delete(std_x, 2, 1)
    logit_mod   = sm.Logit(endog = tr_y, exog=std_x)
    logit_res   = logit_mod.fit(method='newton')
    print(logit_res.summary()) 

    # TODO: Conseguir Odds
    # params = logit_res.params
    # conf = logit_res.conf_int()
    # print(conf)
    # conf['Odds Ratio'] = params
    # conf.columns = ['5%', '95%', 'Odds Ratio']
    # print(f"Odds Radio: {np.exp(conf)}")

    std_ts_x    = std_scaler.transform(X=s_ts_x)
    std_ts_x    = sm.add_constant(std_ts_x)
    std_ts_x    = np.delete(std_ts_x, 2, 1)
    yhat        = logit_res.predict(std_ts_x)
    prediction  = list(map(round, yhat))

    # confusion matrix
    cm = confusion_matrix(ts_y, prediction) 
    print ("Confusion Matrix : \n", cm)
    metrics = models.Metrics(tp=cm[1][1], tn=cm[0][0], fp=cm[1][0], fn=cm[0][1])
    log_long(metrics)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot()
    plt.show()
    # accuracy score of the model
    print('Test accuracy = ', accuracy_score(ts_y, prediction))

    return logit_res, std_scaler

def c(tr_x: np.array, tr_y: np.array, ts_x: np.array, ts_y: np.array):
    tr_idxs = np.argwhere(tr_x[:,0] == 1)[:,0]
    ts_idxs = np.argwhere(ts_x[:,0] == 1)[:,0]

    tr_not_man_idxs = np.argwhere(tr_x[:,0] != 1)[:,0]
    ts_not_man_idxs = np.argwhere(ts_x[:,0] != 1)[:,0]
    
    log_short("MAN")
    m_tr_x, m_tr_y, m_ts_x, m_ts_y = tr_x[tr_idxs], tr_y[tr_idxs], ts_x[ts_idxs], ts_y[ts_idxs]
    print(m_tr_x, m_tr_y, m_ts_x, m_ts_y)
    b(m_tr_x, m_tr_y, m_ts_x, m_ts_y)

    log_short("WOMAN")
    w_tr_x, w_tr_y, w_ts_x, w_ts_y = tr_x[tr_not_man_idxs], tr_y[tr_not_man_idxs], ts_x[ts_not_man_idxs], ts_y[ts_not_man_idxs]
    b(w_tr_x, w_tr_y, w_ts_x, w_ts_y)

def d(logit_res, std_scaler):
    print(logit_res.summary())
    
    c_const = 0.7163
    c_age   = 0.5443
    c_chol  = 0.3302

    age  = 60
    dur  = 2
    chol = 199

    stds = std_scaler.transform(np.array([[age, dur, chol]]))
    std_age, std_dur, std_chol = stds[0,0], stds[0,1], stds[0,2]
    logit = c_const + c_age * std_age + c_chol * std_chol
    odds  = math.exp(logit)
    log(f"Odds: {odds}")
    # p/(1-p) = e^(c_const + c_age * age + c_chol * chol) = odds
    # p     =  odds * (1 - p)
    # (1 + odds) * p = odds
    p = odds/(1+odds)
    log(f"Probability: {p}")
    
def _e(x: np.array, y: np.array, title: str="General"):
    e_analysis(x, y, title=f"Dataset Completo - {title}")

    pctg = 0.4
    n = math.floor(x.shape[0]*pctg)
    rx, ry = random_pick(x, y, n, seed)

    # e_analysis(rx, ry, title=f"Random Subset - {title}")

    kmeans = {
        "k":                4,
        "init_alg":         "distant",
        "iterations":       1000,
        "show_loading_bar": True
    }
    hclustering = {
        "max_clusters": 10
    }
    kohonen={
        "grid_dimension": 7,
        "radius": (False, 2),
        "input_weights": True,
        "learning_rate": 1,
        "epochs": 1000
    }
    km, hc, ko = unsupervised(
        x, 
        y, 
        kmeans=kmeans, 
        hclustering=hclustering, 
        kohonen=kohonen
    )

    # Sick people
    # log_medium("Sick people")
    # sick_idxs = np.argwhere(ry[:] == 1)[:,0]
    # sick_x    = std_x[sick_idxs]
    # log(sick_idxs.shape)
    
    # km, hc, ko = unsupervised(
    #     sick_x, 
    #     kmeans=kmeans, 
    #     hclustering={
    #         "param": "asd"
    #     }, 
    #     kohonen=kohonen
    # )

    # # Non sick people
    # log_medium("Non sick people")
    # non_sick_idxs = np.argwhere(ry[:] == 0)[:,0]
    # non_sick_x    = std_x[non_sick_idxs]
    # log(non_sick_x.shape)

    # km, hc, ko = unsupervised(
    #     non_sick_x, 
    #     kmeans=kmeans, 
    #     hclustering={
    #         "param": "asd"
    #     }, 
    #     kohonen=kohonen
    # )

def e(df: pd.DataFrame):
    # Male
    log_medium("MALE")
    mx = df[df[ATT_SEX] == 0][NUM_VARS_MINUS_CAD_DUR].to_numpy()
    my = df[df[ATT_SEX] == 0][ATT_SIGDZ].to_numpy()
    _e(mx, my, title="Masculino")

    # Female
    log_medium("FEMALE")
    wx = df[df[ATT_SEX] == 1][NUM_VARS_MINUS_CAD_DUR].to_numpy()
    wy = df[df[ATT_SEX] == 1][ATT_SIGDZ].to_numpy()
    _e(wx, wy, title="Femenino")

    # General case
    log_medium("BOTH")
    gx = df[NUM_VARS_MINUS_CAD_DUR].to_numpy()
    gy = df[ATT_SIGDZ].to_numpy()
    _e(gx, gy, title="General")
    

###### Main ######

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=';')
    log_long(f"{df.size} rows loaded")

    # Analysis
    adf = df.astype(object).replace(np.nan, None)
    # analysis(adf)

    df = df.dropna().reset_index(drop=True) #TODO: que hacemos con esto, media, neurona, knn

    x = df[ALL_VARS].to_numpy()
    y = df[ATT_SIGDZ].to_numpy()
    print(f"X SHAPE: {x.shape}")
    print(f"Y SHAPE: {y.shape}")
    # Exercise a
    log_long("Exercise a")
    # (tr_x, tr_y), (ts_x, ts_y) = a(x, y)

    # (best_x_train, best_t_train, best_x_test, best_t_test), error_pctg, max_ = cross_validation(x=df[NUM_VARS].to_numpy(), t=y)
    # print(f"BEST X TEST SHAPE: {best_x_test.shape}")
    # print(f"MAX error is: {max_}")
    # Exercise b
    log_long("Exercise b")
    # logit_res, std_scaler = b(tr_x, tr_y, ts_x, ts_y)

    # Exercise c
    log_long("Exercise c")
    # c(tr_x, tr_y, ts_x, ts_y) # TODO: Preguntar si lo que entendimos es lo que hab??a que hacer (hacer 2 an??lisis, uno de cada)

    # Exercise d
    log_long("Exercise d")
    # d(logit_res, std_scaler)
    
    # Exercise e
    log_long("Exercise e")
    e(df)