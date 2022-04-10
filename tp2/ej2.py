from tkinter import font
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import random
import seaborn as sn

from utils import df_to_np
from knn import KNN
from models import Metrics

FILEPATH        = "reviews_sentiment.csv"

REVIEW_TITLE    = 'Review Title'
REVIEW_TEXT     = 'Review Text'
WORD_COUNT      = 'wordcount'
TITLE_SENTIMENT = 'titleSentiment'
TEXT_SENTIMENT  = 'textSentiment'
STAR_RATING     = "Star Rating"
SENTIMENT_VALUE = "sentimentValue"


def ej_a(df):
    total = len(df.columns)
    stars = [i for i in range(1, 6)]
    proms = [df.loc[df[STAR_RATING] == i][WORD_COUNT].sum() / total for i in stars]
    
    plt.rcParams.update({'font.size': 22})
    plt.bar(stars, proms)
    # fig, ax = plt.subplot()
    # ax.bar(stars, proms)
    # ax.set_xticklabels(x_ticks, rotation=0, fontsize=8)
    # ax.set_yticklabels(y_ticks, rotation=0, fontsize=8)
    plt.show()

def discretize(df):
    df[TITLE_SENTIMENT] = df[TITLE_SENTIMENT].apply(lambda sentiment: 1 if sentiment == "positive" else 0) #TODO: estamos pisando NaNs

CATEGORIES = [1,2,3,4,5]
def metrics(t, results):
    cats = CATEGORIES
    ret = {}

    for cat in cats:
        ret[cat] = Metrics()

    for idx, (k, result, probs, w) in enumerate(results):
            for cat in cats:
                ti = t[idx]
                if ti == cat:
                    if probs.get(cat) != None and probs[cat] == probs[result]:
                        ret[ti].tp += 1
                    else:
                        ret[ti].fn += 1
                else:
                    if probs.get(cat) != None and probs[cat] == probs[result]:
                        ret[cat].fp += 1
                    else:
                        ret[cat].tn += 1
    return ret

def multiple_cross_validations(x, t, from_k=5, to_k=10, step=1):
    all_results = []
    precision_avg = 0
    c = 0
    for k in range(from_k, to_k+step, step):
        (best_x_train, best_t_train, best_x_test, best_t_test), avg_err, max_ = cross_validation(KNN(), x, t, k)
        knn = KNN()
        knn.load(best_x_train, best_t_train)
        results = knn.find(best_x_test)
        all_results.append((best_t_test, results))
        m = metrics(best_t_test, results)
        tp = sum(list(map(lambda cat_m: cat_m.tp, m.values())))
        fp = sum(list(map(lambda cat_m: cat_m.fp, m.values())))
        precision = tp/(tp+fp)
        precision_avg += precision
        c += 1
    precision_avg /= c
    return precision_avg, all_results

RANDOM = random.Random(111)
def shuffle(x, t):
    shuffled_idxs = [i for i in range(x.shape[0])]
    RANDOM.shuffle(shuffled_idxs)
    return np.array([x[i] for i in shuffled_idxs]), np.array([t[i] for i in shuffled_idxs])

def cross_validation(knn, x, t, k=20):
    total = x.shape[0]
    amount = int(total/k)
    acum = 0
    new_total = total
    max_ = math.inf
    best_x_train, best_t_train, best_x_test, best_t_test  = None, None, None, None
    if total % amount != 0:
        new_total -= amount ## in order to add extra cases to last test set
        k -= 1
    for i in range(0, new_total, amount):
        if(i + amount > new_total): ## in order to add extra cases to last test set
            x_test = x[i:]
            t_test = t[i:]
            x_train = x[0:i]
            t_train = t[0:i]
        else: 
            x_test = x[i:i+amount]
            t_test = t[i:i+amount]
            x_train = np.concatenate((x[0:i], x[i+amount:]), axis=0)
            t_train = np.concatenate((t[0:i], t[i+amount:]), axis=0)
        if not (len(set(t_test)) == len(set(t_train)) == len(CATEGORIES)):
            print(f"Skipping at K: {k}, with i: {i}")
            continue
        knn.load(x_train, t_train)
        err = 0
        res = knn.find(x_test)
        for r, cat in zip(res, t_test):
            t_found = r[1]
            err += 1 if cat != t_found else 0
        if(err < max_):
            max_ = err
            best_x_train, best_t_train, best_x_test, best_t_test  = x_train, t_train, x_test, t_test
        acum += err
    return (best_x_train, best_t_train, best_x_test, best_t_test), acum / k, max_

def confusion(all_results):
    cats = CATEGORIES
    cats_len = len(cats)
    
    m = [[0 for i in range(cats_len)] for j in range(cats_len)]

    print(f'CROSS-RESULTS: {len(all_results)}')
    for (best_t_test, results) in all_results:
        print(f'RESULTS: {len(results)}')
        for idx, (k, pred_t, probs, closest) in enumerate(results):
            m[cats.index(best_t_test[idx])][cats.index(pred_t)] += 1

    m = np.array(m)

    df_cm = pd.DataFrame(m, cats, cats)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 22}, cmap=sn.cm.rocket_r, fmt='d') # font size
    plt.xticks(rotation=0)
    plt.show()

    return m

if __name__ == '__main__':
    df = pd.read_csv(FILEPATH, sep=';')
    ej_a(df)
    discretize(df)
    x, t = df_to_np(df, [WORD_COUNT, TITLE_SENTIMENT, SENTIMENT_VALUE], STAR_RATING)
    x, t = shuffle(x, t)
    print(f"------- KNN k = 5 -------")
    knn = KNN(k=5)
    knn.load(x, t)
    res = knn.find([[3, 1, 2]])
    print(res)
    print(f"------- KNN weighted k = 5 -------")
    knn = KNN(k=5, weighted=True)
    knn.load(x, t)
    res = knn.find([[3, 1, 2]])
    print(res)
    precision, all_results = multiple_cross_validations(x, t)
    print(precision)
    confusion(all_results)
