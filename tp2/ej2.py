from tkinter import font
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import random
import seaborn as sn

from utils import df_to_np, mode
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

def dist_analysis(df):
    stars = list(range(1, 6))
    
    # Entries per Stars Rating
    counts = [df.loc[df[STAR_RATING] == i].shape[0] for i in stars]
    plt.rcParams.update({'font.size': 22})
    plt.title("Entries per number of stars rating")
    plt.bar(stars, counts)
    for s, data in zip(stars, counts):
        plt.text(x=s ,y=data+0.5 ,s=f"{data}" ,fontdict=dict(fontsize=20))
    plt.show()

    # Sentiment per Stars Rating
    for s in stars:
        counts = {}
        regs = df.loc[df[STAR_RATING] == s]
        for idx, r in regs.iterrows():
            title_sentiment = r[TITLE_SENTIMENT]
            title_sentiment = 'nan' if pd.isna(title_sentiment) else str(title_sentiment).lower()
            if title_sentiment not in counts:
                counts[title_sentiment] = 0
            counts[title_sentiment] += 1

        counts = dict(sorted(counts.items()))
        x = counts.keys()
        y = counts.values()
        plt.rcParams.update({'font.size': 22})
        plt.title(f"Entries with {s} stars by each title sentiment value")
        plt.bar(x, y)
        for v, data in zip(x, y):
            plt.text(x=v ,y=data ,s=f"{data}" ,fontdict=dict(fontsize=20))
        plt.show()

    pass

def sanitize(df):
    print(f"BEFORE - {df[df[TITLE_SENTIMENT].notnull()].shape[0]} not null")

    # Replace NaN
    stars = list(range(1,6))
    text_sentiments = set(df[TEXT_SENTIMENT])
    mode_by_stars_and_text_sent = {}
    for s in stars:
        regs = df.loc[df[STAR_RATING] == s]
        regs = regs[regs[TITLE_SENTIMENT].notnull()]
        for text_sent in text_sentiments:
            rs = regs[regs[TEXT_SENTIMENT] == text_sent]
            mo = mode(np.array(list(map(lambda i: i[1][TITLE_SENTIMENT], rs.iterrows()))))
            mo = mo if mo is not None else text_sent
            mode_by_stars_and_text_sent[(s, text_sent)] = mo
    
    for idx, row in df.loc[df[TITLE_SENTIMENT].isnull()].iterrows():
        s = row[STAR_RATING]
        text_sent = row[TEXT_SENTIMENT]
        df.loc[idx, TITLE_SENTIMENT] = mode_by_stars_and_text_sent[(s, text_sent)]

    print(f"AFTER - {df[df[TITLE_SENTIMENT].notnull()].shape[0]} not null")

def ej_a(df):
    stars = list(range(1, 6))
    entries_per_stars = [df.loc[df[STAR_RATING] == i].shape[0] for i in stars]
    proms = [df.loc[df[STAR_RATING] == i][WORD_COUNT].sum() / entries_per_stars[i-1] for i in stars]
    
    plt.rcParams.update({'font.size': 22})
    plt.title("Word count average per number of stars rating")
    plt.bar(stars, proms)
    for stars, data in zip(stars, proms):
        plt.text(x=stars ,y=data ,s=f"{data:.2f}" ,fontdict=dict(fontsize=20))
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

def multiple_cross_validations(x, t, from_k=5, to_k=10, step=1, knn_k = 5, weighted=False):
    all_results = []
    precision_avg = 0
    c = 0
    for k in range(from_k, to_k+step, step):
        (best_x_train, best_t_train, best_x_test, best_t_test), avg_err, max_ = cross_validation(KNN(knn_k, weighted=weighted), x, t, k)
        knn = KNN(knn_k, weighted=weighted)
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
    counter = 0
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
            # print(f"Skipping at K: {k}, with i: {i}")
            continue
        counter += 1
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
    return (best_x_train, best_t_train, best_x_test, best_t_test), acum / counter, max_

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
    sn.set(font_scale=1.6) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 22}, cmap=sn.cm.rocket_r, fmt='d') # font size
    plt.xticks(rotation=0)
    plt.show()

    return m

def compare_weighted(x, t, min_k=1, max_k=200, step_k=4, iterations_per_k=4) -> tuple:
    # non_weighted
    ks = list(range(min_k, max_k + step_k, step_k))

    non_weighted_precisions = []
    weighted_precisions     = []
    best = {}
    for knn_k in ks:
        print(f"--> Comparing K = {knn_k}")
        nw_avg = 0
        w_avg = 0
        for it in range(0, iterations_per_k):
            x, t = shuffle(x, t)
            nw_avg += multiple_cross_validations(x, t, knn_k=knn_k, weighted=False)[0]
            w_avg  += multiple_cross_validations(x, t, knn_k=knn_k, weighted=True)[0]
        nw_precision = nw_avg/iterations_per_k
        w_precision = w_avg/iterations_per_k
        non_weighted_precisions.append(nw_precision)
        weighted_precisions.append(w_precision)
        if not best or w_precision > best['w_precision']:
            best['k'] = knn_k
            best['nw_precision'] = nw_precision
            best['w_precision'] = w_precision

    print(f"---\nBest is {best['k']}\nnw_precision: {best['nw_precision']}\nw_precision: {best['w_precision']}\n---")
    plt.plot(ks, non_weighted_precisions, color='r', label='KNN')
    plt.plot(ks, weighted_precisions, color='g', label='KNN con Distancias Ponderadas')

    plt.xlabel("k")
    plt.ylabel("precision")
    plt.title("Precision en funcion de k")
    plt.legend()
    plt.show()
    return best['k'], best['nw_precision'], best['w_precision']
    
if __name__ == '__main__':
    df = pd.read_csv(FILEPATH, sep=';')
    print(f"Loaded {df.shape[0]} rows")
    # print("Before Sanitizing")
    # dist_analysis(df)
    ej_a(df)
    sanitize(df)
    # print("After sanitizing")
    # dist_analysis(df)
    discretize(df)

    x, t = df_to_np(df, [WORD_COUNT, TITLE_SENTIMENT, SENTIMENT_VALUE], STAR_RATING)
    x, t = shuffle(x, t)
    plt.rcParams.update({'font.size': 22})
    
    k, nw_precision, w_precision = compare_weighted(x, t, min_k=1, max_k=200, step_k=5, iterations_per_k=4)
    knn_k, nw_precision, w_precision = compare_weighted(x, t, min_k=k-5 if k > 5 else 1, max_k=k+5, step_k=1, iterations_per_k=4)
    print(f"------- KNN k = {knn_k} -------")
    knn = KNN(k=knn_k, weighted=False)
    knn.load(x, t)
    res = knn.find([[3, 1, 2]])
    print(res)
    precision, all_results = multiple_cross_validations(x, t, knn_k=knn_k, weighted=False)
    print(precision)
    confusion(all_results)
    print(f"------- KNN weighted k = {knn_k} -------")
    knn = KNN(k=knn_k, weighted=True)
    knn.load(x, t)
    res = knn.find([[3, 1, 2]])
    print(res)
    precision, all_results = multiple_cross_validations(x, t, knn_k=knn_k, weighted=True)
    print(precision)
    confusion(all_results)

    
