from hashlib import new
from random import Random
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import math

from bayes_2 import Naive
from words import normalize

FILEPATH = 'Noticias_argentinas.xlsx'

FECHA     = 'fecta'
TITULAR   = 'titular'
FUENTE    = 'fuente'
CATEGORIA = 'categoria'

def build_bags(df):
    results = []
    for i, row in df.iterrows():
        results.append(bagify(row[TITULAR]))
    return results


def bagify(text):
    bag = {}
    words = normalize(text)
    for w in words:
        if w not in bag:
            bag[w] = 0
        bag[w] += 1
    return bag
    
CATEGORIES = ['Salud', 'Entretenimiento', 'Economia', 'Deportes', 'Ciencia y Tecnologia', 'Internacional', 'Nacional']
CATEGORIES_LWR = list(map(lambda c: c.lower(), CATEGORIES))

def select_categories(df):
    return df[df[CATEGORIA].isin(CATEGORIES)]
    # for index, row in df.iterrows():
    #     if str(row[CATEGORIA]).lower() in CATEGORIES:
    #         df.drop(index, inplace=True)

## Returns (best_x_train, best_t_train, best_x_test, best_t_test, median error for k, error for best case
def cross_validation(bayes, bags, t, k=20):
    total = len(bags)
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
            x_test = bags[i:]
            t_test = t[i:]
            x_train = bags[0:i]
            t_train = t[0:i]
        else: 
            x_test = bags[i:i+amount]
            t_test = t[i:i+amount]
            x_train = bags[0:i] + bags[i+amount:]
            t_train = t[0:i] + t[i+amount:]
        if not (len(set(t_test)) == len(set(t_train)) == len(CATEGORIES)):
            print("Skipping at K: {k}, with i: {i}")
            continue
        err = bayes.train(x_train, t_train, x_test, t_test)
        if(err < max_):
            max_ = err
            best_x_train, best_t_train, best_x_test, best_t_test  = x_train, t_train, x_test, t_test
        acum += err
    return (best_x_train, best_t_train, best_x_test, best_t_test), acum / k, max_

def confusion(bags, t, results):
    cats = CATEGORIES_LWR
    cats_len = len(cats)
    
    m = [[0 for i in range(cats_len)] for j in range(cats_len)]

    for idx, (pred_t, prob) in enumerate(results):
        m[cats.index(t[idx])][cats.index(pred_t)] += 1

    m = np.array(m)

    df_cm = pd.DataFrame(m, cats, cats)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=sn.cm.rocket_r, fmt='d') # font size
    plt.xticks(rotation=0)
    plt.show()

    return m

class Metrics():
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
    
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0

    def true_positive_rate(self):
        return self.tp / (self.tp + self.fn)

    def false_positive_rate(self):
        return self.fp / (self.fp + self.tn)
    
    def f1_score(self):
        return 2 * self.precision() * self.true_positive_rate() / (self.precision() + self.true_positive_rate())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{" + f"acc: {self.accuracy():.3f}, prec: {self.precision():.3f}, tpr: {self.true_positive_rate():.3f}, fpr: {self.false_positive_rate():.3f}, f1s: {self.f1_score():.3f}" + "}"


def metrics(bags, t, results, threshold=0):
    cats = CATEGORIES_LWR
    ret = {}

    for cat in cats:
        ret[cat] = Metrics()

    for idx, result in enumerate(results):
            for cat in cats:
                prob = Naive.get_prob_for_class(result, cat)
                ti = t[idx]
                if ti == cat:
                    if prob >= threshold:
                        ret[ti].tp += 1
                    else:
                        ret[ti].fn += 1
                else:
                    if prob >= threshold:
                        ret[cat].fp += 1
                    else:
                        ret[cat].tn += 1
    return ret

def auc(x, y):
    x, y = x.copy(), y.copy()
    x.reverse()
    y.reverse()
    a = 0
    for i in range(len(x)-1):
        a += (x[i+1] - x[i])*(y[i] + (y[i+1] - y[i])/2)
    return a

ROC_COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'black', 'purple']
def roc(bags, t, results, start=0, stop=1, step=0.2):
    categories = CATEGORIES_LWR
    
    r = {}
    xs = {}
    ys = {}

    for cat in categories:
        xs[cat] = []
        ys[cat] = []

    # Evaluating with different alphas
    thresholds = np.arange(start, stop+step, step)
    for threshold in thresholds:
        m = metrics(bags, t, results, threshold=threshold)
        for cat, mc in m.items():
            xs[cat].append(mc.false_positive_rate())
            ys[cat].append(mc.true_positive_rate())
        r[threshold] = m

    # for cat in categories:
    #     xs[cat].append(0)
    #     ys[cat].append(0)

    # Plotting
    fig, ax = plt.subplots()

    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.xlabel('Tasa de Falsos Positivos')

    ax.plot(np.array([0, 1]), np.array([0, 1]), 'k--')

    for i, cat in enumerate(categories):
        x = xs[cat]
        y = ys[cat]
        a = auc(x,y)
        print(f'AUC({cat}) = {a:.3f}')
        # print(f'{cat} - x: {x}, y: {y}')
        ax.scatter(x, y, c=ROC_COLORS[i % len(categories)])
        # for i, alpha in enumerate(alphas):
        #     ax.annotate(f'{alpha:.1f}', (x[i], y[i]))
        ax.plot(x, y, ROC_COLORS[i % len(categories)], label=f'{cat} (AUC={a:.3f})')
        ax.legend()

    plt.show()

    return r

RANDOM = random.Random(111)
def shuffle(bags, t):
    shuffled_idxs = [i for i in range(len(bags))]
    RANDOM.shuffle(shuffled_idxs)
    return [bags[i] for i in shuffled_idxs], [t[i] for i in shuffled_idxs]

if __name__ == '__main__':

    df = pd.read_excel(FILEPATH)
    df = df.iloc[:, :4] ## Trim unnecesary columns

    df = select_categories(df)

    bags = build_bags(df)
    total_words = 0
    s = set()
    for b in bags:
        for w in b:
            total_words += 1
        s.add(w)
    print(f"Analyzing {len(bags)} news, containing a total of {total_words} words ({len(s)} distinct)")
    t = list(map(lambda e: str(e[1]).lower(), df[CATEGORIA].items()))

    # summary = []

    # for i, cat in enumerate(t):
    #     summary.append([bags[i], str(cat).lower()])

    # out_file = open("summary.json", "w")
    # json.dump(summary, out_file)

    # Cross Validation
    bayes = Naive()

    MAX_K = 35
    bags, t = shuffle(bags, t)
    best_err, k = math.inf, 0
    best_bags_train, best_t_train, best_bags_test, best_t_test = bags, t, [], []
    for i in range(10,MAX_K+1,5):
        best_div, mean_err_, err_ = cross_validation(bayes, bags, t, i)
        print(f"Cross validation result for {i}: {mean_err_}, best case: {err_}")
        if(mean_err_ < best_err):
            (best_bags_train, best_t_train, best_bags_test, best_t_test) = best_div
            best_err = mean_err_
            k = i
    print(f"Cross validation FINAL result for {k}: {best_err}")
    # With best configuration, we use testing lists from now on
    best_bayes = Naive()
    best_bayes.train(best_bags_train, best_t_train, best_bags_test, best_t_test)

    # Random evals
    results = best_bayes.eval([bagify("Pol??mica por la SUBE. Faltan tarjetas y se cobran precios exorbitantes: donde recomiendan comprarla")])
    print(Naive.get_most_probable_class(results))
    results = best_bayes.eval(best_bags_test)

    # With testing set
    
    ## Draw confusion matrix
    confusion = confusion(best_bags_test, best_t_test, Naive.get_most_probable_class(results))

    ## Metrics
    print("Metrics results:")
    met = metrics(best_bags_test, best_t_test, results, threshold=0.7)
    print(met)

    # ## ROC curve
    roc(best_bags_test, best_t_test, results, step=0.1)