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
    
CATEGORIES = ['Salud', 'Entretenimiento', 'Economia', 'Deportes', 'Ciencia y Tecnologia', 'Internacional']
CATEGORIES_LWR = list(map(lambda c: c.lower(), CATEGORIES))

def select_categories(df):
    return df[df[CATEGORIA].isin(CATEGORIES)]
    # for index, row in df.iterrows():
    #     if str(row[CATEGORIA]).lower() in CATEGORIES:
    #         df.drop(index, inplace=True)

## Returns (shuffled bag, shuffled t), idx where test set starts, median error for k, error for best case
RANDOM = random.Random(111)
def cross_validation(bayes, bags, t, k=20):
    shuffled_idxs = [i for i in range(len(bags))]
    RANDOM.shuffle(shuffled_idxs)
    bags = [bags[i] for i in shuffled_idxs]
    t = [t[i] for i in shuffled_idxs]
    total = len(bags)
    amount = int(total/k)
    acum = 0
    new_total = total
    max_, test_idx = math.inf, 0
    if total % amount != 0:
        new_total -= amount ## in order to add extra cases to last test set
    for i in range(0, new_total, amount):
        if(i + amount > new_total): ## in order to add extra cases to last test set
            x_test = bags[i:]
            t_test = t[i:]
            x_train = bags[0:i]
            t_train = t[0:i]
        else: 
            x_test = bags[i:amount]
            t_test = t[i:amount]
            x_train = bags[0:i] + bags[i+amount:]
            t_train = t[0:i] + t[i+amount:]
        if len(set(t_test)) == len(set(t_train)) == len(CATEGORIES):
            print("Skipping")
            continue
        print(x_test)
        err = bayes.train(x_train, t_train, x_test, t_test)
        print(err)
        if(err < max_):
            max_ = err
            test_idx = i
        acum += err
    return (bags, t), test_idx, acum / k, max_

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
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.xticks(rotation=70)
    plt.show()

    return m

class Metrics():
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
    
    def accuracy(self):
        return (self.tp + self.fn) / (self.tp + self.tn + self.fp + self.fn)

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
        return "{" + f"acc: {self.accuracy()}, prec: {self.precision()}, tpr: {self.true_positive_rate()}, fpr: {self.false_positive_rate()}, f1s: {self.f1_score()}" + "}"


def metrics(bags, t, results, alpha=0):
    cats = CATEGORIES_LWR
    ret = {}

    # Gato, No Gato
    
    # --> Gato    <-- O_o No Gato o_O  => FN
    # --> Gato    <-- O_o Gato    o_O  => TP
    # --> No Gato <-- O_o Gato    o_O  => FP
    # --> No Gato <-- O_o No Gato o_O  => TN

    for cat in cats:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for idx, (pred_t, prob) in enumerate(results):
            ti = t[idx]
            if pred_t == cat and prob > alpha:
                if ti == cat:
                    tp += 1
                else: # ti != cat
                    fp += 1
            else:
                if ti == cat:
                    fn += 1
                else: # ti != cat
                    tn += 1

        ret[cat] = Metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    return ret


ROC_COLORS = ['r', 'g', 'b', 'y', 'o', 'p']
def roc(bayes, bags, t, left=0, right=1, step=0.2):
    categories = CATEGORIES_LWR
    
    r = {}
    xs = {}
    ys = {}

    for cat in categories:
        xs[cat] = []
        ys[cat] = []

    # Evaluating with different alphas
    for alpha in range(left, right, step):
        results = bayes.eval(bags)
        m = metrics(bags, t, results, alpha=alpha)
        for cat, mc in m.items():
            xs[cat].append(mc.fpr())
            ys[cat].append(mc.tpr())
        r[alpha] = m

    # Plotting
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    for i, cat in enumerate(categories):
        x = xs[cat]
        y = ys[cat]
        plt.plot(x, y, ROC_COLORS[i % len(categories)], label=cat)

    plt.show()

    return r


if __name__ == '__main__':

    df = pd.read_excel(FILEPATH)
    df = df.iloc[:, :4] ## Trim unnecesary columns

    df = select_categories(df)

    bags = build_bags(df)
    t = list(map(lambda e: str(e[1]).lower(), df[CATEGORIA].items()))

    # summary = []

    # for i, cat in enumerate(t):
    #     summary.append([bags[i], str(cat).lower()])

    # out_file = open("summary.json", "w")
    # json.dump(summary, out_file)
    
    bayes = Naive()
    bayes.train(bags, t, [], [])
    results = bayes.eval([bagify("Jair Bolsonaro recicla el discurso antiizquierdista en el inicio anticipado de su campaña por la reelección")])
    print(results)

    # Cross Validation
    CASES = 30
    best_err, t_idx, k = math.inf, 0, 0
    for i in range(2,CASES):
        print(i)
        (b_, t_), t_idx_, mean_err_, err_ = cross_validation(bayes, bags, t, i)
        print(f"Cross validation result for {i}: {mean_err_}, best case: {err_}")
        if(mean_err_ < best_err):
            best_err = mean_err_
            bags = b_
            t = t_
            t_idx = t_idx_
            k = i
    print(f"Cross validation FINAL best result for {k}: {best_err}")
    # With best configuration, we use testing lists from now on

    # bags_testing, t_testing = ...
    # results = bayes.eval(bags)
    results = bayes.eval(bags) ## Solución temporal

    # With testing set
    
    ## Draw confusion matrix
    confusion = confusion(bags, t, results)

    ## Metrics
    print("Metrics results:")
    metrics = metrics(bags, t, results)
    print(metrics)

    ## ROC curve
    roc(bayes, bags, t)