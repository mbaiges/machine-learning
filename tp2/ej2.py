import pandas as pd
import matplotlib.pyplot as plt
import copy

from utils import df_to_np
from knn import KNN

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
    plt.bar(stars, proms)
    plt.show()

def discretize(df):
    for idx, e in df.iterrows():
        e[TITLE_SENTIMENT] = 1 if e[TITLE_SENTIMENT] == 'positive' else 0

def cross_validation_run(df):
    pass
    ## Returns (best_x_train, best_t_train, best_x_test, best_t_test, median error for k, error for best case

# def cross_validation(knn, x, t, k=20):
#     total = len(bags)
#     amount = int(total/k)
#     acum = 0
#     new_total = total
#     max_ = math.inf
#     best_x_train, best_t_train, best_x_test, best_t_test  = None, None, None, None
#     if total % amount != 0:
#         new_total -= amount ## in order to add extra cases to last test set
#         k -= 1
#     for i in range(0, new_total, amount):
#         if(i + amount > new_total): ## in order to add extra cases to last test set
#             x_test = bags[i:]
#             t_test = t[i:]
#             x_train = bags[0:i]
#             t_train = t[0:i]
#         else: 
#             x_test = bags[i:i+amount]
#             t_test = t[i:i+amount]
#             x_train = bags[0:i] + bags[i+amount:]
#             t_train = t[0:i] + t[i+amount:]
#         if not (len(set(t_test)) == len(set(t_train)) == len(CATEGORIES)):
#             print("Skipping at K: {k}, with i: {i}")
#             continue
#         err = bayes.train(x_train, t_train, x_test, t_test)
#         if(err < max_):
#             max_ = err
#             best_x_train, best_t_train, best_x_test, best_t_test  = x_train, t_train, x_test, t_test
#         acum += err
#     return (best_x_train, best_t_train, best_x_test, best_t_test), acum / k, max_

if __name__ == '__main__':
    df = pd.read_csv(FILEPATH, sep=';')
    ej_a(df)
    discretize(df)
    print(df)
    x, t = df_to_np(df, [WORD_COUNT, TITLE_SENTIMENT, SENTIMENT_VALUE], STAR_RATING)
    knn = KNN()
    knn.load(x, t)
    knn.find([[3, 1, 2]])
