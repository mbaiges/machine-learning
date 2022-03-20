from matplotlib.font_manager import json_dump
import pandas as pd
import json

from bayes_set import Naive
from words import normalize

FILEPATH = 'Noticias_argentinas.xlsx'

FECHA     = 'fecta'
TITULAR   = 'titular'
FUENTE    = 'fuente'
CATEGORIA = 'categoria'

def build_set(df):
    results = []
    for i, row in df.iterrows():
        text = row[TITULAR]
        results.append(list(set(normalize(text))))
    return results

if __name__ == '__main__':

    df = pd.read_excel(FILEPATH)
    df = df.iloc[:, :4] ## Trim unnecesary columns

    sets = build_set(df)

    # # See ocurrences
    # master_bag = {}
    # for bag in bags:
    #     for k in bag:
    #         if k not in master_bag:
    #             master_bag[k] = 0
    #         master_bag[k] += bag[k]
    # print(master_bag)

    t = list(map(lambda e: e[1], df[CATEGORIA].items()))

    summary = []

    for i, cat in enumerate(t):
        summary.append([sets[i], str(cat).lower()])

    out_file = open("summary.json", "w")
    json.dump(summary, out_file)
    # bayes = Naive()
    # bayes.train(bags, t, [], [])
    # results = bayes.eval([bags[0]])
    # print(results)