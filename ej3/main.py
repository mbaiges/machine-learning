import pandas as pd
from sklearn import linear_model, model_selection
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import math

import utils
from models import Metrics

seed = 6271834

FILEPATH = 'Advertising.csv'

CSV_HEADER = ["TV","Radio","Newspaper","Sales"]

def model_diag(y_test, y_pred):
    n = y_test.shape[0]
    rse = math.sqrt(1/(n-2)) * np.linalg.norm(y_test - y_pred)
    residue = y_pred - y_test
    std_residue = residue / rse
    print(f"residue mean: {np.mean(std_residue)}, std_dev: {np.std(std_residue)}, rse: {rse}")
    plt.hist(std_residue)
    plt.show()

def stats(mat, classes):
    st = {}
    for i in range(mat.shape[0]):
        clazz = classes[i]
        s = {}
        total = np.sum(mat)
        tp = mat[i,i]
        fn = np.sum(mat[i,:]) - tp
        fp = np.sum(mat[:,i]) - tp
        tn = total - tp - fn - fp
        st[clazz] = Metrics(tp, tn, fp, fn)
    acc = sum([m.accuracy() for m in st.values()]) / mat.shape[0] # promediamos accuracy
    return acc, st

def simple_regression(df, division_iterations, iterations):
    y = df[CSV_HEADER[-1]].to_numpy()
    bins = utils.bins(y, alg='dist', options={'dist': 6})
    for field in CSV_HEADER[:-1]:
        x = np.array([df[field].to_numpy()]).T

        results = []
        for i in range(iterations):
            best_accuracy = -math.inf
            best_stats = None
            best_cfg = None
            for it in range(division_iterations):
                x_train, x_test, y_train, y_test = cfg = model_selection.train_test_split(x, y, test_size=0.4, random_state=seed+it)

                linear = linear_model.LinearRegression()
                linear.fit(x_train, y_train)
                y_pred = linear.predict(x_test)

                # Binning
                y_test_binned = utils.arr_binning(bins, y_test)
                y_pred_binned = utils.arr_binning(bins, y_pred)
                test_bins = set(y_test_binned)
                pred_bins = set(y_pred_binned)
                used_bins = test_bins if len(test_bins) >= len(pred_bins) else pred_bins

                # Confusion Matrix
                mat = confusion_matrix(y_test_binned, y_pred_binned)
                acc, s = stats(mat, list(sorted(used_bins)))
                if acc >= best_accuracy:
                    best_accuracy = acc
                    best_cfg = cfg
                    best_stats = s

            results.append({'acc': best_accuracy, 'stats': best_stats, 'cfg': best_cfg, 'model': linear})


        # Stats average
        average_acc = 0
        average_stats = {}

        best = {'acc': -math.inf}

        for m in results:
            acc = m['acc']
            s = m['stats']

            average_acc += acc
            for k, v in s.items():
                curr_v, c = average_stats.get(k, (Metrics(), 0))
                
                curr_v.tp += v.tp
                curr_v.tn += v.tn
                curr_v.fp += v.fp
                curr_v.fn += v.fn
                average_stats[k] = (curr_v, c + 1)

            if acc >= best['acc']:
                best = m 

        average_acc /= len(results)
        for k in average_stats.keys():
            s, c = average_stats[k]
            s.tp /= c
            s.tn /= c
            s.fp /= c
            s.fn /= c

        print(f"Stats average based on {iterations} iterations:")
        print(f"Average accuracy: {average_acc:.3f}")
        print(f"Average stats:")
        for k, v in average_stats.items():
            print(f"  {k} -> {v}")

        # Best case
        x_train, x_test, y_train, y_test = cfg = best['cfg']
        y_pred = best['model'].predict(x_test)
        print(f"Best accuracy: {best['acc']:.3f}")

        # Binning
        y_test_binned = utils.arr_binning(bins, y_test)
        y_pred_binned = utils.arr_binning(bins, y_pred)
        test_bins = set(y_test_binned)
        pred_bins = set(y_pred_binned)
        used_bins = test_bins if len(test_bins) >= len(pred_bins) else pred_bins

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)

        model_diag(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_binned, y_pred_binned), display_labels=used_bins)
        disp.plot()
        plt.show()
        print(linear.coef_)

def multiple_regression(df, division_iterations, iterations):
    y = df[CSV_HEADER[-1]].to_numpy()
    x = df[CSV_HEADER[:-1]].to_numpy()

    bins = utils.bins(y, alg='dist', options={'dist': 6})

    results = []
    for i in range(iterations):
        best_accuracy = -math.inf
        best_stats = None
        best_cfg = None
        for it in range(division_iterations):
            x_train, x_test, y_train, y_test = cfg = model_selection.train_test_split(x, y, test_size=0.4, random_state=seed+it)

            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)
            y_pred = linear.predict(x_test)

            # Binning
            y_test_binned = utils.arr_binning(bins, y_test)
            y_pred_binned = utils.arr_binning(bins, y_pred)
            test_bins = set(y_test_binned)
            pred_bins = set(y_pred_binned)
            used_bins = test_bins if len(test_bins) >= len(pred_bins) else pred_bins

            # Confusion Matrix
            mat = confusion_matrix(y_test_binned, y_pred_binned)
            acc, s = stats(mat, list(sorted(used_bins)))
            if acc >= best_accuracy:
                best_accuracy = acc
                best_cfg = cfg
                best_stats = s

        results.append({'acc': best_accuracy, 'stats': best_stats, 'cfg': best_cfg, 'model': linear})


    # Stats average
    average_acc = 0
    average_stats = {}

    best = {'acc': -math.inf}

    for m in results:
        acc = m['acc']
        s = m['stats']

        average_acc += acc
        for k, v in s.items():
            curr_v, c = average_stats.get(k, (Metrics(), 0))
            
            curr_v.tp += v.tp
            curr_v.tn += v.tn
            curr_v.fp += v.fp
            curr_v.fn += v.fn
            average_stats[k] = (curr_v, c + 1)

        if acc >= best['acc']:
            best = m 

    average_acc /= len(results)
    for k in average_stats.keys():
        s, c = average_stats[k]
        s.tp /= c
        s.tn /= c
        s.fp /= c
        s.fn /= c

    print(f"Stats average based on {iterations} iterations:")
    print(f"Average accuracy: {average_acc:.3f}")
    print(f"Average stats:")
    for k, v in average_stats.items():
        print(f"  {k} -> {v}")

    # Best case
    x_train, x_test, y_train, y_test = cfg = best['cfg']
    y_pred = best['model'].predict(x_test)
    print(f"Best accuracy: {best['acc']:.3f}")

    # Binning
    y_test_binned = utils.arr_binning(bins, y_test)
    y_pred_binned = utils.arr_binning(bins, y_pred)
    test_bins = set(y_test_binned)
    pred_bins = set(y_pred_binned)
    used_bins = test_bins if len(test_bins) >= len(pred_bins) else pred_bins

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)

    model_diag(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_binned, y_pred_binned), display_labels=used_bins)
    disp.plot()
    plt.show()
    print(linear.coef_)

if __name__ == '__main__':

    df = pd.read_csv(FILEPATH, sep=',')
    
    figure, axs = plt.subplots(3,3)
    for i, f1 in enumerate(CSV_HEADER):
        for j,f2 in enumerate(CSV_HEADER[i+1:]):
            x = df[f1].to_numpy()
            y = df[f2].to_numpy()
            axs[i][j].set_title(f"{f1} vs {f2}")
            axs[i][j].scatter(x,y)
    plt.show()

    division_iterations = 10
    iterations = 20
    simple_regression(df, division_iterations, iterations)

    print(f"-----------------------------")
    print(f"Multiple Regression")
    print(f"-----------------------------")
    multiple_regression(df, division_iterations, iterations)