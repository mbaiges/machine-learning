from cProfile import label
from typing import Iterable, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from utils import bootstrap_df, hist, bins, LoadingBar
from id3 import ID3
from random_forest import RandomForest
from models import Metrics

np.random.seed(1)

########## DATA SET GERMAN CREDIT ##########
DATASET_GERMAN_CREDIT             = "german_credit"
FILEPATH_GERMAN_CREDIT            = "german_credit.csv"

CREDITABILITY                     = "Creditability"
ACCOUNT_BALANCE                   = "Account Balance"
DURATION_OF_CREDIT_MONTH          = "Duration of Credit (month)"
PAYMENT_STATUS_OF_PREVIOUS_CREDIT = "Payment Status of Previous Credit"
PURPOSE                           = "Purpose"
CREDIT_AMOUNT                     = "Credit Amount"
VALUE_SAVINGS_STOCKS              = "Value Savings/Stocks"
LENGTH_OF_CURRENT_EMPLOYMENT      = "Length of current employment"
INSTALMENT_PER_CENT               = "Instalment per cent"
SEX_AND_MARITAL_STATUS            = "Sex & Marital Status"
GUARANTORS                        = "Guarantors"
DURATION_IN_CURRENT_ADDRESS       = "Duration in Current address"
MOST_VALUABLE_AVAILABLE_ASSET     = "Most valuable available asset"
AGE                               = "Age (years)"
CONCURRENT_CREDITS                = "Concurrent Credits"
TYPE_OF_APARTMENT                 = "Type of apartment"
NO_OF_CREDITS_AT_THIS_BANK        = "No of Credits at this Bank"
OCCUPATION                        = "Occupation"
NO_OF_DEPENDENTS                  = "No of dependents"
TELEPHONE                         = "Telephone"
FOREIGN_WORKER                    = "Foreign Worker"

########## DATA SET JUEGA TENNIS ##########
DATASET_TENNIS                    = "tennis"
FILEPATH_TENNIS                   = "juega_tenis.csv"

DIA                               = "Dia" 
PRONOSTICO                        = "Pronostico"
TEMPERATURA                       = "Temperatura"
HUMEDAD                           = "Humedad"
VIENTO                            = "Viento"
JUEGA                             = "Juega"


##### SET CONNSTANTS FOR DATASET #####
DATASET = DATASET_GERMAN_CREDIT

def filepath_for_dataset() -> str:
    return FILEPATH_TENNIS if DATASET == DATASET_TENNIS else FILEPATH_GERMAN_CREDIT
FILEPATH = filepath_for_dataset()

def t_name_for_dataset() -> str:
    return JUEGA if DATASET == DATASET_TENNIS else CREDITABILITY
T_NAME = t_name_for_dataset()

def attributes_names_for_dataset() -> str:
    return [
        PRONOSTICO,
        TEMPERATURA,
        HUMEDAD,
        VIENTO
    ] if DATASET == DATASET_TENNIS else [
        ACCOUNT_BALANCE,
        DURATION_OF_CREDIT_MONTH,
        PAYMENT_STATUS_OF_PREVIOUS_CREDIT,
        PURPOSE,
        CREDIT_AMOUNT,
        VALUE_SAVINGS_STOCKS,
        LENGTH_OF_CURRENT_EMPLOYMENT,
        INSTALMENT_PER_CENT,
        SEX_AND_MARITAL_STATUS,
        GUARANTORS,
        DURATION_IN_CURRENT_ADDRESS,
        MOST_VALUABLE_AVAILABLE_ASSET,
        AGE,
        CONCURRENT_CREDITS,
        TYPE_OF_APARTMENT,
        NO_OF_CREDITS_AT_THIS_BANK,
        OCCUPATION,
        NO_OF_DEPENDENTS,
        TELEPHONE,
        FOREIGN_WORKER
    ]
ATTRIBUTES_NAMES = attributes_names_for_dataset()

def analysis(df: pd.DataFrame) -> None:

    # Credit amount
    credit_amounts = df[CREDIT_AMOUNT].to_numpy()
    hist(credit_amounts, title=CREDIT_AMOUNT)

    # Ages
    ages = df[AGE].to_numpy()
    hist(ages, title=AGE)

    # PCA para ver como se ubican en un diagrama de biplot???????? Si sale super rapido nomas

    # Creditability

def _discretize_with_bins(b, v):
    for i, bv in enumerate(b[1:]):
        if v < bv:
            return i
    return i

def discretize(df) -> None:
    # Duration of Credit Month
    x = df[DURATION_OF_CREDIT_MONTH]
    b = bins(x, alg='dist', options={'min': 0, 'dist': 6})
    print(f"{DURATION_OF_CREDIT_MONTH} - Bins: {b}")
    df[DURATION_OF_CREDIT_MONTH] = df[DURATION_OF_CREDIT_MONTH].apply(lambda v: _discretize_with_bins(b, v))

    # Credit Amount
    x = df[CREDIT_AMOUNT]
    b = bins(x, alg='perc', options={'n': 10})
    print(f"{CREDIT_AMOUNT} - Bins: {b}")
    df[CREDIT_AMOUNT] = df[CREDIT_AMOUNT].apply(lambda v: _discretize_with_bins(b, v))

    # Age
    x = df[AGE]
    b = bins(x, alg='perc', options={'n': 10})
    print(f"{AGE} - Bins: {b}")
    df[AGE] = df[AGE].apply(lambda v: _discretize_with_bins(b, v))

CATEGORIES = [0, 1]
def metrics(t: Iterable[Union[int, float]], results: Iterable[Union[int, float]]) -> Metrics:
    cats = CATEGORIES
    ret = {}

    for cat in cats:
        ret[cat] = Metrics()

    for idx, result in enumerate(results):
        for cat in cats:
            ti = t[idx]
            if ti == cat:
                if result == cat:
                    ret[ti].tp += 1
                else:
                    ret[ti].fn += 1
            else:
                if result == cat:
                    ret[cat].fp += 1
                else:
                    ret[cat].tn += 1
    return ret

def precision(metrics_map: object):
    tp = sum(map(lambda m: m.tp, metrics_map.values()))
    fp = sum(map(lambda m: m.fp, metrics_map.values()))
    return tp/(tp+fp) if (tp + fp) != 0 else 0

def multiple_iterations_id3(x: pd.DataFrame, t: pd.DataFrame, sample_size: int=500, n: int=5, max_depth: int=None, show_loading_bar: bool=False) -> tuple:
    precisions = [] 
    errors = []
    all_t = []
    all_results = []
    if show_loading_bar:
        loading_bar = LoadingBar()
        loading_bar.init()
    for i in range(0, n):
        if show_loading_bar:
            loading_bar.update(i/n)
        (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=sample_size, test_size=sample_size)
        id3 = ID3(max_depth=max_depth)
        id3.load(x_train, t_train)
        # id3.print_tree()
        results, error = id3.eval(x_test, t_test)
        metrics_map = metrics(t_test[CREDITABILITY].to_numpy().tolist(), results)
        prec = precision(metrics_map)
        precisions.append(prec)
        errors.append(error)
        all_t.append(t_test[CREDITABILITY].to_numpy())
        all_results.append(results)
    if show_loading_bar:
        loading_bar.end()
    return np.array(precisions), np.array(errors), (np.array(all_t), np.array(all_results))

def multiple_iterations_random_forest(x: pd.DataFrame, t: pd.DataFrame, sample_size: int=500, n: int=5, trees_per_forest: int=5, max_depth: int=None, show_loading_bar: bool=False) -> tuple:
    precisions = [] 
    errors = []
    all_t = []
    all_results = []
    if show_loading_bar:
        loading_bar = LoadingBar()
        loading_bar.init()
    for i in range(0, n):
        if show_loading_bar:
            loading_bar.update(i/n)
        (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=sample_size, test_size=sample_size)
        rf = RandomForest(n=trees_per_forest, sample_size=sample_size, max_depth=max_depth)
        rf.load(x_train, t_train)
        results, error = rf.eval(x_test, t_test)
        metrics_map = metrics(t_test[CREDITABILITY].to_numpy().tolist(), results)
        prec = precision(metrics_map)
        precisions.append(prec)
        errors.append(error)
        all_t.append(t_test[CREDITABILITY].to_numpy())
        all_results.append(results)
    if show_loading_bar:
        loading_bar.end()
    return np.array(precisions), np.array(errors), (np.array(all_t), np.array(all_results))

def multiple_depths_id3(x: pd.DataFrame, t: pd.DataFrame, sample_size: int=500, min_depth: int=0, max_depth: int=10, iterations_per_depth: int=10, show_loading_bar: bool=False) -> tuple:
    depths = range(min_depth, max_depth+1)
    train_precisions = []
    train_errors = []
    test_precisions = [] 
    test_errors = []
    nodes = []
    iter = 0
    total_iter = (max_depth+1 - min_depth)*iterations_per_depth
    if show_loading_bar:
        loading_bar = LoadingBar()
        loading_bar.init()
    for depth in depths:
        train_prec = []
        train_error = []
        test_prec = []
        test_error = []
        s_nodes = []
        np.random.seed(1)
        for i in range(0, iterations_per_depth):
            iter += 1
            if show_loading_bar:
                loading_bar.update(iter/total_iter)

            (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=sample_size, test_size=sample_size)
            id3 = ID3(max_depth=depth)
            id3.load(x_train, t_train)
            s_nodes.append(id3.count_nodes())
            # id3.print_tree()
            train_results, train_err = id3.eval(x_train, t_train)
            train_metrics_map = metrics(t_train[CREDITABILITY].to_numpy().tolist(), train_results)
            train_prec.append(precision(train_metrics_map))
            train_error.append(train_err)
            
            test_results, test_err = id3.eval(x_test, t_test)
            test_metrics_map = metrics(t_test[CREDITABILITY].to_numpy().tolist(), test_results)
            test_prec.append(precision(test_metrics_map))
            test_error.append(test_err)

        train_precisions.append(np.mean(np.array(train_prec)))
        train_errors.append(np.mean(np.array(train_error)))
        test_precisions.append(np.mean(np.array(test_prec)))
        test_errors.append(np.mean(np.array(test_error)))
        nodes.append(np.mean(np.array(s_nodes)))
    if show_loading_bar:
        loading_bar.end()
    return (np.array(train_precisions), np.array(train_errors)), (np.array(test_precisions), np.array(test_errors)), np.array(depths), np.array(nodes)

def multiple_depths_forest(x: pd.DataFrame, t: pd.DataFrame, sample_size: int=500, min_depth: int=0, max_depth: int=10, iterations_per_depth: int=10, trees_amount: int=3, show_loading_bar: bool=False) -> tuple:
    depths = range(min_depth, max_depth+1)
    train_precisions = []
    train_errors = []
    test_precisions = [] 
    test_errors = []
    nodes = []
    iter = 0
    total_iter = (max_depth+1 - min_depth)*iterations_per_depth
    if show_loading_bar:
        loading_bar = LoadingBar()
        loading_bar.init()
    for depth in depths:
        train_prec = []
        train_error = []
        test_prec = []
        test_error = []
        s_nodes = []
        np.random.seed(1)
        for i in range(0, iterations_per_depth):
            iter += 1
            if show_loading_bar:
                loading_bar.update(iter/total_iter)

            (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=sample_size, test_size=sample_size)
            rf = RandomForest(max_depth=depth, n=trees_amount)
            rf.load(x_train, t_train)
            s_nodes.append(rf.count_nodes())
            # id3.print_tree()
            train_results, train_err = rf.eval(x_train, t_train)
            train_metrics_map = metrics(t_train[CREDITABILITY].to_numpy().tolist(), train_results)
            train_prec.append(precision(train_metrics_map))
            train_error.append(train_err)
            
            test_results, test_err = rf.eval(x_test, t_test)
            test_metrics_map = metrics(t_test[CREDITABILITY].to_numpy().tolist(), test_results)
            test_prec.append(precision(test_metrics_map))
            test_error.append(test_err)

        train_precisions.append(np.mean(np.array(train_prec)))
        train_errors.append(np.mean(np.array(train_error)))
        test_precisions.append(np.mean(np.array(test_prec)))
        test_errors.append(np.mean(np.array(test_error)))
        nodes.append(np.mean(np.array(s_nodes)))
    if show_loading_bar:
        loading_bar.end()
    return (np.array(train_precisions), np.array(train_errors)), (np.array(test_precisions), np.array(test_errors)), np.array(depths), np.array(nodes)

def multiple_trees_and_depths_forest(x: pd.DataFrame, t: pd.DataFrame, min_trees: int=3, max_trees: int=10, show_loading_bar: bool=True) -> tuple:
        trees_amount = range(min_trees, max_trees+1)
        iter = 0
        total_iter = max_trees + 1 - min_trees
        mean_train_precisions = []
        mean_test_precisions = []
        best = {}
        if show_loading_bar:
            loading_bar = LoadingBar()
            loading_bar.init()
        for tree_amount in trees_amount:
            iter += 1
            if show_loading_bar:
                loading_bar.update(iter/total_iter)

            (train_precisions, train_errors), (test_precisions, test_errors), depths, nodes = multiple_depths_forest(x, t, sample_size=500, min_depth=0, max_depth=8, iterations_per_depth=2, trees_amount=tree_amount, show_loading_bar=False)
            train_precisions=list(map(lambda e: 1-e,train_errors))
            test_precisions=list(map(lambda e: 1-e,test_errors))
            mean_train_precision = sum(train_precisions)/len(train_precisions)
            mean_test_precision = sum(test_precisions)/len(test_precisions)
            mean_train_precisions.append(mean_train_precision)
            mean_test_precisions.append(mean_test_precision)
            if not best or best['mean_test_precision'] < mean_test_precision:
                best['mean_test_precision'] = mean_test_precision
                best['mean_train_precision'] = mean_train_precision
                best['test_precision'] = test_precisions
                best['train_precisions'] = train_precisions
                best['nodes'] = nodes
                best['n'] = tree_amount
        print(f"\n---\nBest is {best['n']}\nmean train precision: {best['mean_train_precision']}\nmean test precision: {best['mean_test_precision']}\n---")
    
        plt.figure()
        mean_precision_vs_tree_amount(mean_train_precisions=mean_train_precisions, mean_test_precisions=mean_test_precisions, trees_amount=trees_amount, plot=False)
        plt.figure()
        precision_vs_nodes_plot(f"Random Forest n={best['n']}", train_precisions=best['train_precisions'], test_precisions=best['test_precision'], nodes=best['nodes'], plot=False)
        plt.show()
        return np.array(best['train_precisions']), np.array(best['test_precision']), np.array(best['nodes'])

def mean_precision_vs_tree_amount(mean_train_precisions: list, mean_test_precisions: list, trees_amount: list, plot: bool=True) -> None:
    plt.plot(trees_amount, mean_train_precisions, marker='o', label="entrenamiento")
    plt.plot(trees_amount, mean_test_precisions, marker='o', label="evaluacion")
    plt.title(f"Precision promedio en funcion de la cantidad de arboles")
    plt.xlabel("Cantidad de arboles")
    plt.ylabel("Precision promedio")
    plt.legend()
    if plot: plt.show()

def precision_vs_nodes_plot(method: str, train_precisions: list, test_precisions: list, nodes: list, plot: bool=True, method_location: str="title") -> None:
    ordered = sorted([(i, nodes[i]) for i in range(len(nodes))], key=lambda e: e[1])
    indexes = list(map(lambda e: e[0], ordered))
    nodes            = [nodes[i]            for i in indexes]
    train_precisions = [train_precisions[i] for i in indexes]
    test_precisions  = [test_precisions[i]  for i in indexes]
    plt.plot(nodes, train_precisions, marker='o', label=f"entrenamiento {method}" if method_location == "label" else "entrenamiento")
    plt.plot(nodes, test_precisions, marker='o', label=f"evaluacion {method}" if method_location == "label" else "evaluacion")
    plt.title(f"Grafico de ajuste arbol de decision para {method}" if method_location == "title" else "Grafico de ajuste arbol de decision")
    plt.xlabel("Cantidad de nodos")
    plt.ylabel("Precision")
    plt.legend()
    if plot: plt.show()

def confusion(x: pd.DataFrame, t: pd.DataFrame, sample_size: int=500, iterations: int=5, alg: str='id3', trees_per_forest: int=5, max_depth: int=None, show_loading_bar: bool=False):
    cats = CATEGORIES
    cats_len = len(cats)
    
    m = [[0 for i in range(cats_len)] for j in range(cats_len)]
    
    if alg == 'id3':
        precisions, errors, (all_t, all_results) = multiple_iterations_id3(x, t, n=iterations, sample_size=sample_size, max_depth=max_depth, show_loading_bar=show_loading_bar)
    else: # alg == 'rf'
        precisions, errors, (all_t, all_results) = multiple_iterations_random_forest(x, t, n=iterations, sample_size=sample_size, trees_per_forest=trees_per_forest, max_depth=max_depth, show_loading_bar=show_loading_bar)
    
    print(all_results.shape)
    for it in range(all_results.shape[0]):
        for idx, pred_t in enumerate(all_results[it]):
            true_t = all_t[it][idx]
            m[cats.index(true_t)][cats.index(pred_t)] += 1

    precisions=list(map(lambda e: 1-e,errors))
    print(f"Precision: {np.mean(np.array(precisions))}")

    m = np.array(m)

    df_cm = pd.DataFrame(m, cats, cats)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.8) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=sn.cm.rocket_r, fmt='d') # font size
    plt.xticks(rotation=0)
    plt.show()

    return m

    
if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv(FILEPATH, sep=',')
    print(f"Loaded {df.shape[0]} rows")

    # Analyze
    # analysis(df)

    # ID3
    print("ID3!")
    
    plt.rcParams.update({'font.size': 22})
    # Discretize
    if DATASET == DATASET_TENNIS:
        x = df[ATTRIBUTES_NAMES]
        t = df[T_NAME].to_frame()
        id3 = ID3()
        id3.load(x,t)
        id3.print_tree()
        id3.eval(x,t)
    else:
        discretize(df)

        x = df[ATTRIBUTES_NAMES]
        t = df[T_NAME].to_frame()
        
        # # Train and test with bootstrap
        # list_size = 500
        # (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=list_size, test_size=list_size)
 
        # print(x_train)

        # id3 = ID3()
        # id3.load(x_train, t_train)
        # # id3.print_tree()
        # # print(f"Max Depth: {id3.depth}")
        # # print(f"Nodes: {id3.count_nodes()}")
        # predicted = id3.predict(x_test)
        # results, error = id3.eval(x_test, t_test)
        # metrics_map = metrics(t_test[CREDITABILITY].to_numpy().tolist(), results)
        # prec = precision(metrics_map)
        # print(f"Error: {error}")
        # print(f"Precision: {prec}")

        # Multiple iterations
        # precisions, errors, _ = multiple_iterations_id3(x, t, sample_size=500, n=50, show_loading_bar=True)
        # print(f"Mean Precision: {np.mean(precisions):.3f}")
        # print(f"Mean Error: {np.mean(errors):.3f}")

        # (id3_train_precisions, train_errors), (id3_test_precisions, test_errors), depths, id3_nodes = multiple_depths_id3(x, t, sample_size=500, min_depth=0, max_depth=8, iterations_per_depth=5, show_loading_bar=True)
        # id3_train_precisions=list(map(lambda e: 1-e,train_errors))
        # id3_test_precisions=list(map(lambda e: 1-e,test_errors))
        # precision_vs_nodes_plot("ID3", train_precisions=id3_train_precisions, test_precisions=id3_test_precisions, nodes=depths)
        confusion(x, t, iterations=50, alg='id3', max_depth=5, show_loading_bar=True)

    # Random Forest
    print("Random Forest!")
    if DATASET != DATASET_TENNIS:
        discretize(df)

        x = df[ATTRIBUTES_NAMES]
        t = df[T_NAME].to_frame()
        
        # # Train and test with bootstrap
        # list_size = 500
        # (x_train, t_train), (x_test, t_test) = bootstrap_df(x, t, train_size=list_size, test_size=list_size)

        # rf = RandomForest(n=5, sample_size=500)
        # rf.load(x_train, t_train)
        # results, error = rf.eval(x_test, t_test)
        # metrics_map = metrics(t_test[CREDITABILITY].to_numpy().tolist(), results)
        # prec = precision(metrics_map)
        # print(f"Error: {error}")
        # print(f"Precision: {prec}")

        # # Multiple iterations
        # precisions, errors, _ = multiple_iterations_random_forest(x, t, sample_size=500, n=50, show_loading_bar=True)
        # print(f"Mean Precision: {np.mean(precisions):.3f}")
        # print(f"Mean Error: {np.mean(errors):.3f}")

        # rf_train_precisions, rf_test_precisions, rf_nodes = multiple_trees_and_depths_forest(x, t, min_trees=2, max_trees=15, show_loading_bar=True)
        # precision_vs_nodes_plot("ID3", id3_train_precisions, id3_test_precisions, id3_nodes, plot=False, method_location="label")
        # precision_vs_nodes_plot("RF", rf_train_precisions, rf_test_precisions, rf_nodes, plot=False, method_location="label")
        # plt.show()

        # (train_precisions, train_errors), (test_precisions, test_errors), depths, nodes = multiple_depths_forest(x, t, sample_size=500, min_depth=0, max_depth=8, iterations_per_depth=5, trees_amount=6, show_loading_bar=True)
        # train_precisions=list(map(lambda e: 1-e,train_errors))
        # test_precisions=list(map(lambda e: 1-e,test_errors))
        # precision_vs_nodes_plot("Random Forest", train_precisions=train_precisions, test_precisions=test_precisions, nodes=depths)

        confusion(x, t, iterations=50, alg='id3', trees_per_forest=6, max_depth=5, show_loading_bar=True)
