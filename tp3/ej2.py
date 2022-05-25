import os
from enum import Enum
import random

from PIL import Image
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

RESOURCES_PATH  = "resources/"
CIELO_FILE      = "cielo.jpg"
PASTO_FILE      = "pasto.jpg"
VACA_FILE       = "vaca.jpg"

IMG_DATASET_CSV = "image_dataset.csv" 
CSV_HEADER = ['r', 'g', 'b', 'class']

class ClassType(Enum):
    CIELO   = 0, CIELO_FILE,
    PASTO   = 1, PASTO_FILE,
    VACA    = 2, VACA_FILE

def pixels_with_class(pixmap: np.ndarray, class_: int) -> np.ndarray:
        # transform pixmap (x,y,3) to pixel array (x*y, 3)
        # such as
        # [
        #   [128,128,128],
        #   [255,255,255],
        #   [0,0,0]
        # ]
        pixmap      = pixmap.reshape(pixmap.shape[0] * pixmap.shape[1], pixmap.shape[2])
        # class column
        class_col   = np.ones(shape=(pixmap.shape[0], 1)) * class_
        # append class to each pixel
        pixmap      = np.append(pixmap, class_col, axis=1)
        return pixmap

def build_image_dataset(generate: bool=False) -> pd.DataFrame:
    '''
        Builds image dataset from CIELO, PASTO and VACA images,
        optionally receives boolean to generate dataset or to
        use existing dataset if found
    '''
    if os.path.isfile(IMG_DATASET_CSV) and not generate:
        return pd.read_csv(IMG_DATASET_CSV, sep=',')
    imgs = [ClassType.CIELO, ClassType.PASTO, ClassType.VACA]
    df_values = np.empty(shape=(0,4))
    for img in imgs:
        class_  = img.value[0]
        path    = img.value[1]
        with Image.open(RESOURCES_PATH + path) as im:
            # add all pixels to df
            df_values = np.append(df_values, pixels_with_class(np.asarray(im), class_), axis=0)
    
    df = pd.DataFrame(df_values, columns=CSV_HEADER, dtype=int)
    df.to_csv(IMG_DATASET_CSV, encoding='utf-8', index=False, header=True)
    return df

RANDOM = random.Random(111)
def shuffle(x, t):
    shuffled_idxs = [i for i in range(x.shape[0])]
    RANDOM.shuffle(shuffled_idxs)
    return np.array([x[i] for i in shuffled_idxs]), np.array([t[i] for i in shuffled_idxs])

def multiple_c_svm(x_train: np.ndarray, x_test: np.ndarray, t_train: np.ndarray, t_test: np.ndarray, c_start: float=0.1, c_end: float=2.0, step: float=0.1) -> dict:
    best = {}
    for c in np.arange(c_start, c_end + step, step):
        clf = svm.SVC(C=c, decision_function_shape='ovr', kernel='rbf')
        clf.fit(x_train, t_train)
        accuracy = clf.score(x_test, t_test)
        if not best or best['accuracy'] < accuracy:
            best['accuracy'] = accuracy
            best['c'] = c
    return best

def multiple_kernel_svm(x_train: np.ndarray, x_test: np.ndarray, t_train: np.ndarray, t_test: np.ndarray) -> dict:
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    best = {}
    for kernel in kernels:
        clf = svm.SVC(C=1.0, decision_function_shape='ovr', kernel=kernel)
        clf.fit(x_train, t_train)
        accuracy = clf.score(x_test, t_test)
        if not best or best['accuracy'] < accuracy:
            best['accuracy'] = accuracy
            best['kernel'] = kernel
    return best

if __name__ == '__main__':
    df = build_image_dataset()
    print(df)
    x = df[CSV_HEADER[:-1]]
    print(x)
    t = df[CSV_HEADER[-1]].to_frame()
    x, t = shuffle(x.to_numpy(), t.to_numpy())

    # 40% de los datos se usaran para test
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # https://scikit-learn.org/stable/modules/svm.html#svm-kernels
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.4, shuffle=False)

    # validation.py:1111: DataConversionWarning: A column-vector y was passed when a 1d array was expected. 
    # Please change the shape of y to (n_samples, ), for example using ravel()
    t_train = t_train.flatten()
    t_test  = t_test.flatten()

    #TODO: test differents params C, decision_function_shape, kernel
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # best = multiple_c_svm(x_train, x_test, t_train, t_test)
    best = multiple_kernel_svm(x_train, x_test, t_train, t_test)
    print(best)
    # clf = svm.SVC(C=1.0, decision_function_shape='ovr', kernel='rbf')
    # clf.fit(x_train, t_train)
    # print(clf.score(x_test, t_test))
    