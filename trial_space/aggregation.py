import numpy as np
from numba import jit
import sklearn.model_selection as sk


@jit(nopython=True)
def get_weights_corr_m(scores_train, y_train):
    w = np.ones(scores_train.shape[0])

    corr_old = 0
    corr_new = 1
    c = 0
    while (not np.abs(corr_new - corr_old) < 0.0000001) or c == 1000:
        for m in range(scores_train.shape[0]):
            if abs(np.corrcoef(scores_train[m], y_train)[0, 1]) > 0.5:
                w[m] = w[m] * 2
            else:
                w[m] = w[m] / 2

        weighted_scores_train = (w.reshape(1, -1) @ scores_train)[0]
        corr_old = corr_new
        corr_new = np.corrcoef(weighted_scores_train, y_train)[0, 1]
        c += 1

    flip = False
    print(corr_new)
    if corr_new < 0:
        flip = True
    return w, flip

def summarise_scores_supervised_scores_corr_m(all_scores, labels):
    train_indices = np.array([i for i in range(len(labels))]).reshape(-1, 1)
    X_train_i, X_test_i, y_train, y_test = sk.train_test_split(train_indices, labels, test_size=0.2, stratify=labels)

    scores_train = get_set(all_scores, X_train_i)
    scores_test = get_set(all_scores, X_test_i)

    del all_scores, train_indices, labels

    w, flip = get_weights_corr_m(scores_train, np.array(y_train))
    predict_test = np.array((w.reshape(1, -1) @ scores_test)[0])
    return predict_test, y_test, flip


@jit(nopython=True)
def get_weights(scores_train, y_train):
    w = np.ones(scores_train.shape[0])
    weighted_scores_train = np.full((len(y_train),), 0.0)

    corr_new = 0.0
    c = 0
    while (not abs(corr_new) > 0.4) or c == 10000:
        if c %100 == 0:
            print(c)
        for i, score in enumerate(weighted_scores_train):
            if score == 0 and y_train[i] == 1:
                for j, s in enumerate(scores_train[:, i]):
                    if s == 1:
                        w[j] = w[j] * 2
            if score == 1 and y_train[i] == 0:
                for j, s in enumerate(scores_train[:, i]):
                    if s == 1:
                        w[j] = w[j] / 2

        weighted_scores_train = (w.reshape(1, -1) @ scores_train)[0]
        corr_new = np.corrcoef(weighted_scores_train, y_train)[0, 1]
        c += 1
        if c == 10000:
            print("Max number of iterations.")
    flip = False
    if corr_new < 0:
        flip = True
    return w, flip


def summarise_scores_supervised_scores(all_scores, labels):
    print("aggregating...")
    train_indices = np.array([i for i in range(len(labels))]).reshape(-1, 1)
    X_train_i, X_test_i, y_train, y_test = sk.train_test_split(train_indices, labels, test_size=0.2, stratify=labels)

    scores_train = get_set(all_scores, X_train_i)
    scores_test = get_set(all_scores, X_test_i)

    del all_scores, train_indices, labels

    y_train = np.array(y_train)
    print("getting weights...")
    w, flip = get_weights(scores_train, y_train)

    predict_test = np.array((w.reshape(1, -1) @ scores_test)[0])

    return predict_test, y_test, flip


def select(all_scores):
    final_scores = []

    return final_scores