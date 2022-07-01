import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy as sc
import sklearn.metrics as metrics
import pandas as pd
from numba import jit
from tqdm import tqdm
import time

import ensemble as e
import load_data_NAB as nab
import plot_curves as prc
import load_data_MITBIH as mb

@jit(nopython=True)
def make_windows(data, win_pos, win_length):
    if win_length % 2 == 0 and win_pos == 'mid':
        win_length_con = (win_length + 1) * data.shape[1]
    else:
        win_length_con = win_length * data.shape[1]

    expanded_data = []
    for i in range(len(data)):
        pad_front = 0
        pad_back = 0
        if win_length == 0:
            expanded_data.append(data[i])
            continue
        if win_pos == 'prev':
            previous = i - win_length + 1
            future = i
        elif win_pos == "mid":
            previous = i - win_length // 2
            future = i + win_length // 2
        else:
            previous = i
            future = i + win_length - 1

        if previous < 0:
            pad_front = np.abs(previous)
            previous = 0

        if future >= len(data):
            pad_back = future - len(data)+1
            future = len(data)

        point = data[previous:future + 1]
        if pad_front != 0 :
            point = np.concatenate((np.full((pad_front, data.shape[1]),0.0), point))
        if pad_back != 0:
            point = np.concatenate((point, np.full((pad_back, data.shape[1]), 0.0)))

        point = np.transpose(point).flatten()

        if len(point) != win_length_con:
            print("Error for time point ", i, " with window of length ", len(point), " instead of ", win_length_con)
        else:
            expanded_data.append(point)

    return expanded_data


def plot_scores(scores, labels, win_length, best_threshold, apply_pca, dim):
    outliers = []
    normal = []
    for i, l in enumerate(labels):
        if l == 1:
            outliers.append(scores[i])
        else:
            normal.append(scores[i])
    normal = np.array(normal).reshape(-1,)
    outliers = np.array(outliers).reshape(-1,)
    plt.hist(normal, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outliers, bins='auto', label="Outliers", alpha=0.5, color='purple')
    if apply_pca:
        plt.title(f"Histogram of scores for window length {win_length} with PCA (dim = {dim})")
    else:
        plt.title(f"Histogram of scores for window length {win_length}")
    print(dim)
    # plt.axvline(x=sc.stats.chi2.ppf(0.5, dim), color='red', label='0%')
    # plt.axvline(x=sc.stats.chi2.ppf(1 - best_threshold, dim), color='green',
    #             label=f'{100 * (1 - 2 * best_threshold):.1f}% (Highest G-mean)')
    # plt.axvline(x=sc.stats.chi2.ppf(best_threshold, dim), color='green')
    # plt.axvline(x=sc.stats.chi2.ppf(0.9, dim), color='orange', label='80%')
    # plt.axvline(x=sc.stats.chi2.ppf(0.1, dim), color='orange')
    plt.legend()
    plt.show()


def tpr_fpr_naive(scores, labels, win_length):
    tpr = []
    fpr = []
    precision = []

    labels = np.array(labels)
    thresholds = np.arange(0, 0.5, 0.001).tolist()
    for threshold in thresholds:
        limit_up = sc.stats.chi2.ppf(1 - threshold, win_length)
        limit_down = sc.stats.chi2.ppf(threshold, win_length)

        predicted_label = np.where(np.logical_and(limit_up > np.array(scores), np.array(scores) > limit_down), 0, 1)

        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label, labels=[0, 1]).ravel()
        if TP + FN == 0:
            print("TP and FN zero")
            tpr.append(1)
        else:
            tpr.append(TP / (TP + FN))

        fpr.append(FP / (FP + TN))
        if TP + FP == 0:
            precision.append(1)
        else:
            precision.append(TP / (TP + FP))

    # G-means calculation
    gmeans = np.sqrt(np.array(tpr) * (1 - np.array(fpr)))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # F-score calculation
    fscore = (2 * np.array(precision) * np.array(tpr)) / (np.array(precision) + np.array(tpr))
    # locate the index of the largest f score
    ifs = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ifs], fscore[ifs]))

    return tpr, fpr, precision, thresholds[ix]


@jit(nopython=True)
def get_scores(mean, cov, data):
    scores = np.full((len(data), 1), 0.0)
    if mean.shape[0] == 1:
        invcov = 1 / cov
        for i,d in enumerate(data):
            scores[i] = invcov * ((d-mean) @ (d-mean).T)
    else:
        invcov = np.linalg.inv(cov)
        for i,d in enumerate(data):
            if i % 100000==0:
                print(i)
            scores[i] = (d - mean) @ invcov @ (d-mean).T

    return scores


def naive(data, win_length=10, apply_pca=False, n_comp=0.99):
    signal = e.normalise(data.values)
    expanded_data = make_windows(signal, "mid", win_length)

    expanded_data = np.array(expanded_data)

    if apply_pca:
        pca = PCA(n_components=n_comp)
        expanded_data = pca.fit_transform(expanded_data)
    print(expanded_data.shape)
    mean = np.mean(expanded_data, axis=0)
    print(mean.shape)
    cov = np.cov(expanded_data, rowvar=False)
    print(cov.shape)
    scores = get_scores(mean, cov, expanded_data)

    return scores, expanded_data.shape[1]


def run_NAB(win_length, n_comp, apply_pca):
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
    data = data.reset_index().set_index('timestamp')
    data = data.drop('index', axis=1)
    data.index = pd.to_datetime(data.index)

    scores, dim = naive(data, win_length, apply_pca, n_comp)
    tpr, fpr, precision, best_threshold = tpr_fpr_naive(scores, labels, dim)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    if apply_pca:
        type = f"naive+PCA_{n_comp}_{win_length}"
        print("PCA dimension", dim)
    else:
        type = f"naive_{win_length}"

    # e.plot_roc(name, fpr, tpr, roc_auc, runs="", type=type)
    pr_auc = metrics.auc(tpr, precision)
    print(pr_auc)
    # prc.plot_pr(name, precision, tpr, pr_auc, labels, runs="", type=type)
    # plot_scores(scores, labels, win_length,best_threshold, apply_pca, dim)


def convert(heart_beats_x):
    heart_beats_x_array = []
    for x in heart_beats_x:
        heart_beats_x_array.append([x[0],x[-1]])
    return np.array(heart_beats_x_array)


@jit(nopython=True)
def get_beat_score(scores, heart_beats_x):
    scores_beats = np.full((len(heart_beats_x), 1),0.0)
    for j, x in enumerate(heart_beats_x):
        beat_scores = [scores[k-heart_beats_x[0][0]] for k in range(x[0], x[1])]
        scores_beats[j] = max(beat_scores)
    return scores_beats


def run_MITBIH(name, win_length, n_comp, apply_pca):
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(name, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)

    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)

    scores, dim = naive(signal, win_length, apply_pca, n_comp)
    # np.save(f"./output_scores_MITBIH/naive_MITBIH_{name}", scores)
    # scores = np.load("./output_scores_MITBIH/naive_MITBIH_100.npy")
    # dim = 522
    scores = get_beat_score(scores, convert(heart_beats_x))
    print(scores.shape)
    tpr, fpr, precision, best_threshold = tpr_fpr_naive(scores, labels, dim)
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC: ", roc_auc)
    pr_auc = metrics.auc(tpr, precision)
    print("PRC: ", pr_auc)
    # plot_scores(scores, labels, win_length, best_threshold, apply_pca, dim)
    return roc_auc, pr_auc

if __name__ == '__main__':
    # s = time.time()
    for w in [310,320, 330]:
        run_MITBIH("123", w, 0.99, True)
    # run_NAB(10, 0.99, False)
    # e = time.time()
    # print(e-s)
