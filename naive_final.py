import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy as sc
import sklearn.metrics as metrics
import pandas as pd
from numba import jit

import ensemble_final as e
import load_data_NAB as nab
import load_data_MITBIH as mb


# Constructs windows for each point in time.
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
            pad_back = future - len(data) + 1
            future = len(data)

        point = data[previous:future + 1]
        if pad_front != 0:
            point = np.concatenate((np.full((pad_front, data.shape[1]), 0.0), point))
        if pad_back != 0:
            point = np.concatenate((point, np.full((pad_back, data.shape[1]), 0.0)))

        point = np.transpose(point).flatten()

        if len(point) != win_length_con:
            print("Error for time point ", i, " with window of length ", len(point), " instead of ", win_length_con)
        else:
            expanded_data.append(point)

    return expanded_data


# Plots dataset with corresponding outlierness scores.
def plot_scores(scores, labels, win_length, apply_pca, dim):
    outliers = []
    normal = []
    for i, l in enumerate(labels):
        if l == 1:
            outliers.append(scores[i])
        else:
            normal.append(scores[i])
    normal = np.array(normal).reshape(-1, )
    outliers = np.array(outliers).reshape(-1, )
    plt.hist(normal, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outliers, bins='auto', label="Outliers", alpha=0.5, color='purple')
    if apply_pca:
        plt.title(f"Histogram of scores for window length {win_length} with PCA (dim = {dim})")
    else:
        plt.title(f"Histogram of scores for window length {win_length}")

    plt.legend()
    plt.show()


# Computes True Positive Rate, False Positive Rate, Precision.
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

    return tpr, fpr, precision


# Determine naive classifier outlierness scores for each time point.
@jit(nopython=True)
def get_scores(mean, cov, data):
    scores = np.full((len(data), 1), 0.0)
    if mean.shape[0] == 1:
        invcov = 1 / cov
        for i, d in enumerate(data):
            scores[i] = invcov * ((d - mean) @ (d - mean).T)
    else:
        invcov = np.linalg.inv(cov)
        for i, d in enumerate(data):
            if i % 100000 == 0:
                print(i)
            scores[i] = (d - mean) @ invcov @ (d - mean).T

    return scores


# Run naive classifier on given dataset.
def naive(data, win_length=10, apply_pca=False, n_comp=0.99):
    # Normalise and construct windows.
    signal = e.normalise(data.values)

    expanded_data = make_windows(signal, "mid", win_length)
    expanded_data = np.array(expanded_data)

    # Apply PCA if user selects this option.
    if apply_pca:
        pca = PCA(n_components=n_comp)
        expanded_data = pca.fit_transform(expanded_data)

    # Fit Gaussian.
    mean = np.mean(expanded_data, axis=0)
    cov = np.cov(expanded_data, rowvar=False)
    # Determine outlierness scores.
    scores = get_scores(mean, cov, expanded_data)

    return scores, expanded_data.shape[1]


# Naive classifier run on AMB dataset.
def run_AMB(win_length, n_comp, apply_pca):
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
    data = data.reset_index().set_index('timestamp')
    data = data.drop('index', axis=1)
    data.index = pd.to_datetime(data.index)

    scores, dim = naive(data, win_length, apply_pca, n_comp)
    tpr, fpr, precision = tpr_fpr_naive(scores, labels, dim)

    roc_auc = metrics.auc(fpr, tpr)
    pr_auc = metrics.auc(tpr, precision)
    print("ROC AUC", roc_auc)
    print("PR AUC", pr_auc)

    plot_scores(scores, labels, win_length, apply_pca, dim)


# Convert type for Numba run.
def convert(heart_beats_x):
    heart_beats_x_array = []
    for x in heart_beats_x:
        heart_beats_x_array.append([x[0], x[-1]])
    return np.array(heart_beats_x_array)


# Summarise individual time point scores into beat scores.
@jit(nopython=True)
def get_beat_score(scores, heart_beats_x):
    scores_beats = np.full((len(heart_beats_x), 1), 0.0)
    for j, x in enumerate(heart_beats_x):
        beat_scores = [scores[k - heart_beats_x[0][0]] for k in range(x[0], x[1])]
        scores_beats[j] = max(beat_scores)
    return scores_beats


# Naive classifier run on a sample of the MITBIH dataset.
def run_MITBIH(name, win_length, n_comp, apply_pca):
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(name, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)

    scores, dim = naive(signal, win_length, apply_pca, n_comp)
    scores = get_beat_score(scores, convert(heart_beats_x))
    tpr, fpr, precision = tpr_fpr_naive(scores, labels, dim)
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC: ", roc_auc)
    pr_auc = metrics.auc(tpr, precision)
    print("PRC: ", pr_auc)
    plot_scores(scores, labels, win_length, apply_pca, dim)


# Run Naive classifier on different datasets based on parameter settings.
if __name__ == '__main__':
    run_MITBIH("100", 170, 0.99, True)
    # run_AMB(10, 0.99, False)
