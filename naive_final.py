import ensemble as e
import load_data_NAB as nab
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy as sc
import sklearn.metrics as metrics
import plot_curves as prc


def plot_scores(scores, labels, win_length, best_threshold, apply_pca, dim):

    outliers = []
    normal = []
    for i,l in enumerate(labels):
        if l == 1:
            outliers.append(scores[i])
        else:
            normal.append(scores[i])
    plt.hist(normal, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outliers, bins='auto', label="Outliers", alpha=0.5, color='purple')
    if apply_pca:
        plt.title(f"Histogram of scores for window length {win_length} with PCA (dim = {dim})")
    else:
        plt.title(f"Histogram of scores for window length {win_length}")
    print(dim)
    plt.axvline(x=sc.stats.chi2.ppf(0.5, dim), color='red', label='0%')
    plt.axvline(x=sc.stats.chi2.ppf(1-best_threshold, dim), color='green', label=f'{100*(1-2*best_threshold):.1f}% (Highest G-mean)')
    plt.axvline(x=sc.stats.chi2.ppf(best_threshold, dim), color='green')
    plt.axvline(x=sc.stats.chi2.ppf(0.9, dim), color='orange', label='80%')
    plt.axvline(x=sc.stats.chi2.ppf(0.1, dim), color='orange')
    plt.legend()
    plt.show()

def tpr_fpr_naive(scores, labels, win_length):
    tpr = []
    fpr = []
    precision = []

    print(win_length)
    labels = np.array(labels)
    thresholds = np.arange(0, 0.5, 0.001).tolist()
    for threshold in thresholds:
        limit_up = sc.stats.chi2.ppf(1-threshold, win_length)
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
    print(tpr)
    print(fpr)
    #F-score calculation
    fscore = (2 * np.array(precision) * np.array(tpr)) / (np.array(precision) + np.array(tpr))
    # locate the index of the largest f score
    ifs = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ifs], fscore[ifs]))

    return tpr, fpr, precision, thresholds[ix]


def naive(data, labels, win_length=10, apply_pca=False, n_comp = 0.99):
    signal = e.normalise(np.array(data['value']).reshape(-1, 1))
    expanded_data = e.make_windows(signal, "mid", win_length)
    if win_length % 2 == 0:
        win_length += 1
    trim = [i for i in range(len(expanded_data)) if len(expanded_data[i]) == win_length]
    expanded_data_trimmed = np.asarray([expanded_data[i] for i in range(len(expanded_data)) if i in trim])
    expanded_data_trimmed = expanded_data_trimmed.reshape(-1, win_length)
    labels_trimmed = [labels[i] for i in range(len(expanded_data)) if i in trim]

    if apply_pca:
        pca = PCA(n_components=n_comp)
        expanded_data_trimmed = pca.fit_transform(expanded_data_trimmed)
        print(expanded_data_trimmed.shape)

    mean = np.mean(expanded_data_trimmed, axis=0)
    cov = np.cov(expanded_data_trimmed, rowvar=False)

    scores = []
    if mean.shape[0] == 1:
        invcov = 1/cov
        for d in expanded_data_trimmed:
            scores.append(invcov * ((d - mean) @ (d - mean).T))
    else:
        invcov = np.linalg.inv(cov)
        for d in expanded_data_trimmed:
            scores.append((d - mean) @ invcov @ (d - mean).T)
    return scores, labels_trimmed, expanded_data_trimmed.shape[1]


if __name__ == '__main__':
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

    win_length = 14
    n_comp = 0.99
    apply_pca = True

    scores, labels, dim = naive(data, labels, win_length, apply_pca, n_comp)
    # print(dim)

    tpr, fpr, precision, best_threshold = tpr_fpr_naive(scores, labels, dim)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    if apply_pca:
        type = f"naive+PCA_{n_comp}_{win_length}"
    else:
        type = f"naive_{win_length}"

    # e.plot_roc(name, fpr, tpr, roc_auc, runs="", type=type)
    pr_auc = metrics.auc(tpr, precision)
    print(pr_auc)
    prc.plot_pr(name, precision, tpr,pr_auc, labels, runs="", type=type)
    # plot_scores(scores, labels, win_length,best_threshold, apply_pca, dim)
