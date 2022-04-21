import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn
from numba import jit


folder = "auto-explore"


def get_histo(labels, scores, guesses_index):
    normal = []
    outlier = []
    guesses = []
    for i, v in enumerate(labels):
        if guesses_index is not None and i in guesses_index:
            guesses.append(scores[i])
        elif v == 1:  # outlier
            outlier.append(scores[i])
        else:
            normal.append(scores[i])

    return normal, outlier, guesses


def compute_rates(scores, labels, min, max):
    precision = []
    tpr = []
    fpr = []

    labels = np.array(labels)
    step = (max - min) / 100
    thresholds = np.arange(min, max, step).tolist()
    for threshold in thresholds:
        predicted_label = np.where(np.array(scores) > threshold, 1, 0)

        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label, labels=[0, 1]).ravel()

        # All data points are labelled normal (very low threshold used.)
        if TP + FP == 0:
            print("TP and FP are zero (All points labelled normal).")
            precision.append(1)
        else:
            precision.append(TP / (TP + FP))
        # No outliers in dataset.
        if TP + FN == 0:
            print("Error: TP and FN are zero (No outliers in dataset).")
        else:
            tpr.append(TP / (TP + FN))
        # No normal points in dataset.
        if FP + TN == 0:
            print("Error: FP and TN are zero (No normal points in dataset).")
        else:
            fpr.append(FP / (FP + TN))

    roc_auc = metrics.auc(fpr, tpr)
    print("ROC AUC: ", roc_auc)
    # G-means calculation
    gmeans = np.sqrt(np.array(tpr) * (1 - np.array(fpr)))
    ix = np.argmax(gmeans)
    print('ROC: Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    pr_auc = metrics.auc(tpr, precision)
    print("Precision-recall AUC: ", pr_auc)

    # F-score calculation
    fscore = []
    for i,p in enumerate(precision):
        if p+tpr[i] == 0:
            fscore.append(0)
        else:
            fscore.append(2*p*tpr[i]/(p+tpr[i]))
    # fscore = (2 * np.array(precision) * np.array(tpr)) / (np.array(precision) + np.array(tpr))
    # print(tpr)
    # print(precision)
    # locate the index of the largest f score
    ifs = np.argmax(np.array(fscore))
    print('PRC: Best Threshold=%f, F-Score=%.3f' % (thresholds[ifs], fscore[ifs]))

    return tpr, fpr, precision, roc_auc, pr_auc


def plot_pr(name, precision, recall, pr_auc, labels, runs, type):
    plt.title(f'PR Curve for {str(name).replace("_", " ").capitalize()}')
    plt.plot(recall, precision, 'b', label='AUC = %0.4f' % pr_auc)
    plt.legend(loc='lower right')
    no_skill = sum(labels) / len(labels)
    plt.plot([0, 1], [no_skill, no_skill], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(f"./{folder}/{name}_PRC_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


def plot_roc(name, fpr, tpr, roc_auc, runs, type):
    plt.title(f'ROC Curve for {str(name).replace("_", " ").capitalize()}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f"./{folder}/{name}_ROC_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()

def interval_extract(list):
    length = len(list)
    i = 0
    while (i< length):
        low = list[i]
        while i <length-1 and list[i]+1 == list[i + 1]:
            i += 1
        high = list[i]
        if (high - low >= 1):
            yield [low, high]
        elif (high - low == 1):
            yield [low, ]
            yield [high, ]
        else:
            yield [low, ]
        i += 1

def plot_histogram(name, data, scores, labels, guesses, runs, type):
    data_numbers = data.reset_index()
    if guesses is not None:
        guesses_index = data_numbers.index[data.index.isin(guesses)].tolist()
        normal, outlier, guessed = get_histo(labels, scores, guesses_index)
        plt.hist(guessed, bins='auto', label="Guessed", alpha=0.5)
    else:
        normal, outlier, guessed = get_histo(labels, scores, None)

    plt.hist(normal, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outlier, bins='auto', label="Outlier", alpha=0.5)
    plt.title(f"Densities of Outlier Scores for Ensembles")
    plt.legend()

    plt.savefig(f"./{folder}/{name}_histogram_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


def plot_scores(name, data, scores, labels, guesses, to_add_values, runs, type):
    dim = len(data.columns)
    columns = list(data.columns)
    height_ratios = [3 for _ in range(dim)]
    height_ratios.append(1)
    fig, axs = plt.subplots(dim+1, 1, gridspec_kw={'height_ratios': height_ratios})
    axs[0].set_title(str(name).replace("_", " ").capitalize())
    axs[0].xaxis.set_visible(False)

    outliers = data.index[np.where(np.array(labels) > 0)].tolist()

    if guesses is not None:
        outliers = [i for i in outliers if i not in guesses]
        guesses_values = [[] for _ in range(dim)]

    outliers_values = [[] for _ in range(dim)]

    for d in range(dim):
        outliers_values[d] = data.loc[data.index.isin(outliers), columns[d]]
        axs[d].plot(data.index, data[columns[d]], 'k')

        if guesses is not None:
            guesses_values[d] = [item for sublist in to_add_values[d] for item in sublist]
            axs[d].scatter(guesses, guesses_values[d], color='yellow')

        axs[d].scatter(outliers, outliers_values[d], color='b', s=10, alpha=0.5)

    axs[dim].plot(range(len(scores)), scores)
    axs[dim].xaxis.set_visible(False)
    axs[dim].set_ylabel("Ensemble")

    plt.savefig(f"./{folder}/{name}_plot_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


def plot_heights(name, data, labels, guesses, to_add_values, runs, type, heights, locations):
    fig, axs = plt.subplots(3,1, figsize=(14,5), gridspec_kw={'height_ratios':[2,2,0.25]})

    outliers_index = np.where(np.array(labels) > 0)
    outliers_time = data.index[np.where(np.array(labels) > 0)].tolist()
    outliers_values = [-1 for i, l in enumerate(locations) if i in outliers_index[0]]

    axs[0].set_title(str(name).replace("_", " ").capitalize())
    seaborn.heatmap([heights], fmt='', ax=axs[1], cbar_ax=axs[2],cbar_kws={"orientation": "horizontal"})
    axs[1].set_ylabel("peak heights")
    axs[1].xaxis.set_visible(False)
    axs[0].set_ylabel("peak positions")
    axs[0].set_xlim([min(data.index), max(data.index)])
    axs[0].scatter(outliers_time, outliers_values,color='yellow')
    axs[0].plot(data.index, locations, 'k')
    # axs[1].xaxis.set_visible(False)

    plt.savefig(f"./{folder}/{name}_plot_score_peaks_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


def histo_heights_positions(name, labels, heights, locations, runs, type):
    outliers_index = np.where(np.array(labels) > 0)

    outliers_location = [l for i, l in enumerate(locations) if i in outliers_index[0]]
    outliers_heights = [l for i, l in enumerate(heights) if i in outliers_index[0]]
    normal_location = [l for i, l in enumerate(locations) if i not in outliers_index[0]]
    normal_heights = [l for i, l in enumerate(heights) if i not in outliers_index[0]]

    plt.hist(normal_location, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outliers_location, bins='auto', label="Outliers", alpha=0.5)
    plt.title("Histogram of peak locations")
    plt.legend()
    plt.savefig(f"./{folder}/{name}_histo_locations_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()
    plt.hist(normal_heights, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outliers_heights, bins='auto', label="Outlier", alpha=0.5)
    plt.title("Histogram of peak heights")
    plt.legend()
    plt.savefig(f"./{folder}/{name}_histo_heights_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


def all_plots(name, data, scores, labels, guesses, to_add_values, heights, locations, runs, type):
    plot_scores(name, data, scores, labels, guesses, to_add_values, runs, type)
    plot_histogram(name, data, scores, labels, guesses, runs, type)
    tpr, fpr, precision, roc_auc, pr_auc = compute_rates(scores, labels, min=min(scores), max=max(scores))
    plot_roc(name, fpr, tpr, roc_auc, runs, type)
    plot_pr(name, precision, tpr, pr_auc, labels, runs, type)
    # plot_heights(name, data, labels, guesses, to_add_values, runs, type, heights, locations)
    # histo_heights_positions(name, labels, heights, locations, runs, type)
