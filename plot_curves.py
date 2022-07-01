import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# Counts number of normal points, outliers, and interpolations.
def get_histo(labels, scores, interpolations_index):
    normal = []
    outlier = []
    interpolations = []
    for i, v in enumerate(labels):
        if interpolations_index is not None and i in interpolations_index:
            interpolations.append(scores[i])
        elif v == 1:  # outlier
            outlier.append(scores[i])
        else:
            normal.append(scores[i])

    return normal, outlier, interpolations


# Computes True Positive Rate, False Positive Rate, Precision, ROC AUC, PR AUC.
def compute_rates(scores, labels, min, max):
    precision = []
    tpr = []
    fpr = []

    labels = np.array(labels)
    step = (max - min) / 100
    thresholds = np.arange(min - step, max + step, step).tolist()
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

    pr_auc = metrics.auc(tpr, precision)
    print("Precision-recall AUC: ", pr_auc)

    return tpr, fpr, precision, roc_auc, pr_auc


# Plot Precision-Recall curve.
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
    plt.savefig(f"./final_images/{name}_PRC_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


# Plots ROC curve.
def plot_roc(name, fpr, tpr, roc_auc, runs, type):
    plt.title(f'ROC Curve for {str(name).replace("_", " ").capitalize()}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f"./final_images/{name}_ROC_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


# Plots histogram of the normal, outliers, and interpolated points.
def plot_histogram(name, data, scores, labels, interpolations, runs, type):
    data_numbers = data.reset_index()
    if interpolations is not None:
        interpolations_index = data_numbers.index[data.index.isin(interpolations)].tolist()
        normal, outlier, interpolated = get_histo(labels, scores, interpolations_index)
        plt.hist(interpolated, bins='auto', label="Interpolations", alpha=0.5)
    else:
        normal, outlier, interpolated = get_histo(labels, scores, None)

    plt.hist(normal, bins='auto', label="Normal", alpha=0.5)
    plt.hist(outlier, bins='auto', label="Outlier", alpha=0.5)
    plt.title(f"Densities of Outlier Scores for Ensembles")
    plt.legend()
    plt.show()
    plt.savefig(f"./final_images/{name}_histogram_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


# Plots dataset with corresponding outlierness scores.
def plot_scores(name, data, scores, labels, interpolations, to_add_values, runs, type):
    dim = len(data.columns)
    columns = list(data.columns)
    height_ratios = [3 for _ in range(dim)]
    height_ratios.append(1)
    fig, axs = plt.subplots(dim + 1, 1, gridspec_kw={'height_ratios': height_ratios})
    axs[0].set_title(str(name).replace("_", " ").capitalize())
    axs[0].xaxis.set_visible(False)

    outliers = data.index[np.where(np.array(labels) > 0)].tolist()

    if interpolations is not None:
        outliers = [i for i in outliers if i not in interpolations]
        interpolations_values = [[] for _ in range(dim)]

    outliers_values = [[] for _ in range(dim)]

    for d in range(dim):
        outliers_values[d] = data.loc[data.index.isin(outliers), columns[d]]
        axs[d].plot(data.index, data[columns[d]], 'k')

        if interpolations is not None:
            interpolations_values[d] = [item for sublist in to_add_values[d] for item in sublist]
            axs[d].scatter(interpolations, interpolations_values[d], color='yellow')

        axs[d].scatter(outliers, outliers_values[d], color='b', s=10, alpha=0.5)

    axs[dim].plot(range(len(scores)), scores)
    axs[dim].xaxis.set_visible(False)
    axs[dim].set_ylabel("Ensemble")

    plt.savefig(f"./final_images/{name}_plot_{runs}_{type}.png", bbox_inches='tight')
    plt.clf()


# Plots all the above plots.
def all_plots(name, data, scores, labels, interpolations, to_add_values, heights, locations, runs, type):
    plot_scores(name, data, scores, labels, interpolations, to_add_values, runs, type)
    plot_histogram(name, data, scores, labels, interpolations, runs, type)
    tpr, fpr, precision, roc_auc, pr_auc = compute_rates(scores, labels, min=min(scores), max=max(scores))
    plot_roc(name, fpr, tpr, roc_auc, runs, type)
    plot_pr(name, precision, tpr, pr_auc, labels, runs, type)
