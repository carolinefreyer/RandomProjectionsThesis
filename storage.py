import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

def get_beat_labels(beat_outlier_scores, threshold):
    predicted_labels = []
    for beat in beat_outlier_scores:
        label = 0
        for score in beat:
            if score > threshold:
                label = 1
                break
        predicted_labels.append(label)
    return predicted_labels

def show_roc_curves_basic(labels,beat_scores_avg,beat_scores_max):
    fpr, tpr, _ = metrics.roc_curve(labels, beat_scores_avg)
    roc_auc = metrics.auc(fpr, tpr)
    fpr_max, tpr_max, _ = metrics.roc_curve(labels, beat_scores_max)
    roc_auc_max = metrics.auc(fpr_max, tpr_max)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='(average) AUC = %0.2f' % roc_auc)
    plt.plot(fpr_max, tpr_max, 'k', label='(maximum) AUC = %0.2f' % roc_auc_max)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def get_beat_old(signal, peak, freq):
    # length of heart beat is now randomly set as 3 seconds per side.
    window_size = 3
    window_one_side = window_size * freq
    beat_start = peak - window_one_side
    beat_end = peak + window_one_side + 1
    # this cuts off the last beat incase its not whole
    if beat_end < signal.shape[0]:
        return signal[beat_start:beat_end]
    else:
        return np.array([])


def averages(beat_outlier_scores, labels):
    normal_avg = []
    outlier_avg = []
    for i in range(len(labels)):
        if labels[i] == 0:  # normal
            normal_avg.append(beat_outlier_scores[i])
        else:
            outlier_avg.append(beat_outlier_scores[i])

    print("Normal average:", sum(normal_avg) / len(normal_avg))
    print("Outlier average:", sum(outlier_avg) / len(outlier_avg))

    # plt.hist(outlier_scores, bins = 500)
    # plt.title("Histogram of Outlierness Scores")
    # plt.ylabel("Frequency")
    # plt.show()