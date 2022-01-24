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

def show_pr_curve_basic(labels,beat_scores_avg,beat_scores_max):
    precision, recall, thresholds = metrics.precision_recall_curve(labels, beat_scores_avg)
    roc_auc = metrics.auc(recall, precision)
    precision_max, recall_max, _ = metrics.precision_recall_curve(labels, beat_scores_max)
    roc_auc_max = metrics.auc(recall_max, precision_max)

    plt.title('Receiver Operating Characteristic')
    plt.plot(recall, precision, 'b', label='(average) AUC = %0.2f' % roc_auc)
    plt.plot(recall_max, precision_max, 'k', label='(maximum) AUC = %0.2f' % roc_auc_max)
    plt.legend(loc='lower right')
    no_skill = sum(labels) / len(labels)
    plt.plot([0, 1], [no_skill, no_skill], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


    #max-second max
    # diff_scores_max = np.array(beat_scores_max) - np.array(beat_scores_second_max)
    # diff_scores_max = 1 / max(diff_scores_max) * diff_scores_max
    # tpr_diff_max, fpr_diff_max = compute_tpr_fpr_flipped(diff_scores_max, labels)
    # roc_auc_diff_max = metrics.auc(fpr_diff_max, tpr_diff_max)

def compute_tpr_fpr_flipped(scores, labels):
    tpr = []
    fpr = []

    thresholds = np.arange(0, 1, 0.01).tolist()
    for threshold in thresholds:
        predicted_label = []
        for beat in scores:
            if beat > threshold:
                predicted_label.append(1)
            else:
                predicted_label.append(0)

        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label).ravel()

        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    return tpr, fpr


def compute_f1_score(scores, labels):
    f1 =[]
    thresholds = np.arange(0, 1, 0.01).tolist()
    for threshold in thresholds:
        predicted_label = np.where(np.array(scores) > threshold, 1, 0)
        f1.append(metrics.f1_score(labels, predicted_label))
    ix = np.argmax(f1)
    return f1[ix], thresholds[ix]

def averages(beat_outlier_scores, labels):
    normal_avg = []
    outlier_avg = []
    for i in range(len(labels)):
        if labels[i] == 0:  # normal
            normal_avg.append(beat_outlier_scores[i])
        else:
            outlier_avg.append(beat_outlier_scores[i])

    print("Normal average:", round(sum(normal_avg) / len(normal_avg), 4), "+/-", round(np.std(normal_avg), 4))
    print("Outlier average:", round(sum(outlier_avg) / len(outlier_avg), 4), "+/-", round(np.std(outlier_avg), 4))
    if (sum(normal_avg) / len(normal_avg) + np.std(normal_avg)) > (
            sum(outlier_avg) / len(outlier_avg) - np.std(outlier_avg)):
        print("overlap")

    ##Print Class Averages

    # print("Class averages for average outlierness scores in beat:")
    # averages(beat_scores_avg, labels)
    # print("Class averages for maximum outlierness scores in beat:")
    # averages(beat_scores_max, labels)
    # print("Class averages for second maximum outlierness scores in beat:")
    # averages(beat_scores_second_max, labels)
    # print("Class averages for third maximum outlierness scores in beat:")
    # averages(beat_scores_third_max, labels)
    # print("Class averages for difference between maximum and average outlierness scores in beat:")
    # averages(diff_scores, labels)

    ## F1 scores
    # print("F1 scores:")
    # f1, threshold = compute_f1_score(beat_scores_avg, labels)
    # print("F1 score (Average): ", f1, " for threshold: ", threshold)
    # f1, threshold = compute_f1_score(beat_scores_max, labels)
    # print("F1 score (Maximum): ", f1, " for threshold: ", threshold)
    # f1, threshold = compute_f1_score(beat_scores_second_max, labels)
    # print("F1 score (Second Maximum): ", f1, " for threshold: ", threshold)
    # f1, threshold = compute_f1_score(beat_scores_third_max, labels)
    # print("F1 score (Third Maximum): ", f1, " for threshold: ", threshold)
    # f1, threshold = compute_f1_score(diff_scores, labels)
    # print("F1 score (Difference: Max and avg) : ", f1, " for threshold: ", threshold)

def get_beat_scores_window_auto(record, annotation, sampfrom, win_length, pres_norm, k):
    signal_norm = load_data_MITBIH.normalise(record.p_signal)
    expanded_signal_auto = []
    for i in range(1, len(signal_norm) + 1):
        previous = i - win_length
        if previous < 0:
            previous = 0
        signal = signal_norm[previous:i]
        signal_auto = []
        for s in signal:
            signal_auto.append(sm.tsa.acf(s))
        expanded_signal_auto.append(signal_auto)

    outlier_scores = random_projection_single(expanded_signal_auto, k, pres_norm)
    outlier_scores = np.array(outlier_scores) * 1 / max(outlier_scores)
    heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation, sampfrom)
    beat_outlier_scores = []

    for x in heart_beats_x:
        beat_scores = [outlier_scores[i] for i in x]
        beat_outlier_scores.append(beat_scores)

    return beat_outlier_scores, labels

def get_beat_scores(record, annotation, sampfrom):
    outlier_scores = random_projection_single(record.p_signal, 1, False)
    outlier_scores = np.array(outlier_scores)*1/max(outlier_scores)
    heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation, sampfrom)
    beat_outlier_scores = []
    for x in heart_beats_x:
        beat_scores = [outlier_scores[i] for i in x]
        beat_outlier_scores.append(beat_scores)
    return beat_outlier_scores, labels


def auc_k_test():
    aucs = [[] for _ in range(1, 16)]
    for k in range(1, 16):
        for j in range(10):
            print(k, "   ", j)
            beat_outlier_scores, labels = get_beat_scores_window(record, annotation, sampfrom, win_length, False, k)
            beat_scores_avg = [sum(score) / len(score) for score in beat_outlier_scores]
            beat_scores_max = [max(score) for score in beat_outlier_scores]
            diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
            diff_scores = 1 / max(diff_scores) * diff_scores
            tpr_diff, fpr_diff = compute_tpr_fpr(diff_scores, labels)
            aucs[k - 1].append(metrics.auc(fpr_diff, tpr_diff))

    aucs_final = []
    aucs_error = []
    for auc in aucs:
        aucs_final.append(np.mean(auc))
        aucs_error.append(np.std(auc) / np.sqrt(len(auc)))

    plt.errorbar(range(1, 16), aucs_final, yerr=aucs_error, fmt='.k')
    plt.title("AUC for different k values")
    plt.xlabel("k")
    plt.show()


# def get_beat_scores_window_mean(record, annotation, sampfrom, win_length, k):
#     signal_norm = load_data_MITBIH.normalise(record.p_signal)
#     expanded_signal = []
#     for i in range(1, len(signal_norm) + 1):
#         previous = i - win_length
#         if previous < 0:
#             previous = 0
#         expanded_signal.append(signal_norm[previous:i])
#     outlier_scores_means = mean_single(expanded_signal, k)
#     heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation, sampfrom)
#     beat_outlier_scores_means = []
#
#     for x in heart_beats_x:
#         beat_scores = [outlier_scores_means[i] for i in x]
#         beat_outlier_scores_means.append(beat_scores)
#
#     return beat_outlier_scores_means, labels
#
# def get_beat_scores_window(record, annotation, sampfrom, win_length, pres_norm, k):
#     signal_norm = load_data_MITBIH.normalise(record.p_signal)
#     expanded_signal = []
#     for i in range(1,len(signal_norm)+1):
#         previous = i-win_length
#         if previous < 0:
#             previous = 0
#         expanded_signal.append(signal_norm[previous:i])
#     outlier_scores = random_projection_single(expanded_signal, k, pres_norm)
#     heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation, sampfrom)
#     beat_outlier_scores = []
#
#     for x in heart_beats_x:
#         beat_scores = [outlier_scores[i] for i in x]
#         beat_outlier_scores.append(beat_scores)
#
#     return beat_outlier_scores, labels

    # Sim Investigation
    # print(labels[130], labels[131], labels[132])
    # print(predicted_label[130], predicted_label[131], predicted_label[132])
    # print(labels[137], labels[138], labels[139])
    # print(predicted_label[137], predicted_label[138], predicted_label[139])
    # print(labels[367], labels[368], labels[369], labels[370])
    # print(predicted_label[367], predicted_label[368], predicted_label[369], predicted_label[370])
