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



###############################################################NAB Database#############################################
def make_intervals(length, data, labels, name, plot=False):
    x = np.array(data['timestamp'])
    y = np.array(data['value'])
    y_norm = normalise(np.array(data['value']).reshape(-1, 1))

    if data['timestamp'].iloc[0].day != data['timestamp'].iloc[length-1].day:
        days = data['timestamp'].dt.date
        next_day = (data['timestamp'].iloc[0] + timedelta(days=1)).date()
        start = days[days == next_day].index[0]
        print("Trimmed beginning: did not start at the beginning of the day.")
    else:
        start = 0
    intervals = []
    intervals_x = []
    interval_labels = []
    y_plot = []

    for i in range((len(data) - start) // length):
        if ((i+1)*length + 1 > len(data)):
            break
        intervals.append(y_norm[start + i * length:start + (i + 1) * length + 1])
        intervals_x.append(x[start + i * length:start + (i + 1) * length + 1])
        y_plot.append(y[start + i * length:start + (i + 1) * length + 1])
        label = 0
        for j in range(start + i * length, start + (i + 1) * length):
            if labels[j] == 1:
                label = 1
                break
        interval_labels.append(label)
    if plot:
        colours = []
        for i in interval_labels:
            if i == 0:
                colours.append("k")
            else:
                colours.append("r")

        for i, c in enumerate(colours):
            plt.plot(intervals_x[i], y_plot[i], color=c)
            plt.scatter(intervals_x[i][0], y_plot[i][0], color='b', s=20)

        plt.title(name.replace("_", " ").capitalize())
        plt.xlabel("Timestamp")
        plt.show()

    return y_norm, intervals, intervals_x, interval_labels


def get_interval_scores(data, data_norm, intervals_x, labels, win_length, win_length_mean, k, pres_norm, mean_comparison, method):
    expanded_data = []
    for i in range(len(data_norm)):
        if win_length == 0:
            expanded_data.append(data_norm[i])
            continue
        if method == 'prev':
            previous = i - win_length +1
        else:
            previous = i - win_length //2
        future = i+ win_length//2
        if previous < 0:
            previous = 0
        if future > len(data_norm):
            future = len(data_norm)
        if method == "prev":
            expanded_data.append(data_norm[previous:i+1])
        else:
            expanded_data.append(data_norm[previous:future])
    scores = rp.random_projection_new(expanded_data, k, pres_norm)
    interval_scores = []

    for i,interval in enumerate(intervals_x):
        index = data.index[data['timestamp'] == interval[0]].tolist()[0]
        interval_score = [scores[i] for i in range(index,index +len(interval))]
        interval_scores.append(interval_score)

    if mean_comparison:
        # window_length for mean might be different to RP method.
        if win_length != win_length_mean:
            expanded_data_mean = []
            for i in range(len(data_norm)):
                if win_length == 0:
                    expanded_data.append(data_norm[i])
                    continue
                if method == "prev":
                    previous = i - win_length +1
                else:
                    previous = i - win_length//2
                future = i + win_length //2
                if previous < 0:
                    previous = 0
                if future > len(data_norm):
                    future = len(data_norm)
                if method == "prev":
                    expanded_data_mean.append(data_norm[previous:i+1])
                else:
                    expanded_data_mean.append(data_norm[previous:future])

            outlier_scores_means = rp.mean_single_new(expanded_data_mean, data_norm)
            outlier_scores_means_old = rp.mean_single(expanded_data_mean, k)
        else:
            outlier_scores_means = rp.mean_single_new(expanded_data, data_norm)
            outlier_scores_means_old = rp.mean_single(expanded_data, k)

        interval_outlier_scores_means = []
        interval_outlier_scores_means_old = []


        for interval in intervals_x:
            index = data.index[data['timestamp'] == interval[0]].tolist()[0]
            interval_score = [outlier_scores_means[i] for i in range(index, index + len(interval))]
            interval_outlier_scores_means.append(interval_score)
            interval_score = [outlier_scores_means_old[i] for i in range(index, index + len(interval))]
            interval_outlier_scores_means_old.append(interval_score)
        return interval_scores, labels, interval_outlier_scores_means, interval_outlier_scores_means_old
    else:
        return interval_scores, labels, None, None


##################################### Ensemble ######################################
def summarise_scores_var(all_scores):
    final_scores = []
    for c in range(len(all_scores[0])):
        scores_per_point = [score[c] for score in all_scores]
        final_scores.append(np.var(scores_per_point))
    return final_scores


def list_to_percentiles(numbers):
    pairs = list(zip(numbers, range(len(numbers))))
    pairs.sort(key=lambda p: p[0])
    result = [0 for i in range(len(numbers))]
    for rank in range(len(numbers)):
        original_index = pairs[rank][1]
        result[original_index] = rank * 100.0 / (len(numbers)-1)
    return result


def standardise_scores_relative(scores):
    return list_to_percentiles(scores)


def standardise_scores_maxmin(scores):
    return 1 / (max(scores) - min(scores)) * (np.array(scores) - min(scores))


@jit(nopython=True)
def summarise_scores(all_scores):
    final_scores = np.array([0.0 for _ in range(len(all_scores[0]))])
    # heights = []
    # locations = []
    for c in range(len(all_scores[0])):
        # scores_per_point = [score[c] for score in all_scores]
        scores_per_point = all_scores[:, c]
        # n, bins, _ = axs[0].hist(scores_per_point, bins='auto', alpha = 0.5)
        up = np.quantile(scores_per_point, 0.99)
        low = np.quantile(scores_per_point, 0.01)
        final_scores[c] = max(up, 1-low)
        # final_scores.append(max(scores_per_point))
        # heights.append(max(n))
        # locations.append(round(bins[np.argmax(n)],2))
    return final_scores
