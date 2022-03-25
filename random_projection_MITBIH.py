import load_data_MITBIH
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import heapq
import time


# Computes outlierness score as mean of interval
def mean_single(data_points, k):
    d = max(len(point) for point in data_points)

    outlier_scores = []

    # R just matrix of ones normalised by 1/d
    R = np.ones(shape=(k, d))
    for point in data_points:
        R_prime = R[:, :len(point)]
        mean = (1 / len(point) * R_prime) @ point
        score = np.linalg.norm(mean)
        outlier_scores.append(score)

    # normalise
    outlier_scores = np.array(outlier_scores) * (1 / max(outlier_scores))

    return outlier_scores


def mean_single_new(data_points, signal):
    outlier_scores = []
    for i, point in enumerate(data_points):
        mean = np.mean(point, axis=0)
        score = np.linalg.norm(signal[i] - mean)
        outlier_scores.append(score)

    # normalise
    if max(outlier_scores) != 0:
        outlier_scores = np.array(outlier_scores) * (1 / max(outlier_scores))

    return outlier_scores


def random_projection_new(data_points, k, norm_perservation):
    d = max([len(point) for point in data_points])
    outlier_scores = []

    R = np.random.normal(loc=0.0, scale=1.0, size=(k, d))

    for point in data_points:
        # If window smaller than d
        R_prime = R[:, :len(point)]
        point_proj = (1 / np.sqrt(len(point)) * R_prime) @ point
        point_reconstruct = (1 / np.sqrt(len(point)) * R_prime.T) @ point_proj

        if norm_perservation:
            point_reconstruct = np.sqrt(d / k) * point_reconstruct

        outlier_score = np.linalg.norm(point - point_reconstruct)
        outlier_scores.append(outlier_score)
    # normalise
    outlier_scores = np.array(outlier_scores) * (1 / max(outlier_scores))


    return outlier_scores


def random_projection_single(data_points, k, norm_perservation):
    d = max([len(point) for point in data_points])

    outlier_scores = []

    R = np.random.normal(loc=0.0, scale=1.0, size=(k, d))

    for point in data_points:
        # If window smaller than d
        if len(point) < d:
            R_prime = R[:, :len(point)]
            point_proj = (1 / np.sqrt(d) * R_prime) @ point
            point_reconstruct = (1 / np.sqrt(d) * R_prime.T) @ point_proj
        else:
            point_proj = (1 / np.sqrt(d) * R) @ point
            point_reconstruct = (1 / np.sqrt(d) * R.T) @ point_proj

        if norm_perservation:
            point_reconstruct = np.sqrt(d / k) * point_reconstruct

        outlier_score = np.linalg.norm(point - point_reconstruct)
        outlier_scores.append(outlier_score)

    # normalise
    outlier_scores = np.array(outlier_scores) * (1 / max(outlier_scores))

    return outlier_scores


def get_beat_scores_beats(signal_norm, heart_beats_x, labels, k, pres_norm, mean_comparison):
    expanded_signal = []
    for x in heart_beats_x:
        expanded_signal.append(signal_norm[x])

    beat_outlier_scores = random_projection_new(expanded_signal, k, pres_norm)

    if mean_comparison:
        index = 0
        expanded_signal_means = []
        first = heart_beats_x[0][0]
        last = heart_beats_x[-1][-1]
        for i in range(first, last + 1):
            if i not in heart_beats_x[index]:
                print("error")
            expanded_signal_means.append(signal_norm[heart_beats_x[index]])
            if i == heart_beats_x[index][-1]:
                index += 1

        outlier_scores_means = mean_single_new(expanded_signal_means, signal_norm[first: last + 1])
        outlier_scores_means_old = mean_single(expanded_signal_means, k)

        beat_outlier_scores_means = []
        beat_outlier_scores_means_old = []

        for x in heart_beats_x:
            beat_scores = [outlier_scores_means[i - first] for i in x]
            beat_outlier_scores_means.append(beat_scores)
            beat_scores_old = [outlier_scores_means_old[i - first] for i in x]
            beat_outlier_scores_means_old.append(beat_scores_old)
        return beat_outlier_scores, labels, beat_outlier_scores_means, beat_outlier_scores_means_old
    else:
        return beat_outlier_scores, labels


# Create beats and determine their outlier scores
def get_beat_scores(signal_norm, heart_beats_x, labels, win_length, win_length_mean, k, pres_norm, mean_comparison):
    expanded_signal = []
    for i in range(1, len(signal_norm) + 1):
        previous = i - win_length + 1
        if previous < 0:
            previous = 0
        expanded_signal.append(signal_norm[previous:i+1])

    beat_outlier_scores = []
    outlier_scores = random_projection_single(expanded_signal, k, pres_norm)

    for x in heart_beats_x:
        beat_scores = [outlier_scores[i] for i in x]
        beat_outlier_scores.append(beat_scores)

    if mean_comparison:
        # window_length for mean might be different to RP method.
        if win_length != win_length_mean:
            expanded_signal_mean = []
            for i in range(1, len(signal_norm) + 1):
                previous = i - win_length_mean +1
                if previous < 0:
                    previous = 0
                expanded_signal_mean.append(signal_norm[previous:i+1])

            outlier_scores_means = mean_single_new(expanded_signal_mean, signal_norm)
            outlier_scores_means_old = mean_single(expanded_signal_mean, k)
        else:
            outlier_scores_means = mean_single_new(expanded_signal, signal_norm)
            outlier_scores_means_old = mean_single(expanded_signal, k)

        beat_outlier_scores_means = []
        beat_outlier_scores_means_old = []

        for x in heart_beats_x:
            beat_scores = [outlier_scores_means[i] for i in x]
            beat_outlier_scores_means.append(beat_scores)
            beat_scores_old = [outlier_scores_means_old[i] for i in x]
            beat_outlier_scores_means_old.append(beat_scores_old)
        return beat_outlier_scores, labels, beat_outlier_scores_means, beat_outlier_scores_means_old
    else:
        return beat_outlier_scores, labels

# Create beats and determine their outlier scores
def get_beat_scores_beat_size(signal_norm, heart_beats_x, labels, k, pres_norm, mean_comparison):
    expanded_signal = []
    index = 0
    first = heart_beats_x[0][0]
    last = heart_beats_x[-1][-1]
    for i in range(first, last+1):
        win_length = len(heart_beats_x[index])
        previous = i - win_length+1
        if previous < 0:
            previous = 0
        expanded_signal.append(signal_norm[previous:i+1])
        if i == heart_beats_x[index][-1]:
            index += 1

    beat_outlier_scores = []
    outlier_scores = random_projection_single(expanded_signal, k, pres_norm)

    for x in heart_beats_x:
        beat_scores = [outlier_scores[i-first] for i in x]
        beat_outlier_scores.append(beat_scores)

    if mean_comparison:

        outlier_scores_means = mean_single_new(expanded_signal, signal_norm)
        outlier_scores_means_old = mean_single(expanded_signal, k)

        beat_outlier_scores_means = []
        beat_outlier_scores_means_old = []

        for x in heart_beats_x:
            beat_scores = [outlier_scores_means[i-first] for i in x]
            beat_outlier_scores_means.append(beat_scores)
            beat_scores_old = [outlier_scores_means_old[i-first] for i in x]
            beat_outlier_scores_means_old.append(beat_scores_old)
        return beat_outlier_scores, labels, beat_outlier_scores_means, beat_outlier_scores_means_old
    else:
        return beat_outlier_scores, labels


# Compute TPR and FPR
def compute_tpr_fpr(scores, labels, min, max):
    tpr = []
    fpr = []

    labels = np.array(labels)
    step = (max - min)/100
    thresholds = np.arange(min, max, step).tolist()
    for threshold in thresholds:
        predicted_label = np.where(np.array(scores) > threshold, 1, 0)

        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label, labels=[0, 1]).ravel()
        if TP + FN == 0:
            print("TP and FN zero")
            tpr.append(0)
        else:
            tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    # G-means calculation
    # gmeans = np.sqrt(np.array(tpr) * (1 - np.array(fpr)))
    # ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # For testing for swamping use:
    # predicted_label = np.where(np.array(scores) > thresholds[ix], 1, 0)
    # tpr.reverse()
    # fpr.reverse()

    return tpr, fpr

# Plot ROC curve for different beat summarisation methods
def show_roc_curve(scores, labels, method, sample, win_length):
    print(method)
    # beat summaries
    if type(scores[0]) == np.float64:
        beat_scores_max = scores
        beat_scores_avg = scores
    else:
        beat_scores_avg = [sum(score) / len(score) for score in scores]
        beat_scores_max = [max(score) for score in scores]
    # beat_scores_second_max = [heapq.nlargest(2, score)[1] for score in scores]
    # beat_scores_third_max = [heapq.nlargest(3, score)[2] for score in scores]
    diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
    if max(diff_scores) != 0:
        diff_scores = 1 / max(diff_scores) * diff_scores

    # at least one
    # print("Maximum")
    tpr_max, fpr_max =compute_tpr_fpr (beat_scores_max, labels)
    roc_auc_max = metrics.auc(fpr_max, tpr_max)
    print("Maximum", roc_auc_max)

    # average
    # print("Average")
    tpr_avg, fpr_avg = compute_tpr_fpr(beat_scores_avg, labels)
    roc_auc_avg = metrics.auc(fpr_avg, tpr_avg)

    # max-average
    # print("Diff")
    tpr_diff, fpr_diff = compute_tpr_fpr(diff_scores, labels)
    roc_auc_diff = metrics.auc(fpr_diff, tpr_diff)
    print("diff", roc_auc_diff)

    # get_histo(labels, diff_scores, "Difference")
    #
    # get_histo(labels, beat_scores_max, "Maximum")

    # at least two
    # print("largest two")
    # tpr_second_max, fpr_second_max = compute_tpr_fpr(beat_scores_second_max, labels)
    # roc_auc_second_max = metrics.auc(fpr_second_max, tpr_second_max)

    # at least three
    # print("largest three")
    # tpr_third_max, fpr_third_max = compute_tpr_fpr(beat_scores_third_max, labels)
    # roc_auc_third_max = metrics.auc(fpr_third_max, tpr_third_max)

    #
    #
    # plt.title(f'Receiver Operating Characteristic using {method} with window length {win_length} normalised')
    # plt.plot(fpr_avg, tpr_avg, 'b', label='(average) AUC = %0.4f' % roc_auc_avg)
    # plt.plot(fpr_max, tpr_max, 'k', label='(maximum) AUC = %0.4f' % roc_auc_max)
    # plt.plot(fpr_diff, tpr_diff, 'g', label='(difference) AUC = %0.4f' % roc_auc_diff)
    # plt.plot(fpr_second_max, tpr_second_max, 'y', label='(second maximum) AUC = %0.4f' % roc_auc_second_max)
    # plt.plot(fpr_third_max, tpr_third_max, 'orange', label='(third maximum) AUC = %0.4f' % roc_auc_third_max)

    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
    # plt.clf()
    # plt.savefig(f"./ROC_curves/sample_{sample}_{method}_{win_length}", bbox_inches='tight')
    # plt.clf()
    # print(np.std(beat_scores_avg))
    # print("average", roc_auc_avg,"max", roc_auc_max,"diff", roc_auc_diff)
    # if np.std(beat_scores_avg) > 0.04:
    #     return roc_auc_max, "max"
    # else:
    #     return roc_auc_diff, "diff"
    return roc_auc_max, roc_auc_diff


def get_histo(labels, scores, summ_method, plot=False):
    normal = []
    outlier = []

    for i, v in enumerate(labels):
        if v == 0:  # normal
            normal.append(scores[i])
        else:
            outlier.append(scores[i])

    bins = 20

    if plot:
        plt.hist(normal, bins=bins, density=True, label="Normal", alpha=0.5)
        plt.hist(outlier, bins=bins, density=True, label="Outlier", alpha=0.5)
        plt.title(f"Overlapping Densities for {summ_method} summarisation method")
        plt.xlim([0, 1])
        plt.legend()
        plt.show()

    return normal, outlier


def auc_for_different_window_length(record, annotation, sampfrom, k, pres_norm, mean_comparison):
    aucs_rp_max = []
    aucs_rp_diff = []
    aucs_mean_max = []
    aucs_mean_diff = []

    signal_norm, heart_beats, heart_beats_x, labels, labels_plot = load_data_MITBIH.label_clean_segments_q_points(
        record,
        annotation,
        sampfrom)
    for win_length in range(50, 450, 10):
        print(win_length)
        beat_outlier_scores, labels, beat_outlier_scores_mean,beat_outlier_scores_means_old = get_beat_scores(signal_norm, heart_beats_x, labels,
                                                                                win_length, win_length, k,
                                                                                pres_norm, mean_comparison)

        beat_scores_avg = [sum(score) / len(score) for score in beat_outlier_scores]
        beat_scores_max = [max(score) for score in beat_outlier_scores]
        diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
        diff_scores = 1 / max(diff_scores) * diff_scores

        tpr_diff, fpr_diff = compute_tpr_fpr(diff_scores, labels)
        tpr_max, fpr_max = compute_tpr_fpr(beat_scores_max, labels)

        aucs_rp_diff.append(metrics.auc(fpr_diff, tpr_diff))
        aucs_rp_max.append(metrics.auc(fpr_max, tpr_max))

        # beat_scores_avg_mean = [sum(score) / len(score) for score in beat_outlier_scores_mean]
        # beat_scores_max_mean = [max(score) for score in beat_outlier_scores_mean]
        # diff_scores_mean = np.array(beat_scores_max_mean) - np.array(beat_scores_avg_mean)
        # diff_scores_mean = 1 / max(diff_scores_mean) * diff_scores_mean
        #
        # tpr_diff_mean, fpr_diff_mean = compute_tpr_fpr(diff_scores_mean, labels)
        # tpr_max_mean, fpr_max_mean = compute_tpr_fpr(beat_scores_max_mean, labels)
        #
        # aucs_mean_diff.append(metrics.auc(fpr_diff_mean, tpr_diff_mean))
        # aucs_mean_max.append(metrics.auc(fpr_max_mean, tpr_max_mean))
    print(aucs_rp_max.index(max(aucs_rp_max)), max(aucs_rp_max))
    print(aucs_rp_diff.index(max(aucs_rp_diff)), max(aucs_rp_diff))
    # return aucs_rp_max, aucs_rp_diff, aucs_mean_max, aucs_mean_diff


def run(record, annotation, sampfrom, plot):
    # print(record.record_name)
    # start = time.time()
    signal_norm, heart_beats, heart_beats_x, labels, labels_plot = load_data_MITBIH.label_clean_segments_q_points(
        record, annotation, sampfrom)

    if plot:
        load_data_MITBIH.plot_data(record, heart_beats_x, labels, labels_plot, sampfrom)

    # pre = time.time()
    # print("Preprocessing done: ", pre-start)
    q_points = np.genfromtxt(f'Q_points/{record.record_name}Q.dat', delimiter=',')
    diff = [q_points[i + 1] - q_points[i] for i in range(len(q_points) - 1)]

    win_length = int(np.mean(diff)- np.std(diff))
    win_length_mean = int(np.mean(diff)- np.std(diff))

    # beat_outlier_scores, labels, beat_outlier_scores_mean, beat_outlier_scores_mean_old = get_beat_scores_beats(signal_norm, heart_beats_x, labels,
    #                                                                               k=1, pres_norm=False,
    #                                                                               mean_comparison=True)
    beat_outlier_scores, labels, beat_outlier_scores_mean, beat_outlier_scores_mean_old = get_beat_scores(signal_norm, heart_beats_x, labels,
                                                                            win_length, win_length_mean, k=1,
                                                                            pres_norm=False, mean_comparison=True)

    # beat_outlier_scores, labels, beat_outlier_scores_mean, beat_outlier_scores_mean_old = get_beat_scores_beat_size(signal_norm,heart_beats_x,labels,1,False,True)
    # for i,l in enumerate(labels):
    #     if l == 1:
    #         print(heart_beats_x[i])
    #         print(len(heart_beats_x[i]))
    # ran = time.time()
    # print("Random Projection done", ran - pre)
    # Summary:
    # print(
    #     f"Total: {len(labels)}, Number of outliers: {sum(labels)}, Percent of outliers: "
    #     f"{(sum(labels) / len(labels)) * 100:.3f}%")
    #
    # print("Average difference between peaks:", np.mean(diff), np.std(diff))
    # normal = []
    # outlier = []
    # for i, v in enumerate(labels):
    #     if v == 0:  # normal
    #         normal.append(heart_beats_x[i][-1]-heart_beats_x[i][0])
    #     else:
    #         outlier.append(heart_beats_x[i][-1]-heart_beats_x[i][0])
    # plt.hist(normal, bins=50, label="Normal", alpha=0.5)
    # plt.hist(outlier, bins=50, label="Outlier", alpha=0.5)
    # plt.title(f"Beat lengths")
    # plt.legend()
    # plt.show()


    # ROC Curves
    auc, method = show_roc_curve(beat_outlier_scores, labels, "random projection win_length=avg(beat length)",
                                     record.record_name, win_length)
    # # print("mean")
    # mean_max, mean_diff = show_roc_curve(beat_outlier_scores_mean, labels, "Mean test new", record.record_name,
    #                                      win_length_mean)
    # # print("mean old")
    # mean_max_old, mean_diff_old = show_roc_curve(beat_outlier_scores_mean_old, labels, "Mean test old", record.record_name,
    #                                      win_length_mean)
    # # end = time.time()
    # print("Total time: ", end-start)
    print(record.record_name, method, round(auc, 3))
    # print(record.record_name, round(rp_max,3), round(rp_diff,3), round(mean_max,3), round(mean_diff,3), round(mean_max_old,3), round(mean_diff_old,3))
    return auc
