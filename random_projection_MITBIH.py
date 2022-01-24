import load_data_MITBIH
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import heapq


# Computes outlierness score as mean of interval
def mean_single(data_points, k):
    d = len(data_points[-1])

    outlier_scores = []

    # R just matrix of ones normalised by 1/d
    R = np.ones(shape=(k, d))
    for point in data_points:
        # If window smaller than d
        if len(point) < d:
            R_prime = R[:, :len(point)]
            mean = (1 / len(point) * R_prime) @ point
            score = np.linalg.norm(mean)
        else:
            mean = (1 / d * R) @ point
            score = np.linalg.norm(mean)
        outlier_scores.append(score)

    # normalise
    outlier_scores = np.array(outlier_scores) * (1 / max(outlier_scores))

    return outlier_scores


def random_projection_single(data_points, k, norm_perservation):
    d = len(data_points[-1])

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


# Create beats and determine their outlier scores
def get_beat_scores(signal_norm, heart_beats_x, labels, win_length, win_length_mean, k, pres_norm, mean_comparison):

    expanded_signal = []
    for i in range(1, len(signal_norm) + 1):
        previous = i - win_length
        if previous < 0:
            previous = 0
        expanded_signal.append(signal_norm[previous:i])


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
                previous = i - win_length_mean
                if previous < 0:
                    previous = 0
                expanded_signal_mean.append(signal_norm[previous:i])

            outlier_scores_means = mean_single(expanded_signal_mean, k)
        else:
            outlier_scores_means = mean_single(expanded_signal, k)

        beat_outlier_scores_means = []

        for x in heart_beats_x:
            beat_scores = [outlier_scores_means[i] for i in x]
            beat_outlier_scores_means.append(beat_scores)
        return beat_outlier_scores, labels, beat_outlier_scores_means
    else:
        return beat_outlier_scores, labels


# Compute TPR and FPR
def compute_tpr_fpr(scores, labels):
    tpr = []
    fpr = []

    labels = np.array(labels)

    thresholds = np.arange(0, 1, 0.01).tolist()
    for threshold in thresholds:
        predicted_label = np.where(np.array(scores) > threshold, 1, 0)

        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label, labels=[0, 1]).ravel()
        if TP + FN == 0:
            tpr.append(1)
        else:
            tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    gmeans = np.sqrt(np.array(tpr) * (1 - np.array(fpr)))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # For testing for swamping use:
    # predicted_label = np.where(np.array(scores) > thresholds[ix], 1, 0)

    return tpr, fpr


# Plot ROC curve for different beat summarisation methods
def show_roc_curve(scores, labels, type, sample, win_length):
    # beat summaries
    beat_scores_avg = [sum(score) / len(score) for score in scores]
    beat_scores_max = [max(score) for score in scores]
    beat_scores_second_max = [heapq.nlargest(2, score)[1] for score in scores]
    beat_scores_third_max = [heapq.nlargest(3, score)[2] for score in scores]
    diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
    diff_scores = 1 / max(diff_scores) * diff_scores

    # at least one
    print("Maximum")
    tpr_max, fpr_max = compute_tpr_fpr(beat_scores_max, labels)
    roc_auc_max = metrics.auc(fpr_max, tpr_max)

    # average
    print("Average")
    tpr_avg, fpr_avg = compute_tpr_fpr(beat_scores_avg, labels)
    roc_auc_avg = metrics.auc(fpr_avg, tpr_avg)

    # max-average
    print("Diff")
    tpr_diff, fpr_diff = compute_tpr_fpr(diff_scores, labels)
    roc_auc_diff = metrics.auc(fpr_diff, tpr_diff)

    # at least two
    print("largest two")
    tpr_second_max, fpr_second_max = compute_tpr_fpr(beat_scores_second_max, labels)
    roc_auc_second_max = metrics.auc(fpr_second_max, tpr_second_max)

    # at least three
    print("largest three")
    tpr_third_max, fpr_third_max = compute_tpr_fpr(beat_scores_third_max, labels)
    roc_auc_third_max = metrics.auc(fpr_third_max, tpr_third_max)

    plt.title(f'Receiver Operating Characteristic using {type} with window length {win_length} normalised')
    plt.plot(fpr_avg, tpr_avg, 'b', label='(average) AUC = %0.4f' % roc_auc_avg)
    plt.plot(fpr_max, tpr_max, 'k', label='(maximum) AUC = %0.4f' % roc_auc_max)
    plt.plot(fpr_diff, tpr_diff, 'g', label='(difference) AUC = %0.4f' % roc_auc_diff)
    plt.plot(fpr_second_max, tpr_second_max, 'y', label='(second maximum) AUC = %0.4f' % roc_auc_second_max)
    plt.plot(fpr_third_max, tpr_third_max, 'orange', label='(third maximum) AUC = %0.4f' % roc_auc_third_max)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f"./ROC_curves/sample_{sample}_{type}_{win_length}", bbox_inches='tight')
    plt.clf()


def get_histo(labels, scores, summ_method):
    normal = []
    outlier = []

    for i, v in enumerate(labels):
        if v == 0:  # normal
            normal.append(scores[i])
        else:
            outlier.append(scores[i])

    fig, ax = plt.subplots(1, 2)

    arr_normal = ax[0].hist(normal, bins=20)
    ax[0].set_title("Normal")
    ax[0].set(ylabel="Frequency", xlim=[0, 1])

    ax[1].set_title("Outlier")
    arr_outlier = ax[1].hist(outlier, bins=20)
    ax[1].set(xlim=[0, 1])

    fig.text(0.5, 0.04, f'Histogram of {summ_method} summarisation method', ha='center')

    plt.show()


def auc_for_different_window_length(record, annotation, sampfrom, k, pres_norm):
    aucs_rp_max = []
    aucs_rp_diff = []
    aucs_mean_max = []
    aucs_mean_diff = []

    for win_length in range(150, 450, 10):
        beat_outlier_scores, labels, beat_outlier_scores_mean = get_beat_scores(record, annotation, sampfrom,
                                                                                win_length, k, pres_norm, True,
                                                                                win_length)

        beat_scores_avg = [sum(score) / len(score) for score in beat_outlier_scores]
        beat_scores_max = [max(score) for score in beat_outlier_scores]
        diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
        diff_scores = 1 / max(diff_scores) * diff_scores

        tpr_diff, fpr_diff = compute_tpr_fpr(diff_scores, labels)
        tpr_max, fpr_max = compute_tpr_fpr(beat_scores_max, labels)

        aucs_rp_diff.append(metrics.auc(fpr_diff, tpr_diff))
        aucs_rp_max.append(metrics.auc(fpr_max, tpr_max))

        beat_scores_avg_mean = [sum(score) / len(score) for score in beat_outlier_scores_mean]
        beat_scores_max_mean = [max(score) for score in beat_outlier_scores_mean]
        diff_scores_mean = np.array(beat_scores_max_mean) - np.array(beat_scores_avg_mean)
        diff_scores_mean = 1 / max(diff_scores_mean) * diff_scores_mean

        tpr_diff_mean, fpr_diff_mean = compute_tpr_fpr(diff_scores_mean, labels)
        tpr_max_mean, fpr_max_mean = compute_tpr_fpr(beat_scores_max_mean, labels)

        aucs_mean_diff.append(metrics.auc(fpr_diff_mean, tpr_diff_mean))
        aucs_mean_max.append(metrics.auc(fpr_max_mean, tpr_max_mean))

    plt.plot(range(150, 450, 10), aucs_rp_diff, label="Random projection AUC difference")
    plt.plot(range(150, 450, 10), aucs_rp_max, label="Random projection AUC maximum")
    plt.plot(range(150, 450, 10), aucs_mean_diff, label="Mean AUC difference")
    plt.plot(range(150, 450, 10), aucs_mean_max, label="Mean AUC maximum")
    plt.title("AUC values for different window lengths")
    plt.xlabel("Window length")
    plt.legend()
    plt.savefig(f"./AUC/sample_{record.record_name}", bbox_inches='tight')
    plt.clf()

    return aucs_rp_max, aucs_rp_diff, aucs_mean_max, aucs_mean_diff


def run(record, annotation, sampfrom, plot):

    signal_norm, heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation, sampfrom)
    if plot:
        load_data_MITBIH.plot_data(record, signal_norm, heart_beats, heart_beats_x, labels, sampfrom)

    win_length = 260
    win_length_mean = 260

    beat_outlier_scores, labels, beat_outlier_scores_mean = get_beat_scores(signal_norm, heart_beats_x, labels,
                                                                            win_length, win_length_mean, k=1,
                                                                            pres_norm=False, mean_comparison=True)

    # Summary:
    print(
        f"Total: {len(labels)}, Number of outliers: {sum(labels)}, Percent of outliers: "
        f"{(sum(labels) / len(labels)) * 100:.3f}%")
    diff_anno = [annotation.sample[i + 1] - annotation.sample[i] for i in range(len(annotation.sample) - 1)]
    print("Average difference between peaks:", sum(diff_anno) / len(diff_anno))

    # ROC Curves
    show_roc_curve(beat_outlier_scores, labels, "random projection test", record.record_name, win_length)
    show_roc_curve(beat_outlier_scores_mean, labels, "Mean test", record.record_name, win_length_mean)

