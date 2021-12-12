import load_data_MITBIH
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import heapq


def random_projection_single(data_points, k, norm_perservation):
    d = data_points.shape[1]

    outlier_scores = []

    R = np.random.normal(loc=0.0, scale=1.0, size=(k, d))
    for point in data_points:
        point_proj = (1 / np.sqrt(d) * R) @ point
        point_reconstruct = (1 / np.sqrt(d) * R.T) @ point_proj
        if norm_perservation:
            point_reconstruct = np.sqrt(d / k) * point_reconstruct
        outlier_score = np.linalg.norm(point - point_reconstruct)
        outlier_scores.append(outlier_score)
    return outlier_scores


def get_beat_scores(record, annotation, sampfrom):
    outlier_scores = random_projection_single(record.p_signal, 1, False)
    outlier_scores = np.array(outlier_scores)*1/max(outlier_scores)
    heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation, sampfrom)
    beat_outlier_scores = []
    for x in heart_beats_x:
        beat_scores = [outlier_scores[i] for i in x]
        beat_outlier_scores.append(beat_scores)
    return beat_outlier_scores, labels


def averages(beat_outlier_scores, labels):
    normal_avg = []
    outlier_avg = []
    for i in range(len(labels)):
        if labels[i] == 0:  # normal
            normal_avg.append(beat_outlier_scores[i])
        else:
            outlier_avg.append(beat_outlier_scores[i])

    print("Normal average:", round(sum(normal_avg) / len(normal_avg),4), "+/-", round(np.std(normal_avg),4))
    print("Outlier average:", round(sum(outlier_avg) / len(outlier_avg),4), "+/-", round(np.std(outlier_avg),4))
    if (sum(normal_avg) / len(normal_avg) + np.std(normal_avg)) > (sum(outlier_avg) / len(outlier_avg) - np.std(outlier_avg)):
        print("overlap")


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

def compute_tpr_fpr(scores, labels):
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


def compute_tpr_fpr_flipped(scores, labels):
    tpr = []
    fpr = []

    thresholds = np.arange(0, 1, 0.01).tolist()
    for threshold in thresholds:
        predicted_label = []
        for beat in scores:
            if beat > threshold:
                predicted_label.append(0)
            else:
                predicted_label.append(1)

        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label).ravel()

        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    return tpr, fpr

def show_roc_curve(beat_scores_max, beat_scores_avg, beat_scores_second_max, beat_scores_third_max, labels):
    #at least one
    print("Maximum")
    tpr_max, fpr_max = compute_tpr_fpr(beat_scores_max, labels)
    roc_auc_max = metrics.auc(fpr_max, tpr_max)

    #average
    print("Average")
    tpr_avg, fpr_avg = compute_tpr_fpr(beat_scores_avg, labels)
    roc_auc_avg = metrics.auc(fpr_avg, tpr_avg)

    #max-average
    print("Difference: Maximum and Average")
    diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
    diff_scores = 1/max(diff_scores)*diff_scores
    tpr_diff, fpr_diff = compute_tpr_fpr(diff_scores, labels)
    roc_auc_diff = metrics.auc(fpr_diff, tpr_diff)

    #max-second max
    print("Difference: Maximum and Second Largest")
    diff_scores_max = np.array(beat_scores_max) - np.array(beat_scores_second_max)
    diff_scores_max = 1 / max(diff_scores_max) * diff_scores_max
    tpr_diff_max, fpr_diff_max = compute_tpr_fpr_flipped(diff_scores_max, labels)
    roc_auc_diff_max = metrics.auc(fpr_diff_max, tpr_diff_max)

    #at least two
    print("Second Largest")
    tpr_second_max, fpr_second_max = compute_tpr_fpr(beat_scores_second_max, labels)
    roc_auc_second_max = metrics.auc(fpr_second_max, tpr_second_max)

    #at least three
    print("Third Largest")
    tpr_third_max, fpr_third_max = compute_tpr_fpr(beat_scores_third_max, labels)
    roc_auc_third_max = metrics.auc(fpr_third_max, tpr_third_max)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_avg, tpr_avg, 'b', label='(average) AUC = %0.2f' % roc_auc_avg)
    plt.plot(fpr_max, tpr_max, 'k', label='(maximum) AUC = %0.2f' % roc_auc_max)
    plt.plot(fpr_diff, tpr_diff, 'g', label='(difference) AUC = %0.2f' % roc_auc_diff)
    plt.plot(fpr_second_max, tpr_second_max, 'y', label='(second maximum) AUC = %0.2f' % roc_auc_second_max)
    plt.plot(fpr_third_max, tpr_third_max, 'orange', label='(third maximum) AUC = %0.2f' % roc_auc_third_max)
    # plt.plot(fpr_diff_max, tpr_diff_max, 'purple', label='(Difference maximum flipped) AUC = %0.2f' % roc_auc_diff_max)
    # plt.plot(fpr_diff_max_2, tpr_diff_max_2, 'pink', label='(Difference maximum) AUC = %0.2f' % roc_auc_diff_max_2)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()


def compute_f1_score(scores, labels):
    f1 =[]
    thresholds = np.arange(0, 1, 0.01).tolist()
    for threshold in thresholds:
        predicted_label = []
        for beat in scores:
            if beat > threshold:
                predicted_label.append(0)
            else:
                predicted_label.append(1)
        TN, FP, FN, TP = metrics.confusion_matrix(labels, predicted_label).ravel()
        f1.append(TP/(TP+1/2*(FP+FN)))
    return max(f1), thresholds[f1.index(max(f1))]


def main(name, sampfrom, sampto):

    record, annotation = load_data_MITBIH.load_mit_bih_data(name, True, sampfrom, sampto)
    beat_outlier_scores, labels = get_beat_scores(record,annotation, sampfrom)

    beat_scores_avg = [sum(score)/len(score) for score in beat_outlier_scores]
    beat_scores_max = [max(score) for score in beat_outlier_scores]
    beat_scores_second_max = [heapq.nlargest(2, score)[1] for score in beat_outlier_scores]
    beat_scores_third_max = [heapq.nlargest(3, score)[2] for score in beat_outlier_scores]
    # diff_scores = np.array(beat_scores_max) - np.array(beat_scores_avg)
    # diff_scores = 1 / max(diff_scores) * diff_scores

    ## ROC Curves
    # show_roc_curve(beat_scores_max, beat_scores_avg, beat_scores_second_max,beat_scores_third_max, labels)

    ## Percentage of outliers:
    # print(f"Percent of outliers: {(sum(labels) / len(labels))*100:.2f}%")

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


    # show_pr_curve_basic(labels,beat_scores_avg,beat_scores_max)







