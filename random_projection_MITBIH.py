import load_data_MITBIH
import numpy as np
from sklearn import random_projection


def random_projection_single(data_points, k, norm_perservation):
    d = data_points.shape[1]

    outlier_scores = []
    # generate R (k x d)
    R = np.random.normal(loc=0.0, scale=1.0, size=(k, d))
    for point in data_points:
        point_proj = (1 / np.sqrt(d) * R) @ point
        point_reconstruct = (1 / np.sqrt(d) * R.T) @ point_proj
        if norm_perservation:
            point_reconstruct = np.sqrt(d / k) * point_reconstruct
        outlier_score = np.linalg.norm(point - point_reconstruct)
        outlier_scores.append(outlier_score)
    return outlier_scores


def average_beat_score(beat_outlier_scores, labels):
    beat_scores_avg = [sum(score)/len(score) for score in beat_outlier_scores]

    #Basic Evaluation
    normal_avg = []
    outlier_avg = []
    for i in range(len(labels)):
        if labels[i] == 0:  # normal
            normal_avg.append(beat_scores_avg[i])
        else:
            outlier_avg.append(beat_scores_avg[i])

    print("Normal average:", sum(normal_avg) / len(normal_avg))
    print("Outlier average:", sum(outlier_avg) / len(outlier_avg))


def max_beat_score(beat_outlier_scores, labels):

    beat_scores_max = [max(score) for score in beat_outlier_scores]

    normal_avg = []
    outlier_avg = []
    for i in range(len(labels)):
        if labels[i] == 0:  # normal
            normal_avg.append(beat_scores_max[i])
        else:
            outlier_avg.append(beat_scores_max[i])

    print("Normal average:", sum(normal_avg) / len(normal_avg))
    print("Outlier average:", sum(outlier_avg) / len(outlier_avg))


def get_beat_scores(record, annotation):
    outlier_scores = random_projection_single(record.p_signal, 1, False)
    heart_beats, heart_beats_x, labels = load_data_MITBIH.label_clean_segments(record, annotation)
    beat_outlier_scores = []
    for x in heart_beats_x:
        beat_scores = [outlier_scores[i] for i in x]
        beat_outlier_scores.append(beat_scores)
    return beat_outlier_scores, labels

def main():
    record, annotation = load_data_MITBIH.load_mit_bih_data(False, None)
    beat_outlier_scores, labels = get_beat_scores(record,annotation)
    average_beat_score(beat_outlier_scores, labels)
    max_beat_score(beat_outlier_scores, labels)


    normal_percentage = 1 - (sum(labels) / len(labels))
    # print('normal percentage:', normal_percentage)
