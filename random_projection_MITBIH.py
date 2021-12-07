import load_data_MITBIH
import numpy as np
from sklearn import random_projection


def random_projection(data_points, k, norm_persevation):
    d = data_points.shape[1]

    outlier_scores = []
    # generate R (k x d)
    R = np.random.normal(loc=0.0, scale=1.0, size=(k, d))
    for point in data_points:
        point_proj = (1 / np.sqrt(d) * R) @ point
        point_reconstruct = (1 / np.sqrt(d) * R.T) @ point_proj
        if norm_persevation:
            point_reconstruct = np.sqrt(d / k) * point_reconstruct
        outlier_score = np.linalg.norm(point - point_reconstruct)
        outlier_scores.append(outlier_score)
    return outlier_scores


def main():
    record, annotation = load_data_MITBIH.load_mit_bih_data(True, 20000)
    # Determine dimension k
    outlier_scores = random_projection(record.p_signal, 1, False)
    heart_beats, heart_beats_x, labels = load_data_MITBIH.label_segment(record, annotation)

    beat_outlier_scores = []
    for x in heart_beats_x:
        beat_scores = [outlier_scores[i] for i in x]
        # beat_scores_avg = sum(beat_scores)/len(beat_scores)
        beat_outlier_scores.append(max(beat_scores))

    normal_avg = []
    outlier_avg = []
    for i in range(len(labels)):
        if labels[i] == 0:  # normal
            normal_avg.append(beat_outlier_scores[i])
        else:
            outlier_avg.append(beat_outlier_scores[i])

    print("Normal average:", sum(normal_avg) / len(normal_avg))
    print("Outlier average:", sum(outlier_avg) / len(outlier_avg))

    normal_percentage = 1 - (sum(labels) / len(labels))
    # print('normal percentage:', normal_percentage)
