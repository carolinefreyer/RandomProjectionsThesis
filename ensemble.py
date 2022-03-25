import matplotlib.pyplot as plt
import numpy as np
import random

from numba import jit

import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy as sc
from tqdm import tqdm
import load_data_MITBIH as mb

import load_data_NAB as nab
import plot_curves as pc


def normalise(signal):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


def make_windows(data, win_pos, win_length):
    expanded_data = []
    for i in range(len(data)):
        if win_length == 0:
            expanded_data.append(data[i])
            continue
        if win_pos == 'prev':
            previous = i - win_length + 1
            future = i
        elif win_pos == "mid":
            previous = i - win_length // 2
            future = i + win_length // 2
        else:
            previous = i
            future = i + win_length - 1

        if previous < 0:
            previous = 0
        if future > len(data):
            future = len(data)

        expanded_data.append(data[previous:future+1])

    return expanded_data


def random_projection_single(data_points, k, norm_perservation, norm_power):

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

        outlier_score = np.linalg.norm(point - point_reconstruct) ** norm_power
        outlier_scores.append(outlier_score)

    return outlier_scores

jit(nopython=True)
def mean_prime_2(signal, norm_power_range, win_length_range):
    expanded_data_mean = []
    win_length_range = [win_length_range[i] for i in range(len(win_length_range)) if win_length_range[i] > 1]
    win_length = random.choice(win_length_range)
    for i in range(0, len(signal)):
        previous = i - win_length//2
        future = i + win_length//2
        if previous < 0:
            previous = 0
        if future > len(signal) - 1:
            future = len(signal) - 1
        expanded_data_mean.append(np.append(signal[previous:i], signal[i + 1:future + 1]).reshape(-1, 1))

    outlier_scores = []
    norm_power = random.choice(norm_power_range)
    for i, point in enumerate(expanded_data_mean):
        mean = np.mean(point, axis=0)
        score = np.linalg.norm(signal[i] - mean) ** norm_power
        outlier_scores.append(score)

    # normalise
    if max(outlier_scores) != 0:
        outlier_scores = np.array(outlier_scores) * (1 / max(outlier_scores))

    return outlier_scores


def standardise_scores_z_score(scores):
    return sc.stats.zscore(scores)


def standardise_scores_maxmin(scores):
    return 1 / (max(scores) - min(scores)) * (np.array(scores) - min(scores))


def list_to_percentiles(numbers):
    pairs = list(zip(numbers, range(len(numbers))))
    pairs.sort(key=lambda p: p[0])
    result = [0 for i in range(len(numbers))]
    for rank in range(len(numbers)):
        original_index = pairs[rank][1]
        result[original_index] = rank * 100.0 / (len(numbers)-1)
    print(result)
    return result


def standardise_scores_relative(scores):
    return list_to_percentiles(scores)


@jit(nopython=True)
def summarise_scores(all_scores):
    final_scores = []
    # heights = []
    # locations = []
    for c in range(len(all_scores[0])):
        # scores_per_point = [score[c] for score in all_scores]
        scores_per_point = all_scores[:, c]
        # n, bins, _ = axs[0].hist(scores_per_point, bins='auto', alpha = 0.5)
        up = np.quantile(scores_per_point, 0.99)
        low = np.quantile(scores_per_point, 0.01)
        final_scores.append(max(up, 1-low))
        # heights.append(max(n))
        # locations.append(round(bins[np.argmax(n)],2))
    return final_scores


def summarise_scores_var(all_scores):
    final_scores = []
    for c in range(len(all_scores[0])):
        scores_per_point = [score[c] for score in all_scores]
        final_scores.append(np.var(scores_per_point))
    return final_scores


def random_projection_window(data, k_range, norm_perservation_range, win_pos_range, norm_power_range, win_length_range):
    # Parameters of random projection + window method.
    k = random.choice(k_range)
    norm_perservation = random.choice(norm_perservation_range)
    win_pos = random.choice(win_pos_range)
    norm_power = random.choice(norm_power_range)
    win_length = random.choice(win_length_range)

    expanded_data = make_windows(data, win_pos, win_length)

    outlier_scores = random_projection_single(expanded_data, k, norm_perservation, norm_power)
    return standardise_scores_z_score(outlier_scores)


#labels = df_train['Normal/Attack'].to_numpy().astype('int')
def get_signals(data):
    data = data.drop(columns='timestamp', axis =1)
    signal = normalise(data.values)
    signal_diff_right = normalise(data.diff().fillna(0).values)
    signal_diff_right = [np.abs(i) for i in signal_diff_right]
    signal_diff_left = normalise(data.diff(-1).fillna(0).values)
    signal_diff_left = [np.abs(i) for i in signal_diff_left]

    return signal, signal_diff_right, signal_diff_left


# m = number of components
def run(data, win_length_max, m):
    k_range = [1, 2, 3, 4, 5, 10]
    norm_perservation_range = [True, False]
    win_pos_range = ['mid'] #can add prev (if online) and future.
    norm_power_range = [1, 1.5, 2, 3]
    win_length_range = np.unique(np.logspace(0, np.log(win_length_max), 50, dtype=int, base=np.e))
    outlier_scores_m = [[] for _ in range(m)]

    signal, signal_diff_right, signal_diff_left = get_signals(data)

    for i in tqdm(range(m)):

        mode = random.choice([1, 2, 3])

        if mode == 1:  # mu
            outlier_scores_m[i] = mean_prime_2(signal, norm_power_range, win_length_range)
        elif mode == 2:  # normal RP
            outlier_scores_m[i] = random_projection_window(signal, k_range, norm_perservation_range, win_pos_range,
                                                           norm_power_range, win_length_range)
        else:  # RP on differenced signal
            direction = random.choice(["left", "right"])
            if direction == "right":
                scores = random_projection_window(signal_diff_right, k_range, norm_perservation_range, win_pos_range,
                                                  norm_power_range, win_length_range)
            else:
                scores = random_projection_window(signal_diff_left, k_range, norm_perservation_range, win_pos_range,
                                                  norm_power_range, win_length_range)
            outlier_scores_m[i] = scores

    return summarise_scores(np.array(outlier_scores_m))


def summarise_data(data, labels, guesses):
    print("Total number of data points:", len(data))
    print(f"Total number of outliers: {sum(labels)} ({(sum(labels) / len(labels)) * 100:.3f} %)")
    print(f"Total number of guesses: {len(guesses)} ({(len(guesses) / len(data)) * 100:.3f} %)")

def run_NAB():
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

    guesses = [item for sublist in to_add_times for item in sublist[1:]]
    guesses_index = data.index[data['timestamp'].isin(guesses)].tolist()

    summarise_data(data, labels, guesses)
    m = 1000
    type = "try"

    scores = run(data, 100, m)
    print(scores[810], scores[811], scores[812], scores[813], scores[814], scores[815], scores[816])

    pc.all_plots(name, data, scores, labels, guesses, to_add_values, [], [], runs=m, type=type)


def run_MITBIH():
    plot = False

    sampfrom = 0
    sampto = None

    name = 100

    record, annotation = mb.load_mit_bih_data(name, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels, labels_plot = mb.label_clean_segments_q_points(
        record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))]).reshape(-1,1)
    signal_norm = np.hstack((timestamp, signal_norm))
    signal = pd.DataFrame(signal_norm, columns=np.append("timestamp",record.sig_name))

    if plot:
        mb.plot_data(record, heart_beats_x, labels, labels_plot, sampfrom)

    summarise_data(signal, labels, [])
    m = 1000
    scores = run(signal,400, m)
    pc.all_plots(name, signal, scores, labels, [], [], [], [], runs=m, type="trial")


if __name__ == '__main__':
    run_NAB()
    # run_MITBIH()


