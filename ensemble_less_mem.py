import numpy as np
import random
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.tsa.stattools as sm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import argparse
import time

import load_data_NAB as nab
import plot_curves as pc
import load_data_MITBIH as mb
import aggregation as agr


@jit(nopython=True)
def standardise_scores_z_score(scores):
    return (scores - np.mean(scores)) / np.std(scores)


@jit(nopython=True)
def mean_columns(point):
    means = []
    for c in range(point.shape[1]):
        column = point[:, c]
        means.append(np.mean(column))

    return np.array(means)


@jit(nopython=True)
def mean_prime(signal, norm_power, win_length):
    outlier_scores = np.array([0.0 for _ in range(len(signal))])
    for i in range(0, len(signal)):
        previous = i - win_length // 2
        future = i + win_length // 2
        if previous < 0:
            previous = 0
        if future > len(signal) - 1:
            future = len(signal) - 1
        point = np.concatenate((signal[previous:i], signal[i + 1:future + 1]))
        mean = mean_columns(point)
        score = np.linalg.norm(signal[i] - mean) ** norm_power
        outlier_scores[i] = score

    return standardise_scores_z_score(outlier_scores)


@jit(nopython=True)
def random_projection_single(point, k, norm_perservation, norm_power, R):
    d = R.shape[1]
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

    return outlier_score


@jit(nopython=True)
def random_projection_window(data, k, norm_perservation, win_pos, norm_power, win_length):
    # Parameters of random projection + window method.
    if win_pos == 'mid' and win_length != 1:
        R = np.random.normal(loc=0.0, scale=1.0, size=(k, win_length + 1))
    else:
        R = np.random.normal(loc=0.0, scale=1.0, size=(k, win_length))
    outlier_scores = np.array([0.0 for _ in range(len(data))])
    for i in range(len(data)):
        if win_length == 1:
            point = data[i].reshape(1, -1)
        else:
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
            if future >= len(data):
                future = len(data) - 1

            point = data[previous:future + 1]

        outlier_scores[i] = random_projection_single(point, k, norm_perservation, norm_power, R)
    return standardise_scores_z_score(outlier_scores)


# @jit(nopython=True)
def random_projection_window_auto(data, k, norm_perservation, win_pos, norm_power, win_length):
    # Parameters of random projection + window method.
    if win_pos == 'mid' and win_length != 1:
        R = np.random.normal(loc=0.0, scale=1.0, size=(k, win_length + 1))
    else:
        R = np.random.normal(loc=0.0, scale=1.0, size=(k, win_length))
    outlier_scores = np.array([0.0 for _ in range(len(data))])
    for i in range(len(data)):
        if win_length == 1:
            point = data[i].reshape(1, -1)
        else:
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
            if future >= len(data):
                future = len(data) - 1

            point = data[previous:future + 1]
            point = autocorrelation_sm(point)

        outlier_scores[i] = random_projection_single(point, k, norm_perservation, norm_power, R)
    return standardise_scores_z_score(outlier_scores)


def normalise(signal):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


# Parallel tasks
def task(win_length_max, signal, signal_diff_right, signal_diff_left, signal_auto, i):
    mode = random.choice([1, 2, 3])
    norm_perservation = random.choice([True, False])
    win_pos = random.choice(['mid'])
    norm_power = random.choice([1, 1.5, 2, 3])
    window_length_range = np.unique(np.logspace(0, np.log(win_length_max), 50, dtype=int, base=np.e, endpoint=True))
    k_range = np.array([1, 2, 3, 4, 5, 10])

    if mode == 1:  # mu
        win_length = random.choice(window_length_range[window_length_range > 1])
        scores = mean_prime(signal, norm_power, win_length)
    elif mode == 2:  # normal RP
        win_length = random.choice(window_length_range)
        k = random.choice(k_range[k_range <= min(win_length, max(k_range))])
        scores = random_projection_window(signal, k, norm_perservation, win_pos,
                                          norm_power, win_length)
    elif mode == 3:  # RP on differenced signal
        direction = random.choice(["left", "right"])
        win_length = random.choice(window_length_range)
        k = random.choice(k_range[k_range <= min(win_length, max(k_range))])
        if direction == "right":
            scores = random_projection_window(signal_diff_right, k, norm_perservation, win_pos,
                                              norm_power, win_length)
        else:
            scores = random_projection_window(signal_diff_left, k, norm_perservation, win_pos,
                                              norm_power, win_length)
    else:  # autocorrelation
        win_length = random.choice(window_length_range[window_length_range > len(signal) * 0.00002])
        k = random.choice(k_range[k_range <= min(win_length, max(k_range))])
        scores = random_projection_window_auto(signal, k, norm_perservation, win_pos, norm_power, win_length)

    return scores


def summarise_scores_supervised(all_scores, labels):
    train_indices = np.array([i for i in range(len(labels))]).reshape(-1, 1)
    X_train_i, X_test_i, y_train, y_test = sk.train_test_split(train_indices, labels, test_size=0.2, stratify=labels)

    scores_binary = np.full(all_scores.shape, 0)
    for i, scores in enumerate(all_scores):
        scores_binary[i] = np.array((np.where((scores > 1.96) | (scores < -1.96), 1, 0)))

    scores_train = get_set(scores_binary, X_train_i)
    scores_test = get_set(scores_binary, X_test_i)

    del all_scores, scores_binary, train_indices, labels

    w = np.ones(scores_train.shape[0])

    weighted_scores_binary_train = np.full((len(y_train),), 0)

    while not len(np.where(weighted_scores_binary_train - y_train != 0)[0]) / len(y_train) < 0.02:
        for i, score in enumerate(weighted_scores_binary_train):
            if score == 0 and y_train[i] == 1:
                for j, s in enumerate(scores_train[:, i]):
                    if s == 1:
                        w[j] = w[j] * 2
            if score == 1 and y_train[i] == 0:
                for j, s in enumerate(scores_train[:, i]):
                    if s == 1:
                        w[j] = w[j] / 2
        weighted_scores_train = (w.reshape(1, -1) @ scores_train)[0]
        weighted_scores_binary_train = np.array(np.where(weighted_scores_train > scores_train.shape[0], 1, 0))
    predict_test = np.array(np.where((w.reshape(1, -1) @ scores_test)[0] > scores_train.shape[0], 1, 0))
    return predict_test, y_test

@jit(nopython=True)
def summarise_scores(all_scores):
    final_scores = np.array([0.0 for _ in range(len(all_scores[0]))])
    for c in range(len(all_scores[0])):
        scores_per_point = all_scores[:, c]
        up = np.quantile(scores_per_point, 0.995)
        low = np.quantile(scores_per_point, 0.005)
        final_scores[c] = max(up, 1 - low)
    return final_scores


def autocorrelation_sm(data):
    auto_coefficients = np.full(data.shape, 0.0)
    for c in range(data.shape[1]):
        auto_coefficients[:, c] = sm.acf(data[:, c], nlags=len(data))
    return auto_coefficients


@jit(nopython=True)
def autocorrelation(data):
    mean = mean_columns(data)
    auto_coefficients = np.full(data.shape, 0.0)
    for i, point in enumerate(data):
        sum = np.full(point.shape, 0.0)
        for t in range(i, len(data)):
            sum = sum + ((data[t] - mean) * (data[t - i] - mean))[0]
        auto_coefficients[i] = point.shape[0] * sum / np.sum((data - mean) ** 2)
    return auto_coefficients


# labels = df_train['Normal/Attack'].to_numpy().astype('int')
def get_signals(data):
    signal = normalise(data.values)
    signal_diff_right = normalise(data.diff().fillna(0).values)
    signal_diff_right = np.array([np.abs(i) for i in signal_diff_right])
    signal_diff_left = normalise(data.diff(-1).fillna(0).values)
    signal_diff_left = np.array([np.abs(i) for i in signal_diff_left])
    signal_auto = autocorrelation_sm(signal)

    return signal, signal_diff_right, signal_diff_left, signal_auto


# m = number of components
def run(data, labels, win_length_max, n_runs, parallelise, num_workers):
    outlier_scores_m = []

    signal, signal_diff_right, signal_diff_left, signal_auto = get_signals(data)

    if parallelise:
        with ProcessPoolExecutor(num_workers) as executor:
            for r in tqdm(
                    [executor.submit(task, win_length_max, signal, signal_diff_right, signal_diff_left, signal_auto, i)
                     for i in
                     range(n_runs)]):
                outlier_scores_m.append(r.result())
    else:
        for i in tqdm(range(n_runs)):
            outlier_scores_m.append(task(win_length_max, signal, signal_diff_right, signal_diff_left, signal_auto, i))
    print("Summarising...")
    np.save(f"./output_scores/all_scores_MITBIH_sample_101", np.array(outlier_scores_m))
    return agr.summarise_scores_supervised_scores(np.array(outlier_scores_m), labels)


def summarise_data(data, labels, guesses):
    print("Total number of data points:", len(data))
    print(f"Total number of outliers: {sum(labels)} ({(sum(labels) / len(labels)) * 100:.3f} %)")
    print(f"Total number of guesses: {len(guesses)} ({(len(guesses) / len(data)) * 100:.3f} %)")


def run_NAB(n_runs, max_window_size, type, parallelise=False, num_workers=6):
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

    guesses = [item for sublist in to_add_times for item in sublist[1:]]

    data = data.reset_index().set_index('timestamp')
    data = data.drop('index', axis=1)
    data.index = pd.to_datetime(data.index)

    summarise_data(data, labels, guesses)
    scores_test, y_test = run(data, labels, max_window_size, n_runs, parallelise, num_workers)
    print(scores_test[810], scores_test[811], scores_test[812], scores_test[813], scores_test[814], scores_test[815],
          scores_test[816])
    # diff = len(np.where(scores_test - y_test != 0)[0])
    # print(diff, diff/len(y_test))
    np.save(f"./output_scores/NAB_{name}_{n_runs}_{type}", scores_test)
    pc.all_plots(name, data, scores_test, y_test, None, None, None, None, runs=n_runs, type=type)


def run_MITBIH(sample, n_runs, max_window_size, type, parallelise=False, num_workers=6):
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(sample, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_q_points_single(record, annotation, sampfrom,
                                                                                     sampto)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)

    summarise_data(signal, labels, [])
    # scores = run(signal,max_window_size, n_runs, parallelise, num_workers)
    # print("Plotting...")

    # pc.all_plots(f"sample_{sample}", signal, scores, labels, None, None, None, None, runs=n_runs, type=type)
    scores_test, y_test = run(signal, labels, max_window_size, n_runs, parallelise, num_workers)
    # diff = len(np.where(scores_test - y_test != 0)[0])
    # print(diff, diff / len(y_test))
    np.save(f"./output_scores/MITBIH_sample_{sample}_{n_runs}_{type}", scores_test)
    pc.all_plots(f"sample_{sample}", signal, scores_test, y_test, None, None, None, None, runs=n_runs, type=type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['NAB', 'MITBIH'], default='NAB',
                        help='Which dataset to run: ["NAB","MITBIH"]')
    parser.add_argument('--n_runs', type=int, default=1000,
                        help='Number of iterations')
    parser.add_argument('--max_window_size', type=int, default=100,
                        help='Maximum window size')
    parser.add_argument('--type', type=str, default="testing",
                        help='Test ID')
    parser.add_argument('--parallelise', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to parallelise the iterations or not.')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of parallel workers.')
    parser.add_argument('--sample', type=int, default=100,
                        help='Patient number of MITBIH Dataset.')

    args = parser.parse_args()

    str_print = ""
    for par, arg in vars(args).items():
        str_print = str_print + " {}={} ".format(par, arg)
    print("Ran with parameters:", str_print)

    if args.dataset == "NAB":
        run_NAB(n_runs=args.n_runs, max_window_size=args.max_window_size, type=args.type, parallelise=args.parallelise,
                num_workers=args.num_workers)
    elif args.dataset == "MITBIH":
        run_MITBIH(sample=args.sample, n_runs=args.n_runs, max_window_size=args.max_window_size, type=args.type,
                   parallelise=args.parallelise, num_workers=args.num_workers)
