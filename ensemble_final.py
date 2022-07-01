import numpy as np
import random
from numba import jit
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import sklearn.model_selection as sk
import load_data_NAB as nab
import plot_curves as pc
import load_data_MITBIH as mb


# Computes mean of matrix columns using Numba.
@jit(nopython=True)
def mean_columns(point):
    means = []
    for c in range(point.shape[1]):
        column = point[:, c]
        means.append(np.mean(column))

    return np.array(means)


# Mean Projection method: constructs windows and computes outlierness score for each time point.
@jit(nopython=True)
def mean_projection(signal, win_pos, norm_power, win_length):
    outlier_scores = np.array([0.0 for _ in range(len(signal))])
    for i in range(0, len(signal)):
        if win_pos == 'prev':
            if i == 0:
                continue
            previous = i - win_length + 1
            future = i
        elif win_pos == "mid":
            previous = i - win_length // 2
            future = i + win_length // 2
        else:
            if i == len(signal) - 1:
                continue
            previous = i
            future = i + win_length - 1

        if previous < 0:
            previous = 0
        if future > len(signal) - 1:
            future = len(signal) - 1

        if win_pos == 'prev':
            point = signal[previous:i]
        elif win_pos == "mid":
            point = np.concatenate((signal[previous:i], signal[i + 1:future + 1]))
        else:
            point = signal[i + 1:future + 1]

        mean = mean_columns(point)
        score = np.linalg.norm(signal[i] - mean) ** norm_power
        outlier_scores[i] = score

    return outlier_scores


# Random projection with outlierness score computation for a single window
@jit(nopython=True)
def random_projection_single(point, k, norm_perservation, norm_power, R):
    d = R.shape[1]
    # If window smaller than d (equivalent to padding with zeroes).
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


# Random Projection method: constructs windows and calls random projection per window.
@jit(nopython=True)
def random_projection_window(data, k, norm_perservation, win_pos, norm_power, win_length):
    # Constructs R only once. Same R is used for all time points.
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
    return outlier_scores


# Normalise different components of the time series.
def normalise(signal):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


# Ensemble component.
def task(win_length_max, signal, signal_diff_right, signal_diff_left, i):
    # Mode chooses type of outlier detection method. 1) MP, 2) RP on original time series, 3) RP on differenced time series.
    mode = random.choice([1, 2, 3])

    # Sample method parameters.
    norm_perservation = random.choice([True, False])
    win_pos = random.choice(['prev', 'mid', 'future'])
    norm_power = random.choice([0.5, 1, 2, 3, 4])
    window_length_range = np.concatenate((win_length_max // 2 + np.unique(
        np.logspace(0, np.log(win_length_max // 2), 30, dtype=int, base=np.e, endpoint=True)),
                                          win_length_max // 2 - np.unique(
                                              np.logspace(0, np.log(win_length_max // 2), 30, dtype=int, base=np.e,
                                                          endpoint=True))))
    k_range = np.array([1, 3, 10])

    if mode == 1:  # MP
        win_length = random.choice(window_length_range[window_length_range > 1])
        scores = mean_projection(signal, win_pos, norm_power, win_length)
    elif mode == 2:  # RP on original time series
        win_length = random.choice(window_length_range)
        k = random.choice(k_range[k_range <= min(win_length, max(k_range))])
        scores = random_projection_window(signal, k, norm_perservation, win_pos,
                                          norm_power, win_length)
    else:  # RP on differenced time series
        direction = random.choice(["left", "right"])
        win_length = random.choice(window_length_range)
        k = random.choice(k_range[k_range <= min(win_length, max(k_range))])
        # Randomly choose direction of differencing.
        if direction == "right":
            scores = random_projection_window(signal_diff_right, k, norm_perservation, win_pos,
                                              norm_power, win_length)
        else:
            scores = random_projection_window(signal_diff_left, k, norm_perservation, win_pos,
                                              norm_power, win_length)

    return scores


# Z-score standardisation of training and test set.
@jit(nopython=True)
def standardise_sets(all_scores_train, all_scores_test):
    train_normalised = np.full(all_scores_train.shape, 0.0)
    test_normalised = np.full(all_scores_test.shape, 0.0)
    for i, scores in enumerate(all_scores_train):
        mu = np.mean(scores)
        sigma = np.std(scores)
        train_normalised[i, :] = (scores - mu) / sigma
        # Standardise test set using same mean and standard deviation.
        test_normalised[i, :] = (all_scores_test[i] - mu) / sigma
    return train_normalised, test_normalised


# Binarise training and test set.
@jit(nopython=True)
def get_binary_sets(all_scores, indices_train, indices_test):
    train = all_scores[:, indices_train.reshape(-1, )].reshape(all_scores.shape[0], -1).astype('float64')
    test = all_scores[:, indices_test.reshape(-1, )].reshape(all_scores.shape[0], -1).astype('float64')

    train_normalised, test_normalised = standardise_sets(train, test)

    train_binary = np.full(train_normalised.shape, 0.0)
    for i, scores in enumerate(train_normalised):
        train_binary[i] = np.where((scores > 1.96) | (scores < -1.96), 1.0, 0.0)

    test_binary = np.full(test_normalised.shape, 0.0)
    for i, scores in enumerate(test_normalised):
        test_binary[i] = np.where((scores > 1.96) | (scores < -1.96), 1.0, 0.0)

    return train_binary, test_binary


# WINNOW weighting for each ensemble component.
@jit(nopython=True)
def get_weights(scores_train, y_train, v):
    w = np.ones(scores_train.shape[0])
    weighted_scores_binary_train = np.full((len(y_train),), 0)
    c = 0
    step_size = (sum(y_train) / len(y_train)) / v
    threshold = 0.0
    errors = np.where(weighted_scores_binary_train - y_train != 0)[0]
    # While tolerated error is not reached, reweigh.
    while (len(errors) / len(y_train) > threshold) or (c < np.log2(scores_train.shape[0])):
        for i in errors:
            if y_train[i] == 1:  # Then score is 0 ==> False Negative
                for j, s in enumerate(scores_train[:, i]):
                    if s == 1:
                        w[j] = w[j] * 2
            if y_train[i] == 0:  # Then score is 0 ==> False Positive
                for j, s in enumerate(scores_train[:, i]):
                    if s == 1:
                        w[j] = w[j] / 2
        weighted_scores_train = (w.reshape(1, -1) @ scores_train)[0]
        weighted_scores_binary_train = np.where(weighted_scores_train > scores_train.shape[0], 1, 0)

        errors = np.where(weighted_scores_binary_train - y_train != 0)[0]

        c += 1
        # Increase tolerated error.
        if c == int(50 * np.log2(scores_train.shape[0])):
            print("upped")
            threshold += step_size
            c = 0

    # print("training error", len(errors) / len(y_train))

    return w


# WINNOW aggregation with cross validation.
def winnow_cross_val(all_scores, labels, semi_labels=None, folds=3, v=10):
    roc_aucs = []
    pr_aucs = []
    accuracies = []
    # Split time series into folds.
    kf = sk.KFold(n_splits=folds)
    for train, test in kf.split(np.array([i for i in range(len(labels))]).reshape(-1, 1)):
        y_test = labels[test.reshape(-1, )]
        # Cannot evaluate a split with no outliers.
        if 1 not in y_test:
            print(
                "Warning: Test set contains no outliers so was skipped during evaluation.")
            continue
        # Semi-supervised variant.
        if semi_labels is not None:
            y_train = semi_labels[train.reshape(-1, )]
        else:
            y_train = labels[train.reshape(-1, )]
        scores_train, scores_test = get_binary_sets(all_scores, train.reshape(-1, 1), test.reshape(-1, 1))

        # Determine weights.
        w = get_weights(scores_train, np.array(y_train), v)

        # Predict scores of test set.
        final_test_scores = (w.reshape(1, -1) @ scores_test)[0]
        predict_test = np.array(np.where(final_test_scores > scores_train.shape[0], 1, 0))

        # Compute TPR, FPR, precision, ROC AUC, and PR AUC
        tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(final_test_scores, y_test, min(final_test_scores),
                                                                max(final_test_scores))
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        # Compute accuracy
        diff = len(np.where(predict_test - y_test != 0)[0])
        accuracies.append(1 - (diff / len(y_test)))

    print(f"\n{folds}-fold cross validation:")
    print("ROC AUC: ", np.round(np.mean(roc_aucs), 3), "PR AUC: ", np.round(np.mean(pr_aucs), 3))
    return


# Unsupervised aggregation.
@jit(nopython=True)
def summarise_scores(all_scores):
    final_scores = np.array([0.0 for _ in range(len(all_scores[0]))])
    for c in range(len(all_scores[0])):
        scores_per_point = all_scores[:, c]
        up = np.quantile(scores_per_point, 0.995)
        low = np.quantile(scores_per_point, 0.005)
        final_scores[c] = max(up, 1 - low)
    return final_scores


# Normalise and difference signal.
def get_signals(data):
    signal = normalise(data.values)
    signal_diff_right = normalise(data.diff().fillna(0).values)
    signal_diff_right = np.array([np.abs(i) for i in signal_diff_right])
    signal_diff_left = normalise(data.diff(-1).fillna(0).values)
    signal_diff_left = np.array([np.abs(i) for i in signal_diff_left])

    return signal, signal_diff_right, signal_diff_left


# Runs the ensemble, calling components in parallel.
def run(data, win_length_max, n_runs, parallelise, num_workers):
    outlier_scores_m = []

    # Difference signal once beforehand to save time.
    signal, signal_diff_right, signal_diff_left = get_signals(data)

    if parallelise:
        with ProcessPoolExecutor(num_workers) as executor:
            for r in tqdm(
                    [executor.submit(task, win_length_max, signal, signal_diff_right, signal_diff_left, i)
                     for i in
                     range(n_runs)]):
                outlier_scores_m.append(r.result())
    else:
        for i in tqdm(range(n_runs)):
            outlier_scores_m.append(task(win_length_max, signal, signal_diff_right, signal_diff_left, i))
    print("Summarising...")
    return np.array(outlier_scores_m)


# Outputs summary of dataset.
def summarise_data(data, labels, guesses):
    print("Total number of data points:", len(data))
    print(f"Total number of outliers: {sum(labels)} ({(sum(labels) / len(labels)) * 100:.3f} %)")
    print(f"Total number of guesses: {len(guesses)} ({(len(guesses) / len(data)) * 100:.3f} %)")


# RPOE run on AMB dataset.
def run_AMB(n_runs, max_window_size, semi_supervised=False, parallelise=False, num_workers=6):
    # Load dataset.
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
    guesses = [item for sublist in to_add_times for item in sublist[1:]]

    data = data.reset_index().set_index('timestamp')
    data = data.drop('index', axis=1)
    data.index = pd.to_datetime(data.index)

    summarise_data(data, labels, guesses)

    # Run ensemble.
    all_scores = run(data, max_window_size, n_runs, parallelise, num_workers)

    # Aggregation.
    if semi_supervised:
        sample_size = int(0.8 * len(labels))
        kept_true_labels = np.random.choice(range(len(labels)), sample_size, replace=False)
        semi_supervised_labels = [labels[i] if i in kept_true_labels else 0 for i in range(len(labels))]
        winnow_cross_val(all_scores, np.array(labels),
                         semi_labels=np.array(semi_supervised_labels))
    else:
        winnow_cross_val(all_scores, np.array(labels), semi_labels=np.array(labels))


# Convert type for Numba run.
def convert(heart_beats_x):
    heart_beats_x_array = []
    for x in heart_beats_x:
        heart_beats_x_array.append([x[0], x[-1]])
    return np.array(heart_beats_x_array)


# Summarise individual time point scores into beat scores.
@jit(nopython=True)
def get_beat_score(all_scores, heart_beats_x):
    all_scores_beats = np.full((len(all_scores), len(heart_beats_x)), 0.0)
    for i, score in enumerate(all_scores):
        for j, x in enumerate(heart_beats_x):
            beat_scores = [score[k - heart_beats_x[0][0]] for k in range(x[0], x[1])]
            all_scores_beats[i][j] = max(beat_scores)
    return all_scores_beats


# RPOE run on a sample of the MITBIH dataset.
def run_MITBIH(sample, n_runs, max_window_size, semi_supervised=False, parallelise=False, num_workers=6):
    # Load dataset.
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(sample, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)

    summarise_data(heart_beats, labels, [])

    # Run ensemble.
    all_scores = run(signal, max_window_size, n_runs, parallelise, num_workers)
    # Summarise individual scores into beat scores.
    all_scores_beat = get_beat_score(all_scores, convert(heart_beats_x))

    # Aggregation.
    if semi_supervised:
        sample_size = int(0.8 * len(labels))
        kept_true_labels = np.random.choice(range(len(labels)), sample_size, replace=False)
        semi_supervised_labels = [labels[i] if i in kept_true_labels else 0 for i in range(len(labels))]
        winnow_cross_val(all_scores_beat, np.array(labels),
                         semi_labels=np.array(semi_supervised_labels))
    else:
        winnow_cross_val(all_scores_beat, np.array(labels), semi_labels=np.array(labels))


# Run RPOE on different datasets based on parameter settings.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['NAB', 'MITBIH'], default='NAB',
                        help='Which dataset to run: ["NAB","MITBIH"]')
    parser.add_argument('--n_runs', type=int, default=1000,
                        help='Number of iterations')
    parser.add_argument('--max_window_length', type=int, default=100,
                        help='Maximum window size')
    parser.add_argument('--parallelise', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to parallelise the iterations or not.')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of parallel workers.')
    parser.add_argument('--sample', type=int, default=100,
                        help='Patient number of MITBIH Dataset.')
    parser.add_argument('--semi-supervised', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Supervised or semi-supervised method.')

    args = parser.parse_args()

    str_print = ""
    for par, arg in vars(args).items():
        str_print = str_print + " {}={} ".format(par, arg)
    print("Ran with parameters:", str_print)

    if args.dataset == "NAB":
        run_AMB(n_runs=args.n_runs, max_window_size=args.max_window_length, parallelise=args.parallelise,
                num_workers=args.num_workers)
    elif args.dataset == "MITBIH":
        run_MITBIH(sample=args.sample, n_runs=args.n_runs, max_window_size=args.max_window_length,
                   parallelise=args.parallelise, num_workers=args.num_workers)
