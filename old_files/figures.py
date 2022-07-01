import load_data_MITBIH as mb
from old_files import ensemble_WINNOW_testing as e
import plot_curves as pc
import matplotlib.pyplot as plt
import load_data_NAB as nab

from numba import jit
import numpy as np
import pandas as pd

sampfrom = 0
sampto = None

def convert(heart_beats_x):
    heart_beats_x_array = []
    for x in heart_beats_x:
        heart_beats_x_array.append([x[0],x[-1]])
    return np.array(heart_beats_x_array)

@jit(nopython=True)
def get_scores(scores, heart_beats_x):
    beat_scores = np.full((len(heart_beats_x),1),0.0)
    for i, x in enumerate(heart_beats_x):
        b = [scores[i - heart_beats_x[0][0]] for i in range(x[0], x[1])]
        beat_scores[i] = max(b)
    return beat_scores

def k_dependence():
    record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    # name = "ambient_temperature_system_failure"
    # data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    # data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
    # data = data.reset_index().set_index('timestamp')
    # data = data.drop('index', axis=1)
    # data.index = pd.to_datetime(data.index)
    aucs = [[] for _ in range(10)]
    aucs_p = [[] for _ in range(10)]
    for k in range(1, 16):
        print(k)
        for j in range(10):
            scores = e.standardise_scores_z_score(e.random_projection_window(signal_norm, k, False, "prev", 2, 260))
            beat_scores = get_scores(scores, convert(heart_beats_x))
            tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(beat_scores, labels, min=min(beat_scores), max=max(beat_scores))
            aucs[j].append(roc_auc)
            aucs_p[j].append(pr_auc)

    means = np.mean(aucs, axis = 0)
    stds = np.std(aucs, axis = 0 )
    means_p = np.mean(aucs_p, axis = 0)
    stds_p = np.std(aucs_p, axis = 0 )

    plt.title("ROC AUC values for different projection dimensions k")
    plt.xlabel("Projection dimension (k)")
    plt.ylabel("AUC")
    plt.errorbar(range(1,16), means, yerr=stds, fmt="o")
    plt.show()
    plt.clf()
    plt.title("Precision Recall AUC values for different projection dimensions k")
    plt.xlabel("Projection dimension (k)")
    plt.ylabel("AUC")
    plt.errorbar(range(1,16), means_p, yerr=stds_p, fmt="o")
    plt.show()


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

def random_projection_window(data, k, norm_perservation, win_pos, norm_power, win_length, R):
    # Parameters of random projection + window method.
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


def build_sin():
    x = np.linspace(0,30, 1000)
    y = []
    y_pro_f = []
    y_pro_t = []
    R = np.random.normal(loc=0.0, scale=1.0, size=(1, 3))
    for i in x:
        y.append(np.sin(i) + np.random.normal(0,0.05))

    # y[200] = -0.75
    y_pro_f = random_projection_window(np.array(y), 1, False, "mid", 2, 2, R)
    y_pro_t = random_projection_window(np.array(y), 1, True, "mid", 2, 2, R)



    plt.plot(x,y, label="Original")
    plt.plot(x, y_pro_f, label="No scaling")
    plt.plot(x, y_pro_t, label= "Scaling")
    plt.legend()
    plt.show()

def plot_scores(name, data, scores, labels, guesses, to_add_values, runs, type):

    # Colour outlier red, normal beats black
    colours = []
    for i in labels[:25]:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")

    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3,3,1]})

    x_labels = [i for i in range(sampfrom, heart_beats_x[24][-1] + sampfrom) if i % (10 * record.fs) == 0]
    x_labels_values = [int(i / record.fs) for i in x_labels]

    if sampfrom != 0:
        x_labels = [i - sampfrom for i in x_labels]
    signal = record.p_signal
    signal1 = [signal[heart_beat[0]:heart_beat[-1] + 1][:, 0] for heart_beat in heart_beats_x[:25]]
    signal2 = [signal[heart_beat[0]:heart_beat[-1] + 1][:, 1] for heart_beat in heart_beats_x[:25]]

    fig.suptitle(f"MIT-BIH Arrhythmia Database: Sample {record.record_name}")

    for i, c in enumerate(colours):
        axs[0].plot(heart_beats_x[i], signal1[i], color=c)
        axs[1].plot(heart_beats_x[i], signal2[i], color=c)

    axs[0].set_ylabel(record.sig_name[0])
    axs[0].set_xticks(x_labels, x_labels_values)
    axs[1].set_ylabel(record.sig_name[1])
    axs[1].set_xticks(x_labels, x_labels_values)
    axs[1].set_xlabel("time (seconds)")

    bar_pos = [heart_beat[len(heart_beat)//2] for heart_beat in heart_beats_x[:25]]
    axs[2].bar(bar_pos, scores[:25].reshape(-1,), width=200)
    axs[2].xaxis.set_visible(False)
    axs[2].set_ylabel("Ensemble")

    plt.savefig(f"./final-explore/{name}_plot_{runs}_{type}.png", bbox_inches='tight')


def get_histo(labels, scores, guesses_index):
    normal = []
    outlier = []
    guesses = []
    for i, v in enumerate(labels):
        if v == 1 and i in guesses_index:
            guesses.append(scores[i])
        elif v == 1:  # outlier
            outlier.append(scores[i])
        else:
            normal.append(scores[i])
    # print(scores[812],scores[4124], scores[6801])
    return normal, outlier, guesses

def plot_histogram(name, scores, labels):
    # [4124, 6801]
    normal, outlier, guessed = get_histo(labels, scores, [])
    plt.hist(np.array(normal).reshape(-1,), bins='auto', label="Normal", alpha=0.5)
    plt.hist(np.array(outlier).reshape(-1,), bins='auto', label="Outliers", alpha=0.5)
    # plt.hist(np.array(guessed).reshape(-1, ), bins='auto', label="Interpolations", alpha=0.5)
    plt.title(f"Histogram of Outlierness Scores for {name}")
    plt.xlabel("Standardised MP outlierness scores")
    plt.legend()

    plt.show()
    plt.clf()

def pres_norm():
    record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)
    r = []
    t = []
    for _ in range(50):
        scores = e.standardise_scores_z_score(e.random_projection_window(signal_norm, 1, True, "prev", 2, 260))
        beat_scores = get_scores(scores, convert(heart_beats_x))
        tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(beat_scores, labels, max(beat_scores), min(beat_scores))
        r.append(roc_auc)
        t.append(pr_auc)
        # scores = e.standardise_scores_z_score(e.random_projection_window(signal_norm, 1, False, "prev", 2, 260))
        # beat_scores = get_scores(scores, convert(heart_beats_x))
        # tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(beat_scores, labels, max(beat_scores), min(beat_scores))
        # t.append(roc_auc)
    print(max(r))
    print(max(t))
    r.append(0.92)
    print("with", np.mean(r), min(r), max(r))
    print("with", np.mean(t), min(t), max(t))
    # print("without", np.mean(t), min(t), max(t))
    # plt.boxplot([r,t])
    # plt.xticks([1,2],["with norm preservation", "without norm preservation"])
    # plt.title("Box plot of AUC of 50 random projection runs with and without scaling")
    # plt.ylabel("ROC AUC")
    # plt.show()
    # plot_histogram("Sample 100", signal_norm, beat_scores, labels, [], 1, "np=False")


def windowing():
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Outlier score around contextual outlier for different window choices")
    axs[0].bar(range(5),np.array([20.5, 31, 80.7, 51.2, 44.2]))
    axs[0].set_xticks(range(5), [0,0,1,0,0])
    axs[0].set_xlabel("prev and mid")
    axs[0].set_ylabel("Quantile aggregate ensemble")
    # axs[0].set_ylim([0.4,1.1])
    axs[1].bar(range(5),np.array([32.4, 34.5, 80.7, 42.2, 33.9]))
    axs[1].set_xticks(range(5), [0,0,1,0,0])
    axs[1].set_xlabel("prev, mid, future")
    # axs[1].set_ylim([0.4,1.1])
    axs[2].bar(range(5),np.array([36.4, 38.9, 81.1, 36.6, 34.6]))
    axs[2].set_xticks(range(5), [0,0,1,0,0])
    axs[2].set_xlabel("mid")
    # axs[2].set_ylim([0.4,1.1])
    plt.show()


def window_dependence():
    record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)

    r = []
    t = []
    for i, win in enumerate(range(150, 450, 10)):
        scores = e.standardise_scores_z_score(e.random_projection_window(signal_norm, 1, False, "prev", 2, win))
        beat_scores = get_scores(scores, convert(heart_beats_x))
        tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(beat_scores, labels, max(beat_scores), min(beat_scores))
        r.append(roc_auc)
        t.append(pr_auc)

    x = range(150, 450, 10)
    plt.plot(x, t)
    plt.title("AUC values for different window lengths")
    plt.xlabel("Window length $\ell$")
    plt.ylabel("PR AUC")
    plt.show()

def plot_scores_nab(name, signal, signal_diff, signal_diff_left, scores, labels):

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(str(name).replace("_", " ").capitalize())
    outliers = []
    for i,l in enumerate(labels):
        if l == 1:
            outliers.append(i)
    indices = [i for i in outliers]
    axs[0].plot(range(len(signal)), signal, color= 'k')
    # axs[0].scatter(812,signal[112], color='b', s=10, alpha=0.5)
    axs[0].set_ylabel("Original")
    axs[0].scatter(indices,signal[np.array(indices)], color= 'yellow', s = 10, alpha=0.5)
    # axs[1].plot(range(len(signal)), signal_diff, color='k')
    # axs[1].scatter(indices,signal_diff[np.array(indices) - start], color= 'yellow', s = 10, alpha=0.5)
    # axs[1].set_ylabel("Right Differenced $\mathbf{X}\_\mathbf{r}$")
    axs[1].set_ylabel("Random projection score \n (standardised)")
    axs[1].bar(range(len(signal)),scores)
    # axs[2].plot(range(700, 900), signal_diff_left, color='k')
    # axs[2].scatter(812, signal_diff_left[112], color='b', s=10, alpha=0.5)
    # axs[2].set_ylabel("Left Differenced $\mathbf{X}\_\mathbf{l}$")
    plt.show()


# record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
# # signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
# # scores = e.standardise_scores_z_score(e.mean_prime(signal_norm, "mid", 2, 375))
# beat_scores = get_scores(scores, convert(heart_beats_x))
# plot_histogram("Mean Projection", beat_scores, labels)

# name = "ambient_temperature_system_failure"
# data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
# data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
#
# guesses = [item for sublist in to_add_times for item in sublist[1:]]
#
# data = data.reset_index().set_index('timestamp')
# data = data.drop('index', axis=1)
# data.index = pd.to_datetime(data.index)
#
# signal = e.normalise(data.values)
# signal_diff_right = e.normalise(data.diff().fillna(0).values)
# signal_diff_right = np.array([np.abs(i) for i in signal_diff_right])
# signal_diff_left = e.normalise(data.diff(-1).fillna(0).values)
# signal_diff_left = np.array([np.abs(i) for i in signal_diff_left])
#
# scores = e.standardise_scores_z_score(e.random_projection_window(signal_diff_right, 1, False, "mid", 2, 40))
# # scores = e.standardise_scores_z_score(e.mean_prime(signal, "mid", 2, 5))
# start = 1000
# end = 2000
# # plot_scores_nab("Ambient temperature system failure", np.array(data.values)[start:end], signal_diff_right[start:end], signal_diff_left[start:end], scores[start:end], labels)
# plot_histogram("Random Projection", scores, labels)

# w = np.load("C:/Users/carol/PycharmProjects/RandomProjectionsThesis/weights/weights.npy")
# window = np.load("C:/Users/carol/PycharmProjects/RandomProjectionsThesis/weights/windowing.npy", allow_pickle=True)
#
# plt.plot(range(len(w)), w)

# mids = w[window[0]]
# prevs = w[window[1]]
# futures = w[window[2]]
# print(mids[mids>0.001])
# print(prevs[prevs>0.001])
# print(futures[futures>0.001])
# # print(futures)
# # plt.hist(mids, bins=10, label="Mid", alpha=0.5)
# # plt.hist(prevs, bins=10, label="Prev", alpha=0.5)
# # plt.hist(futures, bins=10, label="Futures", alpha=0.5)
# plt.show()

@jit(nopython=True)
def get_beat_score(all_scores, heart_beats_x):
    all_scores_beats = np.full((len(all_scores),len(heart_beats_x)),0.0)
    for i, score in enumerate(all_scores):
        for j, x in enumerate(heart_beats_x):
            beat_scores = [score[k-heart_beats_x[0][0]] for k in range(x[0], x[1])]
            all_scores_beats[i][j] = max(beat_scores)
    return all_scores_beats

# record, annotation = mb.load_mit_bih_data(sample, sampfrom, sampto)
# signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
# timestamp = np.array([int(i) for i in range(len(signal_norm))])
# signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)
# e.summarise_data(heart_beats, labels, [])
# print("got scores")
# scores_test, y_test, bin_test = e.summarise_scores_supervised(all_scores_beat, labels, test_size=0.2)
# print(np.bincount(bin_test))
# pc.compute_rates(scores_test, y_test, min(scores_test), max(scores_test))
# diff = len(np.where(bin_test - y_test != 0)[0])
# print(diff, diff/len(labels))
def window_range():
    max_window = 400
    exp_range = np.unique(np.logspace(0, np.log(max_window), 70, dtype=int, base=np.e, endpoint=True))
    print(sorted(exp_range),len(exp_range))
    # plt.hist(exp_range, label= "old", alpha=0.5)
    new_range = sorted(np.concatenate((max_window//2 + np.unique(np.logspace(0, np.log(max_window//2), 30, dtype=int, base=np.e, endpoint=True)), max_window//2 - np.unique(np.logspace(0, np.log(max_window//2), 30, dtype=int, base=np.e, endpoint=True)))))
    print(new_range, len(new_range))
    plt.hist(new_range)
    plt.title("Range for $\ell$ for Sample 100")
    plt.ylabel("Frequency")
    plt.xlabel("$\ell$")
    plt.show()


def mp_diff():
    sampfrom = 0
    sampto = None

    # name = "ambient_temperature_system_failure"
    # data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    # data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
    #
    # guesses = [item for sublist in to_add_times for item in sublist[1:]]
    #
    # data = data.reset_index().set_index('timestamp')
    # data = data.drop('index', axis=1)
    # data.index = pd.to_datetime(data.index)

    record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    data = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)

    signal = e.normalise(data.values)
    signal_diff_right = e.normalise(data.diff().fillna(0).values)
    signal_diff_right = np.array([np.abs(i) for i in signal_diff_right])
    signal_diff_left = e.normalise(data.diff(-1).fillna(0).values)
    signal_diff_left = np.array([np.abs(i) for i in signal_diff_left])

    scores = e.mean_prime(signal_diff_right, "prev", 2, 370)
    beat_scores = get_scores(scores, convert(heart_beats_x))
    pc.compute_rates(beat_scores, labels, min(beat_scores), max(beat_scores))
    plot_histogram("Sample 100", e.standardise_scores_z_score(beat_scores), labels)
    # plot_scores_nab("Ambient temperature system failure", signal,signal_diff_right, signal_diff_left, scores, labels)


import statsmodels.tsa.stattools as sm


def autocorrelation_sm(data):
    auto_coefficients = np.full(data.shape, 0.0)
    for c in range(data.shape[1]):
        auto_coefficients[:, c] = sm.acf(data[:, c], nlags=len(data))
    return auto_coefficients


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
    return outlier_scores

import scipy.stats as ss


def standardise_scores_rank(scores):
    return ss.rankdata(scores, method='max')/(len(scores))

def range_scale(scores):
    return (scores - min(scores))/(max(scores)-min(scores))

def ranking_test():
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

    guesses = [item for sublist in to_add_times for item in sublist[1:]]

    data = data.reset_index().set_index('timestamp')
    data = data.drop('index', axis=1)
    data.index = pd.to_datetime(data.index)

    # record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
    # signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    # timestamp = np.array([int(i) for i in range(len(signal_norm))])
    # data = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)
    #
    # signal = e.normalise(data.values)

    # scores = e.standardise_scores_z_score(random_projection_window_auto(e.normalise(data), 1, False, 'prev', 2, 260))
    # beat_scores = get_scores(scores, convert(heart_beats_x))
    # # print(beat_scores)
    # print(len(beat_scores), min(beat_scores), max(beat_scores))
    # # plot_scores_nab(name, data.values, None, None, scores, labels)
    # pc.compute_rates(beat_scores, labels, min(beat_scores), max(beat_scores))
    # plot_histogram("Sample 100", beat_scores, labels)
    signal_diff_right = e.normalise(data.diff().fillna(0).values)
    signal_diff_right = np.array([np.abs(i) for i in signal_diff_right])
    #
    # scores = e.standardise_scores_z_score(e.random_projection_window(signal_diff_right, 1, False, 'mid', 2, 50))
    # plt.hist(scores, bins='auto',  alpha=0.5)
    # plt.show()
    scores = standardise_scores_rank(e.random_projection_window(signal_diff_right, 1, False, 'mid', 2, 50))
    plt.hist(scores, bins=20)
    plt.title("Density of ranked scores of differenced signal (right)")
    plt.ylabel("Frequency")
    plt.xlabel("Ranked score")
    plt.show()
    scores = standardise_scores_rank(e.random_projection_window(e.normalise(data.values), 1, False, 'mid', 2, 50))
    plt.hist(scores, bins=20)
    plt.title("Density of ranked scores of original signal")
    plt.ylabel("Frequency")
    plt.xlabel("Ranked score")
    plt.show()
    # scores_2 = range_scale(e.random_projection_window(signal_diff_right, 1, False, 'mid', 2, 25))
    # plt.hist(scores_2, bins='auto',  alpha=0.5)
    # plt.show()

def amb_average():
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

    guesses = [item for sublist in to_add_times for item in sublist[1:]]

    data = data.reset_index().set_index('timestamp')
    data = data.drop('index', axis=1)
    data.index = pd.to_datetime(data.index)

    scores_1 = e.standardise_scores_z_score(e.random_projection_window(data.values, 1, False, "mid", 2, 1))
    scores_25 = e.standardise_scores_z_score(e.random_projection_window(data.values, 1, False, "mid", 2, 25))
    scores_100 = e.standardise_scores_z_score(e.random_projection_window(data.values, 1, False, "mid", 2, 100))

    dim = 1

    columns = list(data.columns)

    fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3,1,1,1]})
    axs[0].set_title(str(name).replace("_", " ").capitalize())
    axs[0].xaxis.set_visible(False)

    outliers = data.index[np.where(np.array(labels) > 0)].tolist()

    outliers = [i for i in outliers if i not in guesses]
    guesses_values = [[] for _ in range(dim)]

    outliers_values = [[] for _ in range(dim)]

    for d in range(dim):
        outliers_values[d] = data.loc[data.index.isin(outliers), columns[d]]
        axs[d].plot(data.index, data[columns[d]], 'k')

        guesses_values[d] = [item for sublist in to_add_values[d] for item in sublist]
        axs[d].scatter(guesses, guesses_values[d], color='yellow')

        axs[d].scatter(outliers, outliers_values[d], color='b', s=10, alpha=0.5)

    axs[dim].plot(range(len(scores_1[15:-15])), scores_1[15:-15])
    axs[dim].xaxis.set_visible(False)
    axs[dim].set_ylabel("$\ell$=1", rotation=90)
    axs[2].plot(range(len(scores_1[15:-15])), scores_25[15:-15])
    axs[2].xaxis.set_visible(False)
    axs[2].set_ylabel("$\ell$=25", rotation=90)
    axs[3].plot(np.array(range(len(scores_1[50:-50])))+35, scores_100[50:-50])
    axs[3].xaxis.set_visible(False)
    axs[3].set_ylabel("$\ell$=100", rotation=90)
    plt.show()

def plot_histogram_beats():
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(123, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)

    scores = np.load("C:/Users/carol/PycharmProjects/RandomProjectionsThesis/output_scores_MITBIH/scores_final_unstandardised_123_0.npy")
    all_scores_beat = get_beat_score(scores, convert(heart_beats_x))
    pc.plot_histogram(f"sample_123", signal, e.standardise_scores_z_score(all_scores_beat[100]), labels, None, runs=1, type="testing")


mp_diff()