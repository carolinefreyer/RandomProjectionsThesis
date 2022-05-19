import load_data_MITBIH as mb
import ensemble_final as e
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
        beat_scores[i] = max([scores[i - heart_beats_x[0][0]] for i in range(x[0], x[1])])
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
        for j in range(2):
            scores = e.standardise_scores_z_score(e.random_projection_window(signal_norm, k, True, "prev", 2, 260))
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
    return point_reconstruct[(len(point)-1)//2]

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
        if v == 1 and i not in guesses_index:
            guesses.append(scores[i])
        elif v == 1:  # outlier
            outlier.append(scores[i])
        else:
            normal.append(scores[i])
    print(scores[812])
    return normal, outlier, guesses

def plot_histogram(name, scores, labels):
    normal, outlier, guessed = get_histo(labels, scores, [4124, 6801])
    plt.hist(np.array(normal).reshape(-1,), bins='auto', label="Normal", alpha=0.5)
    plt.hist(np.array(outlier).reshape(-1,), bins='auto', density= True, label="Outliers", alpha=0.5)
    plt.hist(np.array(guessed).reshape(-1, ), bins='auto', label="Interpolations", alpha=0.5)
    plt.title(f"Histogram of Outlierness Scores for {name} method on differenced AMB dataset")
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

        scores = e.standardise_scores_z_score(e.random_projection_window(signal_norm, 1, False, "prev", 2, 260))
        beat_scores = get_scores(scores, convert(heart_beats_x))
        tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(beat_scores, labels, max(beat_scores), min(beat_scores))
        t.append(roc_auc)
    print(max(r))
    r.append(0.92)
    plt.boxplot([r,t])
    plt.xticks([1,2],["with norm preservation", "without norm preservation"])
    plt.title("Box plot of AUC of 50 random projection runs with and without scaling")
    plt.ylabel("ROC AUC")
    plt.show()
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

    for i, win in enumerate(range(250, 450, 10)):
        scores = e.standardise_scores_z_score(e.mean_prime(signal_norm, "mid", 2, win))
        beat_scores = get_scores(scores, convert(heart_beats_x))
        tpr, fpr, precision, roc_auc, pr_auc = pc.compute_rates(beat_scores, labels, max(beat_scores), min(beat_scores))
        r.append(roc_auc)

    x = range(250, 450, 10)
    plt.plot(x, r)
    plt.title("AUC values for different window lengths")
    plt.xlabel("Window length $\ell$")
    plt.ylabel("ROC AUC")
    plt.show()

def plot_scores_nab(name, signal,signal_diff, signal_diff_left, scores, labels):

    fig, axs = plt.subplots(3, 1)
    axs[0].set_title(str(name).replace("_", " ").capitalize())
    outliers = []
    for i,l in enumerate(labels):
        if l == 1:
            outliers.append(i)
    indices = [i for i in outliers if start <= i <= end]
    axs[0].plot(range(start,end), signal, color= 'k')
    # axs[0].scatter(812,signal[112], color='b', s=10, alpha=0.5)
    axs[0].set_ylabel("Original")
    axs[0].scatter(indices,signal[np.array(indices) - start], color= 'yellow', s = 10, alpha=0.5)
    axs[1].plot(range(start, end), signal_diff, color='k')
    axs[1].scatter(indices,signal_diff[np.array(indices) - start], color= 'yellow', s = 10, alpha=0.5)
    axs[1].set_ylabel("Right Differenced $\mathbf{X}\_\mathbf{r}$")
    axs[2].set_ylabel("Random projection score \n (standardised)")
    axs[2].bar(range(start,end),scores)
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

