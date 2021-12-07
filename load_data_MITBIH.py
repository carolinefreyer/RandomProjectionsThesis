import numpy as np
import wfdb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# we only consider two classes: normal and abnormal beats.
def classify_beat(symbol):
    if symbol == "N" or symbol == ".":
        return 0
    else:
        return 1


def get_beat_old(signal, peak, freq):
    # length of heart beat is now randomly set as 3 seconds per side.
    window_size = 3
    window_one_side = window_size * freq
    beat_start = peak - window_one_side
    beat_end = peak + window_one_side + 1
    # this cuts off the last beat incase its not whole
    if beat_end < signal.shape[0]:
        return signal[beat_start:beat_end]
    else:
        return np.array([])


def normalise(record):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(record.p_signal)


# Find beat that is labelled.
def label_clean_segments(record, annotation):
    heart_beats = []
    heart_beats_x = []
    labels = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample

    start = 0
    for i, i_sample in enumerate(ann_sample):
        if ann_symbol[i] == "+":
            continue
        label = classify_beat(ann_symbol[i])
        if i != len(ann_sample) - 1:
            end = ann_sample[i + 1] - 10
        else:
            end = len(record.p_signal)
        beat = record.p_signal[start:end]
        beat_x = range(start, end)
        start = end + 1
        if label is not None and beat.size > 0:
            heart_beats.append(beat)
            heart_beats_x.append(beat_x)
            labels.append(label)

    return heart_beats, heart_beats_x, labels


def plot_data_basic(record, annotation):
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10, 4), ecg_grids='all')


def plot_data(record, annotation):
    heart_beats, heart_beats_x, labels = label_clean_segments(record, annotation)
    colours = []
    for i in labels:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")
    fig, axs = plt.subplots(2)
    x_labels = [i for i in range(0, record.p_signal.shape[0], 10 * record.fs)]
    x_labels_values = [10 * i for i in range(len(x_labels))]
    fig.suptitle("MIT-BIH Arrhythmia Database: Sample 100")
    signal1 = [[item[0] for item in heart_beat] for heart_beat in heart_beats]
    signal2 = [[item[1] for item in heart_beat] for heart_beat in heart_beats]
    for i, c in enumerate(colours):
        axs[0].plot(heart_beats_x[i], signal1[i], color=c)
        axs[1].plot(heart_beats_x[i], signal2[i], color=c)
    axs[0].set(ylabel=record.sig_name[0])
    axs[0].set_xticks(x_labels, x_labels_values)
    axs[1].set(xlabel="seconds", ylabel=record.sig_name[1])
    axs[1].set_xticks(x_labels, x_labels_values)
    plt.show()


def load_mit_bih_data(plot, sampto):
    record = wfdb.rdrecord('./data/sample_100/100', sampto=sampto)
    # Annotates each beat
    annotation = wfdb.rdann('./data/sample_100/100', 'atr', sampto=sampto)
    if plot:
        plot_data(record, annotation)
    return record, annotation
