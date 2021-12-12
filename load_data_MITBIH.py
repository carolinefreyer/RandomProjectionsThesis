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

# normalise different components of the time series.
def normalise(record):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(record.p_signal)


# Find beat that is labelled: still a bit premature - maybe weigh peak?
def label_clean_segments(record, annotation, sampfrom):
    heart_beats = []
    heart_beats_x = []
    labels = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample

    start = sampfrom
    for i, i_sample in enumerate(ann_sample):
        if ann_symbol[i] == "+":
            start = ann_sample[i + 1] - 10 - 1
            continue
        label = classify_beat(ann_symbol[i])
        if i != len(ann_sample) - 1:
            end = ann_sample[i + 1] - 10
        else:
            end = len(record.p_signal)+sampfrom

        beat = record.p_signal[start-sampfrom:end-sampfrom]
        beat_x = range(start-sampfrom, end-sampfrom)
        start = end - 1
        if label is not None and beat.size > 0:
            heart_beats.append(beat)
            heart_beats_x.append(beat_x)
            labels.append(label)

    return heart_beats, heart_beats_x, labels

#Plot using package.
def plot_data_basic(record, annotation):
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10, 4), ecg_grids='all')

#Own plot with anomalies coloured red.
def plot_data(record, annotation, sampfrom):
    heart_beats, heart_beats_x, labels = label_clean_segments(record, annotation, sampfrom)

    colours = []
    for i in labels:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")
    fig, axs = plt.subplots(2)
    x_labels = [i for i in range(sampfrom, record.p_signal.shape[0]+sampfrom) if i%(10*record.fs) == 0]
    x_labels_values = [int(i/record.fs) for i in x_labels]
    if sampfrom != 0:
        x_labels = [i-sampfrom for i in x_labels]
    fig.suptitle(f"MIT-BIH Arrhythmia Database: Sample {record.record_name}")
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

#Main function.
def load_mit_bih_data(name, plot, sampfrom, sampto):
    record = wfdb.rdrecord(f'./data/sample_{name}/{name}', sampfrom= sampfrom, sampto=sampto)

    length = record.sig_len + sampfrom

    # Annotates each beat peak.
    annotation = wfdb.rdann(f'./data/sample_{name}/{name}', 'atr', sampfrom=sampfrom, sampto=length)

    if plot:
        plot_data(record, annotation, sampfrom)
    return record, annotation
