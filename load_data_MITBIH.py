import wfdb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter


# we only consider two classes: normal and abnormal beats.
def classify_beat(symbol):
    if symbol == "N" or symbol == ".":
        return 0
    else:
        return 1

# normalise different components of the time series.
def normalise(signal):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


# Find beat that is labelled: still a bit premature, just guessing beat.
def label_clean_segments(record, annotation, sampfrom):

    heart_beats = []
    heart_beats_x = []
    labels = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample

    signal_norm = normalise(record.p_signal)

    #Print types of outliers
    print(Counter(ann_symbol))

    start = sampfrom
    for i, i_sample in enumerate(ann_sample):

        #skipping start up beats labelled "+"
        if ann_symbol[i] == "+":
            start = ann_sample[i + 1] - 10 - 1
            continue

        label = classify_beat(ann_symbol[i])

        if i != len(ann_sample) - 1:
            end = ann_sample[i + 1] - 10
        else:
            end = len(signal_norm)+sampfrom

        beat = signal_norm[start-sampfrom:end-sampfrom]
        beat_x = range(start-sampfrom, end-sampfrom)

        if label is not None and beat.size > 0:
            heart_beats.append(beat)
            heart_beats_x.append(beat_x)
            labels.append(label)

        #update start value
        start = end - 1

    return signal_norm, heart_beats, heart_beats_x, labels

#Plot using package - high runtime!
def plot_data_basic(record, annotation):
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 106',
                   figsize=(10, 4), ecg_grids='all', )

#Own plot with anomalies coloured red.
def plot_data(record, signal_norm, heart_beats, heart_beats_x, labels, sampfrom):

    #Colour outlier red, normal beats black
    colours = []
    for i in labels:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")

    fig, axs = plt.subplots(2)

    x_labels = [i for i in range(sampfrom, signal_norm.shape[0]+sampfrom) if i%(10*record.fs) == 0]
    x_labels_values = [int(i/record.fs) for i in x_labels]

    if sampfrom != 0:
        x_labels = [i-sampfrom for i in x_labels]

    signal1 = [[item[0] for item in heart_beat] for heart_beat in heart_beats]
    signal2 = [[item[1] for item in heart_beat] for heart_beat in heart_beats]

    fig.suptitle(f"MIT-BIH Arrhythmia Database: Sample {record.record_name}")

    for i, c in enumerate(colours):
        axs[0].plot(heart_beats_x[i], signal1[i], color=c)
        axs[1].plot(heart_beats_x[i], signal2[i], color=c)

    axs[0].set(ylabel=record.sig_name[0])
    axs[0].set_xticks(x_labels, x_labels_values)
    axs[1].set(xlabel="seconds", ylabel=record.sig_name[1])
    axs[1].set_xticks(x_labels, x_labels_values)

    plt.show()

#Main function.
def load_mit_bih_data(name, sampfrom, sampto):
    print(name)
    #Read record
    record = wfdb.rdrecord(f'./data/sample_{name}/{name}', sampfrom= sampfrom, sampto=sampto)

    length = record.sig_len + sampfrom

    # Annotates each beat peak.
    annotation = wfdb.rdann(f'./data/sample_{name}/{name}', 'atr', sampfrom=sampfrom, sampto=length)
    #For basic plot
    annotation_basic = wfdb.rdann(f'./data/sample_{name}/{name}', 'atr', sampfrom=sampfrom, sampto=length, shift_samps = True)

    return record, annotation
