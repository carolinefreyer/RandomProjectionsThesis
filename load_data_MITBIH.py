import wfdb
from sklearn.preprocessing import StandardScaler
import os

from QRS_util import *


# Labels all beats normal or outlier.
def classify_beat(symbol):
    if symbol == "N" or symbol == ".":
        return 0
    else:
        return 1


# Normalise different components of the time series.
def normalise(signal):
    # Centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


# Calls QRS detection algorithm implemented by Kemeng Chen.
def QRS_detection(ecg):
    fs = 360
    ecg_100 = ecg * 90

    R_peaks, S_pint, Q_point = EKG_QRS_detect(ecg_100, ecg, fs, True, False)
    return Q_point


# Segments time series into heartbeats and determines labels for each heartbeat.
# Returns normalised signal and segmentation intervals with labels.
def label_clean_segments_q_points(record, annotation, sampfrom):
    heart_beats = []
    heart_beats_x = []
    labels = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample
    signal = record.p_signal
    signal_norm = normalise(signal)

    if os.path.exists(
            f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/Q_points/{record.record_name}Q.dat') and sampfrom == 0:
        q_points = np.genfromtxt(
            f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/Q_points/{record.record_name}Q.dat').astype(
            int)
    else:
        index = record.sig_name.index('MLII')
        q_points = QRS_detection(signal_norm[:, index], record.record_name)

    end = sampfrom
    for i, q_i in enumerate(q_points):
        if end - 1 > sampfrom + q_i:
            continue
        start = sampfrom + q_i
        if i < len(q_points) - 1:
            end = q_points[i + 1] + sampfrom + 1
        else:
            continue

        # Get original labels contained by the segment.
        symbols = [ann_symbol[k] for k in np.where((ann_sample >= start) & (ann_sample <= end))[0]]

        # Merge with next segment if contains no original label.
        count = 0
        while len(symbols) == 0:
            count += 1
            if i < len(q_points) - count - 1:
                end = q_points[i + 1 + count] + sampfrom + 1
            elif i == len(q_points) - count - 1:
                break

            symbols = [ann_symbol[k] for k in np.where((ann_sample >= start) & (ann_sample <= end))[0]]

        if symbols != []:
            temp_labels = []
            for s in symbols:
                temp_labels.append(classify_beat(s))
            # Labels beat an outlier if it contains at least one outlier, else normal.
            label = max(temp_labels)
        else:
            label = None

        beat = signal_norm[start - sampfrom:end - sampfrom]
        beat_x = range(start - sampfrom, end - sampfrom)

        if label is not None and beat.size > 0:
            heart_beats.append(beat)
            heart_beats_x.append(beat_x)
            labels.append(label)

    return signal_norm[q_points[0] - sampfrom:q_points[-1] - sampfrom], heart_beats, heart_beats_x, labels


# Plots dataset with outliers coloured in red.
def plot_data(record, heart_beats_x, labels, sampfrom):
    # Colour outlier red, normal beats black
    colours = []
    for i in labels:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")

    fig, axs = plt.subplots(2)

    x_labels = [i for i in range(sampfrom, heart_beats_x[-1][-1] + sampfrom) if i % (10 * record.fs) == 0]
    x_labels_values = [int(i / record.fs) for i in x_labels]

    if sampfrom != 0:
        x_labels = [i - sampfrom for i in x_labels]
    signal = record.p_signal
    signal1 = [signal[heart_beat[0]:heart_beat[-1] + 1][:, 0] for heart_beat in heart_beats_x]
    signal2 = [signal[heart_beat[0]:heart_beat[-1] + 1][:, 1] for heart_beat in heart_beats_x]

    fig.suptitle(f"MIT-BIH Arrhythmia Database: Sample {record.record_name}")

    for i, c in enumerate(colours):
        axs[0].plot(heart_beats_x[i], signal1[i], color=c)
        axs[1].plot(heart_beats_x[i], signal2[i], color=c)
        axs[0].scatter(heart_beats_x[i][0], signal1[i][0], color='b', s=20)
    axs[0].set_ylabel(record.sig_name[0])
    axs[0].set_xticks(x_labels, x_labels_values)
    axs[1].set_ylabel(record.sig_name[1])
    axs[1].set_xticks(x_labels, x_labels_values)
    axs[1].set_xlabel("time (seconds)")

    plt.show()


# Loads original MITBIH dataset sample.
def load_mit_bih_data(name, sampfrom, sampto):
    print("Sample", name)
    # Read record
    record = wfdb.rdrecord(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/sample_{name}/{name}',
                           sampfrom=sampfrom, sampto=sampto)

    length = record.sig_len + sampfrom

    # Annotates each beat peak.
    annotation = wfdb.rdann(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/sample_{name}/{name}',
                            'atr', sampfrom=sampfrom, sampto=length)

    return record, annotation
