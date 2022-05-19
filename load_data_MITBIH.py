import wfdb
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os

from data.MITBIH.Q_points.QRS_util import *


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


def QRS_detection(ecg, name):
    fs = 360
    ecg_100 = ecg * 90

    R_peaks, S_pint, Q_point = EKG_QRS_detect(ecg_100, ecg, fs, True, False)
    # print("Number of Q points: ", len(Q_point))
    diff = np.array([Q_point[i + 1] - Q_point[i] for i in range(len(Q_point) - 1)])
    # output_df = pd.DataFrame({'top': Q_point})
    # output_df.to_csv(f'Q_points/{name}Q.dat', index=False, header=False)
    return Q_point


def label_clean_segments_q_points(record, annotation, sampfrom):
    heart_beats = []
    heart_beats_x = []
    labels = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample
    signal = record.p_signal
    signal_norm = normalise(signal)

    if os.path.exists(f'data/MITBIH/Q_points/{record.record_name}Q.dat') and sampfrom == 0:
        q_points = np.genfromtxt(f'data/MITBIH/Q_points/{record.record_name}Q.dat').astype(int)
    else:
        index = record.sig_name.index('MLII')
        q_points = QRS_detection(signal_norm[:, index], record.record_name)

    end = sampfrom
    for i, q_i in enumerate(q_points):
        if end - 1 > sampfrom + q_i:
            # print("skipped", q_i, end)
            continue
        start = sampfrom + q_i
        if i < len(q_points) - 1:
            end = q_points[i + 1] + sampfrom + 1
        else:
            # print("else:beat ended prematurely.")
            continue

        symbols = [ann_symbol[k] for k in np.where((ann_sample >= start) & (ann_sample <= end))[0]]

        count = 0
        while len(symbols) == 0:
            # print("Skipped q, ", start, end)
            count += 1
            if i < len(q_points) - count - 1:
                end = q_points[i + 1 + count] + sampfrom + 1
            elif i == len(q_points) - count - 1:
                # print("last beat ended prematurely.")
                break

            symbols = [ann_symbol[k] for k in np.where((ann_sample >= start) & (ann_sample <= end))[0]]

        if symbols != []:
            temp_labels = []
            for s in symbols:
                temp_labels.append(classify_beat(s))
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


def label_clean_q_points_single(record, annotation, sampfrom, sampto):
    heart_beats = []
    heart_beats_x = []
    labels = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample
    signal = record.p_signal
    signal_norm = normalise(signal)

    if os.path.exists(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/Q_points/{record.record_name}Q.dat') and sampfrom == 0 and sampto is None:
        q_points = np.genfromtxt(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/Q_points/{record.record_name}Q.dat').astype(int)
    else:
        index = record.sig_name.index('MLII')
        q_points = QRS_detection(signal_norm[:, index], record.record_name)

    end = sampfrom
    for i, q_i in enumerate(q_points):
        if end - 1 > sampfrom + q_i:
            # print("skipped", q_i, end)
            continue
        start = sampfrom + q_i
        if i < len(q_points) - 1:
            end = q_points[i + 1] + sampfrom + 1
        else:
            # print("else:beat ended prematurely.")
            continue

        symbols = [ann_symbol[k] for k in np.where((ann_sample >= start) & (ann_sample <= end))[0]]

        count = 0
        while len(symbols) == 0:
            # print("Skipped q, ", start, end)
            count += 1
            if i < len(q_points) - count - 1:
                end = q_points[i + 1 + count] + sampfrom + 1
            elif i == len(q_points) - count - 1:
                # print("last beat ended prematurely.")
                break

            symbols = [ann_symbol[k] for k in np.where((ann_sample >= start) & (ann_sample <= end))[0]]

        if symbols != []:
            temp_labels = []
            for s in symbols:
                temp_labels.append(classify_beat(s))
            label = max(temp_labels)
        else:
            label = None

        beat = signal_norm[start - sampfrom:end - sampfrom]
        beat_x = range(start - sampfrom, end - sampfrom)

        if label is not None and beat.size > 0:
            heart_beats.append(beat)
            heart_beats_x.append(beat_x)
            for _ in beat_x[:-1]:
                labels.append(label)

    return signal_norm[q_points[0] - sampfrom:q_points[-1] - sampfrom], heart_beats, heart_beats_x, labels


# Find beat that is labelled: still a bit premature, just guessing beat.
def label_clean_segments(record, annotation, sampfrom):
    heart_beats = []
    heart_beats_x = []
    labels = []
    labels_plot = []

    ann_symbol = annotation.symbol
    ann_sample = annotation.sample

    signal_norm = normalise(record.p_signal)

    # Print types of outliers
    print(Counter(ann_symbol))

    start = sampfrom
    for i, i_sample in enumerate(ann_sample):

        label = classify_beat(ann_symbol[i])
        if label == 1:
            label_plot = ann_symbol[i]
        else:
            label_plot = ""

        if i != len(ann_sample) - 1:
            end = ann_sample[i + 1] - 10
        else:
            end = len(signal_norm) + sampfrom

        beat = signal_norm[start - sampfrom:end - sampfrom]
        beat_x = range(start - sampfrom, end - sampfrom)

        if label is not None and beat.size > 0:
            heart_beats.append(beat)
            heart_beats_x.append(beat_x)
            labels.append(label)
            labels_plot.append(label_plot)

        # update start value
        start = end - 1

    return signal_norm, heart_beats, heart_beats_x, labels, labels_plot


# Plot using package - high runtime!
def plot_data_basic(record, annotation):
    wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title=f'MIT-BIH Record {record.record_name}',
                   figsize=(10, 4), ecg_grids='all', )


# Own plot with anomalies coloured red.
def plot_data(record, heart_beats_x, labels, sampfrom):
    # Colour outlier red, normal beats black
    colours = []
    for i in labels[:25]:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")

    fig, axs = plt.subplots(2)

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

        # axs[0].scatter(heart_beats_x[i][0], signal1[i][0], color='b', s=20)
    axs[0].set_ylabel(record.sig_name[0])
    axs[0].set_xticks(x_labels,x_labels_values)
    axs[1].set_ylabel(record.sig_name[1])
    axs[1].set_xticks(x_labels,x_labels_values)
    axs[1].set_xlabel("time (seconds)")
    # for i, c in enumerate(colours):
    #     axs[0].text(heart_beats_x[i][len(heart_beats_x[i]) // 8], axs[0].get_ylim()[1] + 0.1, str(labels_plot[i]))

    plt.show()




# Main function.
def load_mit_bih_data(name, sampfrom, sampto):
    print(name)
    # Read record
    record = wfdb.rdrecord(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/sample_{name}/{name}', sampfrom=sampfrom, sampto=sampto)

    length = record.sig_len + sampfrom

    # Annotates each beat peak.
    annotation = wfdb.rdann(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/sample_{name}/{name}', 'atr', sampfrom=sampfrom, sampto=length)
    # For basic plot
    annotation_basic = wfdb.rdann(f'C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/MITBIH/sample_{name}/{name}', 'atr', sampfrom=sampfrom, sampto=length,
                                  shift_samps=True)

    return record, annotation
