import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import random_projection_MITBIH as rp
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from datetime import timedelta


def get_histo(labels, scores,guesses_index):
    normal = []
    outlier = []
    guesses = []

    for i, v in enumerate(labels):

        if i in guesses_index:
            guesses.append(scores[i])
        elif v == 1:  # outlier
            outlier.append(scores[i])
        else:
            normal.append(scores[i])

    return normal, outlier, guesses


# normalise different components of the time series.
def normalise(signal):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


def get_scores(data, win_length, k, norm_perservation, mean_comparison, win_length_mean, method):
    expanded_data = []
    for i in range(len(data)):
        if win_length == 0:
            expanded_data.append(data[i])
            continue
        if method == 'prev':
            previous = i - win_length + 1
        else:
            previous = i - win_length // 2
        future = i + win_length // 2
        if previous < 0:
            previous = 0
        if future > len(data):
            future = len(data)
        if method == "prev":

            expanded_data.append(data[previous:i + 1])
        else:
            expanded_data.append(data[previous:future])
    scores = rp.random_projection_new(expanded_data, k, norm_perservation)

    if mean_comparison:
        # window_length for mean might be different to RP method.
        if win_length != win_length_mean:
            expanded_data_mean = []
            for i in range(1, len(data) + 1):
                if win_length == 0:
                    expanded_data.append(data[i])
                    continue
                if method == "prev":
                    previous = i - win_length + 1
                else:
                    previous = i - win_length // 2
                future = i + win_length // 2
                if previous < 0:
                    previous = 0
                if future > len(data):
                    future = len(data)
                if method == "prev":
                    expanded_data.append(data[previous:i + 1])
                else:
                    expanded_data_mean.append(data[previous:future])

            outlier_scores_means = rp.mean_single_new(expanded_data_mean, data)
            outlier_scores_means_old = rp.mean_single(expanded_data_mean, k)
        else:
            outlier_scores_means = rp.mean_single_new(expanded_data, data)
            outlier_scores_means_old = rp.mean_single(expanded_data, k)

    return scores, outlier_scores_means, outlier_scores_means_old


def test(data, name, labels, win_length, k, norm_perservation, mean_comparison, win_length_mean, method, to_add_times, to_add_values, plot=False):
    signal = np.array(data['value'].diff().fillna(0)).reshape(-1,1)
    signal_norm = normalise(signal)
    scores, outlier_scores_means, outlier_scores_means_old = get_scores(signal_norm, win_length, k, norm_perservation,
                                                                        mean_comparison, win_length_mean, method)

    tpr, fpr = rp.compute_tpr_fpr(scores, labels)
    roc_auc = metrics.auc(fpr, tpr)

    tpr_mean_p, fpr_mean_p = rp.compute_tpr_fpr(outlier_scores_means, labels)
    roc_auc_mean_p = metrics.auc(fpr_mean_p, tpr_mean_p)

    tpr_mean, fpr_mean = rp.compute_tpr_fpr(outlier_scores_means_old, labels)
    roc_auc_mean = metrics.auc(fpr_mean, tpr_mean)

    fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    outliers = data.loc[np.where(np.array(labels) > 0), 'timestamp']
    outliers_values = data.loc[data['timestamp'].isin(outliers), 'value']
    # outliers_values = signal_norm[np.where(np.array(labels) > 0)]

    guesses = [item for sublist in to_add_times for item in sublist[1:]]
    guesses_values = [item for sublist in to_add_values for item in sublist]

    guesses_index = data.index[data['timestamp'].isin(guesses)].tolist()

    axs[0].plot(data['timestamp'], signal, 'k')
    axs[0].scatter(outliers, [-7,-7], color='b',s=5)
    # axs[0].scatter(guesses, guesses_values, color='yellow')
    axs[0].set_title(name.replace("_", " ").capitalize())
    axs[0].xaxis.set_visible(False)

    axs[1].bar(range(len(scores)), scores)
    axs[1].xaxis.set_visible(False)
    axs[1].set_ylabel("RP")
    axs[2].bar(range(len(scores)), outlier_scores_means)
    axs[2].xaxis.set_visible(False)
    axs[2].set_ylabel("Mu'")
    axs[3].bar(data['timestamp'], outlier_scores_means_old)
    axs[3].set_ylabel("Mu")

    plt.show()
    plt.cla()
    plt.close()

    print("RP", roc_auc)
    print("mean prime", roc_auc_mean_p)
    print("mean", roc_auc_mean)

    fig, axs = plt.subplots(1, 3)
    bins = 20
    normal, outlier, guessed = get_histo(labels, scores, guesses_index)
    axs[0].hist(normal, bins=bins, label="Normal", alpha=0.5)
    axs[0].hist(guessed, bins=bins, label= "Guessed", alpha=0.5)
    axs[0].hist(outlier, bins=bins, density=True, label="Outlier", alpha=0.5)
    axs[0].set_title(f"Densities for random projection method")
    axs[0].set_xlim([0, 1])
    axs[0].legend()
    normal, outlier,guessed = get_histo(labels, outlier_scores_means, guesses_index)
    axs[1].hist(normal, bins=bins, label="Normal", alpha=0.5)
    axs[1].hist(guessed, bins=bins, label="Guessed", alpha=0.5)
    axs[1].hist(outlier, bins=10, density=True, label="Outlier", alpha=0.5)
    axs[1].set_title(f"Densities for mu' method")
    axs[1].set_xlim([0, 1])
    axs[1].legend()
    normal, outlier,guessed = get_histo(labels, outlier_scores_means_old, guesses_index)
    axs[2].hist(normal, bins=bins, label="Normal", alpha=0.5)
    axs[2].hist(guessed, bins=bins, label="Guessed", alpha=0.5)
    axs[2].hist(outlier, bins=bins, label="Outlier", density=True, alpha=0.5)
    axs[2].set_title(f"Densities for mu method")
    axs[2].set_xlim([0, 1])
    axs[2].legend()

    plt.show()
    plt.cla()

    if plot:
        plt.title(f'Receiver Operating Characteristic for {name} with window length {win_length} normalised')
        plt.plot(fpr, tpr, 'b', label='(average) AUC = %0.4f' % roc_auc)

        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(f"./new-data/ROC_curves/{name}_{win_length}", bbox_inches='tight')
        plt.clf()
    return roc_auc, roc_auc_mean, roc_auc_mean_p


def guess(arr, first, last, labels):
    mid = (last + first) // 2
    arr[mid] = (arr[first] + arr[last]) / 2
    labels[mid] = max(labels[first], labels[last])
    if mid - 1 != first:
        arr, labels = guess(arr, first, mid, labels)
    if mid + 1 != last:
        arr, labels = guess(arr, mid, last, labels)

    return arr, labels


def clean_data(data, labels, name, unit = None, plot=False):
    if unit is not None:
        missing_values = False
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        to_add_times = []
        to_add_labels = []
        to_add_values = []
        for i in range(len(data['timestamp']) - 1):
            if ((data['timestamp'].iloc[i] + timedelta(hours=unit[1], minutes=unit[0])) < data['timestamp'].iloc[i + 1]):
                missing_values=True
                difference = int((data['timestamp'].iloc[i + 1] - data['timestamp'].iloc[i]).total_seconds() // (3600*unit[1]+60*unit[0]))
                values = [data['value'].iloc[i]]
                values.extend([0 for _ in range(difference - 1)])
                values.append(data['value'].iloc[i + 1])

                guess_labels = [labels[i]]
                guess_labels.extend([0 for _ in range(difference - 1)])
                guess_labels.append(labels[i])

                values, guess_labels = guess(values, 0, len(values) - 1, guess_labels)
                if difference > 2:
                    guess_labels = [labels[i]]
                    guess_labels.extend([1 for _ in range(difference - 1)])
                    guess_labels.append(labels[i])

                to_add_values.append(values[1:-1])
                to_add_labels.append(guess_labels[1:-1])

                times = []
                d1 = data['timestamp'].iloc[i]
                d2 = data['timestamp'].iloc[i + 1]
                delta = timedelta(hours=unit[1], minutes=unit[0])
                while d1 < d2:
                    times.append(d1)
                    d1 += delta
                to_add_times.append(times)

        for i, time in enumerate(to_add_times):
            index = data.index[data['timestamp'] == time[0]].tolist()[0] + 1
            indices = [index + i for i in range(1, len(time))]
            addition = pd.DataFrame({"timestamp": time[1:], "value": to_add_values[i]}, index=indices)

            data = pd.concat([data.iloc[:index], addition, data.iloc[index:]]).reset_index(drop=True)
            labels = labels[:index] + to_add_labels[i] + labels[index:]
    if plot:
        guesses = [item for sublist in to_add_times for item in sublist[1:]]
        guesses_values = [item for sublist in to_add_values for item in sublist]

        outliers = data.loc[np.where(np.array(labels) > 0), 'timestamp']
        outliers = [i for i in outliers.values if i not in guesses]
        outliers_values = data.loc[data['timestamp'].isin(outliers), 'value']

        plt.plot(data['timestamp'], data['value'], 'k')
        plt.scatter(guesses, guesses_values, color='yellow')
        plt.scatter(outliers, outliers_values, color='b')

        plt.title(name.replace("_", " ").capitalize())
        plt.xlabel("Timestamp")
        plt.show()
    if missing_values:
        print("Missing values in dataset.")
    return data, labels, to_add_times, to_add_values


def load_data(name, plot):
    path = "C:/Users/carol/OneDrive - Delft University of Technology/Desktop/Masters/Thesis/new-data/"
    data = pd.read_csv(os.path.join(path, f"data/{name}"))
    j = open(os.path.join(path, "combined_labels.json"))
    combined_labels = json.load(j)
    outliers = np.array(combined_labels[name])
    outliers_values = data.loc[data['timestamp'].isin(outliers), 'value']
    labels = []

    for time in data['timestamp']:
        if time not in outliers:
            labels.append(0)
        else:
            labels.append(1)
    if plot:
        plt.plot(data['timestamp'], data['value'], 'k')
        plt.scatter(outliers, outliers_values)
        plt.title(name.split('/')[1].strip()[:-4].replace("_", " ").capitalize())
        x_positions = np.arange(0, len(data['timestamp']) + 1, 500)
        x_labels = [str(i) for i in x_positions]
        plt.xticks(x_positions, x_labels)
        plt.xlabel("Timestamp")
        plt.show()

    # return np.array(data['timestamp']).reshape(-1,1), labels
    return data, labels


if __name__ == '__main__':
    data, labels = load_data("realKnownCause/ambient_temperature_system_failure.csv", False)
    data, labels, to_add_times, to_add_values = clean_data(data, labels, name="ambient_temperature_system_failure", plot=False)
    test(data,"ambient_temperature_system_failure", labels, 25, 1, False, True, 25, "prev", to_add_times, to_add_values, False)
    # data_norm, intervals, intervals_x, interval_labels = make_intervals(288, data, labels,
    #                                                                     "ambient_temperature_system_failure", False)
    # cpu_utilization_asg_misconfiguration
    print(len(data), sum(labels))

    # rp_aucs_max = []
    # rp_aucs_diff = []
    # mean_aucs_max = []
    # mean_aucs_diff = []
    # mean_old_aucs_max = []
    # mean_old_aucs_diff = []
    #
    # win_length = 500
    # win_length_mean = 500
    # for win_length in range(0,300,2):
    #     print(win_length)
    #     rp_scores, labels, interval_outlier_scores_mean, interval_outlier_scores_mean_old = get_interval_scores(data, data_norm,
    #                                                                                                         intervals_x,
    #                                                                                                         interval_labels,
    #                                                                                                         win_length,
    #                                                                                                         win_length,
    #                                                                                                         k=1,
    #                                                                                                         pres_norm=False,
    #                                                                                                         mean_comparison=True, method = "mid")
    #
    #
    #     auc_max, auc_diff = rp.show_roc_curve(rp_scores, labels, "RP", "machine_temperature_system_failure", win_length)
    #     rp_aucs_max.append(auc_max)
    #     rp_aucs_diff.append(auc_diff)
    #     auc_max, auc_diff = rp.show_roc_curve(interval_outlier_scores_mean, labels, "Mean prime", "machine_temperature_system_failure", win_length)
    #     mean_aucs_max.append(auc_max)
    #     mean_aucs_diff.append(auc_diff)
    #     auc_max, auc_diff = rp.show_roc_curve(interval_outlier_scores_mean_old, labels, "Mean", "machine_temperature_system_failure", win_length)
    #     mean_old_aucs_max.append(auc_max)
    #     mean_old_aucs_diff.append(auc_diff)
    # print(rp_aucs_max)
    # print(rp_aucs_diff)
    # print(mean_aucs_max)
    # print(mean_aucs_diff)
    # print(mean_old_aucs_max)
    # print(mean_old_aucs_diff)
    # auc, method = rp.show_roc_curve(beat_outlier_scores, labels, "random projection win_length=avg(beat length)",
    #                              "ambient_temperature_system_failure", win_length)
    # # print("mean")
    # mean_max, mean_diff = rp.show_roc_curve(beat_outlier_scores_mean, labels, "Mean test new", "ambient_temperature_system_failure",
    #                                      win_length_mean)
    # # print("mean old")
    # mean_max_old, mean_diff_old = rp.show_roc_curve(beat_outlier_scores_mean_old, labels, "Mean test old","ambient_temperature_system_failure",
    #                                      win_length_mean)
    # rp_aucs = []
    # mean_p_aucs = []
    # mean_aucs = []
    #
    # x = [1]
    # x.extend(np.arange(10,200, 10))
    #
    # for i in x:
    #     print(i)
    #     roc_auc, roc_auc_mean, roc_auc_mean_p = test(data, "ambient temperature system", labels, win_length=i, k=1, norm_perservation=False, mean_comparison=True,win_length_mean=i)
    #     rp_aucs.append(roc_auc)
    #     mean_p_aucs.append(roc_auc_mean_p)
    #     mean_aucs.append(roc_auc_mean)
    #
    # plt.plot(x, rp_aucs, label="Random projection")
    # # plt.plot(x, mean_p_aucs, label="Mean prime")
    # plt.plot(x, mean_aucs, label="Mean")
    # plt.title(f"AUC values for ambient temperature system for different window lengths")
    # plt.xlabel("Window length")
    # plt.legend()
    # plt.show()
