import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import timedelta


def guess(arr, first, last, labels):
    mid = (last + first) // 2
    arr[mid] = (arr[first] + arr[last]) / 2
    labels[mid] = max(labels[first], labels[last])
    if mid - 1 != first:
        arr, labels = guess(arr, first, mid, labels)
    if mid + 1 != last:
        arr, labels = guess(arr, mid, last, labels)

    return arr, labels


def clean_data(data, labels, name, unit=None, plot=False):
    dim = len(data.columns) - 1
    missing_values = False
    if unit is not None:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        to_add_times = []
        to_add_labels = []
        to_add_values = [[] for _ in range(dim)]
        for i in range(len(data['timestamp']) - 1):
            if ((data['timestamp'].iloc[i] + timedelta(hours=unit[1], minutes=unit[0]))
                    < data['timestamp'].iloc[i + 1]):
                missing_values = True
                difference = int((data['timestamp'].iloc[i + 1] - data['timestamp'].iloc[i]).total_seconds() // (
                        3600 * unit[1] + 60 * unit[0]))
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

                to_add_values[0].append(values[1:-1])
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
            addition = pd.DataFrame({"timestamp": time[1:], "value": to_add_values[0][i]}, index=indices)

            data = pd.concat([data.iloc[:index], addition, data.iloc[index:]]).reset_index(drop=True)
            labels = labels[:index] + to_add_labels[i] + labels[index:]
        if missing_values:
            print("Missing values in dataset.")

    if plot:
        if missing_values:
            guesses = [item for sublist in to_add_times for item in sublist[1:]]
            guesses_values = [item for sublist in to_add_values[0] for item in sublist]

        plt.plot(data['timestamp'], data['value'], 'k')

        outliers = data.loc[np.where(np.array(labels) > 0), 'timestamp']
        if missing_values:
            outliers = [i for i in outliers.values if i not in guesses]
            plt.scatter(guesses, guesses_values, color='yellow')

        outliers_values = data.loc[data['timestamp'].isin(outliers), 'value']
        plt.scatter(outliers, outliers_values, color='b')

        plt.title(name.replace("_", " ").capitalize())
        plt.xlabel("Timestamp")
        plt.show()

    return data, labels, to_add_times, to_add_values


def load_data(name, plot):
    path = "C:/Users/carol/PycharmProjects/RandomProjectionsThesis/data/NAB/"
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

    return data, labels
