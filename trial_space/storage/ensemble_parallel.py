import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import load_data_MITBIH as mb
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

import load_data_NAB as nab
import plot_curves as pc
import ensemble as e


def task(k_range, norm_perservation_range, win_pos_range, norm_power_range, win_length_range, signal, signal_diff_right, signal_diff_left, i):
    mode = random.choice([1, 2, 3])
    if mode == 1:  # mu
        scores = e.mean_prime_2(signal, norm_power_range, win_length_range)
    elif mode == 2:  # normal RP
        scores = e.random_projection_window(signal, k_range, norm_perservation_range, win_pos_range,
                                                         norm_power_range, win_length_range)
    else:  # RP on differenced signal
        direction = random.choice(["left", "right"])
        if direction == "right":
            scores = e.random_projection_window(signal_diff_right, k_range, norm_perservation_range, win_pos_range,
                                                norm_power_range, win_length_range)
        else:
            scores = e.random_projection_window(signal_diff_left, k_range, norm_perservation_range, win_pos_range,
                                                norm_power_range, win_length_range)
    return scores


# m = number of components
def run(data, win_length_max, m):
    k_range = [1, 2, 3, 4, 5, 10]
    norm_perservation_range = [True, False]
    win_pos_range = ['mid']  # can add prev (if online) and future.
    norm_power_range = [1, 1.5, 2, 3]
    win_length_range = np.unique(np.logspace(0, np.log(win_length_max), 50, dtype=int, base=np.e))

    signal, signal_diff_right, signal_diff_left = e.get_signals(data)

    outlier_scores_m = []
    with ProcessPoolExecutor(6) as executor:
        for r in tqdm([executor.submit(task, k_range, norm_perservation_range, win_pos_range, norm_power_range, win_length_range, signal, signal_diff_right, signal_diff_left, i) for i in range(m)]):
        # args = ((k_range, norm_perservation_range, win_pos_range, norm_power_range, win_length_range, signal, signal_diff_right, signal_diff_left, i) for i in range(m))
        # for r in pool.map(lambda p: task(*p), args):
            outlier_scores_m.append(r.result())
    return e.summarise_scores(np.array(outlier_scores_m))


def run_NAB(m, type):
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

    guesses = [item for sublist in to_add_times for item in sublist[1:]]
    guesses_index = data.index[data['timestamp'].isin(guesses)].tolist()

    e.summarise_data(data, labels, guesses)

    scores = run(data, 100, m)
    print(scores[810], scores[811], scores[812], scores[813], scores[814], scores[815], scores[816])

    pc.all_plots(name, data, scores, labels, guesses, to_add_values, [], [], runs=m, type=type)


def run_MITBIH(m, type):
    sampfrom = 0
    sampto = None

    name = 100

    record, annotation = mb.load_mit_bih_data(name, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_q_points_single(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))]).reshape(-1, 1)
    signal_norm = np.hstack((timestamp, signal_norm))
    signal = pd.DataFrame(signal_norm, columns=np.append("timestamp", record.sig_name))

    e.summarise_data(signal, labels, [])
    scores = run(signal, 400, m)
    pc.all_plots(name, signal, scores, labels, [], [], [], [], runs=m, type=type)


if __name__ == '__main__':
    run_NAB(1000, 'trial_space')
    # run_MITBIH(10, 'trial_space')
