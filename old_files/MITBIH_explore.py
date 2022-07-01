import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from numba import jit

import load_data_MITBIH as mb
from old_files import ensemble_WINNOW_testing as e


def convert(heart_beats_x):
    heart_beats_x_array = []
    for x in heart_beats_x:
        heart_beats_x_array.append([x[0],x[-1]])
    return np.array(heart_beats_x_array)


@jit(nopython=True)
def get_beat_score(all_scores, heart_beats_x):
    all_scores_beats = np.full((len(all_scores),len(heart_beats_x)),0.0)
    for i, score in enumerate(all_scores):
        for j, x in enumerate(heart_beats_x):
            beat_scores = [score[k-heart_beats_x[0][0]] for k in range(x[0], x[1])]
            all_scores_beats[i][j] = max(beat_scores)
    return all_scores_beats


def plot():
    fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 3, 1, 1]})

    colours = []
    for i in labels[:25]:
        if i == 0:
            colours.append("k")
        else:
            colours.append("r")

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
    # axs[2].set_xlabel("time (seconds)")

    a = heart_beats_x[0][0]
    b = heart_beats_x[24][-1]
    axs[2].plot(range(a, b), scores_test[a:b].reshape(-1, ))
    axs[2].xaxis.set_visible(False)
    axs[2].set_ylabel("Individual \n scores")
    bar_pos = [heart_beat[len(heart_beat) // 2] for heart_beat in heart_beats_x[:25]]
    axs[3].bar(bar_pos, scores_test_beat[:25].reshape(-1, ), width=200)
    axs[3].xaxis.set_visible(False)
    axs[3].set_ylabel("Beat \n scores")
    plt.show()


def summarisation(sample, max_window_size, n_runs, parallelise, num_workers):
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(sample, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)
    e.summarise_data(heart_beats, labels, [])
    # all_scores = e.run(signal, max_window_size, n_runs, parallelise, num_workers)
    all_scores = np.load("../output_scores_MITBIH/scores_final_unstandardised_123_0.npy")
    print(all_scores.shape)
    all_scores_beat = get_beat_score(all_scores, convert(heart_beats_x))
    # print("got scores")
    # for p in [0, 0.2, 0.4, 0.6, 0.8]:
    #     np.random.seed(0)
    sample_size = int(0.8 * len(labels))
    kept_true_labels = np.random.choice(range(len(labels)), sample_size, replace=False)
    semi_supervised_labels = [labels[i] if i in kept_true_labels else 0 for i in range(len(labels))]
    print(np.bincount(semi_supervised_labels))
    print(e.summarise_scores_semi_supervised_cross_val(all_scores_beat, np.array(labels),
                                                semi_labels=np.array(semi_supervised_labels)))
    # print(e.summarise_scores_supervised_cross_val(all_scores_beat, np.array(labels), semi_labels=np.array(labels)))
    # pc.all_plots(f"sample_{sample}", signal, scores_test, y_test, None, None, None, None, runs=n_runs, type=type)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=1000,
                        help='Number of iterations')
    parser.add_argument('--max_window_size', type=int, default=400,
                        help='Maximum window size')
    parser.add_argument('--parallelise', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to parallelise the iterations or not.')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of parallel workers.')
    parser.add_argument('--sample', type=int, default=100,
                        help='Patient number of MITBIH Dataset.')

    args = parser.parse_args()

    str_print = ""
    for par, arg in vars(args).items():
        str_print = str_print + " {}={} ".format(par, arg)
    print("Ran with parameters:", str_print)

    summarisation(sample=args.sample, n_runs=args.n_runs, max_window_size=args.max_window_size,
                   parallelise=args.parallelise, num_workers=args.num_workers)



