import numpy as np
import pandas as pd
import argparse
from numba import jit

import load_data_MITBIH as mb
import ensemble_less_mem as e


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


def summarisation(sample, max_window_size, n_runs, parallelise, num_workers):
    sampfrom = 0
    sampto = None

    record, annotation = mb.load_mit_bih_data(sample, sampfrom, sampto)
    signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_segments_q_points(record, annotation, sampfrom)
    timestamp = np.array([int(i) for i in range(len(signal_norm))])
    signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)
    e.summarise_data(heart_beats, labels, [])
    # all_scores = e.run(signal, max_window_size, n_runs, parallelise, num_workers)
    all_scores = np.load("./output_scores_MITBIH/scores_sample100_400_1000.npy")
    all_scores_beat = get_beat_score(all_scores, convert(heart_beats_x))
    print("got scores")
    scores_test, y_test = e.summarise_scores_supervised(all_scores_beat, labels)
    print(np.bincount(scores_test))
    diff = len(np.where(scores_test - y_test != 0)[0])
    print(diff, diff/len(labels))




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



