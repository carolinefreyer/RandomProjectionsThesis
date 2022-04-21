import numpy as np
import pandas as pd

import load_data_MITBIH as mb
import matplotlib.pyplot as plt
import ensemble_less_mem as e
import plot_curves as pc
import statsmodels.tsa.stattools as sm
import aggregation as agr


def plot_scores(name, data, scores, labels, guesses, to_add_values, runs, type):
    dim = len(data.columns)
    columns = list(data.columns)
    height_ratios = [3 for _ in range(dim)]
    height_ratios.append(1)
    fig, axs = plt.subplots(dim+1, 1, gridspec_kw={'height_ratios': height_ratios})
    axs[0].set_title(str(name).replace("_", " ").capitalize())
    axs[0].xaxis.set_visible(False)

    outliers = data.index[np.where(np.array(labels) > 0)].tolist()
    if guesses is not None:
        outliers = [i for i in outliers if i not in guesses]
        guesses_values = [[] for _ in range(dim)]

    outliers_values = [[] for _ in range(dim)]

    for d in range(dim):
        outliers_values[d] = data.loc[data.index.isin(outliers), columns[d]]
        axs[d].plot(data.index, data[columns[d]], 'k')

        if guesses is not None:
            guesses_values[d] = [item for sublist in to_add_values[d] for item in sublist]
            axs[d].scatter(guesses, guesses_values[d], color='yellow')

        axs[d].scatter(outliers, outliers_values[d], color='b', s=10, alpha=0.5)

    axs[dim].plot(range(len(scores)), scores)
    axs[dim].xaxis.set_visible(False)
    axs[dim].set_ylabel("Ensemble")

    plt.show()


# scores = np.load("./output_scores/MITBIH_sample_100_1000_1600.npy")

sampfrom = 0
sampto = None

record, annotation = mb.load_mit_bih_data("101", sampfrom, sampto)
signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_q_points_single(record, annotation, sampfrom, sampto)
timestamp = np.array([int(i) for i in range(len(signal_norm))])
signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)
# scores = scores[heart_beats_x[0][0]:heart_beats_x[-1][-1]]

# fig, ax = plt.subplots(2, 1)
# scores_rp = e.random_projection_window(signal.values, 2, False, 'mid', 2, 10)
# scores_auto_10 = e.random_projection_window_auto(signal.values, 2, False, 'mid', 2, 10)
# np.save(f"./output_scores/MITBIH_sample_100_autocorr_explore_10", scores_auto_10)
# print(list(pc.interval_extract(list(np.where(np.array(labels) > 0)[0]))))
# scores_final = np.load("./output_scores/MITBIH_sample_100_1000_final.npy")
# scores_auto = np.load("./output_scores/MITBIH_sample_100_1_autocorr_explore_200.npy")
# print(min(scores_rp), max(scores_rp), min(scores_final), max(scores_final), min(scores_auto), max(scores_auto))

# pc.plot_scores("sample_100",signal, scores, labels, None, None, 1, "autocorr_explore_200")
# pc.plot_scores("sample_100",signal[520000: 522000], scores_final[520000: 522000], labels[520000: 522000], None, None, 1, "autocorr_explore_final_3")
# pc.plot_scores("sample_100",signal[520000: 522000], scores_auto[520000: 522000], labels[520000: 522000], None, None, 1, "autocorr_explore_200_3")
# pc.plot_scores("sample_100",signal[:3000] , scores_auto_10[:3000], labels[:3000], None, None, 1, "RP_auto_10")
# beat_scores = []
# beat_labels = []
# for x in heart_beats_x:
#     beat_scores.append(max(scores[x[0]:x[-1]])-np.average(scores[x[0]:x[-1]]))
#     beat_labels.append(np.argmax(np.bincount(labels[x[0]:x[-1]])))
#
# for i,a in enumerate(beat_labels):
#     if a==1:
#         print(i)

all_scores = np.load("./output_scores/all_scores_MITBIH_sample_101.npy")
scores_test, y_test, flip = agr.summarise_scores_supervised(all_scores, labels)
print("done aggregation")
diff = len(np.where(scores_test - y_test != 0)[0])
print(diff, diff/len(y_test))
# pc.all_plots(f"sample_100", signal, scores_test, y_test, None, None, None, None, flip, runs=0, type="TEST")



