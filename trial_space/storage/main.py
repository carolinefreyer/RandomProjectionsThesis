import random_projection_MITBIH as rp
import load_data_MITBIH as load
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_auc_sd(auc_average, aucs_sd, name, type, color):
    x = range(100, 350, 10)
    plt.plot(x, auc_average, color=color)
    plt.fill_between(x, auc_average - aucs_sd, auc_average + aucs_sd, alpha=0.2, color=color)
    plt.title(f"Average AUC values for all patients for different window lengths: {type} {name} summarisation")
    plt.xlabel("Window length")
    plt.savefig(f"./AUC_averager_{type}_{name}", bbox_inches='tight')
    plt.clf()

def average_auc_all_samples(samples, sampfrom, sampto):

    aucs_rp_max = [[] for _ in samples]
    aucs_rp_diff = [[] for _ in samples]
    aucs_mean_max = [[] for _ in samples]
    aucs_mean_diff = [[] for _ in samples]

    for i, sample in enumerate(samples):
        if sample not in skip:
            record, annotation = load.load_mit_bih_data(sample, sampfrom, sampto)
            aucs_rp_max[i], aucs_rp_diff[i], aucs_mean_max[i], aucs_mean_diff[i] = rp.auc_for_different_window_length(
                record, annotation, sampfrom, k=1, pres_norm=False, mean_comparison=True)

    aucs_rp_max_avg = np.mean(aucs_rp_max, axis=0)
    aucs_rp_max_sd = np.std(aucs_rp_max, axis=0)

    aucs_rp_diff_avg = np.mean(aucs_rp_diff, axis=0)
    aucs_rp_diff_sd = np.std(aucs_rp_diff, axis=0)

    aucs_mean_max_avg = np.mean(aucs_mean_max, axis=0)
    aucs_mean_max_sd = np.std(aucs_mean_max, axis=0)

    aucs_mean_diff_avg = np.mean(aucs_mean_diff, axis=0)
    aucs_mean_diff_sd = np.std(aucs_mean_diff, axis=0)

    x = range(10, 350, 10)
    plt.plot(x, aucs_rp_diff_avg, label ="Random projection difference")
    plt.plot(x, aucs_rp_max_avg, label="Random projection maximum")
    plt.plot(x, aucs_mean_diff_avg, label="Mean difference")
    plt.plot(x, aucs_mean_max_avg, label="Mean maximum")
    plt.title(f"Average AUC values for all patients for different window lengths")
    plt.xlabel("Window length")
    plt.legend()
    plt.savefig(f"./AUC_average", bbox_inches='tight')
    plt.clf()
    # plot_auc_sd(aucs_rp_max_avg, aucs_rp_max_sd, "maximum", "random projection", 'orange')
    # plot_auc_sd(aucs_rp_diff_avg, aucs_rp_diff_sd, "difference", "random projection", 'b')
    # plot_auc_sd(aucs_mean_max_avg, aucs_mean_max_sd, "maximum", "mean", 'r')
    # plot_auc_sd(aucs_mean_diff_avg, aucs_mean_diff_sd, "difference", "mean", 'g')

if __name__ == '__main__':

    sampfrom = 0
    sampto = None

    rp_maxs = []
    rp_diffs = []
    mean_maxs = []
    mean_diffs = []
    mean_maxs_old = []
    mean_diffs_old = []

    samples = []
    skip = [102, 104, 107, 108, 109, 111, 118, 124, 203, 207, 212, 214, 217, 231, 232]
    for root, dirs, files in os.walk("../../data/"):
        for dir in dirs:
            if int(dir[-3:]) not in skip:
                record, annotation = load.load_mit_bih_data(dir[-3:], sampfrom, sampto)
                auc = rp.run(record, annotation, sampfrom, False)
                rp_maxs.append(auc)
                # rp_diffs.append(rp_diff)
                # mean_maxs.append(mean_max)
                # mean_diffs.append(mean_diff)
                # mean_maxs_old.append(mean_max_old)
                # mean_diffs_old.append(mean_diff_old)
                # print(" ")
    print(round(np.mean(rp_maxs),3), round(np.std(rp_maxs),3))
    print(rp_maxs)
    rp_maxs.pop(-3)
    print(round(np.mean(rp_maxs), 3), round(np.std(rp_maxs), 3))
    # print(round(np.mean(rp_diffs),3), round(np.std(rp_diffs),3))
    # print(round(np.mean(mean_maxs),3), round(np.std(mean_maxs),3))
    # print(round(np.mean(mean_diffs),3), round(np.std(mean_diffs),3))
    # print(round(np.mean(mean_maxs_old),3), round(np.std(mean_maxs_old),3))
    # print(round(np.mean(mean_diffs_old),3), round(np.std(mean_diffs_old),3))

    # record, annotation = load.load_mit_bih_data('100', sampfrom, sampto)
    # print(rp.run(record, annotation, sampfrom, True))
    # average_auc_all_samples(samples, sampfrom, sampto)
    # record, annotation = load.load_mit_bih_data('230', sampfrom, sampto)
    # rp.auc_for_different_window_length(record, annotation, sampfrom, k=1, pres_norm=False, mean_comparison=True)
