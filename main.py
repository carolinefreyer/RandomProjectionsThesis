import random_projection_MITBIH as rp
import load_data_MITBIH as load
import numpy as np
import os
import matplotlib.pyplot as plt

def average_auc_all_samples(samples, sampfrom, sampto):

    aucs_rp_max = [[] for _ in samples]
    aucs_rp_diff = [[] for _ in samples]
    aucs_mean_max = [[] for _ in samples]
    aucs_mean_diff = [[] for _ in samples]

    for i, sample in enumerate(samples):
        if sample not in skip:
            record, annotation = load.load_mit_bih_data(sample, sampfrom, sampto)
            aucs_rp_max[i], aucs_rp_diff[i], aucs_mean_max[i], aucs_mean_diff[i] = rp.auc_for_different_window_length(
                record, annotation, sampfrom, k=1, pres_norm=False)

    aucs_rp_max_avg = np.mean(aucs_rp_max, axis=0)
    aucs_rp_diff_avg = np.mean(aucs_rp_diff, axis=0)
    aucs_mean_max_avg = np.mean(aucs_mean_max, axis=0)
    aucs_mean_diff_avg = np.mean(aucs_mean_diff, axis=0)

    plt.plot(range(150, 450, 10), aucs_rp_diff_avg, label="Random projection AUC difference")
    plt.plot(range(150, 450, 10), aucs_rp_max_avg, label="Random projection AUC maximum")
    plt.plot(range(150, 450, 10), aucs_mean_diff_avg, label="Mean AUC difference")
    plt.plot(range(150, 450, 10), aucs_mean_max_avg, label="Mean AUC maximum")
    plt.title("Average AUC values for all patients for different window lengths")
    plt.xlabel("Window length")
    plt.legend()
    plt.savefig(f"./AUC_averager", bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    sampfrom = 0
    sampto = None
    samples = []
    skip = [102, 104, 107, 109, 111, 118,124,207, 212, 214, 217, 231, 232]
    for root, dirs, files in os.walk("./data/"):
        for dir in dirs:
            samples.append(dir[-3:])

    record, annotation = load.load_mit_bih_data(samples[0], sampfrom, sampto)
    rp.run(record, annotation, sampfrom, False)

    # average_auc_all_samples(samples, sampfrom, sampto)

