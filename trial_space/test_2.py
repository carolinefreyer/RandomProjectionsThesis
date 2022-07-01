from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from tqdm import tqdm
import numpy as np
from functools import partial
from numba import jit

def task(name, sign):
    print(f"{name} {sign} Executing our Task on Process: {os.getpid()}")
    return os.getpid()

def main():
    with ProcessPoolExecutor(max_workers=3) as executor:
        task1 = executor.submit(task, sign = "+", name="C")
        task2 = executor.submit(task, name="V", sign = "-")
        print(task1)
        print(task1.result(), task2.result())


def add_prefix(prefix, data, i):
    mean = np.mean(data)
    print("%s: %s%s" % (os.getpid(), prefix, mean))
    return mean, i

@jit(nopython=True)
def get_bin_sets(all_scores, indices_train, indices_test):
    scores_binary = np.full(all_scores.shape, 0.0)
    for i, scores in enumerate(all_scores):
        scores_binary[i] = np.where((scores > 1.96) | (scores < -1.96), 1.0, 0.0)
    train = scores_binary[:, indices_train.reshape(-1,)].reshape(scores_binary.shape[0], -1).astype('float64')
    test = scores_binary[:, indices_test.reshape(-1,)].reshape(scores_binary.shape[0], -1).astype('float64')
    return train, test


@jit(nopython=True)
def get_bin_sets_ranked(all_scores, indices_train, indices_test):
    scores_binary = np.array([0.0 for _ in range(2*len(all_scores[0]))])
    for i, scores in enumerate(all_scores):
        up = np.quantile(scores, 0.99)
        low = np.quantile(scores, 0.001)
        scores_binary[i] = np.where((scores > up), 1.0, 0.0)[0]
        scores_binary[i+len(all_scores[0])] = np.where((scores < low), 1.0, 0.0)[0]
    train = scores_binary[:, indices_train.reshape(-1,)].reshape(scores_binary.shape[0], -1).astype('float64')
    test = scores_binary[:, indices_test.reshape(-1,)].reshape(scores_binary.shape[0], -1).astype('float64')
    return train, test


if __name__ == "__main__":
    # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # m = 10
    # with ProcessPoolExecutor() as executor:
    #     for r in [executor.submit(add_prefix, "hi", data, i) for i in range(m)]:
    #         print(r.result()[0])
    scores = np.array([0,1,2,0.5,3])
    print(get_bin_sets(scores,np.array([[0],[1],[2]]), np.array([[3],[4]])))



