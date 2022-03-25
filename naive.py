import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import scipy as sc

import ensemble as e
import load_data_NAB as nab



def get_random_cov(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())


def initialize_random_params(n):
    params = {'phi': np.random.uniform(0, 1),
              'mu0': np.random.normal(0, 1, size=(n,)),
              'mu1': np.random.normal(0, 1, size=(n,)),
              'sigma0': get_random_cov(n),
              'sigma1': get_random_cov(n)}
    return params


def e_step(x, params):
    np.log([sc.stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x),
            sc.stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)])
    log_p_y_x = np.log([1-params["phi"], params["phi"]])[np.newaxis, ...] + \
                np.log([sc.stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x),
            sc.stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)]).T

    log_p_y_x_norm = sc.special.logsumexp(log_p_y_x, axis=1)
    return log_p_y_x_norm, np.exp(log_p_y_x - log_p_y_x_norm[..., np.newaxis])


def m_step(x, params):
    total_count = x.shape[0]
    _, heuristics = e_step(x, params)
    heuristic0 = heuristics[:, 0]
    heuristic1 = heuristics[:, 1]
    sum_heuristic1 = np.sum(heuristic1)
    sum_heuristic0 = np.sum(heuristic0)
    phi = (sum_heuristic1/total_count)
    mu0 = (heuristic0[..., np.newaxis].T.dot(x)/sum_heuristic0).flatten()
    mu1 = (heuristic1[..., np.newaxis].T.dot(x)/sum_heuristic1).flatten()
    diff0 = x - mu0
    sigma0 = diff0.T.dot(diff0 * heuristic0[..., np.newaxis]) / sum_heuristic0
    diff1 = x - mu1
    sigma1 = diff1.T.dot(diff1 * heuristic1[..., np.newaxis]) / sum_heuristic1
    params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    return params


def get_avg_log_likelihood(x, params):
    loglikelihood, _ = e_step(x, params)
    return np.mean(loglikelihood)


def run_em(x, params):
    avg_loglikelihoods = []
    while True:
        avg_loglikelihood = get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break
        params = m_step(x, params)
    print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
               % (params['phi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']))
    _, posterior = e_step(x, params)
    forecasts = np.argmax(posterior, axis=1)
    return forecasts


def navie_detector(data, labels, win_pos = "mid", win_length = 10):
    signal = e.normalise(np.array(data['value']).reshape(-1, 1))
    expanded_data = e.make_windows(signal, win_pos, win_length)
    trim = [i for i in range(len(expanded_data)) if len(expanded_data[i]) == win_length]
    expanded_data_trimmed = np.asarray([expanded_data[i] for i in range(len(expanded_data)) if i in trim])
    expanded_data_trimmed = expanded_data_trimmed.reshape(-1, win_length)
    labels_trimmed = [labels[i] for i in range(len(expanded_data)) if i in trim]

    random_params = initialize_random_params(expanded_data_trimmed.shape[1])
    scores = run_em(expanded_data_trimmed, random_params)

    return scores, labels_trimmed


def naive_bayes_single(data, labels):
    signal = e.normalise(np.array(data['value']).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(signal, labels, test_size=0.2, random_state=0, stratify=labels)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    plot_naive(y_pred, y_test)
    print(f"Number of mislabeled points out of a total {X_test.shape[0]} points :{(y_test != y_pred).sum()}")
    print(f"Error :{(y_test != y_pred).sum()/X_test.shape[0]*100} %")


def naive_bayes_window(data, labels, win_pos, win_length):
    signal = e.normalise(np.array(data['value']).reshape(-1, 1))
    expanded_data = e.make_windows(signal, win_pos, win_length)
    trim = [i for i in range(len(expanded_data)) if len(expanded_data[i]) == win_length]
    expanded_data_trimmed = np.asarray([expanded_data[i] for i in range(len(expanded_data)) if i in trim])
    expanded_data_trimmed = expanded_data_trimmed.reshape(-1, win_length)
    labels_trimmed = [labels[i] for i in range(len(expanded_data)) if i in trim]
    pca = PCA(n_components = 0.99)
    pca_expanded_data = pca.fit_transform(expanded_data_trimmed)
    X_train, X_test, y_train, y_test = train_test_split(pca_expanded_data, labels_trimmed, test_size=0.2, random_state=0, stratify=labels_trimmed)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print(f"Number of mislabeled points out of a total {X_test.shape[0]} points :{(y_test != y_pred).sum()}")
    print(f"Error :{(y_test != y_pred).sum() / X_test.shape[0] * 100} %")

    plot_naive(y_pred, y_test)


def plot_naive(scores, labels):
    outliers = []
    normal = []

    for i, c in enumerate(scores):
        if labels[i] == 1:
            outliers.append(c)
        else:
            normal.append(c)

    fig, ax = plt.subplots()
    ax.bar([0, 1], [normal.count(0), normal.count(1)], 0.1, label= "Normal")
    ax.bar([0, 1], [outliers.count(0), outliers.count(1)], 0.1, label="Outliers", bottom=[normal.count(0), normal.count(1)])
    ax.legend()
    ax.set_ylabel('Frequency')
    ax.set_title('Naive Bayes Prediction')
    plt.show()

def plot_naive_guesses(scores, labels, guesses_index, runs):
    outliers = []
    normal = []
    guesses = []

    for i, c in enumerate(scores):
        if i in guesses_index:
            guesses.append(c)
        elif labels[i] == 1:
            outliers.append(c)
        else:
            normal.append(c)

    fig, ax = plt.subplots()
    ax.bar([0, 1], [normal.count(0), normal.count(1)], 0.1, label= "Normal")
    ax.bar([0, 1], [guesses.count(0), guesses.count(1)], 0.1, label="Guesses",
           bottom=[normal.count(0), normal.count(1)])
    ax.bar([0, 1], [outliers.count(0)*50, outliers.count(1)*50], 0.1, label="Outliers",
           bottom=[normal.count(0) + guesses.count(0), normal.count(1) + guesses.count(1)])
    ax.legend()
    ax.set_ylabel('Frequency')
    ax.set_title(f'EM Algorithm Prediction for {runs} runs')
    plt.show()

if __name__ == '__main__':
    name = "ambient_temperature_system_failure"
    data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
    data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
    naive_bayes_window(data, labels, "mid", 20)