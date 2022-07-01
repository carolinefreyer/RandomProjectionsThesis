import matplotlib.pyplot as plt
import numpy as np
import ensemble_corrected as e
import pandas as pd
import load_data_NAB as nab

def get_signals(data):
    signal = np.array(data.values)
    signal_diff_right = data.diff().fillna(0).values
    signal_diff_right = np.array([np.abs(i) for i in signal_diff_right])
    signal_diff_left = data.diff(-1).fillna(0).values
    signal_diff_left = np.array([np.abs(i) for i in signal_diff_left])

    return signal, signal_diff_right, signal_diff_left

# x = np.linspace(0,7,1000)
# y = [np.sin(6*i)+np.cos(2*i) for i in x]
# y2 = [np.cos(3*i)+np.cos(4*i) + 2 for i in x]
# y3 = [np.sin(i)+np.sin(3*i) + 1 for i in x]
# y4 = [0.5*(np.cos(i)+np.sin(3*i))+4 for i in x]
# y[300] = 0
# y2[500] = 2
# y3[800] = 2
# for j in range(50):
#     y4[750+j] = 3
#     y2[25+j] = 1
# y[100] = 0
# y2[700] = 3
# y3[200] = 2
# y4[500] = 3
# arr = np.column_stack((y4, y, y3))
# data = pd.DataFrame(arr)
# signal, signal_diff_right, signal_diff_left = e.get_signals(data)
# for y in range(signal.shape[1]):
#     plt.plot(x, signal_diff_left[:,y])
# plt.show()
labels = np.array([0,0,1,1,1, 0, 0,0,0,0,0,0,0,0,0,0,1,0, 1])
all_scores = np.array([[0.1, 0.05, 0.8, 0.03, 0.01], [0.1, 0.05, 0.8, 0.03, 0.01], [0.1, 0.05, 0.8, 0.03, 0.01]])
import sklearn.model_selection as sk

# kf = sk.StratifiedKFold(n_splits=5)
# for train, test in kf.split(np.array([i for i in range(len(labels))]).reshape(-1, 1), labels):
#     y_train = labels[train]
#     y_test = labels[test]
#     e.get_bin_sets(all_scores, train, test)
#     print(train)
#     print(test)
# import numpy as np
# np.random.seed(0)
# print(np.random.choice(len(labels), int(0.2*len(labels))))
import scipy.stats as ss
# x = [0.973, 0.959, 0.955, 0.972, 0.967, 0.957, 0.952, 0.965, 0.959, 0.961, 0.958, 0.957, 0.96,0.959, 0.962]
# y = [0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972, 0.972,0.972, 0.972]
# print(ss.mannwhitneyu(x,y, alternative = "less"))
#
# x =  [0.955,0.965,0.929,0.966,0.975, 0.955,0.965,0.929,0.966,0.975, 0.955,0.965,0.929,0.966,0.975, 0.955,0.965,0.929,0.966,0.975]
# y = [0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642]
# print(ss.mannwhitneyu(x,y, alternative = "less"))
# x = [0.776, 0.749, 0.564, 0.772, 0.617, 0.723]
# y = [0.874, 0.874, 0.874, 0.874, 0.874, 0.874]
# print(ss.mannwhitneyu(x,y, alternative = "less"))
# x = [0.809, 0.74, 0.696, 0.739, 0.794]
# y = [0.395, 0.395, 0.395, 0.395, 0.395]
# print(ss.mannwhitneyu(x,y, alternative = "less"))

x = [0.996,
0.994,
0.972,
0.991,
]
y = [0.959, 0.959, 0.959, 0.959]
print(ss.mannwhitneyu(x,y, alternative = "less"))

x = [0.208,
0.185,
0.217,
0.147]
print(ss.mannwhitneyu(x,y, alternative = "less"))
y = [0.512, 0.512, 0.512, 0.512]

x = [0.92,
0.95,
0.937,
0.885,
0.907]

y = [0.874, 0.874, 0.874, 0.874, 0.874]
print(ss.mannwhitneyu(x,y, alternative = "less"))

x= [0.841,
0.914,
0.982,
0.678,
0.729
]

y = [0.712, 0.712, 0.712, 0.712,0.712]
print(ss.mannwhitneyu(x,y, alternative = "less"))

