import random
import numpy as np
import statsmodels.tsa.stattools as sm
import pandas as pd
import matplotlib.pyplot as plt
import numba

import sys
import os
sys.path.append(os.path.abspath('../'))
import ensemble_less_mem as e
import load_data_NAB as nab
import load_data_MITBIH as mb
from numba import jit


# name = "ambient_temperature_system_failure"
# data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
# data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)
#
# guesses = [item for sublist in to_add_times for item in sublist[1:]]
#
# data = data.reset_index().set_index('timestamp')
# data = data.drop('index', axis=1)
# data.index = pd.to_datetime(data.index)
#
# signal, signal_diff_right, signal_diff_left, signal_auto= e.get_signals(data)
# print(signal.shape, signal_auto.shape)
#
import time
sampfrom = 0
sampto = 6000

# record, annotation = mb.load_mit_bih_data("100", sampfrom, sampto)
# signal_norm, heart_beats, heart_beats_x, labels = mb.label_clean_q_points_single(record, annotation, sampfrom, sampto)
# timestamp = np.array([int(i) for i in range(len(signal_norm))])
# signal = pd.DataFrame(signal_norm, columns=record.sig_name, index=timestamp)


