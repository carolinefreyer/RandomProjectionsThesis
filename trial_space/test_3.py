import numpy as np
from numba import jit
import os
import pandas as pd
import sys
import argparse
sys.path.append(os.path.abspath('../'))
import ensemble_less_mem as e
import load_data_NAB as nab
import matplotlib.pyplot as plt

name = "ambient_temperature_system_failure"
data, labels = nab.load_data(f"realKnownCause/{name}.csv", False)
data, labels, to_add_times, to_add_values = nab.clean_data(data, labels, name=name, unit=[0, 1], plot=False)

guesses = [item for sublist in to_add_times for item in sublist[1:]]

data = data.reset_index().set_index('timestamp')
data = data.drop('index', axis=1)
data.index = pd.to_datetime(data.index)

signal, signal_diff_right, signal_diff_left, signal_auto = e.get_signals(data)

scores_o = e.standardise_scores_z_score(e.random_projection_window(signal, 1, False, 'mid', 2, 50))
scores_d_r = e.standardise_scores_z_score(e.random_projection_window(signal_diff_right, 1, False, 'mid', 2, 50))
plt.hist(scores_o, bins='auto')
plt.title("Density of z-scores of original signal")
plt.show()
plt.clf()
plt.hist(scores_d_r, bins='auto')
plt.title("Density of z-scores of differenced signal (right)")
plt.show()