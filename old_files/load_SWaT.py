import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from old_files import ensemble as e
import plot_curves as pc


def normalise(signal):
    # centering and scaling happens independently on each signal
    scaler = StandardScaler()
    return scaler.fit_transform(signal)


def read_data():
    #Read data
    data_train = "./data/Swat_data/SWaT_train.csv"
    df_train = pd.read_csv(data_train)

    # data_test = "./data/Swat_data/SWaT_test.csv"
    # df_test = pd.read_csv(data_test)

    #Clean column labels
    df_train.columns = [x.replace(' ', '').lower() for x in df_train.columns]
    # df_test.columns = [x.replace(' ', '').lower() for x in df_test.columns]

    # df_test = df_test.drop("unnamed:0", axis = 1)
    # df = df_train.append(df_test)
    df_train.loc[df_train['normal/attack'] == 'Normal', 'normal/attack'] = 0
    df_train.loc[df_train['normal/attack'] == 'Attack', 'normal/attack'] = 1
    labels = np.array(df_train['normal/attack'])
    df_train = df_train.drop('normal/attack', axis=1)

    return df_train, labels

def clean_data(df, labels):
    type = [1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0]
    df_no_burnin = df[18000:58000]
    labels_no_burnin = labels[18000:58000]
    normalised = normalise(df_no_burnin.drop('timestamp', axis=1).values)
    signal = pd.DataFrame(normalised, columns=df.columns[1:], index=np.array(df_no_burnin['timestamp']))
    signal.index = pd.to_datetime(signal.index)
    # signal_resampled= signal.resample('10S').max()
    data = signal[signal.columns[5:8]]
    e.summarise_data(data, labels_no_burnin, [])
    scores = e.run(data, 2000, 100, True, 6)

    pc.plot_scores("SWaT_P2", data, scores, labels_no_burnin, None, None, 100, "testing_sensors")

    return


if __name__ == '__main__':
    data, labels = read_data()
    clean_data(data, labels)