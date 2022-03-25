import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import load_data_NAB as nab
import numpy as np

########### Only run once ##########################################
# tar = tarfile.open("C:/Users/carol/OneDrive - Delft University of Technology/Desktop/Masters/Thesis/new-data/dataset.tgz", "r:gz")
# tar.extractall()
# tar.close()

data = pd.read_csv("./ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/A4Benchmark-TS1.csv")
data['value'] = data['value'].diff()
data = data.iloc[1: , :]

change_points = data.loc[data['changepoint'] == 1, 'value']
change_points_x = data.loc[data['changepoint'] == 1, 'timestamps']

outliers = data.loc[data['anomaly'] == 1, 'value']
outliers_x = data.loc[data['anomaly'] == 1, 'timestamps']

plt.plot(data['timestamps'], data['value'])
plt.scatter(outliers_x,outliers,color = 'r')
plt.title("A4 Yahoo Benchmark-TS1")
plt.show()

print(len(data), len(outliers), len(outliers)/len(data)*100)

roc_aucs = []
roc_auc_means = []
roc_auc_mean_ps = []

for i in range(0,200, 2):
    roc_auc, roc_auc_mean, roc_auc_mean_p = nab.test(np.array(data['value']).reshape(-1,1), "TS1", np.array(data['anomaly']), i, 1, False, True, i, "mid",True)
    roc_aucs.append(roc_auc)
    roc_auc_means.append(roc_auc_mean)
    roc_auc_mean_ps.append(roc_auc_mean_p)

print(roc_aucs)
print(roc_auc_means)
print(roc_auc_mean_ps)

print(max(roc_aucs), roc_aucs.index(max(roc_aucs)))
print(max(roc_auc_means), roc_auc_means.index(max(roc_auc_means)))
print(max(roc_auc_mean_ps), roc_auc_mean_ps.index(max(roc_auc_mean_ps)))

x = range(0,200, 2)
plt.plot(x, roc_aucs, label= "RP")
plt.plot(x, roc_auc_means, label= "Mean")
plt.plot(x, roc_auc_mean_ps, label= "Mean prime")
plt.legend()
plt.title("AUC for different window lengths")
plt.show()
