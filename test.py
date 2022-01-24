import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

d_values = [260, 400, 720]
k_values = [1,2,3,4,5]

for d in d_values:
    for k in k_values:
        print(d, k)
        R = np.random.normal(loc=0.0, scale=1.0, size=(k, d))
        print((1 / np.sqrt(d) * R) @ ((1/np.sqrt(d)) * R.T))

# x = np.linspace(0,10,100)
# y1 = [np.sin(i) for i in x]
# y2 = [0.5*np.cos(i) for i in x]
# y3 = [0.05*np.sin(2*i+1)+0.2 for i in x]
# y4 = [0.5*np.sin(i*np.pi) for i in x]
# y5 = [0.3*np.cos(i*3/4*np.pi) for i in x]
# y6 = [0.1*np.sin(4*i+10)-0.1 for i in x]
#
# # plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)
# # plt.plot(x,y4)
# plt.plot(x,y5)
# plt.plot(x,y6)
# plt.title("Basic example multivarite time series")
# plt.show()
#
#
