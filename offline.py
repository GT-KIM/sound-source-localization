"""

SRP-PHAT OFFLINE

"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from scipy.spatial import KDTree
import pickle
import math
from utils import *
# Microphone array location
# [pos_x, pos_y, pos_z]
# unit : cm

#microphone_array = np.array([[-0.153, 0, 0],[-0.038,0,0.04],[0.038,0,0.04],[0.148,0,0]])
#microphone_array = np.array([[-0.038,0,0.04],[0.038,0,0.04]])
microphone_array = np.array([[-0.170, 0, 0],[-0.04,0,0.04],[0.045,0,0.04],[0.160,0,0]])

# Generate circular Q array
nAngles = 19 # 0 ~ 180 deg
nDist = 5 # 0.1m ~ 1.0m
# Generate Q array
#Q_array = (np.mgrid[-1:1:0.05, 0:1:0.05])
#Q_array = np.reshape(Q_array, [2,-1])
#Q_array = np.concatenate((Q_array, np.zeros((1,Q_array.shape[1]))), axis= 0)
Q_array = np.zeros((3, int(nAngles * nDist)))
Q_polar = np.zeros((3, int(nAngles * nDist)))
for ang in range(nAngles) :
    for dist in range(nDist) :
        rad = math.radians(ang * 10) # rad
        r = dist * 0.25 + 0.1 # m
        x, y = polar2cart(r, rad)
        Q_array[0,ang * nDist + dist] = x
        Q_array[1,ang * nDist + dist] = y

        Q_polar[0,ang * nDist + dist] = r
        Q_polar[1,ang * nDist + dist] = rad


ax = plt.subplot(111, projection = 'polar' )
ax.scatter(Q_polar[1], Q_polar[0], cmap = 'hsv')
plt.show()

# TDOA
Fs = 16000
T = 24 # 실내 온도
c = 331.5+0.61*T # m/s
N = 1024 # STFT window size

num_of_microphone = 4
total_microphone = int(num_of_microphone * (num_of_microphone-1) /2)

# SRP-PHAT Coefficient W 계산
W = np.zeros((Q_array.shape[1], total_microphone, int(N/2+1)), dtype=np.complex_)
for iter_Q in range(Q_array.shape[1]) :
    iter_mic = 0
    for iter_mic1 in range(num_of_microphone) :
        for iter_mic2 in range(iter_mic1+1,num_of_microphone) :
                for iter_N in range(int(N / 2 + 1)) :
                    m1 = microphone_array[iter_mic1,:]
                    m2 = microphone_array[iter_mic2,:]
                    sq = Q_array[:,iter_Q]

                    tau = Fs/c*(np.linalg.norm(sq-m1, 2) - np.linalg.norm(sq-m2, 2))

                    W_k = np.exp(2*np.pi*1j*iter_N*tau/N)
                    y = iter_Q
                    W[y,iter_mic, iter_N] = W_k
                iter_mic += 1

print("DONE")
print(W.shape)
np.savez("E:\\Test/W_fastest.npz", Q=Q_array, Q_polar = Q_polar, W=W)