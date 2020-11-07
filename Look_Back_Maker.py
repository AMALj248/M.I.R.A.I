import pandas as pd
import numpy as np
import math
from keras.preprocessing.sequence import TimeseriesGenerator
def look_back(data, look_back_window) :
    data_X, data_Y = [] , []
    # selecting from i to i+look_back with 1 left
    print("IN\n", data.shape)
    for i in range(len(data)-look_back_window):
        a = data.iloc[i:(i+look_back_window), :-1]
        data_X.append(a)
        b=data.iloc[i+look_back_window,-1]
        data_Y.append(b)
    print("OUT X Shape\n", data_X)
    print("OUT Y Shape\n", data_Y)


    return np.array([np.array(xi) for xi in data_X]), np.array(data_Y)
