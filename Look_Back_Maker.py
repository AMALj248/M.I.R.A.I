import pandas as pd
import numpy as np
import math
from keras.preprocessing.sequence import TimeseriesGenerator
def look_back(data, look_back_window) :
    data_X, data_Y = [] , []
    # selecting from i to i+look_back with 1 left
    print("IN\n", data.shape)
    for i in range(data.shape[0]-look_back_window-1):
        a = data.iloc[i:(i+look_back_window), :-1]
        data_X.append(a)
        b=data.iloc[i+look_back_window, -1]
        data_Y.append(b)
    print("OUT X Shape\n", len(data_X))
    print("OUT Y Shape\n", len(data_Y))
    print("Missing values In Look Back \n", np.isnan(data_Y).sum())

    # There are missing values because of shift like operation
    # We should drop those values

    return np.array([np.array(xi) for xi in data_X ]), np.array(data_Y)

