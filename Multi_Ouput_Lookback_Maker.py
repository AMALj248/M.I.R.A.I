import pandas as pd
import numpy as np
import math


def look_back_multi(data, n_steps_in, n_steps_out) :
    data_X, data_Y = [] , []
    #
    print("IN\n", data.shape)
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(data):
            break

        # gather input and output parts of the pattern
        seq_x = data.iloc[i:end_ix, :-1]
        data_X.append(seq_x)
        seq_y = data.iloc[end_ix:out_end_ix, -1]
        data_Y.append(seq_y)
    return  np.array([np.array(xi) for xi in data_X ]), np.array(data_Y)

