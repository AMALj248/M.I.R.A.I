from datetime import timedelta
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM , Dense , Dropout

# This is where we define the model

# Vanilla Long Sort Term Memory Model
def lstm_model(x_val, y_val, epochs_num, look_back):

    # Sequential LSTM Model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_val, y_val, epochs=epochs_num, batch_size=1, verbose=2)

    return model
