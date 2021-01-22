import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D


# This is where we define the model

# Vanilla Long Sort Term Memory Model
def lstm_model(x_val, y_val, epochs_num, look_back, num_features):
    print("Model Input\n", x_val)
    print("Model Expected Output\n", y_val)
    # Sequential LSTM Model
    model = Sequential()
    model.add(LSTM(120, return_sequences=True, input_shape=(look_back, num_features)))
    model.add(LSTM(120, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(240, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics =['accuracy'] )
    model.fit(x_val, y_val, epochs=5, batch_size=15, verbose=2)
    print(model.summary())

    return model