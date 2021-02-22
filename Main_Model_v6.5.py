from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, LeakyReLU
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ReduceLROnPlateau

# gelu Activation Function
pi = tf.convert_to_tensor(math.pi)
const_gelu =  tf.convert_to_tensor(0.044715)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001, verbose=1)


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf



#  Long Sort Term Memory Model
def lstm_model(x_val, y_val, epochs_num, look_back, num_features, n_steps_out, trail_num, cnrtl_val):

    print("Model V6.5 Running ")
    print("Model Input\n", x_val)
    print("Model Expected Output\n", y_val)
    print('Received Input Shape ->', x_val.shape[1])
    print('Control Value ', cnrtl_val)
    print('Steps_per_Epoch', int(x_val.shape[0]/40))

    # Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


    # Optimizer
    # opt = keras.optimizers.Nadam(learning_rate=0.004)
    # Sequential LSTM Model
    model = Sequential()


    model.add(TimeDistributed(Conv1D(filters=cnrtl_val*1, kernel_size=1, activation=gelu), input_shape=(x_val.shape[1], look_back, n_steps_out)))
    if cnrtl_val > 2:
        model.add((Dropout(0.2)))
        model.add(TimeDistributed(Conv1D(filters=cnrtl_val*2, kernel_size=1, activation=gelu, padding='same')))

    if cnrtl_val > 3:
        model.add((Dropout(0.2)))
        model.add(TimeDistributed(Conv1D(filters=cnrtl_val*3, kernel_size=1, activation=gelu,  padding='same')))

    if cnrtl_val > 4:
        model.add((Dropout(0.2)))
        model.add(TimeDistributed(Conv1D(filters=cnrtl_val*4, kernel_size=1, activation=gelu,  padding='same')))


    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True, activation=gelu ))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True, activation=gelu))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True, activation=gelu))
    model.add(Dropout(0.2))
    model.add(LSTM(70, activation=gelu))
    model.add(Dropout(0.2))
    model.add(Dense(70, activation=gelu))
    model.add(Dense(n_steps_out, activation='linear'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='Nadam', metrics=['accuracy'])
    model.fit(x_val, y_val, epochs=epochs_num, steps_per_epoch=int(x_val.shape[0]/40), batch_size=40, callbacks=[callback, reduce_lr], verbose=1)

    # Model Summary
    print(model.summary())
    # Saving the model
    model.save(str(trail_num) + '_model_.h5')
    print('Using Version V6.0')

    return model