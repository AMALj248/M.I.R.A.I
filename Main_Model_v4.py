from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, LeakyReLU
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


#  Long Sort Term Memory Model
def lstm_model(x_val, y_val, epochs_num, look_back, num_features, n_steps_out, trail_num):
    print("Model V5 Running ")
    print("Model Input\n", x_val)
    print("Model Expected Output\n", y_val)

    # Optimizer
    # opt = keras.optimizers.Nadam(learning_rate=0.004)
    # Sequential LSTM Model
    model = Sequential()
    # Dimension 3,3,2 is harcoded
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(3, 3, 2)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(90, return_sequences=True, activation='relu'))
    model.add(LSTM(120, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(240, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(70, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='softmax'))
    model.add(Dense(n_steps_out))
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'] )
    model.fit(x_val, y_val, epochs=epochs_num, batch_size=40, verbose=2)
    print(model.summary())
    # Saving the model
    model.save(str(trail_num)+'_model_.h5')
    print('Using Version V5.0')

    return model

