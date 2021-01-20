import yfinance as yahoo_data
import finnhub
import pandas as pd
import datetime
from datetime import timezone, datetime
import plotly.graph_objects as go
import math
from datetime import timedelta
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from Look_Back_Maker import *
from Main_Model import *

# Fixing the Seed Vale for Reproducibility
from numpy.random import seed
seed(1)
tf.random.set_seed(1)

# Setting the Initial and Final Date
start_date = datetime(2011, 10, 19)
end_date = datetime.today()

start_date = start_date.replace(tzinfo=timezone.utc).timestamp()
end_date = end_date.replace(tzinfo=timezone.utc).timestamp()
start_date = int(start_date)
end_date = int(end_date)

# Importing data from finnhub API

# Setup Client
fin_client = finnhub.Client(api_key='btthr6748v6or4rae7ag')

# Setting Company Symbol
cmpny = 'AAPL'

# Stock candles
res = fin_client.stock_candles(cmpny, 'D', start_date, end_date)
print(res)

# Getting The Company Profile
# prf = fin_client.company_profile2(symbol=cmpny)
# print('Profile\n', prf)
# norm_nm = prf['name']
# print(norm_nm)

# Converting Unix Timestamp to Normal Date
x_time = []
# For 1 to 30-31 month cycles
x_month = []
for i in res['t']:
    x = datetime.fromtimestamp(i).strftime('%Y-%m-%d')
    x = datetime.strptime(x, '%Y-%m-%d')
    x_month.append(x.day)
    x_time.append(x)
# Setting the Start and End Date
start_date_normal = min(x_time)
end_date_normal = max(x_time)

# Coverting from list to dictionary for Stock Data
stk_df = pd.DataFrame(res)

print("Null Value Check \n", stk_df.isna().sum())
print("DataFrame = \n", stk_df.head(25))


# Plotting function
def plots(data):
    # Plotting the candlestick
    fig = go.Figure(data=[go.Candlestick(x=x_time,
                                         open=data['o'], high=data['h'],
                                         low=data['l'], close=data['c'], name=(cmpny + str(' Prices')))])

    # Plotting the EMA & SMA
    fig.add_trace(go.Scatter(x=stk_df['time'], y=stk_df['sma_close'], text='SMA_C', name=('SMA_C' + str(120))))
    fig.add_trace(go.Scatter(x=stk_df['time'], y=stk_df['ema_open'], text='EMA_O', name=('EMA_O' + str(30))))
    fig.add_trace(go.Scatter(x=stk_df['time'], y=stk_df['ema_close'], text='EMA_C', name=('EMA_C' + str(15))))
    fig.update_layout(title_text='Moving Averages')
    fig.update_layout(title_text=(str(cmpny)))
    #fig.show()

    # Plotting the volume of Trade over time
    fig = go.Figure(go.Scatter(x=x_time, y=stk_df['v'], text='Volume'))
    fig.update_layout(title_text=('Volume of Trades for ' + str(cmpny)))
    #fig.show()


# Making a Moving Average Function

def mov_avg(sma_win_c, sma_win_o, ema_win_c, ema_win_o, data):
    ################################ SMA ################################

    # Creating a Simple Moving Averages for Opening
    open_val = data['o']
    windows1 = open_val.rolling(sma_win_o)
    mv_avg1 = windows1.mean()

    # Converting to list
    sma_lst1 = mv_avg1.tolist()
    sma_data1 = sma_lst1[:]
    sma_val1 = pd.DataFrame(sma_data1)
    print("SMA Opening \n", sma_val1)

    # Creating a Simple Moving Averages for Closing
    close_val = data['c']
    windows = close_val.rolling(sma_win_c)
    mv_avg = windows.mean()

    # Converting to list
    sma_lst = mv_avg.tolist()
    sma_data = sma_lst[:]
    sma_val = pd.DataFrame(sma_data)
    print("SMA Closing \n", sma_val)

    ####################################### EMA #######################################

    # Creating the EMA 1
    exp = open_val.ewm(span=ema_win_o).mean()
    exp_val = pd.DataFrame(exp)
    print("EMA Opening" + str(ema_win_o), exp_val)
    print("EMA SIZE \n", exp_val.shape)

    # Creating the EMA 2
    exp2 = close_val.ewm(span=ema_win_c).mean()
    exp_val2 = pd.DataFrame(exp2)
    print(("EMA Closing" + str(ema_win_c)), exp_val2)
    print("EMA SIZE \n", exp_val2.shape)

    print(len(sma_val), len(exp_val), len(exp_val2), len(sma_val1))

    # Returning The values
    return sma_val1, sma_val, exp_val, exp_val2


# Calling the Moving Function

sml_o, sma_c, exp_o, exp_c = mov_avg(120, 120, 15, 60, stk_df)

# Appending the values to DataFrame Columns
stk_df['sma_open'] = sml_o
stk_df['sma_close'] = sma_c
stk_df['ema_open'] = exp_o
stk_df['ema_close'] = exp_c
stk_df['time'] = x_month

# Replacing Starting SMA with 0
stk_df['sma_open'] = stk_df['sma_open'].fillna(0)
stk_df['sma_close'] = stk_df['sma_close'].fillna(0)

# Calling the Plots Function
plots(stk_df)

print(stk_df)


# Making a function to classify Up & Downs per week
def weekly_classifier(data):
    print("Start Date", start_date_normal)
    print("End Date", end_date_normal)
    cmp_data = data
    # Calculating the number of Weeks to process
    total_weeks = end_date_normal - start_date_normal
    print("Total Weeks\n", (total_weeks.days / 7))

    # Closing values
    cls_values = cmp_data['c']

    # Making EMA7
    ema7 = cls_values.ewm(span=7).mean()
    ema7_val = pd.DataFrame(ema7)
    # print("EMA" + str(7), ema7_val)
    # print("EMA SIZE \n", ema7_val.shape)
    # print("EMA SIZE \n", ema7_val.shape)
    cmp_data['EMA7'] = ema7_val

    print("Data Inside Chk_Val function\n", data.head(20))

    # Logic
    chk_lst = []
    for row in data.itertuples():
        # Bullish Market
        if (row.c >= row.o):
            chk_lst.append(1)
        # Bearish Market
        if (row.c < row.o):
            chk_lst.append(0)

    # cmp_data['Chk_Val'] = chk_lst

    print("Comparison Data\n", cmp_data.tail(25))
    print("Frequency of -1", chk_lst.count(0))
    print("Frequency of +1", chk_lst.count(1))


# Calling the weekly_classifier function


weekly_classifier(stk_df)


# Function to plot the Predictions
def plot_2(pred, real):
    # Plotting the Fit Curve
    tmp_lst = []
    for i in range(0, len(pred)):
        tmp_lst.append(i)
    plt.plot(tmp_lst, pred, color='black', label='Prediction')
    plt.plot(tmp_lst, real, color='red', label='Actual')
    plt.title(f'{cmpny} Prediction V/S Actual')
    plt.legend()
    plt.show()

# Model Input Data Transformation function

def Input_to_Model(data):

    # LookBack Window
    look_back_size = 5
    # Dropping the useless values form Data
    print(data.columns)
    mod_data = data._drop_axis(['sma_close', 'ema_close', 's', 't', 'EMA7'], axis=1)
    print("mod_data null values = \n", mod_data.isnull().sum())
    print(type(mod_data))
    print("mod data", mod_data)

    # Scaling the data for every column except target
    Scaler = MinMaxScaler()
    Scaler_Close = MinMaxScaler()
    close_val = mod_data['c']
    close_val = close_val.values.reshape(-1,1)
    mod_data['c'] = Scaler_Close.fit_transform(close_val)
    mod_data[['time', 'o', 'v', 'sma_open', 'ema_open' ,'l', 'h']] = Scaler.fit_transform(
        mod_data[['time', 'o', 'v', 'sma_open', 'ema_open', 'l', 'h']])

    print("mod data\n", mod_data)

    # Reorganizing the Column Values
    mod_data = mod_data.reindex(columns=['o', 'v', 'sma_open', 'ema_open', 'time','l', 'h', 'c'])

    # Train/Test Split
    train_size = int(len(mod_data) * 0.85)
    print(len(mod_data))
    train = mod_data.iloc[0:train_size, :]
    test = mod_data.iloc[train_size:len(mod_data), :]
    print("train data\n", train)
    print("Train Data\n", len(train))
    print("Test Data\n", len(test))

    # Calling the look_back function
    trainX, trainY = look_back(train, look_back_size)
    testX, testY = look_back(test, look_back_size)

    print("Missing values trainX", np.isnan(trainX).sum())
    print("Missing values trainY", np.isnan(trainY).sum())
    print("Missing values testX", np.isnan(testX).sum())
    print("Missing values testY", np.isnan(testY).sum())
    print()
    print("Len of Train X\n", len(trainX))
    print("Len of Train Y\n", len(trainY))
    print("Len of Test X\n", len(testX))
    print("Len of Test Y\n", len(testY))
    print("trainX Shape = ", trainX.shape)
    print("trainY Shape = ", trainY.shape)
    print("testX Shape = ", testX.shape)
    print("testY Shape = ", testY.shape)
    print()
    print("testX[0]", testX.shape[0])
    print("testX[1]", testX.shape[2])

    # Giving the Input to Model
    mdl = lstm_model(trainX, trainY, 30, look_back_size, testX.shape[2])

    # Testing the model Accuracy
    trainPred = mdl.predict(trainX)
    testPred = mdl.predict(testX)
    # Appending the test & train values to the old data set

    print()
    print("Missing values TestX", np.isnan(testX).sum())
    print("Missing values Test Pred", np.isnan(testPred).sum())
    print("Missing values Train Pred", np.isnan(trainPred).sum())
    print("Missing values Train", np.isnan(trainX).sum())

    # Error Metrics
    print("Predicted Vales\n", testPred)
    print("Actual Values\n", testY)
    trainScore = (mean_squared_error(trainY, trainPred))
    print('Train Error: %.2f MSE' % trainScore)
    testScore = mean_squared_error(testY, testPred)
    print('Test Error: %.2f MSE' % testScore)
    print("R2 Score Test", r2_score(testY, testPred))

    # Printing the Predicted Values
    #print(f'Prediction \n {Scaler_Close.inverse_transform(testPred.reshape(-1,1))}')
    #print(f'Actual Values \n {Scaler_Close.inverse_transform(testY.reshape(-1,1))}')

    # Plotting the predictions
    plot_2(Scaler_Close.inverse_transform(testPred.reshape(-1,1)), Scaler_Close.inverse_transform(testY.reshape(-1,1)))

    return mdl

# Getting the Model
model = Input_to_Model(stk_df)