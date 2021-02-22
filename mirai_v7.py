import sys
import finnhub
import pandas as pd
import datetime
from datetime import timezone, datetime
import plotly.graph_objects as go
import math
from datetime import timedelta
import time
import numpy as np
import pickle
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
import optuna
from optkeras.optkeras import OptKeras
import keras.backend as K
import nsepy as Nse
import pandas as pd
from datetime import timezone, datetime
from datetime import date

from Look_Back_Maker import *
from Main_Model_v6 import lstm_model
from Multi_Ouput_Lookback_Maker import *


# Fixing the Seed Vale for Reproducibility

from numpy.random import seed
seed(1)
tf.random.set_seed(1)

print('MIRAI V7.0')


# Setting the Initial and Final Date
start_date = datetime(2011, 1, 1)
end_date = datetime.today()

start_date = start_date.replace(tzinfo=timezone.utc).timestamp()
end_date = end_date.replace(tzinfo=timezone.utc).timestamp()
start_date = int(start_date)
end_date = int(end_date)

# Setting Company Symbol
cmpny = 'AAPL'


# Function to get Indian NSE data
def get_nse(cmpny):
    # creating a Nse object
    # note index is for index stocks
    end_date = datetime.today()
    end_date = date(end_date.year, end_date.month, end_date.day)
    print(end_date)
    nse = Nse.get_history(symbol=cmpny, start=date(2011, 1, 1), end=end_date)
    print(nse.head(25))

    # Resetting Date from Index
    nse.reset_index(inplace=True)

    # Creating a DataFrame for Indian Market
    col = ['o', 'h', 'l', 'c', 'v', 't']
    indian_data = pd.DataFrame(columns=col)
    print(indian_data)
    indian_data['o'] = nse['Open']
    indian_data['h'] = nse['High']
    indian_data['l'] = nse['Low']
    indian_data['c'] = nse['Close']
    indian_data['v'] = nse['Volume']
    indian_data['t'] = nse['Date']

    # Adding a ok value
    tmp_lst=[]
    for row in range(len(nse['Date'])):
        tmp_lst.append('ok')
    indian_data['s'] = tmp_lst

    # Converting from Timestamp to UNIX Times
    tmp_lst = []
    for row in nse.itertuples():
        unixtime = time.mktime(row.Date.timetuple())
        tmp_lst.append(unixtime)
    indian_data['t'] = tmp_lst

    print(f'Original Length {len(nse["Date"])}  After Conversion {len(indian_data["t"])}')

    return indian_data

# Importing data from finnhub API
# Setup Client
fin_client = finnhub.Client(api_key='c09706n48v6tm13rt11g')

# Stock candles
res = fin_client.stock_candles(cmpny, 'D', start_date, end_date)
print(res)

# Overwriting Values with NSE Values
cmpny = 'HDFCBANK'
res = get_nse(cmpny)




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
    # Passing Normal Time for plot
    stk_df['time'] = x_time

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

def mov_avg(sma_win_o, sma_win_c, ema_win_o, ema_win_o_2, data):
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
    exp2 = close_val.ewm(span=ema_win_o_2).mean()
    exp_val2 = pd.DataFrame(exp2)
    print(("EMA Opening 1" + str(ema_win_o_2)), exp_val2)
    print("EMA SIZE \n", exp_val2.shape)

    print(len(sma_val), len(exp_val), len(exp_val2), len(sma_val1))

    # Returning The values
    return sma_val1, sma_val, exp_val, exp_val2




# Making a function to classify Up & Downs per week
def weekly_classifier(data):
    print("Start Date", start_date_normal)
    print("End Date", end_date_normal)
    cmp_data = data
    # Calculating the number of Weeks to process
    total_weeks = end_date_normal - start_date_normal
    print("Total Weeks\n", (total_weeks.days / 7))

    # Closing values
    cls_values = cmp_data['o']

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
        if row.c >= row.o:
            chk_lst.append(1)
        # Bearish Market
        if row.c < row.o:
            chk_lst.append(0)

    # cmp_data['Chk_Val'] = chk_lst

    print("Comparison Data\n", cmp_data.tail(25))
    print("Frequency of -1", chk_lst.count(0))
    print("Frequency of +1", chk_lst.count(1))



# Function to plot the Predictions
def plot_2(pred, real):
    # Plotting the Fit Curve
    tmp_lst = []
    for i in range(len(pred)):
        tmp_lst.append(i)
    plt.plot(tmp_lst, pred, color='black', label='Prediction')
    plt.plot(tmp_lst, real, color='red', label='Actual')
    plt.title(f'{cmpny} Prediction V/S Actual')
    plt.legend()
    plt.show()

# Model Input Data Transformation function

def Input_to_Model(data, trail_inp, CNTRL_VAL):

    # LookBack Window
    look_back_size = 4
    # Size of Output Window
    n_steps_out=2


    # Dropping the useless values form Data
    print(data.columns)

    # Interchanging Close and EMA 7 value

    mod_data = data._drop_axis(['sma_close', 's', 't', 'EMA7'], axis=1)
    print("mod_data null values = \n", mod_data.isnull().sum())
    print(type(mod_data))
    print("mod data", mod_data)

    # Scaling the data for every column except target
    Scaler = MinMaxScaler()
    Scaler_Close = MinMaxScaler()
    close_val = mod_data['c']
    close_val = close_val.values.reshape(-1,1)
    mod_data['c'] = Scaler_Close.fit_transform(close_val)
    mod_data[['v', 'sma_open', 'ema_open','l','o', 'h', 'ema_open_2', 'time']] = Scaler.fit_transform(
       mod_data[['v', 'sma_open', 'ema_open', 'l','o', 'h', 'ema_open_2', 'time']])

    print("mod data\n", mod_data)

    # Reorganizing the Column Values
    mod_data = mod_data.reindex(columns=['v', 'l', 'h','o', 'sma_open', 'ema_open','ema_open_2','time', 'c'])

    # Train/Test Split
    train_size = int(len(mod_data) * 0.85)
    print(len(mod_data))
    train = mod_data.iloc[0:train_size, :]
    test = mod_data.iloc[train_size
                         :len(mod_data), :]
    print("train data\n", train)
    print("Train Data\n", len(train))
    print("Test Data\n", len(test))

    # Calling the look_back function
    trainX, trainY = look_back_multi(train, look_back_size, n_steps_out)
    testX, testY = look_back_multi(test, look_back_size, n_steps_out)
    print(f'due chk this out  {trainX.shape}')

    #  Converting from Samples, time_steps, features TO Samples, subsequences, timesteps, features
    n_seq = -1

    trainX = trainX.reshape(trainX.shape[0], n_seq, look_back_size, n_steps_out)
    testX = testX.reshape(testX.shape[0], n_seq, look_back_size, n_steps_out)
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
    print("testX[2]", testX.shape[2])

    # Giving the Input to Model
    mdl = lstm_model(trainX, trainY, 30, look_back_size, testX.shape[2], n_steps_out, trail_inp, CNTRL_VAL)

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
    plot_2(Scaler_Close.inverse_transform(testPred.reshape(-1, 1)), Scaler_Close.inverse_transform(testY.reshape(-1, 1)))

    # Returning The Loss value
    return mean_squared_error(testY, testPred)


# Initialize Optuna
study = optuna.create_study(direction='minimize')


def objective(trial):

    # Clear Backend
    K.clear_session()

    # Variables
    sma_o_tr = trial.suggest_int('sma_o_tr',20,200)
    ema_o_tr = trial.suggest_int('ema_o_tr',10,100)
    ema_o_2_tr = trial.suggest_int('ema_o_2_tr',5,95)
    CNTRL_VAL = trial.suggest_int('CNTRL_VAL', 1,5)

    print()
    print(f'Value for SMA_Open = {sma_o_tr} EMA_Open = {ema_o_tr}, EMA_OPEN_2 = {ema_o_2_tr}  CNTRL_VAL = {CNTRL_VAL}, ')
    # Calling the Moving Function
    sma_o, sma_c, exp_o, exp_o_2 = mov_avg(sma_o_tr, 120,ema_o_tr, ema_o_2_tr, stk_df)

    # Appending the values to DataFrame Columns
    stk_df['sma_open'] = sma_o
    stk_df['sma_close'] = sma_c
    stk_df['ema_open'] = exp_o
    stk_df['ema_open_2'] = exp_o_2

    # Replacing Starting SMA with 0
    stk_df['sma_open'] = stk_df['sma_open'].fillna(0)
    stk_df['sma_close'] = stk_df['sma_close'].fillna(0)

    # Calling the Plots Function
    # plots(stk_df)

    # Passing Weekly Number as time
    stk_df['time'] = x_month

    # Calling the weekly_classifier function
    weekly_classifier(stk_df)

    # Training the Model
    score = Input_to_Model(stk_df, trial.number, CNTRL_VAL)

    return score


# Running the Training

study.optimize(objective, n_trials=5)

# Opening the best model
from Main_Model_v6 import gelu
customObjects = {
    'gelu': gelu
}
best_model = keras.models.load_model(str(study.best_trial.number)+'_model_.h5', custom_objects = customObjects)


# Trading Script


class Trader:
    def __init__(self, data, model):
     self.data = data
     self.model = model

    # Making the EMA & SMA Windows

    def prep_data(self, sma_o_tr, ema_o_tr,ema_o_2_tr, CNTRL_VAL):

        print('Preparing the Data')
        # Calling the Moving Function
        sma_o, sma_c, exp_o, exp_o_2 = mov_avg(sma_o_tr, 120,ema_o_tr, ema_o_2_tr, stk_df)

        # Appending the values to DataFrame Columns
        stk_df['sma_open'] = sma_o
        stk_df['sma_close'] = sma_c
        stk_df['ema_open'] = exp_o
        stk_df['ema_open_2'] = exp_o_2


        # Replacing Starting SMA with 0
        stk_df['sma_open'] = stk_df['sma_open'].fillna(0)
        stk_df['sma_close'] = stk_df['sma_close'].fillna(0)

        # Calling the Plots Function
        # plots(stk_df)

        # Passing Weekly Number as time
        stk_df['time'] = x_month

        # Calling the weekly_classifier function
        weekly_classifier(stk_df)
        print('Prep Complete')
        return stk_df

    # Function to Run back testing
    def back_test(self, data):
        print('Running Back Test')
        # LookBack Window
        look_back_size = 4
        # Size of Output Window
        n_steps_out = 2

        # Dropping the useless values form Data
        print(data.columns)
        # Interchanging Close and EMA 7 value
        mod_data = data.drop(['sma_close', 's', 't', 'EMA7'], axis=1)
        print("mod_data null values = \n", mod_data.isnull().sum())
        print(type(mod_data))
        print("mod data", mod_data)

        # Scaling the data for every column except target
        Scaler = MinMaxScaler()
        Scaler_Close = MinMaxScaler()
        close_val = mod_data['c']
        close_val = close_val.values.reshape(-1, 1)
        mod_data['c'] = Scaler_Close.fit_transform(close_val)
        mod_data[['v', 'sma_open', 'ema_open','l','o', 'h', 'ema_open_2', 'time']] = Scaler.fit_transform(
            mod_data[['v', 'sma_open', 'ema_open','l','o', 'h', 'ema_open_2', 'time']])

        print("mod data\n", mod_data)

        # Reorganizing the Column Values
        mod_data = mod_data.reindex(columns=['v', 'l', 'h','o', 'sma_open', 'ema_open','ema_open_2','time', 'c'])

        # Making a Real Time Prediction Set
        real_data = mod_data[len(data)-(look_back_size*6):]
        print('Real Data Length = ', len(real_data))
        print(real_data.head())

        # Calling the look_back function
        X_data, Y_data = look_back_multi(mod_data, look_back_size, n_steps_out)

        n_seq = -1
        X_data = X_data.reshape(X_data.shape[0], n_seq, look_back_size, n_steps_out)


        print("Missing values X_data", np.isnan(X_data).sum())
        print("Missing values Y_data", np.isnan(Y_data).sum())
        print()
        print("Len of X_data\n", len(X_data))
        print("Len of Y_data\n", len(Y_data))
        print("X_data Shape = ", X_data.shape)
        print("Y_data Shape = ", Y_data.shape)
        print("X_data[0]", X_data.shape[0])
        print("X_data[1]", X_data.shape[2])

        # Predicting the Values
        X_Pred = self.model.predict(X_data)

        # Error Metrics
        print("Predicted Vales\n", X_Pred)
        print("Actual Values\n", Y_data)
        trainScore = (mean_squared_error(Y_data, X_Pred))

        # Plotting the predictions
        plot_2(Scaler_Close.inverse_transform(X_Pred.reshape(-1, 1)),
               Scaler_Close.inverse_transform(Y_data.reshape(-1, 1)))
        print('Back Test Complete!')
        print('Test Error: %.2f MSE' % trainScore)
        print("R2 Score Test", r2_score(Y_data, X_Pred))
        print()

        # Real Time Predictions

        # Calling the look_back function
        X_real, Y_real = look_back_multi(real_data, look_back_size, n_steps_out)
        X_real = X_real.reshape(X_real.shape[0], n_seq, look_back_size, n_steps_out)

        X_future = self.model.predict(X_real)

        print("Predicted Values\n", Scaler_Close.inverse_transform(X_future.reshape(-1, 1)))
        print("Actual values\n", Scaler_Close.inverse_transform(Y_real))

        # Making a Final DataFrame
        # Selecting the last few days
        # Taking values from Prediction

        print(f'real_data length ->${len(real_data)}')
        real_data = Scaler.inverse_transform(real_data[['v', 'sma_open', 'ema_open','l','o', 'h', 'ema_open_2', 'time']])
        real_data = pd.DataFrame(real_data, columns=[['v', 'sma_open', 'ema_open','l','o', 'h', 'ema_open_2', 'time']])

        real_data['o'] = data['o'].iloc[-look_back_size * 6:].values.reshape(-1,1)
        real_data = real_data[-len(real_data) + look_back_size:]


        # Because of Multi Output there is copies of values so removing them

        temp_1 = Scaler_Close.inverse_transform(Y_real.tolist())

        print('Real Values Format ->', temp_1)


        close_act = []
        for i in range(len(temp_1)):
            if i == 0:
                close_act.append(temp_1[i])
            else:
                close_act.append(temp_1[i][n_steps_out-1:])

        temp_2 =  Scaler_Close.inverse_transform(X_future.tolist())
        print('Predicted Values Format ->', temp_2)
        close_pred = []
        for i in range(len(temp_2)):
            if i == 0:
                close_pred.append(temp_2[i])
            else:
                close_pred.append(temp_2[i][n_steps_out-1:])

        print('Close Actual Length->', len(np.array(close_act).ravel()))
        print('Close Predicted Length->', len(np.array(close_pred).ravel()))
        print('real_Data length', len(real_data))

        real_data['Pred_Close'] = [item for sublist in close_pred for item in sublist]
        real_data['Close'] = [item for sublist in close_act for item in sublist]

        print(real_data.head(25))
        print()
        print('Program Complete')

# Running Back test

t = Trader(stk_df, best_model)
pass_data = t.prep_data(study.best_params['sma_o_tr'], study.best_params['ema_o_tr'], study.best_params['ema_o_2_tr'], study.best_params['CNTRL_VAL'])
t.back_test(pass_data)
