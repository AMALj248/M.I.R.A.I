import yfinance as yahoo_data
import finnhub
import pandas as pd
import datetime
import requests
from datetime import timezone , datetime
import plotly.graph_objects as go
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

# Setting the Initial and Final Date
start_date = datetime(2011 , 10 , 19)
end_date = datetime.today()

start_date = start_date.replace(tzinfo=timezone.utc).timestamp()
end_date = end_date.replace(tzinfo=timezone.utc).timestamp()
start_date = int(start_date)
end_date = int(end_date)

# Importing data from finnhub API

# Setup Client
fin_client = finnhub.Client(api_key='btthr6748v6or4rae7ag')

# Setting Compnay SYMOBOL
cmpny = 'AAPL'


# Stock candles
res = fin_client.stock_candles(cmpny, 'D',start_date, end_date)
print(res)

# Getting The Company Profile
prf=fin_client.company_profile2(symbol = cmpny)
print(prf)
norm_nm = prf['name']
print(norm_nm)

# Conbverting Unix Timestamp to Normal Date
x_time=[]
for i in res['t'] :
    x=datetime.fromtimestamp(i).strftime('%Y-%m-%d')
    x=datetime.strptime(x , '%Y-%m-%d')
    x_time.append(x)
# Setting the Start and End Date
start_date_normal = min(x_time)
end_date_normal = max(x_time)


# Coverting from list to dictionary for Stock Data
stk_df = pd.DataFrame(res)

print("Null Value Check \n" , stk_df.isna().sum())
print("DataFrame = \n" , stk_df.head(25))

# Making a Moving Average Function
def mov_avg(sma_window , ema_window , ema_window2 , data) :
    # Creating a Simple Moving Averages
    close_val = data['c']
    windows = close_val.rolling(sma_window)
    mv_avg = windows.mean()

    # Converting to list
    sma_lst = mv_avg.tolist()
    sma_data = sma_lst[sma_window - 1:]
    sma_val = pd.DataFrame(sma_data)
    print("SMA\n", sma_val)

    # Creating the EMA 1
    exp = close_val.ewm(span= ema_window ).mean()
    exp_val=pd.DataFrame(exp)
    print("EMA"+str(ema_window) , exp_val)
    print("EMA SIZE \n" , exp_val.shape)
    print("SMA SIZE \n", sma_val.shape)
    print("Total Size\n" , stk_df['c'].shape)

    # Creating the EMA 2
    exp2 = close_val.ewm(span=ema_window2).mean()
    exp_val2 = pd.DataFrame(exp2)
    print(("EMA"+str(ema_window2)), exp_val2)
    print("EMA SIZE \n", exp_val2.shape)
    print("SMA SIZE \n", exp_val2.shape)

    # Returning The values
    return sma_val , exp_val , exp_val2


def plots(data) :
    # Plotting the candlestick
    fig = go.Figure(data = [go.Candlestick( x=x_time,
                     open = data['o'] , high = data['h'],
                    low = data['l'] , close=data['c'] , name=(cmpny+str(' Prices')) )])

    # Plotting the EMA & SMA
    fig.add_trace(go.Scatter(x=stk_df['time'], y=stk_df['sma_close'], text='SMA', name=('SMA' + str(100))))
    fig.add_trace(go.Scatter(x=stk_df['time'], y=stk_df['ema_close1'], text='EMA', name=('EMA' + str(50))))
    fig.add_trace(go.Scatter(x=stk_df['time'], y=stk_df['ema_close2'], text='EMA', name=('EMA' + str(200))))
    fig.update_layout(title_text='Moving Averages')
    fig.update_layout(title_text=(str(norm_nm)))
    #fig.show()

    # Plotting the volume of Trade over time
    fig = go.Figure(go.Scatter(x=x_time, y=stk_df['v'], text='Volume'))
    fig.update_layout( title_text = ('Volume of Trades for '+str(norm_nm)))
    #fig.show()


# Taking the user input for windows for SMA & EMA
# sma_win = int(input('Enter The Window Size for SMA\n'))
# ema_win = int(input('Enter The Window Size for EMA\n'))
# Calling the Moving Funtion

sma_cls, exp_cls ,exp2_cls = mov_avg(20,50,200,stk_df)

# Appending the values to Dataframe Columns
stk_df['sma_close'] = sma_cls
stk_df['ema_close1'] = exp_cls
stk_df['ema_close2'] = exp2_cls
stk_df['time'] = x_time


print("After Appending\n" , stk_df.head(25))
print("Columns\n" , stk_df.columns)
print(stk_df['c'])

# Calling the Plots Function
plots(stk_df)

# Making a function to classify Up & Downs per week
def weekly_classifier(data) :
    print("Start Date" , start_date_normal)
    print("End Date" , end_date_normal)
    cmp_data = data
    # Calculating the number of Weeks to process

    total_weeks = end_date_normal - start_date_normal
    print("Total Weeks\n" , (total_weeks.days/7))

    # Closing values
    cls_values = cmp_data['c']

    # Making EMA7
    ema7 = cls_values.ewm(span=7).mean()
    ema7_val = pd.DataFrame(ema7)
    #print("EMA" + str(7), ema7_val)
    #print("EMA SIZE \n", ema7_val.shape)
    #print("EMA SIZE \n", ema7_val.shape)
    cmp_data['EMA7'] = ema7_val


    # Logic
    chk_lst=[]
    for row  in cmp_data.itertuples():

         # Bullish Market
        if row.c >= row.EMA7:
            chk_lst.append(1)
        # Bearish Market
        else :
            chk_lst.append(-1)

    cmp_data['Chk_Val'] = chk_lst
    print("Comparison Data\n", cmp_data.tail(25))
    print("Frequency of -1" , chk_lst.count(-1))
    print("Frequency of +1", chk_lst.count(1))
# Calling the weekly_classifier function
weekly_classifier(stk_df)

# Model Input Data Transformation function
def Input_to_Model(data):
    # Defining the X and Y Values
    print(data.columns)
    X = data._drop_axis (['Chk_Val', 't', 's' , 'time' ], axis=1)
    print("X = \n", X)
    print(X.columns)
    Y = data.iloc[:,-1]
    print("Y = \n", Y)

    # Defining The MinMax Scaler
    Scaler1 = MinMaxScaler()

    # Scaling the X value with MinMax Scaling
    X = Scaler1.fit_transform(X)
    print("X Scaled = \n", X)

Input_to_Model(stk_df)



