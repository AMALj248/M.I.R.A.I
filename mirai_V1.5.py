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

#Setting the Initial and Final Date
start_date = datetime(2011 , 10 , 19)
end_date = datetime.today()

start_date = start_date.replace(tzinfo=timezone.utc).timestamp()
end_date = end_date.replace(tzinfo=timezone.utc).timestamp()

start_date = int(start_date)
end_date = int(end_date)

#print(start_date)
#print(end_date)
print()
#Importing data from finnhub API

#Setup Client
fin_client = finnhub.Client(api_key='btthr6748v6or4rae7ag')

#Setting Compnay SYMOBOL
cmpny = 'AAPL'


# Stock candles
res = fin_client.stock_candles(cmpny, 'D',start_date, end_date)
print(res)

#Getting The Company Profile
prf=fin_client.company_profile2(symbol = cmpny)
print(prf)
norm_nm = prf['name']
print(norm_nm)

#Conbverting Unix Timestamp to Normal Date
x_time=[]
for i in res['t'] :
    x=datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S')
    x_time.append(x)

#Plotting the candlestick
fig = go.Figure(data = [go.Candlestick( x=x_time,
                 open = res['o'] , high = res['h'],
                low = res['l'] , close=res['c'] )])

#Support and Resistance
#Getting Support and Resistance Lines
sup_res = fin_client.support_resistance('AAPL' , 'D')

#Setting a year back date
plot_min_date = datetime.now()-timedelta(days=365)
plot_max_date = max(x_time)
#Drawing Support & Resistance Lines
#for i in range(len(sup_res['levels'])):
# fig.add_shape(type='line', x0=plot_min_date , y0 = sup_res['levels'][i] ,x1=plot_max_date , y1 = sup_res['levels'][i], )

#Headline
fig.update_layout(title_text=(str(norm_nm)))
fig.show()

#Comparision Module
#Enter The Target Company
target = 'AAPL'

#Setting Base Exchange
excng = 'NYSE'

#Finding Similar Stocks
ct_stk=fin_client.stock_symbols('US')
#Coverting from list to dictionary for Stock Data 
stk_df = pd.DataFrame(res)

print("Null Value Check \n" , stk_df.isna().sum())
print("Dataframe = \n" , stk_df.head(25))

#Makifg a Moving Average Function 
def mov_avg(sma_window , ema_window , data) :
    # Creating a Simple Moving Averages
    close_val = data['c']
    windows = close_val.rolling(sma_window)
    mv_avg = windows.mean()

    # Converting to list
    sma_lst = mv_avg.tolist()
    sma_data = sma_lst[sma_window - 1:]
    sma_val = pd.DataFrame(sma_data)
    print("SMA\n", sma_val)

    #Creating the EMA
    exp = close_val.ewm(span= ema_window ).mean()
    exp_val=pd.DataFrame(exp)
    print("EMA Values\n" , exp_val)

    #Returning The values
    return sma_val , exp_val


#Taking the user input for windows for SMA & EMA
sma_win = int(input('Enter The Window Size for SMA\n'))
ema_win = int(input('Enter The Window Size for EMA\n'))

#Calling the Moving Fucntion
sma_cls, exp_cls = mov_avg(sma_win,ema_win,stk_df)

#Appending the values to Dataframe Columns
stk_df['sma_close'] = sma_cls
stk_df['ema_close'] = exp_cls
stk_df['time'] = x_time

print("After Appending\n" , stk_df.head(25))
print("Columns\n" , stk_df.columns)
#Plotting the EMA & SMA
fig = go.Figure()
fig.add_trace(go.Scatter( x=stk_df['time'] , y=stk_df['sma_close'] ,text='SMA' , name=('SMA'+str(sma_win))))
fig.add_trace(go.Scatter(x=stk_df['time'] , y=stk_df['ema_close'] ,text='EMA' , name=('EMA'+str(ema_win))))
fig.update_layout(title_text= 'Moving Averages')
fig.show()

#Plotting the voulume of Trade over time
fig=go.Figure(go.Scatter(x=x_time , y=stk_df['v'] , text='Volume'))
fig.show()


