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
fin_client = finnhub.Client(api_key='c09706n48v6tm13rt11g')

#Setting Compnay SYMOBOL
cmpny = 'KO'


# Stock candles
res = fin_client.stock_candles(cmpny, 'M',start_date, end_date)
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
for i in range(len(sup_res['levels'])):
 fig.add_shape(type='line', x0=plot_min_date , y0 = sup_res['levels'][i] ,x1=plot_max_date , y1 = sup_res['levels'][i], )

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
#Coverting from list to dictionary
stk_df = pd.DataFrame(ct_stk)
print("Null Value Check \n" , stk_df.isna().sum())

print("The Number of Companies = " ,len(stk_df['displaySymbol']))

tst = stk_df['displaySymbol'][1]
print(tst)
#Pulling Data for a Company
start_time = time.time()
ref_i = fin_client.stock_candles(tst, 'D',start_date, end_date)
end_time = time.time() - start_time
print(res)
print(end_time)