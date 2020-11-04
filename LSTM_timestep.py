import yfinance as yahoo_data
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import plotly
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM , Dense , Dropout
# fix random seed for reproducibility
np.random.seed(7)

data = yahoo_data.download('msft', period='6mo')
data = data.reset_index()
print(data.head())
print(data.columns)

#Checking Null Values
print("Null Values = " , data.isna().sum())

#We will Select Closing Prices as a Target
#Its a bit easier to model

#Creating a Dataframe
cnames2 = ['Date', 'Days' , 'Y_Closing' ]
df=pd.DataFrame(columns=cnames2)
df['Y_Closing'] = data['Close'].values
df['Date'] = data['Date'].values
df=df.drop('Date' , axis = 1)
print("DF \n" , df.head())

X = df['Days'].values
Y= df['Y_Closing'].values

#Scalling the data
Scaler = MinMaxScaler(feature_range= (0,1))

X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

train_size = int(len(df) * 0.67)
test_size = len(df) - train_size
train, test = df[0:train_size], df[train_size:len(df)]


X_train1 = []
y_train2 = []
for i in range(60, 2035):
    X_train1.append(train[i-60:i, 0])
    y_train2.append(train[i, 0])
X_train1, y_train2 = np.array(X_train1), np.array(y_train2)


print(X_train1)
