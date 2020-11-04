import yfinance as yahoo_data
import pandas as pd
import datetime
from datetime import timedelta
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
# fix random seed for reproducibility
np.random.seed(7)
#downloading the data for Microsoft for Past 6 Months
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

#Creating a Days Columns
day_lst = []
min_date = df['Date'].min()
for row in df['Date'] :
    x = row-min_date
    day_lst.append(x.days)

df['Days'] = day_lst
#Dropping Column
df=df.drop('Date' , axis = 1)
print("DF \n" , df.head())


#Taking Simple Moving Average of the Data

sma_num = df['Y_Closing'].values
sma_num = pd.Series(sma_num)
#defining the window size
window_size = 5

#Creating Moving Averages
windows = sma_num.rolling(window_size)
mv_avg = windows.mean()
#Converting to list
sma_lst = mv_avg.tolist()
sma_data = sma_num[window_size -1: ]
sma=pd.DataFrame(sma_data)
sma.rename(columns={0 : 'Y_values'} , inplace=True)
print("SMA \n" , sma.head(25))

#Plotting the SMA Values
fig = px.line(sma , x =sma.index , y = sma['Y_values'] , title=("SMA "+str(window_size)))
#fig.show()

#Plotting without SMA
fig = px.line(df , x = df['Days'] , y=df['Y_Closing'] , title="Without SMA")
#fig.show()


#Creating Input format for LSTM
# convert an array of values into a dataset matrix
look_back = window_size

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):

		a = dataset.iloc[i:(i + look_back), 0]
		dataX.append(a)
		#print("heyyy\n" , dataX)
		b=dataset.iloc[i+look_back , 0]
		#print("heyyy2\n", dataX)
		dataY.append(b)
	return (np.array(dataX), np.array(dataY))




#Defining the Input/Output Varibales
X = sma.index.values
Y= sma['Y_values'].values

#Scalling the data
Scaler = MinMaxScaler(feature_range= (0,1))

X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

X =Scaler.fit_transform(X)
Y =Scaler.fit_transform(Y)

imp_len=len(X)


#Converting to Float Values for Deep Learning Model
#Converting the data to float32
X = X.astype(np.float32)
Y = Y.astype(np.float32)

print("X after float32 conversion\n" , X)
print("Y after float32 conversion\n" , Y)

#creating a Test/Train Split

train_size = int(len(df) * 0.67)
test_size = len(df) - train_size
train, test = df[0:train_size], df[train_size:len(df)]

print("Total Size" , len(df))
print("Train Size = " , train_size)
print("Test Size = " , test_size)

train_X , train_Y = create_dataset(train , 5)
test_X , test_Y = create_dataset(test , 5)


print("train_X" ,  train_X)
print("train_Y" ,  train_Y)

print("test_X" ,  test_X)
print("test_Y" ,  test_Y)

print()
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1],1)



#LSTM Model anbd Architecture

#Note -> Input Shape's Second Argument has to match Timestep
model = Sequential()
#Input LSTM LAYER
model.add(LSTM(150 ,return_sequences=True , input_shape = (5 ,1)))
#Hidden Layer
model.add(LSTM(150,return_sequences=True))
#last Hidden Layer for giving a 2d Output or else R2 will fail
model.add(LSTM(1,return_sequences=False))
#Output Layer
model.add(Dense(1))

model.compile(loss= "mean_squared_error" , optimizer='adam')
model.fit(train_X , train_Y , batch_size=25 , verbose = 2 , epochs=15)

#making the predictions
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

print("MAE" , mean_absolute_error(test_Y , test_pred))
print("R2 Score" , r2_score(test_Y , test_pred))

print((train_X.shape) , (train_Y.shape))
#Plotting the model
plt.scatter(X , Y , color = 'black')
plt.plot(train_X ,train_pred  , color = 'red')
plt.plot(test_X ,test_pred  , color = 'red')
plt.scatter()
