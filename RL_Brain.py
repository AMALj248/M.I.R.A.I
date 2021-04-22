import pandas as pd
from datetime import timezone, datetime
from datetime import date
import time
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import Bounds
from scipy.optimize import minimize, minimize_scalar, dual_annealing


# Getting the Stock Data

def get_nse(comp_name):
    # creating a Nse object
    # note index is for index stocks

    comp_name = yf.Ticker(comp_name)

    end_date = datetime.today()
    end_date = date(end_date.year, end_date.month, end_date.day)

    # get historical market data
    recv_data = comp_name.history( start=date(2015, 1, 1), end=end_date, interval='1d')
    print(recv_data)

    # Resetting Date from Index
    recv_data.reset_index(inplace=True)

    # Creating a DataFrame for Indian Market
    col = ['o', 'h', 'l', 'c', 'v', 't']
    stock_data = pd.DataFrame(columns=col)
    print(stock_data)
    stock_data['o'] = recv_data['Open']
    stock_data['h'] = recv_data['High']
    stock_data['l'] = recv_data['Low']
    stock_data['c'] = recv_data['Close']
    stock_data['v'] = recv_data['Volume']
    stock_data['t'] = recv_data['Date']

    print(stock_data.head(25))

    print(f'Original Length {len(recv_data["Date"])}  After Conversion {len(stock_data["t"])}')

    # If Nan Row, Drop the Entire Row
    stock_data.dropna(inplace=True)

    return stock_data


# Getting the Stock Data

cmpny = 'TATASTEEL.NS'

res = get_nse(cmpny)


# Function to make Moving Averages

def mov_avg(ema_win_c, ema_win_c_2, data):
    ################################ SMA ################################

    # Creating a Simple Moving Averages for Opening
    close_val = data['c']

    ####################################### EMA #######################################

    # Creating the EMA 1
    exp = close_val.ewm(span=ema_win_c).mean()
    exp_val = pd.DataFrame(exp)
    # print("EMA Closing" + str(ema_win_c), exp_val)
    # print("EMA SIZE \n", exp_val.shape)

    # Creating the EMA 2
    exp2 = close_val.ewm(span=ema_win_c_2).mean()
    exp_val2 = pd.DataFrame(exp2)
    # print(("EMA Closing 2" + str(ema_win_c_2)), exp_val2)
    # print("EMA SIZE \n", exp_val2.shape)

    # Returning The values
    return exp_val, exp_val2



print('After adding EMA\n', res.head(25))

# Function to  Find The Crossover Points

def cross_over(data):

    temp_lst = []
    for row in data.itertuples():
        # Bearish Market
        if row.ema_s <= row.ema_L:
            temp_lst.append(0)


        # Bullish Market
        if row.ema_s > row.ema_L:
            temp_lst.append(1)

    # print('res len', len(res))
    # print('tmep_list len', len(temp_lst))
    res['status'] = temp_lst


def objective_fun(x):
    x1 = x[0]
    x2 = x[1]
    # print('EMA VALUES ')
    # print(x1, x2)
    ema_1, ema_2 = mov_avg(x1, x2, res)
    res['ema_s'] = ema_1
    res['ema_L'] = ema_2

    # Finding the Crossover Points
    cross_over(res)
    res['signal'] = res['status'].diff()

    # Dropping first Nan row in temporary DataFrame
    cln = res.dropna()
    # print(cln.head())

    sig_cln = cln[cln['signal'] != 0]
    sig_cln.reset_index(inplace=True)
    # Dropping the Index Column
    del sig_cln['index']


    PROFIT = 0
    sig_trk = []
    for row in sig_cln.itertuples():
        sig_trk.append(row.signal)
        # Sell Preceded by Buy
        if row.Index != 0 and row.signal == -1 and sig_trk[row.Index-1] == 1:
            PROFIT += row.c-sig_cln['c'][row.Index-1]
            #print('PROFIT', PROFIT)

    #print('PROFIT =',PROFIT )

    return -1 * PROFIT


# Defining the Lower and Upper Bounds
lw = [20, 50]
up = [40, 125]
sol = dual_annealing(objective_fun, bounds= list(zip(lw, up)))
print(sol)
print('EMA Solution = ', sol.x[0], sol.x[1])
print('Real Profit = ', objective_fun(sol.x))


print()

# # plot close price, short-term and long-term moving averages
# res['c'].plot(color = 'k', label= 'Close Price')
# res['ema_s'].plot(color = 'b',label = 'EMA Low')
# res['ema_L'].plot(color = 'g', label = 'EMA High')
#
# # plot ‘buy’ signals
# plt.plot(res['t'][res['signal'] == 1].index,
#          res['c'][res['signal'] == 1],
#          '^', markersize = 15, color = 'g', label = 'buy')
# # plot ‘sell’ signals
# plt.plot(res['t'][res['signal'] == -1].index,
#          res['c'][res['signal']  == -1],
#          'v', markersize = 15, color = 'r', label = 'sell')
# plt.ylabel('Price', fontsize = 15 )
# plt.xlabel('Date', fontsize = 15 )
# plt.title(str(cmpny), fontsize = 20)
# plt.legend()
# plt.grid()
# plt.show()
#
