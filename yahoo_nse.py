import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import timezone, datetime
from datetime import date
import time

# Getting Data from Yahoo Finance

cmpny = yf.Ticker('ICICIBANK.NS')

end_date = datetime.today()
end_date = date(end_date.year, end_date.month, end_date.day)

# get historical market data
nse = cmpny.history(period="max", start=date(2011, 1, 1), end=end_date)
print(nse)


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

print(indian_data.head(25))

print(f'Original Length {len(nse["Date"])}  After Conversion {len(indian_data["t"])}')
