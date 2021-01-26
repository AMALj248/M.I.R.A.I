import nsepy as Nse
import pandas as pd
from datetime import timezone, datetime
from datetime import date
import time

# Setting the Initial and Final Date
start_date = datetime(2011, 10, 19)
end_date = datetime.today()

start_date = start_date.replace(tzinfo=timezone.utc).timestamp()
end_date = end_date.replace(tzinfo=timezone.utc).timestamp()
start_date = int(start_date)
end_date = int(end_date)

# Function to get Indian NSE data
# Function to get Indian NSE data
def get_nse():

    # creating a Nse object
    # note index is for index stocks
    nse = Nse.get_history(symbol='HDFCBANK', start=date(2018, 10, 19), end=date(2021, 1, 23))
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

data = get_nse()
print(data)