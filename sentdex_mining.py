import pandas as pd
import numpy as np
import bs4
from urllib.request import urlopen
import json
import csv
from bs4 import BeautifulSoup
from requests import get
url = 'http://sentdex.com/financial-analysis/?i=AAPL&tf=all'
#Sending Request and opening
response = BeautifulSoup(urlopen(url) , features="lxml")

#Printing the HTML Strucuture
print(response)

#Finding the base class for data
base_class = response.find_all('script')
print("Len" , len(base_class))

print(type(base_class))

#To find the correct Slice of Data
# for i in range(0,14) :
#     print(i)
#     print(base_class[i])

cluttered_data = str(base_class[4])#.replace('  ','').replace('\t','').replace('\n','')

#Starting Split
val = cluttered_data.split('var series =')
#print(val[1])

#Ending Split
val2=val[1].split('var title')
print("In Slice Form" , val2[0])

#Converting to Dataframe
data=val2[0]
print("In String Form" , data)
print('Type' , type(data) )
print('Len' , len(data) )
def data_converter (df):
    print()
    print("Type of Data Recieved" ,type(df) )
    print("Data Recieved\n" , df)
    #Removing all blank spcaes
    df_in=df.replace(' ' , '').replace('\n','')
    print("Input \n" , df_in)

    #Extracting Long Term Sentiment
    lts = df_in.split(",'name':'Long-TermSentiment'")
    lts2 = lts[0].split("yAxis':1},")
    lts3 = lts2[1].split("{'data':")
    lts_clean = lts3[1]
    print()
    print(lts_clean)
    print(len(lts_clean))
    #Converting to List of list
    #final_lts=lts_clean.split(', ')

    print("Long Term Sentiment\n" ,lts_clean)
    print(len(lts_clean))


    #Extracting the Short Term Sentiment
    print()
    #print("Input" , df_in)
    sts = df_in.split("'name':'Mid-TermSentiment'")
    sts2=sts[1].split(",'name':'Short-TermSentiment'")
    sts3 = sts2[0].split(",{'data':")
    final_sts = sts3[1]
    print("Short Term\n" , final_sts)
    print(len(final_sts))


#Calling the Function
data_converter(data)


