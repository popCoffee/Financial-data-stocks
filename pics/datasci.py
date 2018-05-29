# -*- coding: utf-8 -*-
"""
Created in 2017
Edited on Sun Mar 18 2018

@author: Andrew Kubal
"""

from pandas_datareader import data, wb
import pandas as pd
import fix_yahoo_finance as yf
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


#start = datetime.date(2006, 1, 1)
#end   = datetime.date(2016, 1, 1)
start  = "2016-01-01"
end    = "2018-01-01"
    
##get stock data from google finance
    ##warning: google api can be unstable 
dimensions  = ['dayOfWeek']
#BAC = data.DataReader("BAC", 'google', start, end)
#CC = data.DataReader("C", 'google', start, end)
#GS = data.DataReader("GS", 'google', start, end)
#JPM = data.DataReader("JPM", 'google', start, end)
#MS = data.DataReader("MS", 'google', start, end)
#WFC = data.DataReader("WFC", 'google', start, end)
yf.pdr_override() 
BAC = data.get_data_yahoo("BAC", start, end)
CC = data.get_data_yahoo("C",  start, end)
GS = data.get_data_yahoo("GS",  start, end)
JPM = data.get_data_yahoo("JPM", start, end)
MS = data.get_data_yahoo("MS", start, end)
WFC = data.get_data_yahoo("WFC", start, end)

list1= ['BAC','C','GS','JPM','MS','WFC']

bank_stocks= pd.concat([BAC,CC,GS,JPM,MS,WFC],axis=1,keys=list1)

bank_stocks.columns.names = ['Bank Ticker','Stock Info']

##closing price list
bank_stocks.xs('Close', level='Stock Info', axis=1, drop_level=True).max()

##returns for each bank stock
bank_stocks['returns']=pd.DataFrame({'returns' : []})
bank_stocks.xs('Close', level='Stock Info', axis=1, drop_level=True)[list1[1]][2]
#bank_stocks.head(2)
bsfiltered = bank_stocks.index
bsfiltered = pd.DataFrame({'returns' : []})

##column for returns
for i in range(0,len(list1)):
    bsfiltered['{} Returns' .format(list1[i])]=bank_stocks.xs('Close', level='Stock Info', axis=1, drop_level=True)[list1[i]].pct_change()
bsfiltered=bsfiltered.loc[:,'BAC Returns':]
bsfiltered.head(3)

##pairplot for returns
bsfiltered = bsfiltered[np.isfinite(bsfiltered['BAC Returns' or 'C Returns'])]
sns.pairplot(bsfiltered,plot_kws={"s": 12})
# JPM has the best return value, with a slightly positive return.

bsfiltered.idxmin() # worst return days
bsfiltered.idxmax() #best return day

#closing price for each stock
bank_stocks.xs('Close', level='Stock Info', axis=1, drop_level=True).plot()
plt.show()

#rolling 30day average in 2008 for bac
plt.figure(figsize=(12,6))
BAC['Close'].loc['2016-02-01':].rolling(window=10).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2016-02-01':].plot(label='BAC Closing')
plt.legend()
plt.show()

