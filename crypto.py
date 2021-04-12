#!/usr/bin/env python3

from pandas_datareader import data,wb,quandl
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy import stats
#import PyQt5
import matplotlib
#print(matplotlib.rcsetup.all_backends)
#print(matplotlib.get_backend())
#matplotlib.rcParams['backend'] = "Qt4Agg"
#matplotlib.use('Qt5Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
import statsmodels.formula.api as fsm

from statsmodels.tsa.stattools import adfuller
##autoregressive models: AR(p) = ARIMA(p,0,0)
##moving average models: MA(q) = ARIMA(0,0,q)
##mixed autoregressive moving average models: ARMA(p, q) = ARIMA(p,0,q)
##integration models: ARIMA(p, d, q)
##seasonal models: SARIMA(P, D, Q, s)

yf.pdr_override() # <== that's all it takes :-)

l=[]
l_names=[]
for s in ['BTC-EUR','BCH-EUR','XTZ-EUR','ETH-EUR','LTC-EUR','GC=F','SI=F','^IXIC','ADA-EUR','DOT1-EUR','PHA-EUR']:
    s_small = s.lower().replace('-','')
    df = data.get_data_yahoo(s, start="2019-01-01") #, end="2021-01-10")
    df = df['Close']
    l += [df]
    l_names += [s_small]

df = pd.concat(l, axis=1, join='outer')
df.columns = l_names

print(df)

df.to_pickle('/mnt/f/investments/crypto.pkl') 
df_orig = pd.read_pickle('/mnt/f/investments/crypto.pkl')
df = df_orig.replace(method='ffill')

print(df)
print(df.index.dayofweek.value_counts())

fig = df.plot(linewidth=0.1).get_figure()
fig.savefig('/mnt/f/investments/crypto.jpg', dpi=600)
plt.clf()
###
#####A PLOT WITH TWO AXES:
####fig, ax1 = plt.subplots()
####
####s = 'etheur'
####ax1.set_ylabel(s, color='tab:red')
####ax1.plot(df.index, df[s], color='tab:red')
####ax1.tick_params(axis='y', labelcolor='tab:red')
####ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
####
####s = 'adaeur'
####ax2.set_ylabel(s, color='tab:blue')
####ax2.plot(df.index, df[s], color='tab:blue')
####ax2.tick_params(axis='y', labelcolor='tab:blue')
####
####fig.tight_layout()  # otherwise the right y-label is slightly clipped
####fig.savefig('/mnt/f/investments/crypto.jpg', dpi=600)
####fig.clf()
###

print(df.cov())
print(df.corr())

###[print(e+str(df[e].std()/df[e].iloc[-1])) for e in list(df)]
###
###df_ = pd.concat([df['btceur'].shift(periods=1),df['xtzeur']], axis=1, join='inner') 
###print(df_.corr())
###
####A = np.vstack([df['xtzeur'].to_numpy(), np.ones(len(df['xtzeur'].to_numpy()))]).T
####k, d = np.linalg.lstsq(A, df['btceur'].to_numpy(), rcond=None)[0]
####print('k,d:'+str(k)+'|'+str(d))
###
####print(df['etheur'].rolling(20).corr(df['adaeur']))
###fig = df['etheur'].rolling(20).corr(df['adaeur']).plot(linewidth=0.1).get_figure()
###fig.savefig('/mnt/f/investments/roll_correl.jpg', dpi=600)
###fig.clf()
###
###
###df_btc = df.div(df['btceur'], axis=0)
###fig = df_btc.plot(linewidth=0.5).get_figure()
####fig.yscale('log')
###
####fig = fig.figure()
###ax = fig.add_subplot(1, 1, 1)
###ax.plot(df_btc)
###ax.set_yscale('log')
###
###fig.savefig('/mnt/f/investments/df_btc.jpg', dpi=600)
###fig.clf()
###
print(df['btceur'][df['btceur'].notna() & df['etheur'].notna()])
#print(df['etheur'])
#df['btceur'][not df['btceur'].isnull() and not df['etheur'].isnull()] 
plt.xcorr(df['btceur'][df['btceur'].notna() & df['etheur'].notna()], df['etheur'][df['btceur'].notna() & df['etheur'].notna()], normed=True, usevlines=True, maxlags=50)
plt.title("XCorr btc v. eth (eth lag)")
fig = plt.gcf()
fig.savefig('/mnt/f/investments/xcorr.jpg', dpi=600)
plt.clf()

####COUNT YOUTUBE-VIDEO CLICKS -> correlated with price data?
####import requests
####from bs4 import BeautifulSoup
####url = 'https://www.youtube.com/watch?v=Ja9D0kpksxw&ab_channel=IOHK'
####soup = BeautifulSoup(requests.get(url).text, 'lxml')
####print(soup.select_one('meta[itemprop="interactionCount"][content]')['content'])
###
###a=pd.to_datetime('2020-05-10')
###b=pd.to_datetime('2021-08-18')
####print(a)
####print((df_orig.index - a).days/(b-a).days*72000+8000)
####df_orig['StockFlow'] = ( (np.exp( (df_orig.index - a).days/(b-a).days /3.36) - 1) / (np.exp(1)-1) *72000+8000).values.clip(8000, 80000)
###df_orig['StockFlow'] = ( (np.exp( (df_orig.index - a).days/(b-a).days *3.36) - 1) *72000 / (np.exp(3.36)-1) +8000).values.clip(8000, 80000)
###
###ax = plt.gca()
###fig =df_orig['btceur'].plot(linewidth=0.5,logy=True,ax=ax).get_figure()
###fig =df_orig['StockFlow'].plot(linewidth=0.5,logy=True,ax=ax).get_figure()
###ax.text(0.05, 0.95, 'Delta='+str(int(df_orig['btceur'].iloc[-1]-df_orig['StockFlow'].iloc[-1])), transform=ax.transAxes, fontsize=10,verticalalignment='top')
###target = df_orig['btceur'].index[-1] + datetime.timedelta(days=np.log((df_orig['btceur'].iloc[-1] - 8000) /72000 * (np.exp(3.36)-1))+1/3.36 * (b-a).days)
###ax.text(0.05, 0.85, 'Target='+str(target.date()), transform=ax.transAxes, fontsize=10,verticalalignment='top')
####exp(-1,84+ 3,36 log(SF))
###fig.savefig('/mnt/f/investments/df_btc_orig.jpg', dpi=600)
###fig.clf()
###
####plt.show()

