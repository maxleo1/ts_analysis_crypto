#!/usr/bin/env python3
import json
#pip3 install python-binance!
from binance.client import Client
import binance.helpers
import datetime

client = Client("", "")

symbol = "ADAEUR"
interval = Client.KLINE_INTERVAL_30MINUTE
start = "1 Dec, 2020"
end = "1 Aug, 2021"
 
klines = client.get_historical_klines(symbol, interval, start, end)

import tzlocal
local_tz = tzlocal.get_localzone()
local_tz

with open(
    "/mnt/f/investments/cardano/Binance_{}_{}_{}-{}.csv".format(
        symbol, 
        interval, 
        binance.helpers.date_to_milliseconds(start),
        binance.helpers.date_to_milliseconds(end)
    ),
    'w' # set file write mode
) as f:
    title= ["Open time","Open","High","Low","Close","Volume","Close time","Quote asset volume","Number of trades","Taker buy base asset volume","Taker buy quote asset volume","Ignore"]
    f.write("|".join(title)+"\n")
    for e in klines:
        st = datetime.datetime.fromtimestamp(e[0]/1000.0)
        end = datetime.datetime.fromtimestamp(e[6]/1000.0)
        st_s = st.strftime("%Y%m%d%H%M%S")
        end_s = end.strftime("%Y%m%d%H%M%S")
        em = [st_s]+e[1:6]+[end_s]+e[7:]
        f.write("|".join([str(x) for x in em])+"\n")
#    f.write(json.dumps(klines))


##1499040000000,      # Open time
##"0.01634790",       # Open
##"0.80000000",       # High
##"0.01575800",       # Low
##"0.01577100",       # Close
##"148976.11427815",  # Volume
##1499644799999,      # Close time
##"2434.19055334",    # Quote asset volume
##308,                # Number of trades
##"1756.87402397",    # Taker buy base asset volume
##"28.46694368",      # Taker buy quote asset volume
##"17928899.62484339" # Ignore

