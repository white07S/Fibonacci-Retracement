from pyparsing import alphanums
import yfinance as yf
import pandas as pd
import yahoo_fin.stock_info as si
from datetime import datetime


def getSP500List():
    '''
    Return a Python list containing all the tickers from SP500
    If include_company_data is set to True, the tickers, company names, and sector information is returned as a data frame.
    '''
    tickers = si.tickers_sp500(include_company_data=True)
    return tickers


def getTickerData(ticker:str,interval:str,start:datetime,end:datetime):
    '''
    Ticker: Company symbol for example: MSFT for microsoft
    period: duration of historic data
    interval: time interval of data
    Some data format: {day:d,month:mo,minute:min,hour:h}
    Example: period=3d,interval=5min
    NOTE: There are some restriction in data
    For 1 min time interval : 1 week
    5 min : 30 days
    1d: 10 years 
    We will be using 1d to train the model and may use 5 min to train the model
    '''
    df = yf.download(ticker=ticker,start=start,end=end,interval=interval).reset_index()
    return df

