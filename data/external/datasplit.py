import yahooData as yd
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


# NOTE - initial commit of the repo - 04.10.2022 - was chosen to be the end for train
#   that is, the 10y worth of data is pulled wrt. the above date
end_train = datetime(2022, 10, 4)
# fixme - these dates do not take into account the trader holidays and days free from trading
start_1d = end_train + relativedelta(years=-10)
start_1h = end_train + relativedelta(months=-6)  # 30min has the same restrictions as 15min; 1h interval works tho
start_15min = end_train + relativedelta(months=-1)  # up to 60 days of 15min -> could go for 1 month


def pull_data_split(ticker: str, save_data: bool = False) -> dict:
    """
    Pulls the data in our data split format for a given stock
    - 10y worth of 1d interval data
    - 6m worth of 1h interval data
    - 1m worth of 15min interval data
    @param ticker: ticker of the stock
    @param save_data: whether the downloaded data should be stored or not; False by default
    @return: dict of dataframes in the data split format
    """
    def merge_data(dataframes: list):
        # pd.merge(data_1d, data_1h, how='outer') TODO - may use merge instead
        return pd.concat(dataframes).drop_duplicates().reset_index(drop=True)

    # note - naming convention of yfinance suck.
    data_1d = yd.get_ticker_data(ticker, '1d', start_1d, end_train).rename(columns={'Date': 'datetime'})
    data_1h = yd.get_ticker_data(ticker, '1h', start_1h, end_train).rename(columns={'index': 'datetime'})
    data_15min = yd.get_ticker_data(ticker, '15m', start_15min, end_train).rename(columns={'Datetime': 'datetime'})
    # merging
    # FIXME - handle duplicates (1d has 00:00:00 time, so no overlap with 1h or 15min)
    data_1d_1h = merge_data([data_1d, data_1h])
    data_1d_15min = merge_data([data_1d, data_15min])

    if save_data:
        data_1d.to_csv(f'{ticker}-1d.csv')
        data_1h.to_csv(f'{ticker}-1h.csv')
        data_15min.to_csv(f'{ticker}-15min.csv')
        data_1d_1h.to_csv(f'{ticker}-1d_1h.csv')
        data_1d_15min.to_csv(f'{ticker}-1d_15min.csv')

    return {
        'data_1d': data_1d,
        'data_1h': data_1h,
        'data_15min': data_15min,
        'data_1d_1h': data_1d_1h,
        'data_1d_15min': data_1d_15min
        # TODO - do we need other dataframes?
    }


def pull_all_stocks(save_data: bool = False) -> dict:
    """
    Pulls the data split for all the sp500 stocks
    @return: dictionary with all the stocks data split format data
    """
    return {symbol: pull_data_split(symbol, save_data=save_data) for symbol in yd.get_sp500_tickers()}


if __name__ == '__main__':
    pull_data_split(ticker='AMD', save_data=True)
