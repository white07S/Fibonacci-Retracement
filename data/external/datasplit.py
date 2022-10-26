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


def pull_data_split(ticker: str, save_data: bool = False) -> pd.DataFrame:
    """
    Pulls the data in our data split format for a given stock
    - 10y worth of 1d interval data
    - 6m worth of 1h interval data
    - 1m worth of 15min interval data
    @param ticker: ticker of the stock
    @param save_data: whether the downloaded data should be stored or not; False by default
    @return: dataframe containing the combined data
    """
    # note - naming convention of yfinance suck.
    data_1d = yd.get_ticker_data(ticker, '1d', start_1d, start_1h).rename(columns={'Date': 'datetime'})
    data_1h = yd.get_ticker_data(ticker, '1h', start_1h, start_15min).rename(columns={'index': 'datetime'})
    data_15min = yd.get_ticker_data(ticker, '15m', start_15min, end_train).rename(columns={'Datetime': 'datetime'})
    data_combined = pd.concat([data_1d, data_1h, data_15min]).reset_index(drop=True)

    if save_data:
        data_combined.to_csv(f'{ticker}-combined.csv')
    return data_combined


def pull_all_stocks(save_data: bool = False) -> dict:
    """
    Pulls the data split for all the sp500 stocks
    @return: dictionary with all the stocks data split format data
    """
    # FIXME - yahoofinance - downloading multiple data at once
    return {symbol: pull_data_split(symbol, save_data=save_data) for symbol in yd.get_sp500_tickers()}


if __name__ == '__main__':
    pull_data_split(ticker='AMD', save_data=True)
