from preprocessor.yahoodownloader import YahooDownloader
from apps.RLTrader import config
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = YahooDownloader(start_date='2016-01-01', end_date='2021-01-01', ticker_list=config.TICKERS).fetch_data()
    df.to_pickle('data.pkl')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
