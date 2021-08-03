from preprocessor.yahoodownloader import YahooDownloader
from apps.RLTrader import config
import pandas as pd
import os.path

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if not os.path.exists(config.DATA_PATH):
        df = YahooDownloader(start_date='2016-01-01', end_date='2021-01-01', ticker_list=config.TICKERS).fetch_data()
        df.to_pickle(config.DATA_PATH)
    else:
        df = pd.read_pickle(config.DATA_PATH)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
