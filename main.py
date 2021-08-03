from preprocessor.yahoodownloader import YahooDownloader
import pandas as pd

tickers = []
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tickers = pd.read_csv('Stocks 100c.csv', header=None)
    tickers[0] = tickers[0].apply(lambda x: x.replace('"', ''))
    tickers = tickers[0].values

    df = YahooDownloader(start_date='2016-01-01', end_date='2021-01-01', ticker_list=tickers).fetch_data()
    df.to_pickle('data.pkl')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
