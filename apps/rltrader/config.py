TICKERS_GROUP = 'DOW30'
START_DATE = '2016-01-01'
END_DATE = '2021-01-01'

##
ALL_TICKERS = ['AMZN', 'MCD', 'DRI', 'F', 'ADM', 'COST', 'PEP', 'WMT', 'ABBV',
               'AMGN', 'COO', 'JNJ', 'AAL', 'CAT', 'EMR', 'ROP', 'AAPL', 'IBM',
               'MA', 'PYPL', 'COG', 'EOG', 'HP', 'XOM', 'AIG', 'BAC', 'JPM',
               'USB', 'CMS', 'ED', 'DUK', 'PCG', 'DIS', 'V', 'HD', 'KO', 'FB',
               'INTC', 'TMUS', 'C', 'PM', 'MS', 'SBUX', 'GS', 'T', 'AMT', 'CVS',
               'AXP', 'BLK', 'SHOP', 'SAP', 'NKE', 'TM', 'RY', 'GOOGL', 'NFLX',
               'AIV', 'BXP', 'DRE', 'IRM', 'ALB', 'IFF', 'PKG', 'IP', 'VZ', 'LIN',
               'NEE', 'BA', 'TD', 'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD',
               'LTC-USD', 'ADA-USD', 'DOGE-USD', 'BCH-USD', 'XLM-USD', 'ETC-USD',
               'XMR-USD', 'TRX-USD', 'EURUSD=X', 'GBPUSD=X', 'JPY=X', 'NZDUSD=X',
               'AUDUSD=X', '^FVX', '^TNX', '^TYX', '^IRX', 'GC=F', 'CL=F', 'ZC=F',
               'CT=F', 'PL=F', 'SI=F', 'HG=F', 'RB=F', 'HO=F', 'NG=F']

DOW_30_TICKERS = ['AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP', 'HD', 'INTC', 'WMT', 'IBM', 'MRK',
                  'UNH', 'KO', 'CAT', 'TRV', 'JNJ', 'CVX', 'MCD', 'VZ', 'CSCO', 'XOM', 'BA', 'MMM', 'PFE', 'WBA', 'DD']

if TICKERS_GROUP == 'ALL':
    TICKERS = ALL_TICKERS
elif TICKERS_GROUP == 'DOW30':
    TICKERS = DOW_30_TICKERS

DATA_PATH = '{}.pkl'.format(TICKERS_GROUP)
