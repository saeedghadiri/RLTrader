from preprocessor.yahoodownloader import YahooDownloader
from preprocessor.preprocessors import data_split
from apps.rltrader import config
import pandas as pd
import os.path
from env.env import StockTradingEnv
import numpy as np

if __name__ == '__main__':

    if not os.path.exists(config.DATA_PATH):
        df = YahooDownloader(start_date=config.START_DATE, end_date=config.END_DATE,
                             ticker_list=config.TICKERS).fetch_data()
        df.to_pickle(config.DATA_PATH)
    else:
        df = pd.read_pickle(config.DATA_PATH)

    del df['day']
    df = data_split(df, config.START_DATE, config.END_DATE)
    df['close_pct_change'] = df.groupby('tic')['close'].pct_change() + 1

    features = ['open', 'high', 'low', 'close']
    env_kwargs = {

        "initial_asset": 1000000,
        "stock_dim": len(config.TICKERS),
        "state_dim": len(config.TICKERS) * len(features) + len(config.TICKERS) + 1,
        "action_dim": len(config.TICKERS) + 1,
        "features": features,
        "reward_scaling": 100,
        "model_name": 'mamad'

    }

    e_train_gym = StockTradingEnv(df=df, **env_kwargs)
    for episode in range(100):
        e_train_gym.reset()
        terminal = False
        while not terminal:
            action_dim = len(config.TICKERS) + 1
            actions = np.random.rand(action_dim)
            actions = actions / sum(actions)
            actions = list(actions)
            state, reward, terminal, _ = e_train_gym.step(actions)
