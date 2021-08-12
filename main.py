from preprocessor.yahoodownloader import YahooDownloader
from preprocessor.preprocessors import data_split, FeatureEngineer
from apps.rltrader import config
import pandas as pd
import os.path
from env.env import StockTradingEnv
import numpy as np
from datetime import datetime, timedelta

features = ['open', 'high', 'low', 'close']

if __name__ == '__main__':

    start_date_download = datetime.strptime(config.START_DATE, '%Y-%m-%d') - timedelta(days=100)
    df = YahooDownloader(start_date=start_date_download, end_date=config.END_DATE,
                         ticker_list=config.TICKERS, data_path=config.DATA_PATH).fetch_data()

    fe = FeatureEngineer(features, sequence_length=5)
    df, data = fe.create_data(df)

    df, data = data_split(df, data, config.START_DATE, config.END_DATE)

    env_kwargs = {

        "initial_asset": 1000000,
        "stock_dim": len(config.TICKERS),
        "state_dim": len(config.TICKERS) * len(features) + len(config.TICKERS) + 1,
        "action_dim": len(config.TICKERS) + 1,
        "features": features,
        "reward_scaling": 100,
        "model_name": 'mamad'

    }

    e_train_gym = StockTradingEnv(df=df, data=data, **env_kwargs)

    tic = datetime.now()
    for episode in range(100):
        e_train_gym.reset()
        terminal = False
        while not terminal:
            action_dim = len(config.TICKERS) + 1
            actions = np.random.rand(action_dim)
            actions = actions / sum(actions)
            actions = list(actions)
            state, reward, terminal, _ = e_train_gym.step(actions)

    print(datetime.now() - tic)
