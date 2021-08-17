from RLTrader.apps.rltrader import config
from RLTrader.env.env import StockTradingEnv
import numpy as np
from datetime import datetime

features = ['open', 'high', 'low', 'close']

if __name__ == '__main__':

    env_kwargs = {

        "initial_asset": 1000000,
        "stock_dim": len(config.TICKERS),
        "state_dim": (len(config.TICKERS), config.SEQUENCE, len(features)),
        "action_dim": len(config.TICKERS) + 1,
        "features": features,
        "reward_scaling": 100,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "data_path": config.DATA_PATH,
        "sequence": config.SEQUENCE,
        "tickers": config.TICKERS,
    }

    e_train_gym = StockTradingEnv(**env_kwargs)

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
