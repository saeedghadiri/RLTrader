from RLTrader.apps.rltrader import config
from RLTrader.env.env import StockTradingEnv
from RLTrader.agent.models import Agent
from RLTrader.apps.rltrader.config import TF_LOG_DIR, CHECKPOINTS_PATH, TOTAL_EPISODES, RENDER_ENV, SAVE_WEIGHTS, \
    LOAD_LAST, UNBALANCE_P
import random

import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
from RLTrader.agent.utils import Tensorboard

if __name__ == '__main__':
    if config.DNN_TYPE == 'LSTM':
        state_dim = (config.SEQUENCE, len(config.TICKERS) * len(config.features))
    elif config.DNN_TYPE == 'CNN':
        state_dim = (config.SEQUENCE, len(config.TICKERS), len(config.features))

    env_kwargs = {

        "initial_asset": 1000000,
        "stock_dim": len(config.TICKERS),
        "state_dim": state_dim,
        "action_dim": len(config.TICKERS) + 1,
        "features": config.features,
        "reward_scaling": 1000,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "data_path": config.DATA_PATH,
        "sequence": config.SEQUENCE,
        "tickers": config.TICKERS,
        "test_env": False
    }

    env_kwargs_test = env_kwargs.copy()
    env_kwargs_test["start_date"] = '2019-01-02'
    env_kwargs_test["end_date"] = '2021-08-01'
    env_kwargs_test["test_env"] = True

    env = StockTradingEnv(**env_kwargs)

    env_test = StockTradingEnv(**env_kwargs_test)
    agent = Agent(env, env_test, config.DNN_TYPE)
    agent.learn()
