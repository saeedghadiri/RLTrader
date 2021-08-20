from RLTrader.apps.rltrader import config
from RLTrader.env.env import StockTradingEnv
from RLTrader.agent.models import Brain
from RLTrader.apps.rltrader.config import TF_LOG_DIR, CHECKPOINTS_PATH, TOTAL_EPISODES, RENDER_ENV, SAVE_WEIGHTS, \
    LOAD_LAST, UNBALANCE_P
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
from RLTrader.agent.utils import Tensorboard

features = ['high_price', 'low_price', 'close_price']

if __name__ == '__main__':
    env_kwargs = {

        "initial_asset": 1000000,
        "stock_dim": len(config.TICKERS),
        "state_dim": (config.SEQUENCE, len(features) * len(config.TICKERS)),
        "action_dim": len(config.TICKERS) + 1,
        "features": features,
        "reward_scaling": 100,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "data_path": config.DATA_PATH,
        "sequence": config.SEQUENCE,
        "tickers": config.TICKERS,
    }

    env = StockTradingEnv(**env_kwargs)

    env_kwargs = {

        "initial_asset": 1000000,
        "stock_dim": len(config.TICKERS),
        "state_dim": (config.SEQUENCE, len(features) * len(config.TICKERS)),
        "action_dim": len(config.TICKERS) + 1,
        "features": features,
        "reward_scaling": 100,
        "start_date": '2019-01-02',
        "end_date": '2021-08-01',
        "data_path": config.DATA_PATH,
        "sequence": config.SEQUENCE,
        "tickers": config.TICKERS,
    }

    env_test = StockTradingEnv(**env_kwargs)

    brain = Brain(env.observation_space, env.action_space)

    tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

    # load weights if available
    logging.info("Loading weights from %s*, make sure the folder exists", CHECKPOINTS_PATH)
    if LOAD_LAST:
        brain.load_weights(CHECKPOINTS_PATH)

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    test_reward = tf.keras.metrics.Sum('test_reward', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # run iteration

    for ep in range(TOTAL_EPISODES):
        tic = datetime.now()
        print('******************************************************')
        print(ep)
        prev_state = env.reset()
        acc_reward.reset_states()
        test_reward.reset_states()
        Q_loss.reset_states()
        A_loss.reset_states()

        done = False

        while not done:
            if RENDER_ENV:  # render the environment into GUI
                env.render()

            # Recieve state and reward from environment.
            cur_act = brain.act(prev_state, _notrandom=random.random() < ep / TOTAL_EPISODES, noise=True)

            state, reward, done, _ = env.step(cur_act)
            brain.remember(prev_state, reward, state, int(done))

            # update weights
            c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))

            Q_loss(c)
            A_loss(a)

            # post update for next step
            acc_reward(reward)

            prev_state = state

        prev_state = env_test.reset()
        done = False
        while not done:
            cur_act = brain.act(prev_state, _notrandom=random.random() < ep / TOTAL_EPISODES, noise=True)
            state, reward, done, _ = env_test.step(cur_act)
            test_reward(reward)

        ep_reward_list.append(acc_reward.result().numpy())
        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward_list.append(avg_reward)

        # print the average reward
        tensorboard(ep, acc_reward, test_reward, Q_loss, A_loss, brain.buffer.get_buffer_size())

        # save weights
        if ep % 5 == 0 and SAVE_WEIGHTS:
            brain.save_weights(CHECKPOINTS_PATH)
        print(datetime.now() - tic)

    brain.save_weights(CHECKPOINTS_PATH)

    logging.info("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards

    # tic = datetime.now()
    # for episode in range(100):
    #     env.reset()
    #     terminal = False
    #     while not terminal:
    #         action_dim = len(config.TICKERS) + 1
    #         actions = np.random.rand(action_dim)
    #         actions = actions / sum(actions)
    #         actions = list(actions)
    #         state, reward, terminal, _ = env.step(actions)
    #
    # print(datetime.now() - tic)
