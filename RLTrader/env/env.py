import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
from datetime import datetime, timedelta
from RLTrader.preprocessor.yahoodownloader import YahooDownloader
from RLTrader.preprocessor.preprocessors import data_split, FeatureEngineer
from RLTrader.apps.rltrader.config import ALL_TICKERS
import os
from itertools import product

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    # df should have data for every tic every date
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 stock_dim,
                 initial_asset,
                 reward_scaling,
                 state_dim,
                 action_dim,
                 features,
                 start_date,
                 end_date,
                 sequence,
                 data_path,
                 tickers,
                 print_verbosity=10,
                 model_name='',
                 test_env=False):

        self.stock_dim = stock_dim

        self.initial_asset = initial_asset
        self.reward_scaling = reward_scaling
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.test_env = test_env

        if type(start_date) is str:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # if we already downloaded the file, we use our local cache
        if os.path.exists(data_path):
            df = pd.read_pickle(data_path)
        else:
            df = YahooDownloader(start_date="2012-01-01", end_date="2021-08-10",
                                 ticker_list=ALL_TICKERS).fetch_data()
            del df['day']
            df.to_pickle(data_path)

        df = df[df.tic.isin(tickers)]
        start_date_temp = start_date - timedelta(days=100 + sequence)

        df = df[(df.date >= start_date_temp) & (df.date <= end_date)]

        # check whether every date for every ticker exists
        df_check = pd.DataFrame(product(tickers, df['date'].unique()))
        df_check.columns = ['tic', 'date']
        df_check['dummy'] = True
        df_check = pd.merge(df, df_check, on=['tic', 'date'])
        # assert pd.isna(df_check['dummy']).any()
        # assert pd.isna(df_check).any().any()
        del df_check

        fe = FeatureEngineer(features, sequence_length=sequence)
        df, data = fe.create_data(df)

        df, data = data_split(df, data, start_date, end_date)

        self.df = df
        self.data = data

        self.observation_space = spaces.Dict({"market": spaces.Box(low=0, high=np.inf, shape=self.state_dim),
                                              'portfo': spaces.Box(low=0, high=1, shape=(self.action_dim,))})
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,))

        self.print_verbosity = print_verbosity

        self.model_name = model_name

        # initalize state
        self.portfo = None
        self.state = None

        # initialize
        self.asset = None
        self.day = None
        self.stop_day = None
        self.total_days = None
        self.df_today = None
        self.terminal = None
        self.reward = None
        self.cost = None
        self.trades = None
        self.episode = -1
        # memorize all the total balance change
        self.asset_memory = None
        self.rewards_memory = None
        self.actions_memory = None
        self.date_memory = None
        self._seed()
        self.reset()

    def step(self, actions):

        self.day += 1
        self.terminal = self.day >= self.total_days

        self.df_today = self.df.loc[self.day, :]

        asset = np.sum(np.array(actions)[:self.stock_dim] * self.asset * self.df_today.close_pct_change.values) + \
                actions[-1] * self.asset

        self.reward = ((asset / self.asset) / self.df_today.close_pct_change.values.mean() - 1) * self.reward_scaling

        self.asset = asset
        self.portfo = actions

        self.state = self._update_state()

        self.actions_memory.append(actions)
        self.asset_memory.append(asset)
        self.date_memory.append(self._get_date())
        self.rewards_memory.append(self.reward)
        if self.terminal:
            # print(f"Episode: {self.episode}")
            self._terminal_result()

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        self.asset = self.initial_asset
        self.asset_memory = [self.asset]

        self.total_days = len(self.df.index.unique()) - 1
        if self.test_env:
            self.day = 0
            self.stop_day = self.total_days
        else:
            self.day = np.random.randint(0, self.total_days - 100)
            self.stop_day = self.day + 100

        self.df_today = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        # initiate state
        self._initiate_portfo()
        self.state = self._update_state()

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _terminal_result(self):

        df_total_asset = pd.DataFrame(self.asset_memory)
        df_total_asset.columns = ['account_value']
        df_total_asset['date'] = self.date_memory
        df_total_asset['daily_return'] = df_total_asset['account_value'].pct_change(1)
        if df_total_asset['daily_return'].std() != 0:
            sharpe = (252 ** 0.5) * df_total_asset['daily_return'].mean() / \
                     df_total_asset['daily_return'].std()
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ['account_rewards']
        df_rewards['date'] = self.date_memory[:-1]
        if self.episode % self.print_verbosity == 0 and self.test_env:
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {self.asset_memory[-1]:0.2f}")
            print(f"total_reward: {np.sum(self.rewards_memory):0.2f}")
            # print(f"total_cost: {self.cost:0.2f}")
            # print(f"total_trades: {self.trades}")
            if df_total_asset['daily_return'].std() != 0:
                print(f"Sharpe: {sharpe:0.3f}")
            print("=================================")

        if self.model_name != '':
            df_actions = self.save_action_memory()
            df_actions.to_csv('results/actions_{}.csv'.format(self.model_name))
            df_total_asset.to_csv(
                'results/asset_value_{}.csv'.format(self.model_name),
                index=False)
            df_rewards.to_csv(
                'results/rewards_{}.csv'.format(self.model_name),
                index=False)
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/asset_value_{}_{}.png'.format(self.model_name, self.episode))
            plt.close()

    def _initiate_portfo(self):
        self.portfo = np.random.rand(self.action_dim)
        self.portfo = self.portfo / sum(self.portfo)
        self.portfo = list(self.portfo)

    def _update_state(self):
        ind = self.df_today['ind']
        d_data = self.data[ind, :]

        # for lstm
        d_data = d_data.transpose(1, 0, 2).reshape(d_data.shape[1], -1)
        portfo = np.array(self.portfo)

        # d_data = d_data[np.newaxis, ...]
        # portfo = portfo[np.newaxis, ...]

        state = [d_data, portfo]
        # state = {'market': self.data[ind, :], 'portfo': self.portfo}

        # state = {'market': self.data[ind, :].transpose(2, 0, 1), 'portfo': self.portfo}

        return state

    def _get_date(self):

        date = self.df_today.date.unique()[0]

        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame({'date': date_list, 'account_value': asset_list})
        return df_account_value

    def save_action_memory(self):

        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.df_today.tic.values.tolist() + ['cash']
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})

        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def get_sb_env(self):
    #     e = DummyVecEnv([lambda: self])
    #     obs = e.reset()
    #     return e, obs
