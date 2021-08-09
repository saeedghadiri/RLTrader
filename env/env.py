import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


# from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    # df should have data for every tic every date
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 data,
                 stock_dim,
                 initial_asset,
                 reward_scaling,
                 state_dim,
                 action_dim,
                 features,
                 print_verbosity=10,
                 model_name=''):

        self.df = df
        self.data = data
        self.stock_dim = stock_dim

        self.initial_asset = initial_asset
        self.reward_scaling = reward_scaling
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.features = features
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))

        self.print_verbosity = print_verbosity

        self.model_name = model_name

        # initalize state
        self.portfo = None
        self.state = None

        # initialize
        self.asset = None
        self.day = None
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

        self.reward = (asset / self.asset - 1) * self.reward_scaling

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

        self.day = 0
        self.total_days = len(self.df.index.unique()) - 1
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
        if self.episode % self.print_verbosity == 0:
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

        state = [self.portfo, self.data[ind, :]]

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
