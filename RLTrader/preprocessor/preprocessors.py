import numpy as np
import pandas as pd
import datetime
import itertools


def data_split(df, data, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    df = df[(df.date >= start) & (df.date <= end)]
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    data = data[df['ind'], :, :]

    df['ind'] = np.arange(len(df))

    # this needs to be calculated for price changes
    df['close_pct_change'] = df.groupby('tic')['close'].pct_change() + 1
    return df, data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


def _2d_to_3d(X, n_steps=10):
    _X = np.zeros((X.shape[0], n_steps, X.shape[1])) * np.nan
    for i in range(n_steps - 1, len(X)):
        _X[i, :] = X[i - n_steps + 1: i + 1]
    return _X


class FeatureEngineer:
    def __init__(self, features, sequence_length):
        self.features = features
        self.sequence_length = sequence_length
        pass

    def create_data(self, df_input):
        df = df_input.copy()
        df['day'] = df.date.factorize()[0]
        df = df.reset_index(drop=True)

        df = df.sort_values(by=['tic', 'day']).reset_index(drop=True)

        df['base_normal'] = df.groupby('tic')['close'].shift(1)
        features = []
        for f in self.features:
            if 'price' in f:
                feature = f.replace('_price', '')
                # df[feature + '_pct'] = df.groupby('tic')[feature].pct_change()
                df[feature + '_pct'] = df[feature] / df['base_normal'] - 1
                features.append(feature + '_pct')

        if len(features) > 0:
            df[features] = df[features] / df[features].std().mean()

        # convert data to 3d
        data = []
        for tic in df['tic'].unique():
            data.append(_2d_to_3d(df.loc[df['tic'] == tic, features].values, self.sequence_length))

        # concatenate data
        data = np.concatenate(data)

        # sort by time
        df = df.sort_values(by=['day', 'tic'])
        data = data[df.index, :, :]
        df = df.reset_index(drop=True)

        df['ind'] = np.arange(len(df))

        del df['day']

        return df, data
