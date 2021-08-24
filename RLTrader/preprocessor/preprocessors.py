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
        # drop before start date for each tic
        # ind_first = df.groupby('tic')[features].agg(lambda x: x.first_valid_index()).max(axis=1).astype(int)
        # ind_first = pd.Series(data=df.loc[ind_first, 'ind_matlab_date'].values, index=ind_first.index.values)

        # df = df.groupby('tic').apply(lambda x: x.loc[x['day'] >= ind_first.loc[x.name]]).reset_index(drop=True)

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

# class FeatureEngineer:
#     """Provides methods for preprocessing the stock price data
#     Attributes
#     ----------
#         use_technical_indicator : boolean
#             we technical indicator or not
#         tech_indicator_list : list
#             a list of technical indicator names (modified from config.py)
#         use_turbulence : boolean
#             use turbulence index or not
#         user_defined_feature:boolean
#             user user defined features or not
#     Methods
#     -------
#     preprocess_data()
#         main method to do the feature engineering
#     """
#
#     def __init__(
#             self,
#             use_technical_indicator=True,
#             tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
#             use_vix=False,
#             use_turbulence=False,
#             user_defined_feature=False,
#     ):
#         self.use_technical_indicator = use_technical_indicator
#         self.tech_indicator_list = tech_indicator_list
#         self.use_vix = use_vix
#         self.use_turbulence = use_turbulence
#         self.user_defined_feature = user_defined_feature
#
#     def preprocess_data(self, df):
#         """main method to do the feature engineering
#         @:param config: source dataframe
#         @:return: a DataMatrices object
#         """
#         # clean data
#         df = self.clean_data(df)
#
#         # add technical indicators using stockstats
#         if self.use_technical_indicator == True:
#             df = self.add_technical_indicator(df)
#             print("Successfully added technical indicators")
#
#         # add vix for multiple stock
#         if self.use_vix == True:
#             df = self.add_vix(df)
#             print("Successfully added vix")
#
#         # add turbulence index for multiple stock
#         if self.use_turbulence == True:
#             df = self.add_turbulence(df)
#             print("Successfully added turbulence index")
#
#         # add user defined feature
#         if self.user_defined_feature == True:
#             df = self.add_user_defined_feature(df)
#             print("Successfully added user defined features")
#
#         # fill the missing values at the beginning and the end
#         df = df.fillna(method="bfill").fillna(method="ffill")
#         return df
#
#     def clean_data(self, data):
#         """
#         clean the raw data
#         deal with missing values
#         reasons: stocks could be delisted, not incorporated at the time step
#         :param data: (df) pandas dataframe
#         :return: (df) pandas dataframe
#         """
#         df = data.copy()
#         df = df.sort_values(['date', 'tic'], ignore_index=True)
#         df.index = df.date.factorize()[0]
#         merged_closes = df.pivot_table(index='date', columns='tic', values='close')
#         merged_closes = merged_closes.dropna(axis=1)
#         tics = merged_closes.columns
#         df = df[df.tic.isin(tics)]
#         # df = data.copy()
#         # list_ticker = df["tic"].unique().tolist()
#         # only apply to daily level data, need to fix for minute level
#         # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
#         # combination = list(itertools.product(list_date,list_ticker))
#
#         # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
#         # df_full = df_full[df_full['date'].isin(df['date'])]
#         # df_full = df_full.sort_values(['date','tic'])
#         # df_full = df_full.fillna(0)
#         return df
#
#     def add_technical_indicator(self, data):
#         """
#         calculate technical indicators
#         use stockstats package to add technical inidactors
#         :param data: (df) pandas dataframe
#         :return: (df) pandas dataframe
#         """
#         df = data.copy()
#         df = df.sort_values(by=['tic', 'date'])
#         stock = Sdf.retype(df.copy())
#         unique_ticker = stock.tic.unique()
#
#         for indicator in self.tech_indicator_list:
#             indicator_df = pd.DataFrame()
#             for i in range(len(unique_ticker)):
#                 try:
#                     temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
#                     temp_indicator = pd.DataFrame(temp_indicator)
#                     temp_indicator['tic'] = unique_ticker[i]
#                     temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
#                     indicator_df = indicator_df.append(
#                         temp_indicator, ignore_index=True
#                     )
#                 except Exception as e:
#                     print(e)
#             df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
#         df = df.sort_values(by=['date', 'tic'])
#         return df
#
#     def add_user_defined_feature(self, data):
#         """
#          add user defined features
#         :param data: (df) pandas dataframe
#         :return: (df) pandas dataframe
#         """
#         df = data.copy()
#         df["daily_return"] = df.close.pct_change(1)
#         # df['return_lag_1']=df.close.pct_change(2)
#         # df['return_lag_2']=df.close.pct_change(3)
#         # df['return_lag_3']=df.close.pct_change(4)
#         # df['return_lag_4']=df.close.pct_change(5)
#         return df
#
#     def add_vix(self, data):
#         """
#         add vix from yahoo finance
#         :param data: (df) pandas dataframe
#         :return: (df) pandas dataframe
#         """
#         df = data.copy()
#         df_vix = YahooDownloader(start_date=df.date.min(),
#                                  end_date=df.date.max(),
#                                  ticker_list=["^VIX"]).fetch_data()
#         vix = df_vix[['date', 'close']]
#         vix.columns = ['date', 'vix']
#
#         df = df.merge(vix, on="date")
#         df = df.sort_values(["date", "tic"]).reset_index(drop=True)
#         return df
#
#     def add_turbulence(self, data):
#         """
#         add turbulence index from a precalcualted dataframe
#         :param data: (df) pandas dataframe
#         :return: (df) pandas dataframe
#         """
#         df = data.copy()
#         turbulence_index = self.calculate_turbulence(df)
#         df = df.merge(turbulence_index, on="date")
#         df = df.sort_values(["date", "tic"]).reset_index(drop=True)
#         return df
#
#     def calculate_turbulence(self, data):
#         """calculate turbulence index based on dow 30"""
#         # can add other market assets
#         df = data.copy()
#         df_price_pivot = df.pivot(index="date", columns="tic", values="close")
#         # use returns to calculate turbulence
#         df_price_pivot = df_price_pivot.pct_change()
#
#         unique_date = df.date.unique()
#         # start after a year
#         start = 252
#         turbulence_index = [0] * start
#         # turbulence_index = [0]
#         count = 0
#         for i in range(start, len(unique_date)):
#             current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
#             # use one year rolling window to calcualte covariance
#             hist_price = df_price_pivot[
#                 (df_price_pivot.index < unique_date[i])
#                 & (df_price_pivot.index >= unique_date[i - 252])
#                 ]
#             # Drop tickers which has number missing values more than the "oldest" ticker
#             filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)
#
#             cov_temp = filtered_hist_price.cov()
#             current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
#             # cov_temp = hist_price.cov()
#             # current_temp=(current_price - np.mean(hist_price,axis=0))
#
#             temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
#                 current_temp.values.T
#             )
#             if temp > 0:
#                 count += 1
#                 if count > 2:
#                     turbulence_temp = temp[0][0]
#                 else:
#                     # avoid large outlier because of the calculation just begins
#                     turbulence_temp = 0
#             else:
#                 turbulence_temp = 0
#             turbulence_index.append(turbulence_temp)
#
#         turbulence_index = pd.DataFrame(
#             {"date": df_price_pivot.index, "turbulence": turbulence_index}
#         )
#         return turbulence_index
