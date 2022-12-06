import pandas as pd
from typing import Tuple


class DataCleaner(object):
    @staticmethod
    def get_baseball_x_and_y() -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read in the 2 csv files; 2022 will be used for predictions
        df_x = pd.read_csv('./data/batting_2021.csv')
        df_y = pd.read_csv('./data/batting_2022.csv')

        # Clean the df for x
        df_x.drop(['Name', 'Team', 'xwOBA', 'wRC+'], axis=1, inplace=True)
        df_x['BB%'] = df_x['BB%'].str.rstrip("%").astype(float) / 100
        df_x['K%'] = df_x['K%'].str.rstrip("%").astype(float) / 100

        # Only keep the player id (so we can match with x) and HR (the target) for y
        df_y = df_y[['playerid', 'HR']]

        # Join the 2 dfs on player id (use an inner join to avoid null values) and drop it afterwards
        df = pd.merge(df_x, df_y, on='playerid', how='inner')
        df.drop(['playerid'], axis=1, inplace=True)

        # Create, check, and return x and y
        x = df.loc[:, df.columns != 'HR_y']
        y = df['HR_y']

        assert len(x) == len(y)

        return x, y


# X, Y = DataCleaner.get_baseball_x_and_y()
#
# print(X.dtypes)
# print(X['PA'].mean())
