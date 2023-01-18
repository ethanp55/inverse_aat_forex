import pandas as pd
from typing import Tuple


class DataCleaner(object):
    @staticmethod
    def get_credit_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Dataset found at http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        df = pd.read_csv('./data/taiwan.csv')
        x, y = df.iloc[:, :-1], pd.get_dummies(df.iloc[:, -1])

        return x, y


# x, y = DataCleaner.get_credit_data()
# print(x.columns)
# print(y.head())
