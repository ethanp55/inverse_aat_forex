import pandas as pd
from typing import Tuple


class DataCleaner(object):
    @staticmethod
    def get_score_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        return DataCleaner._get_data('./data/compas-scores-two-years.csv')

    @staticmethod
    def get_violent_score_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        return DataCleaner._get_data('./data/compas-scores-two-years-violent.csv')

    @staticmethod
    def _get_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read in the data
        df = pd.read_csv(file_path)

        # Convert jail dates to length of stay
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

        # Filter out any unnecessary instances
        df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != '0']
        df = df[df['score_text'] != 'N/A']

        # Only grab the columns we are interested in
        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'sex', 'priors_count',
                 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid']]

        # Convert categorical features to one-hot encodings
        df = pd.get_dummies(df, columns=['c_charge_degree', 'race', 'age_cat', 'sex'])

        # Drop any nulls and reset the indices (housekeeping work)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Extract and return the x and y datasets
        x, y = df.loc[:, df.columns != 'decile_score'], df['decile_score']

        return x, y


# x, y = DataCleaner.get_score_data()
