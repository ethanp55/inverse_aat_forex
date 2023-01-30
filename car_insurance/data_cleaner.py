import pandas as pd
from typing import Tuple


class DataCleaner(object):
    @staticmethod
    def get_insurance_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Dataset found at https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4954928053318020/1058911316420443/167703932442645/latest.html
        df = pd.read_csv('./data/insurance.csv')
        x, y = df.iloc[:, :-1], pd.get_dummies(df.iloc[:, -1])

        # Drop any insignificant columns
        x.drop(['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_date', 'collision_type',
                'authorities_contacted', 'incident_state', 'incident_city', 'incident_location', 'auto_make',
                'auto_model', 'insured_hobbies', 'property_damage'],
               axis=1, inplace=True)

        # Convert column of fractions to float
        x['policy_csl'] = x['policy_csl'].apply(pd.eval)

        # Expand any categorical columns
        x = pd.get_dummies(x, columns=['insured_sex', 'insured_education_level', 'insured_occupation',
                                       'insured_relationship', 'incident_type', 'incident_severity',
                                       'police_report_available'])

        return x, y


# x, y = DataCleaner.get_insurance_data()
# print(x.head())
# print(x.columns)
# print(y.head())
