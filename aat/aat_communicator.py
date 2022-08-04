from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
import pickle
from typing import List


class AatCommunicator:
    def __init__(self, currency_pair: CurrencyPairs, top_n_features: int, iqr_multiplier: float = 1.0) -> None:
        feature_names_path = f'../aat/training_data/{currency_pair.value}_training_features.pickle'
        feature_importances_path = f'../aat/training_data/{currency_pair.value}_rf_feature_importances.pickle'
        training_data_path = f'../aat/training_data/{currency_pair.value}_training_data.pickle'

        self.feature_names = pickle.load(open(feature_names_path, 'rb'))
        feature_importances = pickle.load(open(feature_importances_path, 'rb'))
        self.training_data = np.array(pickle.load(open(training_data_path, 'rb')))
        self.top_n_feature_indices = [feature_importances.index(x) for x in
                                      sorted(feature_importances, reverse=True)[:top_n_features]]
        self.iqr_multiplier = iqr_multiplier
        self.quartile_info = {}

        for i in range(self.training_data.shape[1] - 1):
            q1 = np.percentile(self.training_data[:, i], 25, interpolation='midpoint')
            q3 = np.percentile(self.training_data[:, i], 75, interpolation='midpoint')
            iqr = q3 - q1

            self.quartile_info[i] = (q1, q3, iqr)

        self.outliers_when_pred_same, self.outliers_when_pred_diff = [], []
        self.outlier_feature_name_counts = {}

    def communicate(self, aat_tup: List[float], pred: int, actual: int) -> None:
        message, outlier_count = '', 0
        message += 'PREDICTION MATCHED\n' if pred == actual else 'PREDICTION DID NOT MATCH\n'

        for i in range(self.training_data.shape[1] - 1):
            q1, q3, iqr = self.quartile_info[i]
            feature_name, feature_val = self.feature_names[i], aat_tup[i]
            is_outlier = feature_val > q3 + self.iqr_multiplier * iqr or feature_val < q1 - self.iqr_multiplier * iqr
            outlier_count += 1 if is_outlier else 0

            if i in self.top_n_feature_indices or is_outlier:
                message += f'{feature_name} = {feature_val}'
                message += ' -- OUTLIER\n' if is_outlier else '\n'

            if pred != actual and is_outlier:
                self.outlier_feature_name_counts[feature_name] = \
                    self.outlier_feature_name_counts.get(feature_name, 0) + 1

        if pred == actual:
            self.outliers_when_pred_same.append(outlier_count)

        else:
            self.outliers_when_pred_diff.append(outlier_count)

        print(message)

    def outlier_results(self) -> None:
        outlier_message = ''
        pred_same_outliers, pred_diff_outliers = np.array(self.outliers_when_pred_same), \
                                                 np.array(self.outliers_when_pred_diff)

        outlier_message += 'OUTLIER STATISTICS WHEN THE PREDICTION MATCHED:\n'
        outlier_message += f'Min = {min(pred_same_outliers)}, max = {max(pred_same_outliers)}, mean = ' \
                           f'{pred_same_outliers.mean()}\n'
        outlier_message += 'OUTLIER STATISTICS WHEN THE PREDICTION WAS DIFFERENT:\n'
        outlier_message += f'Min = {min(pred_diff_outliers)}, max = {max(pred_diff_outliers)}, mean = ' \
                           f'{pred_diff_outliers.mean()}\n'

        for feature_name, outlier_count in self.outlier_feature_name_counts.items():
            outlier_message += f'{feature_name} outlier count when pred was different = {outlier_count}\n'

        print(outlier_message)
