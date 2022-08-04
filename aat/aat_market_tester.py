from aat.assumptions import Assumptions, TechnicalIndicators
from aat.aat_communicator import AatCommunicator
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType
import numpy as np
from pandas import DataFrame
import pickle
from typing import List


class AatMarketTester:
    def __init__(self, currency_pair: CurrencyPairs) -> None:
        self.currency_pair = currency_pair
        self.n_correct_none_preds, self.n_correct_buy_preds, self.n_correct_sell_preds = 0, 0, 0
        self.n_none_preds, self.n_buy_preds, self.n_sell_preds = 0, 0, 0

    def make_prediction(self, curr_idx: int, market_data: DataFrame, true_trade_type: TradeType) -> float:
        pass

    def print_results(self) -> None:
        n_correct = self.n_correct_none_preds + self.n_correct_buy_preds + self.n_correct_sell_preds
        n_preds = self.n_none_preds + self.n_buy_preds + self.n_sell_preds

        print('AAT PREDICTION RESULTS:')
        print(f'Predicted {self.n_correct_none_preds} out of {self.n_none_preds} for no trades -- accuracy = '
              f'{self.n_correct_none_preds / self.n_none_preds}')
        print(f'Predicted {self.n_correct_buy_preds} out of {self.n_buy_preds} for buy trades -- accuracy = '
              f'{self.n_correct_buy_preds / self.n_buy_preds}')
        print(f'Predicted {self.n_correct_sell_preds} out of {self.n_sell_preds} for sell trades -- accuracy = '
              f'{self.n_correct_sell_preds / self.n_sell_preds}')
        print(f'Predicted {n_correct} out of {n_preds} overall trades -- accuracy = '
              f'{n_correct / n_preds}\n')


class AatKnnMarketTesterForCnnModel(AatMarketTester):
    def __init__(self, currency_pair: CurrencyPairs) -> None:
        AatMarketTester.__init__(self, currency_pair)

        scaler_path = f'../aat/training_data/{self.currency_pair.value}_trained_knn_scaler_aat.pickle'
        knn_path = f'../aat/training_data/{self.currency_pair.value}_trained_knn_aat.pickle'
        data_path = f'../aat/training_data/{self.currency_pair.value}_training_data.pickle'

        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.knn_model = pickle.load(open(knn_path, 'rb'))
        self.training_data = np.array(pickle.load(open(data_path, 'rb')))

    def make_prediction(self, curr_idx: int, market_data: DataFrame, true_trade_type: TradeType) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
            vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]

        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, true_trade_type)
        new_tup = new_assumptions.create_aat_tuple()

        x = np.array(new_tup[0:-1]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        neighbor_distances, neighbor_indices = self.knn_model.kneighbors(x_scaled, 15)

        distances = []

        for i in range(len(neighbor_distances[0])):
            neighbor_dist = neighbor_distances[0][i]
            distances.append(neighbor_dist)

        inverse_distance_sum = 0

        for dist in distances:
            inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

        none_votes, buy_votes, sell_votes = 0, 0, 0

        for i in range(len(distances)):
            distance_i = distances[i]
            inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
            distance_weight = inverse_distance_i / inverse_distance_sum

            neighbor_idx = neighbor_indices[0][i]
            neighbor_pred = self.training_data[neighbor_idx, -1]

            if neighbor_pred == TradeType.NONE.value:
                none_votes += distance_weight

            elif neighbor_pred == TradeType.BUY.value:
                buy_votes += distance_weight

            elif neighbor_pred == TradeType.SELL.value:
                sell_votes += distance_weight

            else:
                raise Exception(f'Invalid trade type: {neighbor_pred}')

        pred = np.argmax([none_votes, buy_votes, sell_votes])

        # Update results
        self.n_none_preds += 1 if pred == TradeType.NONE.value else 0
        self.n_buy_preds += 1 if pred == TradeType.BUY.value else 0
        self.n_sell_preds += 1 if pred == TradeType.SELL.value else 0

        self.n_correct_none_preds += 1 if pred == TradeType.NONE.value and pred == true_trade_type.value else 0
        self.n_correct_buy_preds += 1 if pred == TradeType.BUY.value and pred == true_trade_type.value else 0
        self.n_correct_sell_preds += 1 if pred == TradeType.SELL.value and pred == true_trade_type.value else 0


class AatMarketTesterWithCommunicator(AatMarketTester):
    def __init__(self, currency_pair: CurrencyPairs, communicate_assumptions: bool, top_n_features: int) -> None:
        AatMarketTester.__init__(self, currency_pair)
        self.aat_communicator = AatCommunicator(currency_pair, top_n_features) if communicate_assumptions else None

    def communicate(self, aat_tup: List[float], pred: int, actual: int):
        if self.aat_communicator is not None:
            self.aat_communicator.communicate(aat_tup, pred, actual)

    def communicate_outlier_results(self):
        if self.aat_communicator is not None:
            self.aat_communicator.outlier_results()


class AatRfMarketTesterForCnnModel(AatMarketTesterWithCommunicator):
    def __init__(self, currency_pair: CurrencyPairs, communicate_assumptions: bool = False,
                 top_n_features: int = 0) -> None:
        AatMarketTesterWithCommunicator.__init__(self, currency_pair, communicate_assumptions, top_n_features)

        scaler_path = f'../aat/training_data/{self.currency_pair.value}_trained_rf_scaler_aat.pickle'
        rf_path = f'../aat/training_data/{self.currency_pair.value}_trained_rf_aat.pickle'

        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.rf_model = pickle.load(open(rf_path, 'rb'))

    def make_prediction(self, curr_idx: int, market_data: DataFrame, true_trade_type: TradeType) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
            vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]

        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, true_trade_type)
        new_tup = new_assumptions.create_aat_tuple()

        x = np.array(new_tup[0:-1]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        pred = self.rf_model.predict(x_scaled)[0]

        # Update results
        self.n_none_preds += 1 if pred == TradeType.NONE.value else 0
        self.n_buy_preds += 1 if pred == TradeType.BUY.value else 0
        self.n_sell_preds += 1 if pred == TradeType.SELL.value else 0

        self.n_correct_none_preds += 1 if pred == TradeType.NONE.value and pred == true_trade_type.value else 0
        self.n_correct_buy_preds += 1 if pred == TradeType.BUY.value and pred == true_trade_type.value else 0
        self.n_correct_sell_preds += 1 if pred == TradeType.SELL.value and pred == true_trade_type.value else 0

        # Communicate any values
        self.communicate(new_tup, pred, true_trade_type.value)


class AatReducedRfMarketTesterForCnnModel(AatMarketTesterWithCommunicator):
    def __init__(self, currency_pair: CurrencyPairs, communicate_assumptions: bool = False,
                 top_n_features: int = 0) -> None:
        AatMarketTesterWithCommunicator.__init__(self, currency_pair, communicate_assumptions, top_n_features)

        scaler_path = f'../aat/training_data/{self.currency_pair.value}_trained_reduced_rf_scaler_aat.pickle'
        rf_path = f'../aat/training_data/{self.currency_pair.value}_trained_reduced_rf_aat.pickle'
        feature_names_path = f'../aat/training_data/{self.currency_pair.value}_training_features.pickle'
        features_used_path = f'../aat/training_data/{self.currency_pair.value}_reduced_rf_features_used.pickle'

        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.rf_model = pickle.load(open(rf_path, 'rb'))
        feature_names = pickle.load(open(feature_names_path, 'rb'))
        features_used = pickle.load(open(features_used_path, 'rb'))
        self.feature_indices = [feature_names.index(feature_used) for feature_used in features_used]

    def make_prediction(self, curr_idx: int, market_data: DataFrame, true_trade_type: TradeType) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
            vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]

        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, true_trade_type)
        new_tup = new_assumptions.create_aat_tuple()

        x = np.array([new_tup[idx] for idx in self.feature_indices]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        pred = self.rf_model.predict(x_scaled)[0]

        # Update results
        self.n_none_preds += 1 if pred == TradeType.NONE.value else 0
        self.n_buy_preds += 1 if pred == TradeType.BUY.value else 0
        self.n_sell_preds += 1 if pred == TradeType.SELL.value else 0

        self.n_correct_none_preds += 1 if pred == TradeType.NONE.value and pred == true_trade_type.value else 0
        self.n_correct_buy_preds += 1 if pred == TradeType.BUY.value and pred == true_trade_type.value else 0
        self.n_correct_sell_preds += 1 if pred == TradeType.SELL.value and pred == true_trade_type.value else 0

        # Communicate any values
        self.communicate(new_tup, pred, true_trade_type.value)
