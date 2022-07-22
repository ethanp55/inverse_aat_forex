from aat.assumptions import Assumptions, TechnicalIndicators
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType
import numpy as np
from pandas import DataFrame
import pickle


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
              f'{n_correct / n_preds}')


class AatMarketTesterForCnnModel(AatMarketTester):
    def __init__(self, currency_pair: CurrencyPairs, near_level_pips: float) -> None:
        AatMarketTester.__init__(self, currency_pair)
        self.near_level_pips = near_level_pips

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

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, is_support, self.near_level_pips,
                                      true_trade_type)
        new_tup = new_assumptions.create_aat_tuple()

        x = np.array(new_tup[0:-1]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        neighbor_distances, neighbor_indices = self.knn_model.kneighbors(x_scaled, 15)

        # corrections, distances = [], []
        #
        # for i in range(len(neighbor_indices[0])):
        #     neighbor_idx = neighbor_indices[0][i]
        #     neighbor_dist = neighbor_distances[0][i]
        #     corrections.append(self.training_data[neighbor_idx, -2])
        #     distances.append(neighbor_dist)
        #
        # trade_amount_pred, inverse_distance_sum = 0, 0
        #
        # for dist in distances:
        #     inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)
        #
        # for i in range(len(corrections)):
        #     distance_i, cor = distances[i], corrections[i]
        #     inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
        #     distance_weight = inverse_distance_i / inverse_distance_sum
        #
        #     trade_amount_pred += (self.baseline * cor * distance_weight)
        #
        # return trade_amount_pred

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
