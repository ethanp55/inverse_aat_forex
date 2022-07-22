from aat.assumptions import Assumptions, TechnicalIndicators
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType
import numpy as np
from pandas import DataFrame
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AatMarketTrainer:
    def __init__(self, currency_pair: CurrencyPairs) -> None:
        self.currency_pair = currency_pair

    def record_tuple(self, curr_idx: int, market_data: DataFrame, trade_type: TradeType) -> None:
        pass

    def save_data(self) -> None:
        pass


class AatMarketTrainerForCnnModel(AatMarketTrainer):
    def __init__(self, currency_pair: CurrencyPairs, near_level_pips: float) -> None:
        AatMarketTrainer.__init__(self, currency_pair)
        self.near_level_pips = near_level_pips
        self.training_data = []

    def record_tuple(self, curr_idx: int, market_data: DataFrame, trade_type: TradeType) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
            vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]
        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, is_support, self.near_level_pips,
                                      trade_type)
        new_tup = new_assumptions.create_aat_tuple()

        self.training_data.append(new_tup)

    def save_data(self) -> None:
        data_dir = '../aat/training_data'

        file_path = f'{data_dir}/{self.currency_pair.value}_training_data.pickle'

        with open(file_path, 'wb') as f:
            pickle.dump(self.training_data, f)

        x = np.array(self.training_data)[:, 0:-1]
        y = np.array(self.training_data)[:, -1]

        print('X train shape: ' + str(x.shape))
        print('Y train shape: ' + str(y.shape))

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = NearestNeighbors(n_neighbors=15)
        model.fit(x_scaled)

        trained_knn_file = f'{self.currency_pair.value}_trained_knn_aat.pickle'
        trained_knn_scaler_file = f'{self.currency_pair.value}_trained_knn_scaler_aat.pickle'

        with open(f'{data_dir}/{trained_knn_file}', 'wb') as f:
            pickle.dump(model, f)

        with open(f'{data_dir}/{trained_knn_scaler_file}', 'wb') as f:
            pickle.dump(scaler, f)
