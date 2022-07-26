from aat.aat_market_trainer import AatMarketTrainer
from aat.aat_market_tester import AatMarketTester
from cnn.cnn_utilities import CNN_LOOKBACK, grab_image_data
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade, TradeCalculations, TradeType
from ml_models.learner import Learner
import numpy as np
from pandas import DataFrame
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from tensorflow.keras.models import load_model
from typing import Optional


class CnnStrategy(Strategy):
    def __init__(self, starting_idx: int, risk_reward_ratio: float, currency_pair: CurrencyPairs,
                 proba_threshold: float, lookback: int, spread_cutoff: float) -> None:
        description = f'CNN on {currency_pair.value} with {risk_reward_ratio} risk/reward, {spread_cutoff} spread ' \
                      f'ratio, stop loss lookback of {lookback}, proba threshold of {proba_threshold}'
        Strategy.__init__(self, description, starting_idx)

        if not 0.0 <= proba_threshold <= 1.0:
            raise Exception(f'Probability threshold for predictions is not between 0 and 1: {proba_threshold}')

        self.risk_reward_ratio = risk_reward_ratio
        self.proba_threshold = proba_threshold
        self.lookback = lookback
        self.spread_cutoff = spread_cutoff
        self.model = load_model(f'../cnn/training_data/{currency_pair.value}_trained_cnn')

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        # Use CNN to make a prediction
        curr_slice = market_data.drop(['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 'Ask_Open', 'Ask_High',
                                       'Ask_Low', 'Ask_Close', 'Mid_Open', 'Mid_High', 'Mid_Low', 'Mid_Close',
                                       'Volume', 'Date'], axis=1, inplace=False)
        curr_slice = curr_slice.iloc[curr_idx - CNN_LOOKBACK:curr_idx, :]
        curr_slice = grab_image_data(curr_slice)

        pred = self.model(curr_slice.reshape(1, curr_slice.shape[0], curr_slice.shape[1], curr_slice.shape[2]))
        argmax = np.argmax(pred)
        proba = pred[0][argmax].numpy()

        cnn_buy_signal = argmax == TradeType.BUY.value and proba >= self.proba_threshold
        cnn_sell_signal = argmax == TradeType.SELL.value and proba >= self.proba_threshold

        trade = None

        if cnn_buy_signal or cnn_sell_signal:
            # Get needed prices for placing a trade
            curr_bid_open, curr_ask_open, curr_mid_open, curr_date = \
                market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open', 'Mid_Open', 'Date']]

            spread = abs(curr_ask_open - curr_bid_open)

            mid_highs = list(
                market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_High'])
            mid_lows = list(market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_Low'])

            highest_high, lowest_low = max(mid_highs), min(mid_lows)

            # Try to place a trade (if the levels are set properly and the spread is small enough)
            if cnn_buy_signal:
                open_price = curr_ask_open
                stop_loss = lowest_low

                if stop_loss < open_price:
                    curr_pips_to_risk = open_price - stop_loss

                    if spread <= curr_pips_to_risk * self.spread_cutoff:
                        stop_gain = open_price + (self.risk_reward_ratio * curr_pips_to_risk)
                        trade_type = TradeType.BUY
                        n_units = TradeCalculations.get_n_units(trade_type, stop_loss, curr_ask_open, curr_bid_open,
                                                                curr_mid_open, self.currency_pair)

                        trade = Trade(trade_type, open_price, stop_loss, stop_gain, n_units, n_units, curr_pips_to_risk,
                                      curr_date)

            elif cnn_sell_signal:
                open_price = float(curr_bid_open)
                stop_loss = highest_high

                if stop_loss > open_price:
                    curr_pips_to_risk = stop_loss - open_price

                    if spread <= curr_pips_to_risk * self.spread_cutoff:
                        stop_gain = open_price - (self.risk_reward_ratio * curr_pips_to_risk)
                        trade_type = TradeType.SELL
                        n_units = TradeCalculations.get_n_units(trade_type, stop_loss, curr_ask_open, curr_bid_open,
                                                                curr_mid_open, self.currency_pair)

                        trade = Trade(trade_type, open_price, stop_loss, stop_gain, n_units, n_units, curr_pips_to_risk,
                                      curr_date)

        return trade

    def run_strategy(self, currency_pair: CurrencyPairs, date_range: str,
                     learner: Optional[Learner] = None,
                     aat_trainer: Optional[AatMarketTrainer] = None,
                     aat_tester: Optional[AatMarketTester] = None) -> StrategyResults:
        self.currency_pair = currency_pair
        market_data = DataRetriever.get_data_for_pair(currency_pair, date_range) if learner is None else \
            learner.market_data

        return MarketSimulator.run_simulation(self, market_data, learner, aat_trainer, aat_tester)
