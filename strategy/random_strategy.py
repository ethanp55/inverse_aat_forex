from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.data_retriever import DataRetriever
from market_proxy.market_simulator import MarketSimulator
from market_proxy.trades import Trade, TradeCalculations, TradeType
from ml_models.learner import Learner
import random
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from typing import Optional


class RandomStrategy(Strategy):
    def __init__(self, starting_idx: int, risk_reward_ratio: float, spread_cutoff: float, lookback: int):
        description = f'Random strategy with {risk_reward_ratio} risk/reward, {spread_cutoff} spread ratio, ' \
                      f'stop loss lookback of {lookback}'
        Strategy.__init__(self, description, starting_idx)
        self.risk_reward_ratio = risk_reward_ratio
        self.spread_cutoff = spread_cutoff
        self.lookback = lookback

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        curr_bid_open, curr_ask_open, curr_mid_open, curr_date = \
            market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open', 'Mid_Open', 'Date']]
        spread = abs(curr_ask_open - curr_bid_open)

        mid_highs = list(market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_High'])
        mid_lows = list(market_data.loc[market_data.index[curr_idx - self.lookback:curr_idx], 'Mid_Low'])

        highest_high, lowest_low = max(mid_highs), min(mid_lows)

        signal = random.choice(['buy', 'sell', 'none'])
        buy_signal, sell_signal = signal == 'buy', signal == 'sell'

        trade = None

        if buy_signal:
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
                                  curr_date, None)

        elif sell_signal:
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
                                  curr_date, None)

        return trade

    def run_strategy(self, currency_pair: CurrencyPairs, date_range: str,
                     learner: Optional[Learner] = None) -> StrategyResults:
        self.currency_pair = currency_pair
        market_data = DataRetriever.get_data_for_pair(currency_pair, date_range) if learner is None else \
            learner.market_data

        return MarketSimulator.run_simulation(self, market_data, learner)
