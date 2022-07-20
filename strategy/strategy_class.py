from pandas import DataFrame
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import Trade
from ml_models.learner import Learner
from strategy.strategy_results import StrategyResults
from typing import Optional


class Strategy:
    def __init__(self, description: str, starting_idx: int) -> None:
        self.description = description
        self.starting_idx = starting_idx
        self.currency_pair = CurrencyPairs.AUD_USD

    def place_trade(self, curr_idx: int, market_data: DataFrame) -> Optional[Trade]:
        pass

    def run_strategy(self, currency_pair: CurrencyPairs, date_range: str,
                     learner: Optional[Learner] = None) -> StrategyResults:
        pass
