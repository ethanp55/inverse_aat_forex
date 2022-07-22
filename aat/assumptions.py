from dataclasses import dataclass
from market_proxy.trades import TradeType
from typing import List


@dataclass
class TechnicalIndicators:
    ema200: float
    ema100: float
    atr: float
    atr_sma: float
    rsi: float
    rsi_sma: float
    adx: float
    macd: float
    macdsignal: float
    slowk_rsi: float
    slowd_rsi: float
    vo: float
    willy: float
    willy_ema: float

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]


class Assumptions:
    def __init__(self, ti_vals: TechnicalIndicators, bid_open: float, ask_open: float, key_level: float,
                 is_support: int, near_level_pips: float, prediction: TradeType) -> None:
        self.ti_vals = ti_vals

        mid_open = (ask_open + bid_open) / 2

        self.near_level = abs(mid_open - key_level) <= near_level_pips
        self.above_level = mid_open > key_level
        self.level_is_support = is_support
        self.up_trend = ti_vals.ema100 > ti_vals.ema200
        self.spread = abs(ask_open - bid_open)
        self.spread_atr_percentage = self.spread / self.ti_vals.atr
        self.prediction = prediction.value

    def create_aat_tuple(self) -> List[float]:
        return self.ti_vals.get_values() + [self.near_level, self.above_level, self.level_is_support, self.up_trend,
                                            self.spread, self.spread_atr_percentage, self.prediction]


