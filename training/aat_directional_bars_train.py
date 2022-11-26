from aat.aat_market_trainer import AatMarketTrainer
from market_proxy.currency_pairs import CurrencyPairs
from strategy.directional_bars_strategy import DirectionalBarsStrategy

RISK_REWARD_RATIO = 1.5
SPREAD_PERCENTAGE = 0.10
EACH_BAR = False
N_BARS = 3
PIP_MOVEMENT = 20
USE_PULLBACK = True
SL_LOOKBACK = 12
CURRENCY_PAIR = CurrencyPairs.EUR_USD
FILE_SPECIFIER = 'bars'
DATE_RANGE = '2018-2020'

bars_strategy = DirectionalBarsStrategy(SL_LOOKBACK, RISK_REWARD_RATIO, SPREAD_PERCENTAGE, EACH_BAR, N_BARS,
                                        PIP_MOVEMENT, USE_PULLBACK, SL_LOOKBACK)
aat_trainer = AatMarketTrainer(CURRENCY_PAIR, FILE_SPECIFIER)

results = bars_strategy.run_strategy(CURRENCY_PAIR, DATE_RANGE, aat_trainer=aat_trainer)
print(results)
