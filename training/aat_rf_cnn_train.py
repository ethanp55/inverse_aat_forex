from aat.aat_market_trainer import AatRfMarketTrainerForCnnModel
from market_proxy.currency_pairs import CurrencyPairs
from strategy.cnn_strategy import CnnStrategy, CNN_LOOKBACK

RISK_REWARD_RATIO = 1.5
SPREAD_PERCENTAGE = 0.10
STOPLOSS_LOOKBACK = 12
CURRENCY_PAIR = CurrencyPairs.EUR_USD
PROBA_THRESHOLD = 0.0
DATE_RANGE = '2018-2020'

cnn_strategy = CnnStrategy(CNN_LOOKBACK, RISK_REWARD_RATIO, CURRENCY_PAIR, PROBA_THRESHOLD,
                           STOPLOSS_LOOKBACK, SPREAD_PERCENTAGE)
aat_trainer = AatRfMarketTrainerForCnnModel(CURRENCY_PAIR)

results = cnn_strategy.run_strategy(CURRENCY_PAIR, DATE_RANGE, aat_trainer=aat_trainer)
print(results)
