from aat.aat_market_tester import AatRfMarketTesterForCnnModel
from market_proxy.currency_pairs import CurrencyPairs
from strategy.cnn_strategy import CnnStrategy, CNN_LOOKBACK

RISK_REWARD_RATIO = 1.5
SPREAD_PERCENTAGE = 0.10
STOPLOSS_LOOKBACK = 12
CURRENCY_PAIR = CurrencyPairs.EUR_USD
PROBA_THRESHOLD = 0.0
DATE_RANGE = '2020-2022'

cnn_strategy = CnnStrategy(CNN_LOOKBACK, RISK_REWARD_RATIO, CURRENCY_PAIR, PROBA_THRESHOLD,
                           STOPLOSS_LOOKBACK, SPREAD_PERCENTAGE)
aat_tester = AatRfMarketTesterForCnnModel(CURRENCY_PAIR, True, 5)

results = cnn_strategy.run_strategy(CURRENCY_PAIR, DATE_RANGE, aat_tester=aat_tester)
print(results)
aat_tester.print_results()
aat_tester.communicate_outlier_results()
