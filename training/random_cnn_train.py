from cnn.cnn_market_trainer import CnnMarketTrainer, CNN_LOOKBACK
from market_proxy.currency_pairs import CurrencyPairs
from strategy.random_strategy import RandomStrategy

RISK_REWARD_RATIO = 1.5
SPREAD_PERCENTAGE = 0.10
STOPLOSS_LOOKBACK = 12
CURRENCY_PAIR = CurrencyPairs.EUR_USD
DATE_RANGE = '2018-2020'
TRAINING_DATA_PERCENTAGE = 0.7

random_strategy = RandomStrategy(CNN_LOOKBACK, RISK_REWARD_RATIO, SPREAD_PERCENTAGE, STOPLOSS_LOOKBACK)
cnn_trainer = CnnMarketTrainer(TRAINING_DATA_PERCENTAGE, CURRENCY_PAIR, DATE_RANGE)

results = random_strategy.run_strategy(CURRENCY_PAIR, DATE_RANGE, cnn_trainer)
print(results)

cnn_trainer.save_data()

cnn_trainer.train()
