from pandas import DataFrame
from market_proxy.trades import TradeType
from ml_models.learner import Learner
import numpy as np
from strategy.strategy_class import Strategy
from strategy.strategy_results import StrategyResults
from typing import Optional


class MarketSimulator(object):
    @staticmethod
    def run_simulation(strategy: Strategy, market_data: DataFrame,
                       learner: Optional[Learner] = None) -> StrategyResults:
        print(f'Running simulation for strategy with description: {strategy.description}')

        reward, n_wins, n_losses, win_streak, loss_streak, curr_win_streak, curr_loss_streak, n_buys, n_sells, \
            day_fees = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Numerical results we keep track of
        pips_risked, trade, n_candles = [], None, 0

        for idx in range(strategy.starting_idx, len(market_data)):
            n_candles += 1

            # If there is no open trade, check to see if we should place one
            if trade is None:
                trade = strategy.place_trade(idx, market_data)

                if trade is not None:
                    pips_risked.append(trade.pips_risked)
                    n_candles = 0

                    if trade.trade_type == TradeType.BUY:
                        n_buys += 1

                    elif trade.trade_type == TradeType.SELL:
                        n_sells += 1

                    else:
                        raise Exception(f'Invalid trade type on the following trade: {trade}')

            # If there is an open trade, check to see if it should close out (there are 4 conditions)
            # For each condition, we set the trade's end date; update the reward, num wins, num losses, etc.; close
            # the trade (set it to None); and continue to the next iteration in the simulation loop (continue to the
            # next candle)
            if trade is not None:
                stop_idx = min(idx + 1, len(market_data)) if learner is None else len(market_data)

                for j in range(idx, stop_idx):
                    curr_bid_open, curr_bid_high, curr_bid_low, curr_ask_open, curr_ask_high, curr_ask_low, \
                        curr_mid_open, curr_date = market_data.loc[market_data.index[j], ['Bid_Open', 'Bid_High',
                                                                                          'Bid_Low', 'Ask_Open',
                                                                                          'Ask_High', 'Ask_Low',
                                                                                          'Mid_Open', 'Date']]

                    # Determine if trade should close out; if so, calculate the profit, day fees, etc.
                    trade.calculate_trade(curr_bid_low, curr_bid_high, curr_ask_low, curr_ask_high, curr_date)

                    # Trade closed out
                    if trade.end_date is not None:
                        reward += trade.reward
                        day_fees += trade.day_fees
                        net_profit = trade.net_profit

                        n_wins += 1 if net_profit > 0 else 0
                        n_losses += 1 if net_profit < 0 else 0
                        curr_win_streak = 0 if net_profit < 0 else curr_win_streak + 1
                        curr_loss_streak = 0 if net_profit > 0 else curr_loss_streak + 1

                        if curr_win_streak > win_streak:
                            win_streak = curr_win_streak

                        if curr_loss_streak > loss_streak:
                            loss_streak = curr_loss_streak

                        if learner is not None:
                            learner.trade_finished(net_profit, trade.start_date, trade.trade_type)

                        trade = None

                        break

        # Return the simulation results once we've iterated through all the data
        avg_pips_risked = np.array(pips_risked).mean() if len(pips_risked) > 0 else np.nan

        results = StrategyResults(reward, day_fees, reward + day_fees, avg_pips_risked, n_buys, n_sells, n_wins,
                                  n_losses, win_streak, loss_streak)

        return results
