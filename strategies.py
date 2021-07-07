import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import pyfolio as pf
from IPython.display import display


class _BaseStrategy():
    def __init__(self, df):
        """Basic init, should be modified to accept the strategy arguments."""
        self.df = df


    def _strategy(self):
        """This is where you define the strategy. Should have:
            - self.indicator: indicator name (string)
            - self.df.['POSITION']: 1 for long and -1 for short (series)"""
        pass


    def _calculate_buy_sell_signals(self):

        # Calculate Buy and Sell signals
        self.df['BUY_SIGNAL'] = np.where(
            (self.df['POSITION'] == 1) & (self.df['POSITION'].shift(1) != 1),
            'BUY', np.nan)
        self.df['SELL_SIGNAL'] = np.where(
            (self.df['POSITION'] == -1) & (self.df['POSITION'].shift(1) != -1),
            'SELL', np.nan)
        self.df['NEUTRAL_SIGNAL'] = np.where(
            (self.df['POSITION'] == 0) & \
            (self.df['POSITION'].shift(1) == 0) & \
            (self.df['POSITION'].shift(2) != 0),
            'NEUTRAL', np.nan)
        self.df['BUY_PRICE'] = np.where(self.df['BUY_SIGNAL'].shift(1) == 'BUY', self.df['CLOSE'], np.nan)
        self.df['SELL_PRICE'] = np.where(self.df['SELL_SIGNAL'].shift(1) == 'SELL', self.df['CLOSE'], np.nan)
        self.df['NEUTRAL_PRICE'] = np.where(self.df['NEUTRAL_SIGNAL'].shift(1) == 'NEUTRAL', self.df['CLOSE'], np.nan)


    def _calculate_stats(self):

        # Calculate simple benchmarks
        self.df['STOCK_RETURNS'] = np.log(self.df['CLOSE'] / self.df['CLOSE'].shift(1))
        self.df['STRATEGY_RETURNS'] = self.df['POSITION'].shift(1) * self.df['STOCK_RETURNS']

        # Stock vs Strategy Gross Performance
        self.stock_gross_performance = float(self.df[['STOCK_RETURNS']].sum().apply(np.exp))
        self.strategy_gross_performance = float(self.df[['STRATEGY_RETURNS']].sum().apply(np.exp))

        # PnL
        self._calculate_pnl_stats()


    def _calculate_pnl_stats(self):

        # Get prices
        buy_prices = self.df[['BUY_PRICE']].dropna()
        buy_prices['ACTION'] = 'BUY'
        buy_prices.columns = ['PRICE', 'ACTION']
        sell_prices = self.df[['SELL_PRICE']].dropna()
        sell_prices['ACTION'] = 'SELL'
        sell_prices.columns = ['PRICE', 'ACTION']
        neutral_signals = self.df[['NEUTRAL_PRICE']].dropna()
        neutral_signals['ACTION'] = 'NEUTRAL'
        neutral_signals.columns = ['PRICE', 'ACTION']

        # Get signals
        signals = pd.concat(
            [buy_prices, sell_prices, neutral_signals])
        signals = signals.sort_index()
        signals = signals.reset_index().reset_index()

        # Get post signals
        post_signals = signals.copy()
        post_signals['index'] = post_signals['index'] - 1
        post_signals.columns = ['index', 'POST_DATE', 'POST_PRICE', 'POST_ACTION']

        # Get trades
        trades = pd.merge(signals, post_signals, on='index', how='left')
        trades = trades.dropna()
        del trades['index']

        # Drop trades that are caused by market neutral actions
        trades = trades[trades['ACTION'] != trades['POST_ACTION']]
        trades = trades[~(trades['ACTION'] == 'NEUTRAL')]
        # trades = trades[~(trades['POST_ACTION'] == 'NEUTRAL')]

        # Add calculations to the trades
        def calc_trade_profit(action, cost, post_cost):
            if action == 'BUY':
                return post_cost - cost
            else:
                return cost - post_cost

        trades['UNITS'] = 200 / trades['PRICE'] # 200 dollars per trade
        trades['COST'] = trades['UNITS'] * trades['PRICE']
        trades['POST_COST'] = trades['UNITS'] * trades['POST_PRICE']
        trades['SUBTOTAL'] = 0
        if len(trades) > 0:
            trades['SUBTOTAL'] = trades.apply(
                lambda x: calc_trade_profit(x['ACTION'], x['COST'], x['POST_COST']),
                axis=1)
        trades['PROFIT'] = trades[trades['SUBTOTAL'] > 0][['SUBTOTAL']]
        trades['LOSS'] = trades[trades['SUBTOTAL'] <= 0][['SUBTOTAL']]
        trades['FEES'] = ((trades['COST'] + trades['POST_COST']) / 2) * 0.0004 * 2 # Binance Futures fees
        trades['TRADE_PROFIT'] = trades['PROFIT'].fillna(0) + trades['LOSS'].fillna(0) - trades['FEES'].fillna(0)
        del trades['SUBTOTAL']
        

        # Calculate PnL metrics
        self.trades = trades
        self.number_of_trades = len(trades)
        self.number_of_profit_trades = trades['PROFIT'].count()
        self.number_of_loss_trades = trades['LOSS'].count()
        self.profit = trades['PROFIT'].sum()
        self.loss = trades['LOSS'].sum()
        self.fees = trades['FEES'].sum()
        self.total_profit = self.profit + self.loss - self.fees

    
    def _show_pyfolio_tearsheet(self, benchmark_returns=None):
        pf.create_returns_tear_sheet(self.df['STRATEGY_RETURNS'],
            benchmark_rets=self.df['STOCK_RETURNS'])


    def run_strategy(self):
        self.df = self.df.dropna()
        self._strategy()
        self._calculate_buy_sell_signals()
        self._calculate_stats()


    def get_data(self):
        """Returns a dictionary with data used by the strategy"""

        return {'df': self.df,
                'stock_returns': self.df['STOCK_RETURNS'],
                'strategy_returns': self.df['STRATEGY_RETURNS'],
                'stock_gross_performance': self.stock_gross_performance,
                'strategy_gross_performance': self.strategy_gross_performance,
                'trades': self.trades,
                'number_of_trades': self.number_of_trades,
                'number_of_profit_trades': self.number_of_profit_trades,
                'number_of_loss_trades': self.number_of_loss_trades,
                'profit': self.profit,
                'loss': self.loss,
                'fees': self.fees,
                'total_profit': self.total_profit}


    def _show_results_pnl(self):
        print('Total Profit:', self.total_profit)
        print('Profit:', self.profit)
        print('Loss:', self.loss)
        print('Fees', self.fees)
        print('Number of trades:', self.number_of_trades)
        print('Number of profitable trades:', self.number_of_profit_trades)
        print('Number of lost trades:', self.number_of_loss_trades)
        print('PnL Ratio:', self.number_of_profit_trades / self.number_of_loss_trades)
        print('Trades:'); display(self.trades)


    def _show_results_gross_performance(self, plot_pos):

        # Text
        print('Stock Gross Performance:', self.stock_gross_performance)
        print('Strategy Gross Performance:', self.strategy_gross_performance)

        # Plot
        plt.subplot(plot_pos)
        plt.plot(self.df[['STOCK_RETURNS']].cumsum().apply(np.exp),
            label='STOCK RETURNS')
        plt.plot(self.df[['STRATEGY_RETURNS']].cumsum().apply(np.exp),
            label='STRATEGY RETURNS')
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('GROSS PERFORMANCE')


    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)
        plt.plot(self.df['CLOSE'], linewidth=0.5, color='black', label='STOCK')
        plt.plot(self.df[self.indicator], label=self.indicator)
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


    def _results(self):

        f = plt.figure()
        f.set_figwidth(14)
        f.set_figheight(10)

        # PnL
        self._show_results_pnl()

        # Gross performance
        self._show_results_gross_performance(211)
        
        # Buy/Sell signals
        self._show_results_buy_sell_indicators(212)
        
        plt.show()

        # Pyfolio
        self._show_pyfolio_tearsheet()


    def show_results(self):
        self._results()


class SMA(_BaseStrategy):
    
    def __init__(self, df, n):
        self.df = df
        self.n = n


    def _strategy(self):

        # Calculate SMA
        self.indicator = f'SMA({self.n})'
        self.df[self.indicator] = self.df['CLOSE'].rolling(self.n).mean()

        # Calculate position
        self.df['POSITION'] = np.where(self.df['CLOSE'] > self.df[self.indicator], 1, -1)


class DoubleSMACrossover(_BaseStrategy):
    """n1 is the faster (smaller) SMA window size and 
       n2 is the slower (larger) SMA window size"""
    
    def __init__(self, df, n1, n2):
        self.df = df
        self.n1 = n1
        self.n2 = n2


    def _strategy(self):

        # Calculate SMA
        self.indicator_1 = f'SMA({self.n1})'
        self.indicator_2 = f'SMA({self.n2})'
        self.df[self.indicator_1] = self.df['CLOSE'].rolling(self.n1).mean()
        self.df[self.indicator_2] = self.df['CLOSE'].rolling(self.n2).mean()

        # Calculate position
        self.df['POSITION'] = np.where(self.df[self.indicator_1] > self.df[self.indicator_2], 1, -1)


    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)
        plt.plot(self.df['CLOSE'], linewidth=0.5, color='black', label='STOCK')
        plt.plot(self.df[self.indicator_1], label=self.indicator_1)
        plt.plot(self.df[self.indicator_2], label=self.indicator_2)
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


class TripleSMACrossover(_BaseStrategy):
    """n1 is the faster (smaller) SMA window size and 
       n2 is the normal (mid) SMA window size and
       n3 is the slower (larger) SMA window size."""
    
    def __init__(self, df, n1, n2, n3):
        self.df = df
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3


    def _strategy(self):

        # Calculate SMA
        self.indicator_1 = f'SMA({self.n1})'
        self.indicator_2 = f'SMA({self.n2})'
        self.indicator_3 = f'SMA({self.n3})'
        self.df[self.indicator_1] = self.df['CLOSE'].rolling(self.n1).mean()
        self.df[self.indicator_2] = self.df['CLOSE'].rolling(self.n2).mean()
        self.df[self.indicator_3] = self.df['CLOSE'].rolling(self.n3).mean()

        # Calculate position
        self.df['LONG_POSITION'] = np.where(
            (self.df[self.indicator_1] > self.df[self.indicator_2]) & \
            (self.df[self.indicator_2] > self.df[self.indicator_3]), 
            1,0)
        self.df['SHORT_POSITION'] = np.where(
            (self.df[self.indicator_1] < self.df[self.indicator_2]) & \
            (self.df[self.indicator_2] < self.df[self.indicator_3]), 
            -1,0)
        self.df['POSITION'] = self.df['LONG_POSITION'] + self.df['SHORT_POSITION']
        self.df['POSITION'] = self.df['POSITION'].replace(0, np.nan)
        self.df['POSITION'].ffill(inplace=True)
        # print(self.df['POSITION'])


    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)
        plt.plot(self.df['CLOSE'], linewidth=0.5, color='black', label='STOCK')
        plt.plot(self.df[self.indicator_1], label=self.indicator_1)
        plt.plot(self.df[self.indicator_2], label=self.indicator_2)
        plt.plot(self.df[self.indicator_3], label=self.indicator_3)
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


class Momentum(_BaseStrategy):
    
    def __init__(self, df, n):
        self.df = df
        self.n = n


    def _strategy(self):

        # Calculate Momentum
        self.indicator = f'Momentum({self.n})'
        self.df['returns'] = np.log(self.df['CLOSE'] / self.df['CLOSE'].shift(1))
        self.df[self.indicator] = np.sign(self.df['returns'].rolling(self.n).mean())

        # Calculate position
        self.df['POSITION'] = self.df[self.indicator]

    
    # Do not plot the indicator, only the buy/sell signals
    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)
        plt.plot(self.df['CLOSE'], linewidth=0.5, color='black', label='STOCK')
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


class MeanReversion(_BaseStrategy):
    
    def __init__(self, df, n, th):
        self.df = df
        self.n = n # SMA
        self.th = th # Absolute deviation from the mean


    def _strategy(self):

        # Calculate SMA
        self.indicator = f'MeanReversion({self.n}, {self.th})'
        self.df[f'SMA({self.n})'] = self.df['CLOSE'].rolling(self.n).mean()
        self.df['DISTANCE'] = self.df['CLOSE'] - self.df[f'SMA({self.n})']

        # Calculate position
        self.df['POSITION'] = np.where(self.df['DISTANCE'] >= self.th,
            -1, np.nan) # short
        self.df['POSITION'] = np.where(self.df['DISTANCE'] <= -self.th,
            1, self.df['POSITION']) # long
        self.df['POSITION'] = np.where(
            self.df['DISTANCE'] * self.df['DISTANCE'].shift(1) < 0, 0,
            self.df['POSITION']) # Go market neutral if no change in sign
        self.df['POSITION'] = self.df['POSITION'].ffill().fillna(0)

    
    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)
        plt.plot(self.df['CLOSE'], linewidth=0.5, color='black', label='STOCK')
        plt.plot(self.df[f'SMA({self.n})'], linewidth=1, color='blue', label=f'SMA({self.n})')
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


    def _show_results_distance(self, plot_pos):
        plt.subplot(plot_pos)
        self.df['DISTANCE'].plot(legend=True)
        plt.scatter(self.df.index, (self.df['BUY_PRICE'] * 0) + self.df['DISTANCE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, (self.df['SELL_PRICE'] * 0) + self.df['DISTANCE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.axhline(self.th, color='black')
        plt.axhline(-self.th, color='black')
        plt.axhline(0, color='black')
        plt.xlabel('TIME')
        plt.title('DISTANCE')
        plt.legend()


    def _results(self):

        f = plt.figure()
        f.set_figwidth(14)
        f.set_figheight(15)

        # PnL
        self._show_results_pnl()

        # Gross performance
        self._show_results_gross_performance(311)
        
        # Buy/Sell signals
        self._show_results_buy_sell_indicators(312)

        # Show additional plots
        self._show_results_distance(313)
        
        plt.show()

        # Pyfolio
        self._show_pyfolio_tearsheet()


class BollingerBands(_BaseStrategy):
    
    def __init__(self, df, n, std):
        self.df = df
        self.n = n
        self.std = std


    def _strategy(self):

        # Calculate BollingerBands
        self.indicator = f'BollingerBands({self.n}, {self.std})'
        ta_indicator = ta.volatility.BollingerBands(
            self.df['CLOSE'], window=self.n,
            window_dev=self.std)
        self.df[self.indicator] = ta_indicator.bollinger_hband_indicator()
        self.df['HI_BAND'] = ta_indicator.bollinger_hband()
        self.df['LO_BAND'] = ta_indicator.bollinger_lband()

        # Calculate position
        self.df['POSITION'] = np.where(
            (self.df['CLOSE'] <= self.df['HI_BAND']) & \
            (self.df['CLOSE'].shift(1) >= self.df['HI_BAND']),
            -1, np.nan) # short
        self.df['POSITION'] = np.where(
            (self.df['CLOSE'] >= self.df['LO_BAND']) & \
            (self.df['CLOSE'].shift(1) <= self.df['LO_BAND']),
            1, self.df['POSITION']) # long
        self.df['POSITION'] = self.df['POSITION'].ffill()


    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)

        # Close
        plt.plot(self.df['CLOSE'], linewidth=1,
            color='black', label='STOCK')

        # Bollinger Bands
        plt.plot(self.df['HI_BAND'], linewidth=0.5,
            color='red', label=f'SMA({self.n})')
        plt.plot(self.df['LO_BAND'], linewidth=0.5,
            color='red', label=f'SMA({self.n})')

        # Everything else
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


class DoubleSMACrossoverLong(_BaseStrategy):
    """n1 is the faster (smaller) SMA window size and 
       n2 is the slower (larger) SMA window size"""
    
    def __init__(self, df, n1, n2):
        self.df = df
        self.n1 = n1
        self.n2 = n2


    def _strategy(self):

        # Calculate SMA
        self.indicator_1 = f'SMA({self.n1})'
        self.indicator_2 = f'SMA({self.n2})'
        self.df[self.indicator_1] = self.df['CLOSE'].rolling(self.n1).mean()
        self.df[self.indicator_2] = self.df['CLOSE'].rolling(self.n2).mean()

        # Calculate position
        self.df['POSITION'] = np.where(self.df[self.indicator_1] > self.df[self.indicator_2], 1, 0)


    def _show_results_buy_sell_indicators(self, plot_pos):
        plt.subplot(plot_pos)
        plt.plot(self.df['CLOSE'], linewidth=0.5, color='black', label='STOCK')
        plt.plot(self.df[self.indicator_1], label=self.indicator_1)
        plt.plot(self.df[self.indicator_2], label=self.indicator_2)
        plt.scatter(self.df.index, self.df['BUY_PRICE'], 
            label='BUY', color='green', s=25, marker="^")
        plt.scatter(self.df.index, self.df['SELL_PRICE'], 
            label='SELL', color='red', s=25, marker="v")
        plt.xlabel('TIME')
        plt.ylabel('CLOSE PRICE')
        plt.legend()
        plt.title('BUY/SELL MOMENTS')


