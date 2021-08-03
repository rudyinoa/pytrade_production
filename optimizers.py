import pandas as pd
import numpy as np
import utils as ut
import strategies
import time
import sys
from IPython.display import display


class SimpleOptimizer():
    """
    Very simple optimizer class, trying to maximize profits
    with the least amount of trades.
    
    Arguments (all required):
        - df:           (pandas df) df to test on.
        - strategy:     (str) strategy class name with no arguments. EX: 'BollingerBands'
        - min_st_args:  (list of floats) minimum values for strategy arguments. EX: [0, 0.1]
        - max_st_args:  (list of floats) maximum values for strategy arguments. EX: [10, 1]
        - increments:   (list of floats) setp increments for strategy arguments. EX: [1, 0.1]       
        """


    def __init__(self, df, strategy, min_st_args, max_st_args, increments):
        self.df = df
        self.strategy = strategy
        self.min_st_args = min_st_args
        self.max_st_args = max_st_args
        self.increments = increments

    
    def run(self):
        
        # Create dataframe with possible argument values
        self.pos_args = []
        for n in range(len(self.min_st_args)):
            self.min = self.min_st_args[n]
            self.max = self.max_st_args[n]
            self.step = self.increments[n]
            arg_df = pd.DataFrame({f'ARG_{n}': np.arange(self.min, self.max, self.step)})
            self.pos_args.append(arg_df)
        self.opt_df = self.pos_args[0]
        if len(self.min_st_args) > 1:
            for n in range(1, len(self.min_st_args)):
                self.opt_df = pd.merge(self.opt_df, self.pos_args[n], how='cross')
        self.opt_df = self.opt_df.applymap(str)
        self.arg_column_names = [col for col in self.opt_df.columns if 'ARG' in col]
        self.opt_df['ARGS'] = self.opt_df[self.arg_column_names].agg(','.join, axis=1)
        self.opt_df['STRATEGY_NAME'] = self.strategy + '(' + self.opt_df['ARGS'] + ')'

        # Run the strategy for every combination
        def get_st_results(ser):
            st, st_args = ut.get_strategy(ser['STRATEGY_NAME'])
            bt = st(self.df, *st_args)
            bt.run_strategy()
            results = bt.get_data()

            ser['STOCK_GROSS_PERFORMANCE'] = results['stock_gross_performance']
            ser['STRATEGY_GROSS_PERFORMANCE'] = results['strategy_gross_performance']
            ser['NUMBER_OF_TRADES'] = results['number_of_trades']
            ser['NUMBER_OF_PROFIT_TRADES'] = results['number_of_profit_trades']
            ser['NUMBER_OF_LOSS_TRADES'] = results['number_of_loss_trades']
            ser['PROFIT'] = results['profit']
            ser['LOSS'] = results['loss']
            ser['FEES'] = results['fees']
            ser['TOTAL_PROFIT'] = results['total_profit']
            
            return ser

        self.opt_df = self.opt_df.apply(get_st_results, axis=1)

        # Sort by max profit
        self.opt_df = self.opt_df.sort_values(by=['TOTAL_PROFIT'],
            ascending=[False])

        # Reset index
        self.opt_df = self.opt_df.reset_index(drop=True)


    def show_results(self, n=10):
        """Show top n results from the sorted opt_df."""
        print('----- Top', n, 'strategies:')
        display(self.opt_df.iloc[:n,:])


    def show_strategy_results(self, n=0):
        """n is strategy number from the sorted opt_df starting from 0."""
        print(f'\n----- Showing results for strategy {n}:\n')
        st, st_args = ut.get_strategy(self.opt_df['STRATEGY_NAME'][n])
        bt = st(self.df, *st_args)
        bt.run_strategy()
        bt.show_results()

    def get_data(self):
        return {'opt_df': self.opt_df}
            


