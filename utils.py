from os import close
import pandas as pd
import numpy as np
import binance as bn
from constants import RESAMPLE_COL_AGG
import strategies
import optimizers as op
from config import *
from email_body_templates import *
import configparser
import time
import logging
import yagmail
from datetime import datetime


def _load_csv(datasource, startdate, enddate, timeframe):

        # Load csv
        df = pd.read_csv(datasource)

        # Set all column names as uppercase
        df.columns = [x.upper() for x in df.columns]

        # Set date column as index
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        # Time slice
        if startdate is not None:
            df = df.loc[startdate:]
        if enddate is not None:
            df = df.loc[:enddate]

        # Resample
        if timeframe:
            df = resample_df(df, timeframe)

        return df


def _get_binance_history_data(datasource, startdate, enddate, timeframe, type, client):

    SYMBOL = datasource
    INTERVAL = timeframe # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    if type == 'futures':
        TYPE = bn.enums.HistoricalKlinesType.FUTURES
    else:
        TYPE = bn.enums.HistoricalKlinesType.SPOT

    # Create client
    if client == None:
        api_key = get_config('binance_user', 'api_key')
        api_secret = get_config('binance_user', 'secret_key')
        client = bn.Client(api_key, api_secret)

    # request historical candle (or klines) data
    bars = client.get_historical_klines(symbol=SYMBOL, interval=INTERVAL, 
        start_str=startdate, end_str=enddate, klines_type=TYPE)

    # Create pandas dataframe
    df = pd.DataFrame(bars)
    df = df.iloc[:,:6]
    df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    df['DATE'] = pd.to_datetime(df['DATE'], unit='ms')
    df.set_index('DATE', inplace=True)

    # Covert all values to float
    df = df.apply(pd.to_numeric)

    return df


def data_loader(dstype, datasource, broker, startdate, enddate, timeframe, client=None):

    if dstype == 'csv':
        df = _load_csv(datasource, startdate, enddate, timeframe)
    elif dstype == 'broker':
        if 'binance' in broker:
            if 'futures' in broker:
                type = 'futures'
            else:
                type = 'spot'
            df = _get_binance_history_data(datasource, startdate, enddate, 
                timeframe, type=type, client=client)

    return df


def resample_df(df, timeframe):

    resampled_df = pd.DataFrame()
    for col in df.columns:
        resampled_col = df[col].resample(timeframe)
        agg_func = RESAMPLE_COL_AGG.get(col, 'last')
        if agg_func == 'sum':
            resampled_df[col] = resampled_col.sum()
        elif agg_func == 'min':
            resampled_df[col] = resampled_col.min()
        elif agg_func == 'max':
            resampled_df[col] = resampled_col.max()
        elif agg_func == 'first':
            resampled_df[col] = resampled_col.first()
        elif agg_func == 'last':
            resampled_df[col] = resampled_col.last()

    return resampled_df


def get_strategy(strategy):

    strategy_split = strategy.split('(')

    # Get strategy
    strategy_class = strategy_split[0]
    st = getattr(strategies, strategy_class)

    # Get strategy args
    strategy_args = ()
    if len(strategy_split) > 1:
        strategy_args = strategy_split[1] \
                        .replace(')', '') \
                        .split(',')
    
    # Cast strategy args to a number
    casted_args = []
    for arg in strategy_args:
        if '.' in arg:
            try:
                casted_args.append(float(arg))
            except:
                casted_args.append(arg)
        else:
            try:
                casted_args.append(int(arg))
            except:
                casted_args.append(arg)

    return st, casted_args


def get_config(filepath, section, key):
    parser = configparser.ConfigParser()
    parser.read(filepath)
    return parser.get(section, key)


def binance_create_futures_order_with_tpsl(
        client, symbol, action, order_size, sl_pct, tp_pct, precision):
    """https://binance-docs.github.io/apidocs/futures/en/#new-order-trade""" 
    
    # Close all open orders and positions
    logging.info('Pytrade: Closing all open orders and positions.')
    def _close_open_orders_and_positions():
        client.futures_cancel_all_open_orders(symbol=symbol)
        open_positions = client.futures_position_information(symbol=symbol)
        if float(open_positions[0]['positionAmt']) != 0:
            close_position = client.futures_create_order( 
                symbol=symbol,
                side=action,
                type='MARKET',
                quantity=order_size)
            time.sleep(1)
    retry(_close_open_orders_and_positions,
        f'Pytrade: Could not close all open orders and positions. Trying again in {RETRY_TIME} seconds.',
        'Pytrade: Could not close all open orders and positions. Closing Pytrade.')
    time.sleep(1)
    
    # Create market order (TODO: use limit order for lower fees)
    logging.info('Pytrade: Creating MARKET order.')
    def _create_market_order():
        client.futures_create_order( 
            symbol=symbol,
            side=action,
            type='MARKET',
            quantity=order_size)
    retry(_create_market_order,
        f'Pytrade: Could not create {action} MARKET order. Trying again in {RETRY_TIME} seconds.',
        f'Pytrade: Could not create {action} MARKET order. Closing Pytrade.')
    time.sleep(1)

    # Get position's info
    def _get_position_data():
        open_positions = client.futures_position_information(symbol=symbol)
        price = float(open_positions[0]['entryPrice'])
        cost = float(open_positions[0]['notional'])
        margin = float(open_positions[0]['isolatedWallet'])
        open_time = datetime.fromtimestamp(open_positions[0]['updateTime']/1000)
        return price, cost, margin, open_time
    price, cost, margin, open_time = retry(_get_position_data,
        f'Pytrade: Could not get newly opened position data. Trying again in {RETRY_TIME} seconds.',
        f'Pytrade: Could not get newly opened position data. Closing Pytrade.')
    logging.info(
        f'Pytrade: Executed {action} MARKET ORDER at {open_time} with size {order_size} at {price} USD with a margin of {margin} USD.')

    # Get stop loss and take profit price
    if action == 'BUY':
        sl_price = price - price * sl_pct
        tp_price = price + price * tp_pct
    else:
        sl_price = price + price * sl_pct
        tp_price = price - price * tp_pct
    sl_price = int(sl_price * precision) / precision
    tp_price = int(tp_price * precision) / precision

    # Create stop loss and take profit orders
    other_side = 'SELL' if action == 'BUY' else 'BUY'
    sl_sign = '-' if action == 'BUY' else '+'
    tp_sign = '+' if action == 'BUY' else '-'

    if sl_pct > 0:
        logging.info(f'Pytrade: Creating STOP LOSS MARKET order with {sl_sign}{sl_pct * 100}% current price at {sl_price} USD.')
        def _create_sl_order():
            client.futures_create_order(
                symbol=symbol,
                side=other_side,
                type='STOP_MARKET',
                quantity=order_size,
                stopPrice=sl_price, 
                reduceOnly=True)
        retry(_create_sl_order,
        f'Pytrade: Could not create STOP LOSS MARKET order. Trying again in {RETRY_TIME} seconds.',
        f'Pytrade: Could not create STOP LOSS MARKET order. Closing Pytrade.')

    if tp_pct > 0:
        logging.info(f'Pytrade: Creating TAKE PROFIT MARKET order with {tp_sign}{tp_pct * 100}% current price at {tp_price} USD.')
        def _create_tp_order():
            client.futures_create_order(
                symbol=symbol,
                side=other_side,
                type='TAKE_PROFIT_MARKET',
                quantity=order_size,
                stopPrice=tp_price, 
                reduceOnly=True)
        retry(_create_tp_order,
        f'Pytrade: Could not create TAKE PROFIT MARKET order. Trying again in {RETRY_TIME} seconds.',
        f'Pytrade: Could not create TAKE PROFIT MARKET order. Closing Pytrade.')

    # Send email with trade info
    assets = client.futures_account_balance()
    for asset in assets:
        if 'USDT' in asset['asset']:
            balance = asset['balance']
    income_history = client.futures_income_history(symbol=SYMBOL)
    pnl = 0
    commissions = 0
    for trade in income_history: # Get trades in last minute
        if time.time() - trade['time']/1000 <= 60:
            if trade['incomeType'] == 'REALIZED_PNL':
                pnl = pnl + float(trade['income'])
            elif trade['incomeType'] == 'COMMISSION':
                commissions = commissions + float(trade['income'])
    mail_subject = f'New trade on {SYMBOL}'
    mail_body = EMAIL_BODY_TRADE_INFO.format(
        action, SYMBOL, open_time, price, ORDER_SIZE, cost,
        sl_price, tp_price, pnl, commissions, balance)
    send_email(mail_subject, mail_body)


def retry(func, warning_msg, error_msg):
    """Tries to run a function a couple of times. Returns whatever the function returns.
    Raises an exception and exits the program if the function does not run properly."""
    rt = MAX_RETRIES
    while True:
        try:
            return func()
        except Exception as e:
            if rt > 0:
                logging.warning(warning_msg, e)
                rt = rt - 1
                time.sleep(RETRY_TIME)
                continue
            else:
                logging.error(error_msg)
                quit()
        break


def send_email(subject, msg, attachment=None):

    logging.info(f'Pytrade: Sending email to {RECEIVER_EMAIL}.')
    sender = get_config(CREDENTIALS_FILE, 'email_credentials', 'email')
    pswd = get_config(CREDENTIALS_FILE, 'email_credentials', 'password')
    yag = yagmail.SMTP(sender, pswd)
    yag.send(
        to=RECEIVER_EMAIL,
        subject=subject,
        contents=msg, 
        attachments=attachment)


def daily_optimize_strategy(df):

    # Get daily dates
    dates = list(set(df.index.date))
    dates = [datetime.strftime(x, format='%Y-%m-%d') for x in dates]
    dates.sort()
    dates = dates[-BACK_DAYS - 1: -1]
    start_date = dates[0]
    end_date = dates[-1]

    # Optimize by the previous days
    prev_days_df = df.loc[start_date:end_date]
    opt = op.SimpleOptimizer(prev_days_df, DAILY_OPT_STRATEGY,
        MIN_ARGS, MAX_ARGS, STEP_ARGS)
    opt.run()
    opt_df = opt.get_data()['opt_df']
    try:
        opt_strategy = opt_df.iloc[0]['STRATEGY_NAME']
    except Exception as e:
        logging.warning('PyTrade: Could not optimize strategy. Using default DOS strategy.', e)
        opt_strategy = DEFAULT_DOS_STRATEGY
    logging.info(f'PyTrade: Optimized strategy using DOS -> {opt_strategy}.')
    
    return opt_strategy

    


