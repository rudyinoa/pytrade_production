import utils as ut; from config import *
import pandas as pd; import binance as bn
from datetime import datetime; import time; import logging; import pprint


# Configure pprint
pp = pprint.PrettyPrinter().pprint

# Global variables
client = None
action = None
last_action = None
df = None
loop_count = 1
strategy = None
last_day = None


def setup():

    global client, df, strategy

    # Create client
    logging.info('Pytrade: Creating client.')
    api_key = ut.get_config(CREDENTIALS_FILE, CREDENTIALS_NAME, 'api_key') 
    api_secret = ut.get_config(CREDENTIALS_FILE, CREDENTIALS_NAME, 'secret_key')
    client = bn.Client(api_key, api_secret, testnet=TESTNET)

    # Set margin mode
    logging.info(f'Pytrade: Setting Margin Mode to {MARGIN_MODE}.')
    try: 
        client.futures_change_margin_type(symbol=SYMBOL, marginType=MARGIN_MODE)
    except bn.exceptions.BinanceAPIException as e: #bn.exceptions.BinanceAPIException
        if 'APIError(code=-4046)' in str(e):
            logging.info(f'Pytrade: {MARGIN_MODE} Margin Mode was already set in binance.')

    # Set leverage
    logging.info(f'Pytrade: Setting leverage to {LEVERAGE}x.')
    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)

    # Get past price data
    logging.info(f'Pytrade: Getting past price data for {SYMBOL} {TYPE} using {INTERVAL} interval from {STARTDATE} to {ENDDATE}.')
    df = ut.data_loader(DSTYPE, SYMBOL, BROKER, STARTDATE, ENDDATE, INTERVAL, client)

    # Safety check for ENDDATE variable
    if 'now' in ENDDATE:
        logging.error(f'Pytrade: ENDDATE parameter cannot be "{ENDDATE}". Please change it in config.py.')
        quit()

    # Check for open positions
    if CHECK_FOR_OPEN_POSITIONS:
        logging.info(f'Pytrade: Checking if there are open positions for {SYMBOL}.')
        open_positions = client.futures_position_information(symbol=SYMBOL)
        if float(open_positions[0]['positionAmt']) != 0:
            logging.error('There are open positions. Please close them manually to use the bot.')
            quit()

    # Get strategy
    if DOS:
        strategy = ut.daily_optimize_strategy(df)
    else:
        strategy = STRATEGY

    print(strategy)


def loop():

    global df, action, last_action, loop_count, strategy, last_day

    # Get new price data
    logging.info(f'Pytrade: Getting current price data since {FETCH_TIME_START}.')
    def _new_data(): 
        return ut.data_loader(
            DSTYPE, SYMBOL, BROKER, FETCH_TIME_START, FETCH_TIME_END, INTERVAL, client)
    new_data = ut.retry(_new_data,
        f'Pytrade: Could not get new price data. Trying again in {RETRY_TIME} seconds.',
        'Pytrade: Could not get new price data. Closing Pytrade.')
    new_data = new_data.iloc[:-1, :] # Discard last bar because its not fully closed
    df = pd.concat([df, new_data])
    df.drop_duplicates(inplace=True)
    df = df.iloc[max(0, len(df) - MAX_BARS):, :] # Keep max bars

    # DOS
    if DOS:
        current_day = datetime.strftime(df.index[-1], format='%Y-%m-%d')
        if current_day != last_day and last_day != None:
            logging.info(f'Pytrade: Running DOS.')
            strategy = ut.daily_optimize_strategy(df)
        last_day = current_day

    # Run strategy
    logging.info(f'Pytrade: Running strategy {strategy}.')
    st, st_args = ut.get_strategy(strategy)
    bt = st(df, *st_args)
    bt.run_strategy()

    # Check for actions
    logging.info(f'Pytrade: Checking for actions.')
    df_processed = bt.get_data()['df']
    buy_signal = df_processed['BUY_SIGNAL'][-1]
    sell_signal = df_processed['SELL_SIGNAL'][-1]
    neutral_signal = df_processed['NEUTRAL_SIGNAL'][-1]
    if last_action == None:
        if buy_signal != 'nan':
            action = 'BUY'
        elif sell_signal != 'nan':
            action = 'SELL'
    elif last_action == 'BUY':
        if sell_signal != 'nan':
            action = 'SELL'
    elif last_action == 'SELL':
        if buy_signal != 'nan':
            action = 'BUY'
    if action != last_action:
        logging.info(f'Pytrade: Executing {action} order for {SYMBOL}.')
        ut.binance_create_futures_order_with_tpsl(
            client, SYMBOL, action, ORDER_SIZE, SL_PCT, TP_PCT, PRECISION)
        last_action = action

    # Wait
    logging.info(f'Pytrade: Entering wait mode for {LOOP_WAIT_TIME} seconds.')
    time.sleep(LOOP_WAIT_TIME)


if __name__ == '__main__':

    try:

        # Configure logging
        log_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/pytrade_{log_date}.log"
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s [%(threadName)-10.10s] [%(levelname)-7.7s]  %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()])

        # Run setup one time
        logging.info('Pytrade: Running setup.')
        setup()

        # Info
        logging.info('Pytrade: Starting bot. *******************************************************************')
        logging.info(f'Pytrade: Broker {BROKER}')
        logging.info(f'Pytrade: Testnet {TESTNET}')
        logging.info(f'Pytrade: DOS Mode {DOS}')
        logging.info(f'Pytrade: Symbol {SYMBOL}')
        logging.info(f'Pytrade: History Data Start Date {STARTDATE}')
        logging.info(f'Pytrade: History Data End Date {ENDDATE}')
        logging.info(f'Pytrade: Interval {INTERVAL}')
        logging.info(f'Pytrade: Fetch Time Start {FETCH_TIME_START}')
        logging.info(f'Pytrade: Fetch Time End {FETCH_TIME_END}')
        logging.info(f'Pytrade: Strategy {strategy}')
        logging.info(f'Pytrade: Leverage {LEVERAGE}')
        logging.info(f'Pytrade: Margin Mode {MARGIN_MODE}')
        logging.info(f'Pytrade: Order Size {ORDER_SIZE}')
        logging.info(f'Pytrade: Stop Loss {SL_PCT * 100}%')
        logging.info(f'Pytrade: Take Profit {TP_PCT * 100}%')
        logging.info(f'Pytrade: Price Precision {PRECISION}')

        # Send startup email
        ut.send_email('Pytrade started succesfully!', 'Good luck!', [log_file, 'config.py'])

        # Run loop
        logging.info('Pytrade: Running in loop mode.')
        while True:
            loop()

    except KeyboardInterrupt as e:
        logging.info(f'Pytrade: Exiting pytrade because of a KeyboardInterrupt.')
    except:
        error_msg = 'Exiting pytrade because an exception ocurred and was not handled properly.'
        logging.critical(f'Pytrade: {error_msg}', exc_info=True)
        ut.send_email('Pytrade stoped working', error_msg, log_file)
  