
# Binance Live Trading
CREDENTIALS_FILE = 'credentials.txt'
CREDENTIALS_NAME = 'binance_user'
CHECK_FOR_OPEN_POSITIONS = False
DSTYPE = 'broker'
BROKER = 'binance-futures'
TESTNET = False
DOS = True # Daily Optimized Strategy
SYMBOL = 'BTCUSDT'
TYPE = 'FUTURES'
INTERVAL = '5m'
STARTDATE = '5 days ago -0400' # https://dateparser.readthedocs.io/en/latest/
ENDDATE = '10 mins ago -0400' # DO NOT USE CURRENT TIME
FETCH_TIME_START = '15 mins ago -0400'
FETCH_TIME_END = 'now -0400'
STRATEGY = 'DoubleSMACrossover(13,21)'
LOOP_WAIT_TIME = 60
RETRY_TIME = 5
MAX_RETRIES = 3
LEVERAGE = 10
MARGIN_MODE = 'ISOLATED'
ORDER_SIZE = 0.001
SL_PCT = 0.005
TP_PCT = 0.02
PRECISION = 100 # For calculating Stop Loss and Take Profit orders price
MAX_BARS = 1000

# Email configuration
EMAIL_PORT = 465  # For SSL
SMTP_SERVER = "smtp.gmail.com"
RECEIVER_EMAIL = "botpytrade@gmail.com"

# Backtest
BACKTEST_ORDER_SIZE = 0.01
ENABLE_BACKTEST_SLTP = True
BACKTEST_SL_PCT = 0.005
BACKTEST_TP_PCT = 0.02

# Daily Optimized Strategy (DOS)
BACK_DAYS = 1
DAILY_OPT_STRATEGY = 'MeanReversion'
MIN_ARGS = [8,100]
MAX_ARGS = [15,1000]
STEP_ARGS = [1,100]
DEFAULT_DOS_STRATEGY = 'MeanReversion(7,100)'

