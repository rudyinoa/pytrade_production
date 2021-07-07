
# Binance Live Trading
CREDENTIALS_FILE = 'credentials.txt'
CREDENTIALS_NAME = 'binance_testnet_futures_user'
CHECK_FOR_OPEN_POSITIONS = False
DSTYPE = 'broker'
BROKER = 'binance-futures'
TESTNET = True
SYMBOL = 'BTCUSDT'
TYPE = 'FUTURES'
INTERVAL = '5m'
STARTDATE = '1 day ago -0400' # https://dateparser.readthedocs.io/en/latest/
ENDDATE = '10 mins ago -0400' # DO NOT USE CURRENT TIME
FETCH_TIME_START = '10 mins ago -0400'
FETCH_TIME_END = 'now -0400'
STRATEGY = 'DoubleSMACrossover(13,21)'
LOOP_WAIT_TIME = 300
RETRY_TIME = 1
MAX_RETRIES = 1
LEVERAGE = 30
MARGIN_MODE = 'ISOLATED'
ORDER_SIZE = 0.01
SL_PCT = 0.01
TP_PCT = 0
PRECISION = 100 # For calculating Stop Loss and Take Profit orders price

# Email configuration
EMAIL_PORT = 465  # For SSL
SMTP_SERVER = "smtp.gmail.com"
RECEIVER_EMAIL = "botpytrade@gmail.com"

