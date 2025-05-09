# old, wrong one:
# pip install git+https://github.com/irgb/yfinance.git@4bc7eb174f14c4c668fd2bf94b865717797b4ea7#egg=yfinance
# new, correct one (newer commit with hash 0a71f93e925274f770302af640548811621b83fb)
# pip install git+https://github.com/ranaroussi/yfinance.git@0a71f93e925274f770302af640548811621b83fb#egg=yfinance
# --no-cache-dir and --force-reinstall
# still gave error

# now, uninstall yfinance first
# pip uninstall yfinance

# next step: use --no-cache-dir and --force-reinstall on the specific link given by the guy on stackoverflow
# pip install --force-reinstall --no-cache-dir git+https://github.com/irgb/yfinance.git@4bc7eb174f14c4c668fd2bf94b865717797b4ea7#egg=yfinance


import yfinance as yf
from curl_cffi import requests

instrumentIds = ['SPY']
startDateStr = '2008-10-01'
endDateStr = '2018-10-02' 

# session = requests.Session(impersonate="chrome") # custom fix for rate limiting, based on issue: https://github.com/ranaroussi/yfinance/issues/2422
data = yf.download(
    tickers=instrumentIds,
    start=startDateStr,
    end=endDateStr,
    group_by='ticker',
    auto_adjust=False,
    threads=True,
    # session=session
)
print(data)