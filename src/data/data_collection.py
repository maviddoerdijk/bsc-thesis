import pandas as pd
import yfinance as yf
from curl_cffi import requests


def gather_data(startDateStr, endDateStr, instrumentIds):
    session = requests.Session(impersonate="chrome") # custom fix for rate limiting, based on issue: https://github.com/ranaroussi/yfinance/issues/2422
    data = yf.download(
        tickers=instrumentIds,
        start=startDateStr,
        end=endDateStr,
        group_by='ticker',
        auto_adjust=False,
        threads=True,
        session=session
    )
    data_close = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in instrumentIds if ticker in data.columns.levels[0]})
    data_open = pd.DataFrame({ticker: data[ticker]['Open'] for ticker in instrumentIds if ticker in data.columns.levels[0]})
    data_high = pd.DataFrame({ticker: data[ticker]['High'] for ticker in instrumentIds if ticker in data.columns.levels[0]})
    data_low = pd.DataFrame({ticker: data[ticker]['Low'] for ticker in instrumentIds if ticker in data.columns.levels[0]})
    data_vol = pd.DataFrame({ticker: data[ticker]['Volume'] for ticker in instrumentIds if ticker in data.columns.levels[0]})
    
    return {
        'close': data_close,
        'open': data_open,
        'high': data_high,
        'low': data_low,
        'vol': data_vol,
        'yfinance_formatted': data
    }
    
if __name__ == "__main__":
    startDateStr = '2008-10-01'
    endDateStr = '2018-10-02' # documentation said that endDateStr is exclusive for both yahoofinance and the original code, but actually printing the shapes showed otherwise..
    instrumentIds = list(set(['ITOT', 'ACWI', 'IWV', 'VT', 'VTI',
                    'DIA', 'RSP', 'IOO', 'IVV', 'SPY',
                    'SHE', 'IWM', 'OEF', 'QQQ',
                    'CVY', 'RPG', 'RPV', 'IWB', 'IWF',
                    'IWD', 'IVW', 'IVE', 'PKW',
                    'PRF', 'SDY', 'VV', 'VUG',
                    'VTV', 'MGC', 'MGK', 'MGV', 'VIG',
                    'VYM', 'DTN', 'DLN', 'MDY', 'DVY',
                    'IWR', 'IWP', 'IWS', 'IJH', 'IJK',
                    'IJJ', 'PDP', 'DON', 'IWC', 'IWM',
                    'IWO', 'IWN', 'IJR', 'IJT', 'IJS',
                    'EEB', 'IDV', 'ACWX', 'BKF', 'EFA',
                    'EFG', 'EFV', 'SCZ', 'EEM', 'PID',
                    'DWX', 'DEM', 'DGS', 'AAXJ', 'EZU',
                    'EPP', 'IEV', 'ILF', 'FEZ', 'VGK',
                    'VPL', 'DFE', 'EWA', 'EWC', 'EWG',
                    'EWI', 'EWJ', 'EWD', 'EWL', 'EWP',
                    'EWU', 'DXJ', 'EWZ', 'FXI', 'EWH',
                    'EWW', 'RSX', 'EWS', 'EWM','EWY',
                    'EWT', 'EPI', 'XLY', 'IYC', 'ITB',
                    'XHB', 'VCR','XLP', 'IYK', 'VDC',
                    'XLE', 'IYE', 'IGE',
                    'VDE', 'QCLN', 'XLF','IYF', 'KBE',
                    'KRE', 'VFH']))
    
    
    data = gather_data(startDateStr, endDateStr, instrumentIds)
    data_close = data['close']
    data_open = data['open']
    data_high = data['high']
    data_low = data['low']
    data_vol = data['vol']
    