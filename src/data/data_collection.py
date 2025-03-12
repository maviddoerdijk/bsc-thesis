import yfinance as yf
import pandas as pd
from config.config import START_DATE, END_DATE, ETF_LIST

def download_etf_data(etf, start=START_DATE, end=END_DATE):
    """
    Downloads historical ETF data from Yahoo Finance.
    """
    df = yf.download(etf, start=start, end=end)
    df["Ticker"] = etf
    return df

def collect_data(etf_list=ETF_LIST):
    """
    Collects historical data for a list of ETFs and concatenates into a single DataFrame.
    """
    all_data = []
    for etf in etf_list:
        df = download_etf_data(etf)
        all_data.append(df)
    combined_df = pd.concat(all_data)
    return combined_df

if __name__ == "__main__":
    data = collect_data()
    print("Data shape:", data.shape)