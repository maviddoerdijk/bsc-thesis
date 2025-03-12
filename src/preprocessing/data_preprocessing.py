import pandas as pd
from preprocessing.technical_indicators import add_technical_indicators
from preprocessing.wavelet_denoising import denoise_series

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs cleaning steps such as filtering for active ETFs and adjusting for dividends/splits.
    """
    # For simplicity, assume data is already adjusted.
    # You might want to drop rows with missing values or inactive periods.
    df = df.dropna()
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing: cleaning, adding technical indicators, and denoising.
    """
    df_clean = clean_data(df)
    df_features = add_technical_indicators(df_clean)
    # Apply denoising to the Close price column
    df_features["Close_denoised"] = denoise_series(df_features["Close"])
    return df_features

if __name__ == "__main__":
    import data.data_collection as dc
    raw_data = dc.collect_data()
    processed_data = preprocess_data(raw_data)
    print("Processed data shape:", processed_data.shape)