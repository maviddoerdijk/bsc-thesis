import pywt
import numpy as np
import pandas as pd

def denoise_series(series: pd.Series, wavelet: str = "haar", level: int = 1) -> pd.Series:
    """
    Uses Haar wavelet denoising to reduce noise in the series.
    """
    coeff = pywt.wavedec(series, wavelet, mode="per")
    # Apply a simple thresholding: set coefficients below a threshold to zero
    threshold = np.std(coeff[-level]) * 0.5  # threshold factor can be tuned
    coeff_thresholded = [pywt.threshold(c, threshold, mode="soft") for c in coeff]
    denoised = pywt.waverec(coeff_thresholded, wavelet, mode="per")
    # Ensure the length matches
    denoised = pd.Series(denoised[:len(series)], index=series.index)
    return denoised

if __name__ == "__main__":
    import data.data_collection as dc
    raw_data = dc.collect_data()
    denoised = denoise_series(raw_data["Close"])
    print(denoised.head())