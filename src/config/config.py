# config/config.py
import os

# Define paths and global settings
DATA_DIR = os.path.join(os.getcwd(), "data")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
PLOT_DIR = os.path.join(os.getcwd(), "plots")

# Data collection parameters
START_DATE = "2008-10-01"
END_DATE = "2018-10-01"
ETF_LIST = ["IVE", "IJJ", "VYM", "DVY"]  # example tickers

# LSTM Hyperparameters
LSTM_CONFIG = {
    "neurons": 256,
    "dropout": 0.2,
    "epochs": 200,
    "batch_size": 20,
    "optimizer": "adam",
    "loss": "mse"
}

# Other model params (statistical models OR other ML models) could later be added here as well.