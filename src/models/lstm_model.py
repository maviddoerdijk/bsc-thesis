import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config.config import LSTM_CONFIG

def create_lstm_model(input_shape):
    """
    Builds and compiles a stacked LSTM model.
    """
    model = Sequential()
    model.add(LSTM(LSTM_CONFIG["neurons"], activation="tanh",
                   return_sequences=True, input_shape=input_shape))
    model.add(Dropout(LSTM_CONFIG["dropout"]))
    model.add(LSTM(LSTM_CONFIG["neurons"], activation="tanh"))
    model.add(Dropout(LSTM_CONFIG["dropout"]))
    model.add(Dense(1))  # single spread prediction
    model.compile(optimizer=Adam(), loss=LSTM_CONFIG["loss"])
    return model

def prepare_lstm_data(series: pd.Series, look_back: int = 10):
    """
    Prepares data for LSTM: sequences of look_back steps.
    """
    X, y = [], []
    values = series.values
    for i in range(len(values) - look_back):
        X.append(values[i:i+look_back])
        y.append(values[i+look_back])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def train_lstm(series: pd.Series):
    """
    Trains the LSTM model on the given time series.
    """
    X, y = prepare_lstm_data(series)
    model = create_lstm_model((X.shape[1], 1))
    history = model.fit(X, y, epochs=LSTM_CONFIG["epochs"],
                        batch_size=LSTM_CONFIG["batch_size"],
                        validation_split=0.2, verbose=1)
    return model, history

if __name__ == "__main__":
    # For testing, use a sample series
    import data.data_collection as dc
    raw_data = dc.collect_data()
    close_series = raw_data["Close"].dropna()
    model, history = train_lstm(close_series)
    print("LSTM training complete.")