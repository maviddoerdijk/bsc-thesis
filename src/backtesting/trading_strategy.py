import pandas as pd

def trade(
    S1: pd.Series,
    S2: pd.Series,
    spread: pd.Series, # model-predicted spread for the strategy
    window_long: int,
    window_short: int,
    initial_cash: float = 100000,
    position_threshold: float = 1.0,
    clearing_threshold: float = 0.5,
    risk_fraction: float = 0.1
) -> list:
    ma_long = spread.rolling(window=window_long, center=False).mean()
    ma_short = spread.rolling(window=window_short, center=False).mean()
    std = spread.rolling(window=window_short, center=False).std()
    zscore = (ma_long - ma_short)/std

    # starting with initial_cash allows us to calculate the returns as a percentage of the initial cash, over a time period (YoY are not equal to QoQ for example)
    cash = initial_cash
    qty_s1 = 0
    qty_s2 = 0
    returns = [initial_cash]
    position = 0 # 0: neutral, 1: long, -1: short

    for i in range(len(spread)): # each iteration of the for loop is a new time step, in this case often a single day
        price_s1 = S1.iloc[i]
        price_s2 = S2.iloc[i]
        beta = spread.iloc[i]
        equity = cash + qty_s1 * price_s1 - qty_s2 * price_s2

        # Enter short spread (short S1, long beta S2)
        if position == 0 and zscore.iloc[i] > position_threshold:
            position = -1
            position_size = equity * risk_fraction
            qty_s1 = -position_size / price_s1
            qty_s2 = (position_size * beta) / price_s2
            cash -= (qty_s1 * price_s1 - qty_s2 * price_s2)

        # Enter long spread (long S1, short beta S2)
        elif position == 0 and zscore.iloc[i] < -position_threshold:
            position = 1
            position_size = equity * risk_fraction
            qty_s1 = position_size / price_s1
            qty_s2 = - (position_size * beta) / price_s2
            cash -= (qty_s1 * price_s1 - qty_s2 * price_s2)

        # Exit to neutral when spread reverts
        elif position != 0 and abs(zscore.iloc[i]) < clearing_threshold:
            cash += qty_s1 * price_s1 - qty_s2 * price_s2
            qty_s1 = 0
            qty_s2 = 0
            position = 0

        equity = cash + qty_s1 * price_s1 - qty_s2 * price_s2
        returns.append(equity)

    return returns
