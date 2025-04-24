import statsmodels.api as sm

def add_OLS(pairs_timeseries):
  pairs_timeseries_including_ols = pairs_timeseries.copy()
  est = sm.OLS(pairs_timeseries_including_ols['S1_close'], pairs_timeseries_including_ols['S2_close'])
  est = est.fit()
  alpha = -est.params[0]
  pairs_timeseries_including_ols['Spread_Close'] = pairs_timeseries_including_ols['S1_close'] + (pairs_timeseries_including_ols['S2_close'] * alpha)

  est_op = sm.OLS(pairs_timeseries_including_ols['S1_open'], pairs_timeseries_including_ols['S2_open'])
  est_op = est_op.fit()
  alpha_op = -est_op.params[0]
  pairs_timeseries_including_ols['Spread_Open'] = pairs_timeseries_including_ols['S1_open'] + (pairs_timeseries_including_ols['S2_open'] * alpha_op)
  est_hi = sm.OLS(pairs_timeseries_including_ols['S1_high'], pairs_timeseries_including_ols['S2_high'])
  est_hi = est_hi.fit()
  alpha_hi = -est_hi.params[0]
  pairs_timeseries_including_ols['Spread_High'] = pairs_timeseries_including_ols['S1_high'] + (pairs_timeseries_including_ols['S2_high'] * alpha_hi)
  est_lo = sm.OLS(pairs_timeseries_including_ols['S1_low'], pairs_timeseries_including_ols['S2_low'])
  est_lo = est_lo.fit()
  alpha_lo = -est_lo.params[0]
  pairs_timeseries_including_ols['Spread_Low'] = pairs_timeseries_including_ols['S1_low'] + (pairs_timeseries_including_ols['S2_low'] * alpha_lo)
  return pairs_timeseries_including_ols
