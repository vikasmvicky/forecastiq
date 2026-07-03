"""
ForecastIQ — Backtest Validation
=================================
Holds out the last N weeks of historical data, forecasts them,
and compares against actuals. Calculates MAPE and RMSE.
This is what separates "we ran a model" from "we proved it works."
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import curve_fit

def _fit_log_curve(spend, revenue):
    """Fit a * ln(spend + 1) + b"""
    def log_curve(x, a, b): return a * np.log(x + 1) + b
    valid = (spend > 0) & (revenue > 0)
    if valid.sum() < 4: return None
    try:
        popt, _ = curve_fit(log_curve, spend[valid], revenue[valid], maxfev=5000)
        return popt
    except Exception:
        return None

def run_backtest(weekly_df, holdout_weeks=4):
    """
    Runs a historical backtest.
    Returns dict with metrics, actuals, and predictions.
    """
    # Aggregate total weekly revenue across all channels
    total_weekly = weekly_df.groupby(pd.Grouper(key="date", freq="W-MON")).agg(
        spend=("spend", "sum"), revenue=("revenue", "sum")
    ).reset_index().sort_values("date")
    
    if len(total_weekly) < holdout_weeks + 8:
        return None # Not enough data to backtest
        
    train = total_weekly.iloc[:-holdout_weeks]
    test = total_weekly.iloc[-holdout_weeks:]
    
    # 1. Fit Holt-Winters on train
    values = train["revenue"].values.astype(float)
    if len(values) < 52:
        model = ExponentialSmoothing(values, trend="add", initialization_method="estimated")
    else:
        try:
            model = ExponentialSmoothing(values, trend="add", seasonal="add", seasonal_periods=52, initialization_method="estimated")
        except Exception:
            model = ExponentialSmoothing(values, trend="add", initialization_method="estimated")
    
    result = model.fit()
    point_forecast = result.forecast(holdout_weeks)
    
    # 2. Scale forecast using fitted log curve (unified with forecast.py)
    avg_train_spend = train["spend"].mean()
    avg_test_spend = test["spend"].mean()
    if avg_train_spend > 0 and avg_test_spend > 0:
        popt = _fit_log_curve(train["spend"], train["revenue"])
        if popt:
            a, b = popt
            expected_rev = a * np.log(avg_test_spend + 1) + b
            actual_train_rev = a * np.log(avg_train_spend + 1) + b
            if actual_train_rev > 0:
                scale_factor = expected_rev / actual_train_rev
                point_forecast = point_forecast * scale_factor
                
    # 3. Naive Baseline (Just repeat the last known week)
    naive_forecast = np.full(holdout_weeks, train["revenue"].iloc[-1])
    
    # 4. Calculate Metrics
    actuals = test["revenue"].values
    
    mape_model = np.mean(np.abs((actuals - point_forecast) / actuals)) * 100
    rmse_model = np.sqrt(np.mean((actuals - point_forecast)**2))
    
    mape_naive = np.mean(np.abs((actuals - naive_forecast) / actuals)) * 100
    rmse_naive = np.sqrt(np.mean((actuals - naive_forecast)**2))
    
    return {
        "holdout_weeks": holdout_weeks,
        "dates": test["date"].tolist(),
        "actuals": actuals.tolist(),
        "predicted": point_forecast.tolist(),
        "naive": naive_forecast.tolist(),
        "mape_model": round(mape_model, 2),
        "rmse_model": round(rmse_model, 2),
        "mape_naive": round(mape_naive, 2),
        "rmse_naive": round(rmse_naive, 2),
        "improvement_mape": round(mape_naive - mape_model, 2)
    }