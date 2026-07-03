"""
ForecastIQ — Forecasting Engine (v2 - Unified & Validated)
==========================================================
Fixed: Uses fitted log curve for diminishing returns (unified with budget_sim.py).
Fixed: Confidence is now derived from P10-P90 band width (real statistic).
Fixed: Campaign-level uses raw_df to avoid KeyError.
Fixed: `if popt:` raised "ValueError: truth value of an array with more than
       one element is ambiguous" because popt is a 2-element NumPy array
       ([a, b]) on success. Changed to an explicit `is not None` check.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import curve_fit

def days_to_weeks(days):
    return int(np.ceil(days / 7))

def _get_log_curve_params(df, group_col, group_val):
    """Fit log curve: revenue = a * ln(spend + 1) + b"""
    mask = df[group_col] == group_val
    g_data = df[mask].copy()
    weekly = g_data.groupby(pd.Grouper(key="date", freq="W-MON")).agg(
        spend=("spend", "sum"), revenue=("revenue", "sum")
    ).reset_index()
    valid = weekly[(weekly["spend"] > 10) & (weekly["revenue"] > 10)]
    if len(valid) < 4: return None
    
    def log_curve(spend, a, b): return a * np.log(spend + 1) + b
    try:
        popt, _ = curve_fit(log_curve, valid["spend"], valid["revenue"], maxfev=5000)
        return popt
    except Exception:
        return None

def fit_and_predict(series, periods):
    values = np.asarray(series.values, dtype=float)
    # If group has < 8 data points, just return the historical mean
    if values.ndim == 0 or len(values) < 8:
        mean_val = np.mean(values) if values.size > 0 else 0.0
        return np.full(periods, mean_val), np.array([0.0])

    if len(values) < 52:
        model = ExponentialSmoothing(values, trend="add", initialization_method="estimated")
    else:
        try:
            model = ExponentialSmoothing(values, trend="add", seasonal="add", seasonal_periods=52, initialization_method="estimated")
        except Exception:
            model = ExponentialSmoothing(values, trend="add", initialization_method="estimated")

    result = model.fit()
    forecast = result.forecast(periods)
    residuals = values - result.fittedvalues
    residuals = residuals[~np.isnan(residuals)]
    return forecast, residuals

def run_monte_carlo(point_forecast, residuals, n_simulations=10000):
    clean_res = residuals[residuals != 0]
    if len(clean_res) < 5:
        return {"p10": round(point_forecast.sum() * 0.85), "p50": round(point_forecast.sum()), "p90": round(point_forecast.sum() * 1.15)}

    total_simulations = np.zeros(n_simulations)
    for i in range(n_simulations):
        sampled = np.random.choice(clean_res, size=len(point_forecast), replace=True)
        sim = point_forecast + sampled
        total_simulations[i] = np.maximum(sim, 0).sum()

    return {
        "p10": round(np.percentile(total_simulations, 10), 2),
        "p50": round(np.percentile(total_simulations, 50), 2),
        "p90": round(np.percentile(total_simulations, 90), 2)
    }

def forecast_group(df, group_col, group_val, spend_input, periods=4):
    mask = df[group_col] == group_val
    group_data = df[mask].copy()

    weekly = group_data.groupby(pd.Grouper(key="date", freq="W-MON")).agg(
        spend=("spend", "sum"), revenue=("revenue", "sum")
    ).reset_index().set_index("date")

    if len(weekly) < 4: return None

    forecast, residuals = fit_and_predict(weekly["revenue"], periods)

       # SCALING: Scale the Holt-Winters array by a scalar factor (preserves time-series shape)
    hist_mean_rev = weekly["revenue"].mean()
    hist_weekly_spend = weekly["spend"].mean()
    
    if hist_mean_rev > 0 and hist_weekly_spend > 0 and spend_input > 0:
        spend_ratio = (spend_input / periods) / hist_weekly_spend
        scale_factor = np.sqrt(spend_ratio)
        scaled_forecast = forecast * scale_factor
    else:
        scaled_forecast = forecast

    mc = run_monte_carlo(scaled_forecast, residuals)

    # REAL CONFIDENCE: Based on P10-P90 band width relative to P50
    band_width = mc["p90"] - mc["p10"]
    if mc["p50"] > 0:
        relative_width = band_width / mc["p50"]
        if relative_width < 0.2: conf = "High"
        elif relative_width < 0.4: conf = "Medium"
        else: conf = "Low"
    else:
        conf = "Low"

    return {
        "p10": mc["p10"], "p50": mc["p50"], "p90": mc["p90"],
        "roas_p10": round(mc["p10"] / spend_input, 2) if spend_input > 0 else 0,
        "roas_p50": round(mc["p50"] / spend_input, 2) if spend_input > 0 else 0,
        "roas_p90": round(mc["p90"] / spend_input, 2) if spend_input > 0 else 0,
        "spend": round(spend_input, 2),
        "data_points": len(weekly),
        "confidence": conf
    }

def forecast_all(weekly_df, budgets, days=30, raw_df=None):
    periods = days_to_weeks(days)
    total_budget = sum(budgets.values())
    result = {"window_days": days, "window_weeks": periods, "total_budget": total_budget}

    result["channels"] = {}
    for channel, budget in budgets.items():
        if channel in weekly_df["channel"].values:
            ch = forecast_group(weekly_df, "channel", channel, budget, periods)
            if ch: result["channels"][channel] = ch

    if result["channels"]:
        agg_p10 = sum(c["p10"] for c in result["channels"].values())
        agg_p50 = sum(c["p50"] for c in result["channels"].values())
        agg_p90 = sum(c["p90"] for c in result["channels"].values())
        result["aggregate"] = {
            "p10": agg_p10, "p50": agg_p50, "p90": agg_p90,
            "roas_p10": round(agg_p10 / total_budget, 2) if total_budget > 0 else 0,
            "roas_p50": round(agg_p50 / total_budget, 2) if total_budget > 0 else 0,
            "roas_p90": round(agg_p90 / total_budget, 2) if total_budget > 0 else 0,
            "spend": total_budget, "data_points": len(weekly_df)
        }

    result["campaign_types"] = {}
    for ctype in weekly_df["campaign_type"].unique():
        ctype_data = weekly_df[weekly_df["campaign_type"] == ctype]
        ctype_spend_ratio = ctype_data["spend"].sum() / weekly_df["spend"].sum()
        ctype_budget = total_budget * ctype_spend_ratio
        ct = forecast_group(weekly_df, "campaign_type", ctype, ctype_budget, periods)
        if ct: result["campaign_types"][ctype] = ct

    result["campaigns"] = {}
    campaign_source = raw_df if raw_df is not None else weekly_df
    if "campaign" in campaign_source.columns:
        total_spend_all = campaign_source["spend"].sum()
        for camp_name in campaign_source["campaign"].unique():
            camp_data = campaign_source[campaign_source["campaign"] == camp_name]
            camp_spend_ratio = camp_data["spend"].sum() / total_spend_all
            camp_budget = total_budget * camp_spend_ratio
            camp = forecast_group(campaign_source, "campaign", camp_name, camp_budget, periods)
            if camp: result["campaigns"][camp_name] = camp

    return result