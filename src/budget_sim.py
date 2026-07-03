"""
ForecastIQ — Budget Simulator
==============================
Fixed: Calculates actual Marginal ROAS to find the true Efficient Frontier.
Fixed: marginal_roas at the first curve point (50% budget) now measures
       the true marginal rate from a zero-spend baseline, instead of
       falling back to average ROAS (which made it inconsistent with
       every other point on the curve).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fit_spend_curve(df, channel_name):
    ch = df[df["channel"] == channel_name].copy()
    weekly = ch.groupby(pd.Grouper(key="date", freq="W-MON")).agg(
        spend=("spend", "sum"), revenue=("revenue", "sum")
    ).reset_index()
    weekly = weekly[weekly["spend"] > 10].copy()

    if len(weekly) < 4:
        return None

    def log_curve(spend, a, b):
        return a * np.log(spend + 1) + b

    try:
        popt, _ = curve_fit(log_curve, weekly["spend"], weekly["revenue"], maxfev=5000)
        return popt
    except Exception:
        return None

def predict_rev(a, b, spend):
    return a * np.log(spend + 1) + b

def simulate_budget(df, budgets, days=30):
    weeks = int(np.ceil(days / 7))
    results = {"proposed": {}, "scenarios": {}}

    for channel, budget in budgets.items():
        params = fit_spend_curve(df, channel)
        if params is None: continue

        a, b = params
        weekly_spend = budget / weeks
        weekly_rev = predict_rev(a, b, weekly_spend)
        total_rev = weekly_rev * weeks
        roas = total_rev / budget if budget > 0 else 0

        results["proposed"][channel] = {
            "budget": budget, "revenue": round(total_rev), "roas": round(roas, 2),
            "curve_params": {"a": round(a, 2), "b": round(b, 2)}
        }

        curve = []
        prev_budget = 0
        prev_rev = predict_rev(a, b, 0) * weeks
        for level in np.arange(0.5, 1.6, 0.25):
            test_budget = budget * level
            test_rev = predict_rev(a, b, test_budget / weeks) * weeks
            test_roas = test_rev / test_budget if test_budget > 0 else 0
            budget_delta = test_budget - prev_budget
            marginal_roas = (test_rev - prev_rev) / budget_delta if budget_delta > 0 else test_roas

            curve.append({
                "budget_pct": f"{int(level * 100)}%",
                "budget": round(test_budget),
                "revenue": round(test_rev),
                "roas": round(test_roas, 2),
                "marginal_roas": round(marginal_roas, 2)
            })
            prev_budget = test_budget
            prev_rev = test_rev

        results["scenarios"][channel] = curve

    return results

def get_efficient_frontier(sim_results):
    best_roas = 0
    best_channel = ""
    best_budget = 0
    total_marginal = 0
    count = 0

    marginals = {}
    for ch, points in sim_results["scenarios"].items():
        for pt in points:
            if pt["budget_pct"] == "100%":
                marginals[ch] = pt["marginal_roas"]
                total_marginal += pt["marginal_roas"]
                count += 1
                break

    avg_marginal = total_marginal / count if count > 0 else 0

    if marginals:
        best_channel = max(marginals, key=marginals.get)
        best_roas = marginals[best_channel]
        best_budget = sim_results["proposed"][best_channel]["budget"]

    insight = f"At current budgets, average marginal ROAS is {avg_marginal:.2f}x. "
    if avg_marginal < 1.0:
        insight += "Overall budget is past the efficient frontier. Consider reducing total spend or pausing lowest-performing campaigns."
    else:
        insight += f"{best_channel} has the highest marginal return ({best_roas:.2f}x). Shift budget from channels with marginal ROAS < {avg_marginal:.2f}x to {best_channel} for optimal allocation."

    return {
        "best_channel": best_channel,
        "best_roas": best_roas,
        "best_budget": best_budget,
        "avg_marginal_roas": round(avg_marginal, 2),
        "insight": insight
    }