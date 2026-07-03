"""
ForecastIQ — Statistical Anomaly Detection
==========================================
Finds real outliers in historical data so the LLM has actual facts to explain.
Uses Z-score analysis on weekly aggregated data.

Fixed: deviation_pct is now the true percent deviation from the mean.
Fixed: Enforces standard Python floats so JSON serialization never crashes.
"""

import pandas as pd
import numpy as np

def detect_anomalies(weekly_df, z_threshold=2.0):
    anomalies = []

    for channel in weekly_df["channel"].unique():
        ch_data = weekly_df[weekly_df["channel"] == channel].copy()
        ch_data = ch_data.set_index("date").sort_index()

        for metric in ["revenue", "roas", "conversions"]:
            if metric not in ch_data.columns:
                continue

            series = ch_data[metric].dropna()
            if len(series) < 4:
                continue

            mean = float(series.mean())
            std = float(series.std())
            if std == 0:
                continue

            z_scores = (series - mean) / std
            outlier_dates = z_scores[z_scores.abs() > z_threshold].index

            for date in outlier_dates:
                # FORCE standard Python floats to prevent JSON crashes on duplicate dates
                raw_val = series.loc[date]
                raw_z = z_scores.loc[date]
                
                val = float(raw_val.iloc[0]) if hasattr(raw_val, 'iloc') else float(raw_val)
                z = float(raw_z.iloc[0]) if hasattr(raw_z, 'iloc') else float(raw_z)
                
                direction = "spike" if z > 0 else "drop"
                deviation_pct = round(abs(val - mean) / mean * 100, 1) if mean != 0 else None

                anomalies.append({
                    "date": str(date.date()),
                    "channel": str(channel),
                    "metric": str(metric),
                    "direction": direction,
                    "value": round(val, 2),
                    "expected": round(mean, 2),
                    "deviation_pct": deviation_pct,
                    "z_score": round(z, 2)
                })

    anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return anomalies[:5]