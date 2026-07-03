import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def seasonality(day_of_year):
    yearly = 0.15 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    half_yearly = 0.05 * np.sin(2 * np.pi * (day_of_year - 100) / 182.5)
    bf = 0.0
    if 325 <= day_of_year <= 335:
        bf = 0.25 * np.exp(-0.5 * ((day_of_year - 330) / 3) ** 2)
    return 1.0 + yearly + half_yearly + bf

def spend_to_revenue(spend, efficiency):
    base = efficiency * np.log(spend + 1)
    noise = np.random.normal(0, 0.12 * base)
    return max(0, base + noise)

def generate(campaigns, channel_name, path):
    dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    total = len(dates)
    rows = []
    for c in campaigns:
        start, end = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        for i, d in enumerate(dates):
            if d < start or d > end:
                continue
            trend = 1.0 + (i / total) * 0.15
            seas = seasonality(d.dayofyear)
            spend = c["spend"] * np.random.uniform(0.8, 1.2)
            if np.random.random() < 0.05:
                spend = 0
            rev = spend_to_revenue(spend, c["eff"]) * trend * seas
            clicks = int(spend / np.random.uniform(*c["cpc"])) if spend > 0 else 0
            convs = int(clicks * np.random.uniform(*c["cvr"])) if clicks > 0 else 0
            rows.append({
                "date": d, "campaign": c["name"], "campaign_type": c["type"],
                "impressions": int(clicks / np.random.uniform(0.01, 0.05)) if clicks else 0,
                "clicks": clicks, "conversions": convs,
                "spend": round(spend, 2), "revenue": round(rev, 2), "channel": channel_name
            })
    df = pd.DataFrame(rows)
    miss = np.random.choice(df.index, int(len(df) * 0.02), replace=False)
    df.loc[miss, "revenue"] = np.nan
    miss2 = np.random.choice(df.index, int(len(df) * 0.01), replace=False)
    df.loc[miss2, "conversions"] = np.nan
    out = np.random.choice(df.index, 5, replace=False)
    df.loc[out, "revenue"] = df.loc[out, "revenue"] * 5
    if channel_name == "Google Ads":
        df["quality_score"] = np.random.uniform(1, 10, len(df)).round(1)
        df["avg_position"] = np.random.uniform(1, 8, len(df)).round(2)
    elif channel_name == "Meta Ads":
        df["frequency"] = np.random.uniform(1, 8, len(df)).round(2)
        df["reach"] = np.random.randint(1000, 50000, len(df))
    else:
        df["bid_strategy"] = np.random.choice(["MaxClicks", "MaxConv", "TargetCPA"], len(df))
    df["roas"] = df["revenue"] / df["spend"]
    df["roas"] = df["roas"].replace([np.inf, -np.inf], np.nan)
    df.to_csv(path, index=False)
    print(f"{channel_name}: {len(df):,} rows | ${df['spend'].sum():,.0f} spend | ${df['revenue'].sum():,.0f} rev | {(df['revenue'].sum()/df['spend'].sum()):.2f}x ROAS")
    return df

GOOGLE = [
    {"name": "GA_Search_Brand_US_v2", "type": "Search - Brand", "eff": 400, "spend": 550, "cpc": (0.8,1.5), "cvr": (0.08,0.15), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "GA_Search_NonBrand_US_v3", "type": "Search - Non-Brand", "eff": 500, "spend": 1100, "cpc": (1.5,3.5), "cvr": (0.02,0.05), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "GA_Shopping_NonBrand_US_v1", "type": "Shopping", "eff": 460, "spend": 850, "cpc": (0.6,1.8), "cvr": (0.03,0.06), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "GA_PMax_All_v3", "type": "Performance Max", "eff": 300, "spend": 700, "cpc": (1.0,2.5), "cvr": (0.02,0.05), "start": "2025-03-01", "end": "2026-06-30"},
    {"name": "GA_Display_Remarketing_v1", "type": "Display", "eff": 200, "spend": 300, "cpc": (0.3,1.0), "cvr": (0.01,0.03), "start": "2025-01-01", "end": "2026-04-30"},
]
META = [
    {"name": "Meta_Prospecting_LAL_US_v2", "type": "Prospecting", "eff": 320, "spend": 800, "cpc": (0.8,2.0), "cvr": (0.01,0.03), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "Meta_Retargeting_Website_v1", "type": "Retargeting", "eff": 380, "spend": 450, "cpc": (0.5,1.2), "cvr": (0.04,0.09), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "Meta_Advantage_Plus_Shopping_v1", "type": "Advantage+ Shopping", "eff": 270, "spend": 550, "cpc": (0.7,1.8), "cvr": (0.02,0.04), "start": "2025-04-01", "end": "2026-06-30"},
]
MS = [
    {"name": "MS_Search_Brand_US_v1", "type": "Search - Brand", "eff": 180, "spend": 200, "cpc": (0.5,1.2), "cvr": (0.07,0.14), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "MS_Search_NonBrand_US_v1", "type": "Search - Non-Brand", "eff": 200, "spend": 400, "cpc": (1.0,2.5), "cvr": (0.015,0.04), "start": "2025-01-01", "end": "2026-06-30"},
    {"name": "MS_Shopping_US_v1", "type": "Shopping", "eff": 170, "spend": 250, "cpc": (0.4,1.2), "cvr": (0.02,0.05), "start": "2025-01-01", "end": "2026-06-30"},
]

if __name__ == "__main__":
    out = Path(__file__).parent / "raw"
    out.mkdir(parents=True, exist_ok=True)
    print("Generating...")
    g = generate(GOOGLE, "Google Ads", out / "google_ads.csv")
    m = generate(META, "Meta Ads", out / "meta_ads.csv")
    s = generate(MS, "Microsoft Ads", out / "microsoft_ads.csv")
    all_dfs = [g, m, s]
    ts = sum(d["spend"].sum() for d in all_dfs)
    tr = sum(d["revenue"].sum() for d in all_dfs)
    print(f"\nTotal: {sum(len(d) for d in all_dfs):,} rows | ${ts:,.0f} spend | ${tr:,.0f} revenue | {tr/ts:.2f}x ROAS")
    print(f"Files saved to: {out}/")