"""
ForecastIQ - Preprocessing
===========================
UPGRADED: Handles Google Ads API format, Meta API format, and Bing format.
Automatically converts Google micros to real dollars.
Intelligently backfills missing revenue (e.g., Meta pixel issues).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# The exact columns our system needs
INTERNAL_SCHEMA = ["date", "campaign", "campaign_type", "impressions",
                   "clicks", "conversions", "spend", "revenue", "channel"]

# Maps common raw export names to our internal names
COLUMN_MAP = {
    # --- Standard Platform Exports ---
    "Day": "date", "Date": "date", "Time period": "date", "TimePeriod": "date",
    "Day of week": "date", "Week": "date", "Report period": "date",
    "Campaign": "campaign", "Campaign name": "campaign", "CampaignName": "campaign",
    "Campaign type": "campaign_type", "CampaignType": "campaign_type",
    "Impressions": "impressions", "Impr.": "impressions",
    "Clicks": "clicks",
    "Conversions": "conversions", "Conv.": "conversions", "Transactions": "conversions", "Purchases": "conversions",
    "Cost": "spend", "Spend": "spend", "Amount spent": "spend",
    "Revenue": "revenue", "Conversion value": "revenue", "Sales": "revenue", "Total revenue": "revenue",

    # --- Google Ads API Format ---
    "segments_date": "date",
    "campaign_name": "campaign",
    "campaign_advertising_channel_type": "campaign_type",
    "metrics_impressions": "impressions",
    "metrics_clicks": "clicks",
    "metrics_conversions": "conversions",
    "metrics_conversions_value": "revenue",

    # --- Meta Ads API Format ---
    "date_start": "date",
    "conversion": "conversions",
}

def detect_channel_from_filename(filename):
    """Guess the channel from the filename."""
    name = filename.lower()
    if "google" in name or "ggl" in name:
        return "Google Ads"
    elif "meta" in name or "facebook" in name or "fb" in name:
        return "Meta Ads"
    elif "bing" in name or "microsoft" in name or "ms" in name:
        return "Microsoft Ads"
    return "Unknown Channel"

def standardize_columns(df, filename):
    """Rename raw export columns to internal schema."""
    df.columns = df.columns.str.strip()
    
    # GOOGLE ADS API: Convert micros to real dollars
    if "metrics_cost_micros" in df.columns:
        df["spend"] = df["metrics_cost_micros"] / 1_000_000.0
        
    df.rename(columns=COLUMN_MAP, inplace=True)
    
    if "channel" not in df.columns:
        df["channel"] = detect_channel_from_filename(filename)
        
    if "campaign_type" not in df.columns:
        df["campaign_type"] = "Unknown Type"
        
    return df

def load_and_clean(filepath):
    """Load one channel CSV, auto-map columns, clean it, return DataFrame."""
    if hasattr(filepath, 'name'):
        filename = filepath.name
    else:
        filename = Path(filepath).name

    df = pd.read_csv(filepath)
    df = standardize_columns(df, filename)

    missing = [c for c in INTERNAL_SCHEMA if c not in df.columns]
        # Foolproof: If revenue is missing, set to 0 so the file uploads
    if "revenue" in missing:
        df["revenue"] = 0.0
        missing.remove("revenue")
    if missing:
        raise ValueError(f"Missing columns even after auto-mapping: {missing}. Please ensure file has date, spend, revenue, and campaign data.")

    df = df[INTERNAL_SCHEMA].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    num_cols = ["impressions", "clicks", "conversions", "spend", "revenue"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["campaign", "date"])
    df[num_cols] = df.groupby("campaign")[num_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    for col in num_cols:
        medians = df.groupby("campaign")[col].transform("median")
        df[col] = df[col].fillna(medians)

    for col in ["revenue", "conversions"]:
        p99 = df.groupby("campaign")[col].transform(lambda x: x.quantile(0.99))
        df[col] = df[col].clip(upper=p99)

    for col in num_cols:
        df[col] = df[col].clip(lower=0)

    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0.0)

    return df


def load_all(raw_dir):
    """Load and clean all CSVs found in directory. Return combined DataFrame."""
    raw = Path(raw_dir)
    csv_files = list(raw.glob("*.csv"))
    
    if not csv_files:
        print("  WARNING: No CSV files found in directory!")
        return pd.DataFrame()

    dfs = []
    for filepath in csv_files:
        try:
            print(f"  Loading {filepath.name}...")
            dfs.append(load_and_clean(filepath))
        except Exception as e:
            print(f"  ERROR skipping {filepath.name}: {str(e)[:80]}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    
    # --- INTELLIGENT REVENUE BACKFILL ---
    for ch in combined["channel"].unique():
        ch_data = combined[combined["channel"] == ch]
        if ch_data["revenue"].sum() == 0:
            other_data = combined[combined["channel"] != ch]
            if other_data["revenue"].sum() > 0 and other_data["spend"].sum() > 0:
                blended_roas = other_data["revenue"].sum() / other_data["spend"].sum()
            else:
                blended_roas = 3.5 
                
            mask = combined["channel"] == ch
            combined.loc[mask, "revenue"] = combined.loc[mask, "spend"] * blended_roas
            print(f"  WARNING: '{ch}' missing revenue. Estimated using {blended_roas:.2f}x blended ROAS.")

    combined["roas"] = np.where(combined["spend"] > 0, combined["revenue"] / combined["spend"], 0.0)

    print(f"\n  Combined: {len(combined):,} rows")
    print(f"  Missing values: {combined.isna().sum().sum()}")
    print(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"  Total spend: ${combined['spend'].sum():,.0f}")
    print(f"  Total revenue: ${combined['revenue'].sum():,.0f}")
    print(f"  Blended ROAS: {combined['revenue'].sum() / combined['spend'].sum():.2f}x")
    return combined


def aggregate_weekly(df):
    """Aggregate daily to weekly. Less noise, better for modeling."""
    weekly = df.groupby([
        pd.Grouper(key="date", freq="W-MON"),
        "channel",
        "campaign_type"
    ]).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    ).reset_index()
    weekly["roas"] = np.where(weekly["spend"] > 0, weekly["revenue"] / weekly["spend"], 0)
    return weekly


if __name__ == "__main__":
    print("=" * 50)
    print("ForecastIQ - Preprocessing")
    print("=" * 50)
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    df = load_all(raw_dir)

    if not df.empty:
        print("\n  Weekly aggregation:")
        weekly = aggregate_weekly(df)
        print(f"    {len(weekly):,} weekly rows")
        print(f"    {weekly['channel'].nunique()} channels")
        print(f"    {weekly['campaign_type'].nunique()} campaign types")

    print("\n  Done.")