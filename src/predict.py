import argparse
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from preprocess import load_all, aggregate_weekly
from forecast import forecast_all


def main():
    parser = argparse.ArgumentParser(
        description="ForecastIQ Prediction Pipeline"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing CSV files"
    )

    parser.add_argument(
        "--model",
        default="./pickle/model.pkl",
        help="Reserved for hackathon compatibility"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ForecastIQ Prediction Pipeline")
    print("=" * 60)

    # ---------------------------------------------------
    # Step 1
    # ---------------------------------------------------
    print("\n[1/6] Loading CSV files...")

    df = load_all(args.data_dir)

    if df.empty:
        raise ValueError("No valid CSV files found.")

    print(f"Loaded {len(df):,} rows")

    # ---------------------------------------------------
    # Step 2
    # ---------------------------------------------------
    print("\n[2/6] Aggregating weekly...")

    weekly = aggregate_weekly(df)

    print(f"Weekly rows: {len(weekly):,}")

    # ---------------------------------------------------
    # Step 3
    # ---------------------------------------------------
    print("\n[3/6] Creating budgets...")

    budgets = {
        "Google Ads": 40000,
        "Meta Ads": 22000,
        "Microsoft Ads": 11000
    }

    # ---------------------------------------------------
    # Step 4
    # ---------------------------------------------------
    print("\n[4/6] Running Forecast Engine...")

    result = forecast_all(
        weekly_df=weekly,
        budgets=budgets,
        days=30,
        raw_df=df
    )

    print("Forecast completed.")

    # ---------------------------------------------------
    # Step 5
    # ---------------------------------------------------
    print("\n[5/6] Creating prediction file...")

    agg = result["aggregate"]

    rows = []

    rows.append({
        "Forecast": "Revenue P10",
        "Value": agg["p10"]
    })

    rows.append({
        "Forecast": "Revenue P50",
        "Value": agg["p50"]
    })

    rows.append({
        "Forecast": "Revenue P90",
        "Value": agg["p90"]
    })

    rows.append({
        "Forecast": "ROAS P10",
        "Value": agg["roas_p10"]
    })

    rows.append({
        "Forecast": "ROAS P50",
        "Value": agg["roas_p50"]
    })

    rows.append({
        "Forecast": "ROAS P90",
        "Value": agg["roas_p90"]
    })

    output = pd.DataFrame(rows)

    # ---------------------------------------------------
    # Step 6
    # ---------------------------------------------------
    print("\n[6/6] Saving predictions...")

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_path, index=False)

    print(f"\nPredictions saved successfully:")
    print(output_path.resolve())

    print("\nDone.")


if __name__ == "__main__":
    main()