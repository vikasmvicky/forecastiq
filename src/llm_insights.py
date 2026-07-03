"""
ForecastIQ — LLM Insights
==========================
Upgraded: Now ingests real statistical anomalies.
"""

import os
import json
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from anomaly_detection import detect_anomalies

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent.parent / ".env")

def build_historical_summary(weekly_df):
    lines = []
    total_spend = weekly_df["spend"].sum()
    total_rev = weekly_df["revenue"].sum()
    lines.append(f"Period: {weekly_df['date'].min().date()} to {weekly_df['date'].max().date()}")
    lines.append(f"Total spend: ${total_spend:,.0f}, Revenue: ${total_rev:,.0f}, ROAS: {total_rev/total_spend:.2f}x")

    for ch in weekly_df["channel"].unique():
        ch_data = weekly_df[weekly_df["channel"] == ch]
        lines.append(f"{ch}: ${ch_data['spend'].sum():,.0f} spend -> ${ch_data['revenue'].sum():,.0f} rev ({ch_data['revenue'].sum()/ch_data['spend'].sum():.2f}x)")

    return "\n".join(lines)

def get_llm_insights(forecast_result, historical_summary, budget, weekly_df):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {"summary": "LLM not configured.", "drivers": [], "risks": [{"risk": "No API Key", "severity": "high", "description": "Set GROQ_API_KEY", "mitigation": ""}], "recommendation": ""}

    # GET REAL ANOMALIES
    anomalies = detect_anomalies(weekly_df)
    anomaly_text = "None detected." if not anomalies else json.dumps(anomalies, indent=2)

    prompt = f"""You are a senior e-commerce marketing analyst. Analyze this forecast data.

FORECAST RESULTS:
{json.dumps(forecast_result, indent=2)}

HISTORICAL CONTEXT:
{historical_summary}

STATISTICAL ANOMALIES DETECTED IN HISTORICAL DATA:
{anomaly_text}

PROPOSED BUDGET: ${budget:,}

Respond ONLY with valid JSON in this exact structure. No markdown, no code blocks:
{{
  "summary": "2-3 sentences. Mention exact revenue P50, one specific driver, and one specific risk.",
  "drivers": [
    {{"factor": "Name", "impact": "up/down/stable", "description": "Explanation with specific metric.", "confidence": "high/medium/low"}}
  ],
  "risks": [
    {{"risk": "Name", "severity": "high/medium/low", "description": "What could go wrong.", "mitigation": "Exact action to take."}}
  ],
  "recommendation": "1-2 sentences. Specific dollar amounts and channel names."
}}"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.2, # Lower temp for stricter JSON
                "max_tokens": 800,
                "response_format": {"type": "json_object"}, # Force JSON mode
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code != 200:
            return {"summary": f"API Error {response.status_code}", "drivers": [], "risks": [], "recommendation": ""}

        return json.loads(response.json()["choices"][0]["message"]["content"])

    except Exception as e:
        return {"summary": f"LLM Error: {str(e)[:50]}", "drivers": [], "risks": [], "recommendation": ""}