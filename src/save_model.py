import pickle
from pathlib import Path

model = {
    "name": "ForecastIQ Statistical Forecasting",
    "type": "Holt-Winters",
    "version": "1.0"
}

Path("pickle").mkdir(exist_ok=True)

with open("pickle/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl created successfully.")