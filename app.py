from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_churn_pipeline.pkl"
DATA_PATH = BASE_DIR / "customer_data.csv"

AGE_BINS = [0, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
TENURE_BINS = [-1, 0, 2, 5, 10, 100]
TENURE_LABELS = ["0", "1-2", "3-5", "6-10", "10+"]
HIGH_BALANCE_THRESHOLD = 127644.24

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df_out = frame.copy()

    df_out["balance_per_product"] = (
        df_out["balance"] / df_out["products_number"].replace(0, np.nan)
    )
    df_out["balance_per_product"] = df_out["balance_per_product"].fillna(0)

    df_out["salary_balance_ratio"] = (
        df_out["estimated_salary"] / df_out["balance"].replace(0, np.nan)
    )
    df_out["salary_balance_ratio"] = df_out["salary_balance_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )
    df_out["salary_balance_ratio"] = df_out["salary_balance_ratio"].fillna(
        df_out["salary_balance_ratio"].median()
    )

    df_out["age_group"] = pd.cut(
        df_out["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
    )
    df_out["tenure_bucket"] = pd.cut(
        df_out["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )
    df_out["high_balance"] = (
        df_out["balance"] > HIGH_BALANCE_THRESHOLD
    ).astype(int)

    return df_out


def build_input_frame(form_data: dict[str, str]) -> pd.DataFrame:
    row = {
        "credit_score": int(form_data["credit_score"]),
        "country": form_data["country"],
        "gender": form_data["gender"],
        "age": int(form_data["age"]),
        "tenure": int(form_data["tenure"]),
        "balance": float(form_data["balance"]),
        "products_number": int(form_data["products_number"]),
        "credit_card": int(form_data["credit_card"]),
        "active_member": int(form_data["active_member"]),
        "estimated_salary": float(form_data["estimated_salary"]),
    }
    frame = pd.DataFrame([row])
    return add_engineered_features(frame)


def get_form_defaults() -> dict[str, str]:
    return {
        "credit_score": "650",
        "country": "France",
        "gender": "Male",
        "age": "40",
        "tenure": "3",
        "balance": "50000",
        "products_number": "2",
        "credit_card": "1",
        "active_member": "1",
        "estimated_salary": "60000",
    }


def get_balance_threshold() -> float:
    if DATA_PATH.exists():
        try:
            return float(pd.read_csv(DATA_PATH)["balance"].quantile(0.75))
        except Exception:
            return HIGH_BALANCE_THRESHOLD
    return HIGH_BALANCE_THRESHOLD


HIGH_BALANCE_THRESHOLD = get_balance_threshold()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    risk_band = None
    error = None
    form_values = get_form_defaults()

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            input_frame = build_input_frame(form_values)
            prediction = int(model.predict(input_frame)[0])
            probability = float(model.predict_proba(input_frame)[0, 1])

            if probability >= 0.7:
                risk_band = "High risk"
            elif probability >= 0.4:
                risk_band = "Moderate risk"
            else:
                risk_band = "Low risk"
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        form_values=form_values,
        prediction=prediction,
        probability=probability,
        risk_band=risk_band,
        error=error,
        balance_threshold=HIGH_BALANCE_THRESHOLD,
    )


if __name__ == "__main__":
    app.run(debug=True)
