"""
forecast.py - Future Direction Projection (Classification)

Classification mode:
  Predict probability of price going up/down ->
  Use probability to estimate expected return ->
  Project a price path based on expected direction

Note: Uncertainty grows with forecast horizon.
"""

import numpy as np
import pandas as pd
from features import build_feature_matrix


def _to_daily_return(period_return: float, horizon: int) -> float:
    """Convert an N-day return into an equivalent 1-day compounded return."""
    horizon = max(int(horizon), 1)
    if horizon == 1:
        return float(period_return)

    gross_return = max(1.0 + float(period_return), 1e-6)
    return gross_return ** (1 / horizon) - 1


def _estimate_expected_horizon_return(
    prob_up: float,
    avg_up: float,
    avg_down: float,
    fallback_return: float,
) -> float:
    """Map a classification probability into an expected horizon return."""
    fallback = 0.0 if np.isnan(fallback_return) else float(fallback_return)
    up_return = fallback if np.isnan(avg_up) else float(avg_up)
    down_return = fallback if np.isnan(avg_down) else float(avg_down)
    return (prob_up * up_return) + ((1.0 - prob_up) * down_return)


def _get_model_classes(model) -> np.ndarray | None:
    """Read class labels from a classifier or pipeline when available."""
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)

    named_steps = getattr(model, "named_steps", None)
    if named_steps is None:
        return None

    final_model = named_steps.get("model")
    if final_model is None:
        return None

    classes = getattr(final_model, "classes_", None)
    return None if classes is None else np.asarray(classes)


def _extract_positive_class_probability(model, X) -> float | None:
    """Return the probability of the positive class for a single prediction row."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        prob = np.asarray(model.predict_proba(X), dtype=float)
    except Exception:
        return None

    if prob.ndim != 2 or len(prob) != 1 or prob.shape[1] == 0:
        return None

    classes = _get_model_classes(model)
    if prob.shape[1] == 1:
        if classes is not None and len(classes) == 1:
            return 1.0 if int(classes[0]) == 1 else 0.0
        return float(prob[0, 0])

    if classes is not None and 1 in classes:
        positive_idx = int(np.where(classes == 1)[0][0])
        return float(prob[0, positive_idx])

    return float(prob[0, -1])


def forecast_future(
    model,
    df_original: pd.DataFrame,
    feature_cols: list[str],
    days: int = 5,
    horizon: int = 1,
    external_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generate N-day ahead price-path projections using the trained classifier.

    Args:
        model: Trained sklearn Pipeline model
        df_original: OHLCV DataFrame (raw data)
        feature_cols: Feature column names used by the model
        days: Number of business days to project forward
        horizon: Target horizon used during training
        external_df: Optional external data for feature computation

    Returns:
        DataFrame: Date, projected price path, probabilities, and projected changes
    """
    df = df_original.copy()
    predictions = []
    last_date = df.index[-1]
    current_price = df["Close"].iloc[-1]
    horizon = max(int(horizon), 1)

    horizon_returns = df["Close"].pct_change(horizon)
    hist_avg_return = horizon_returns.mean()
    hist_avg_up = horizon_returns.where(horizon_returns > 0).mean()
    hist_avg_down = horizon_returns.where(horizon_returns < 0).mean()

    for day in range(1, days + 1):
        df_feat = build_feature_matrix(df, external_df=external_df)
        if df_feat.empty:
            break

        missing_cols = [col for col in feature_cols if col not in df_feat.columns]
        if missing_cols:
            break

        last_row = df_feat.loc[[df_feat.index[-1]], feature_cols]
        next_date = last_date + pd.tseries.offsets.BDay(day)

        try:
            pred_class = int(model.predict(last_row)[0])
            prob_up = _extract_positive_class_probability(model, last_row)
            if prob_up is None:
                prob_up = float(pred_class)

            prob_up = float(np.clip(prob_up, 0.0, 1.0))
            est_horizon_return = _estimate_expected_horizon_return(
                prob_up,
                hist_avg_up,
                hist_avg_down,
                hist_avg_return,
            )
            applied_daily_return = _to_daily_return(est_horizon_return, horizon)
            projected_price = current_price * (1 + applied_daily_return)

            row = {
                "Date": next_date,
                "Projected Price": round(projected_price, 2),
                "Probability (%)": round(prob_up * 100, 1),
            }
            if horizon > 1:
                row[f"Est. {horizon}-Day Return (%)"] = round(est_horizon_return * 100, 3)
                row["Applied Daily Return (%)"] = round(applied_daily_return * 100, 3)
            else:
                row["Est. Return (%)"] = round(est_horizon_return * 100, 3)
            predictions.append(row)
        except Exception:
            break

        new_row = pd.DataFrame({
            "Open": [projected_price],
            "High": [projected_price * 1.01],
            "Low": [projected_price * 0.99],
            "Close": [projected_price],
            "Volume": [df["Volume"].tail(5).mean()],
        }, index=[next_date])

        df = pd.concat([df, new_row])
        current_price = projected_price

    forecast_df = pd.DataFrame(predictions)

    if not forecast_df.empty:
        base_price = df_original["Close"].iloc[-1]
        forecast_df["Projected Change (%)"] = (
            (forecast_df["Projected Price"] - base_price) / base_price * 100
        ).round(2)
        forecast_df.index = range(1, len(forecast_df) + 1)
        forecast_df.index.name = "Day"

    return forecast_df


def get_confidence_band(
    forecast_df: pd.DataFrame,
    historical_volatility: float,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute projection confidence intervals using volatility scaling.

    Args:
        forecast_df: Projection DataFrame
        historical_volatility: Historical daily volatility (std of returns)
        confidence_level: Confidence level (0.80, 0.90, 0.95, 0.99)

    Returns:
        DataFrame: Projection table with Lower Bound and Upper Bound columns
    """
    from scipy import stats

    df = forecast_df.copy()
    if df.empty:
        return df

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    prices = df["Projected Price"].values
    for i in range(len(prices)):
        spread = prices[i] * historical_volatility * np.sqrt(i + 1) * z_score
        df.loc[df.index[i], "Lower Bound"] = round(prices[i] - spread, 2)
        df.loc[df.index[i], "Upper Bound"] = round(prices[i] + spread, 2)

    return df
