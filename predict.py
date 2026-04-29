import torch
import numpy as np
import pandas as pd
import joblib

from nhits import NHiTS


# ─── Config ─────────────────────────────────────────────────────────────────

INPUT_SIZE = 72
HORIZON    = 12
N_FEATURES = 13

FEATURE_COLS = ["glucose", "heart_rate", "calories", "steps",
                "basal_rate", "bolus_volume_delivered", "carb_input",
                "iob", "cob", "glucose_roc", "glucose_accel",
                "hour_sin", "hour_cos"]

SCALE_COLS = ["glucose", "heart_rate", "calories", "steps",
              "basal_rate", "bolus_volume_delivered", "carb_input",
              "iob", "cob"]


# ─── Feature helpers ─────────────────────────────────────────────────────────

def _compute_iob(bolus: pd.Series, half_life_steps: int = 11) -> pd.Series:
    decay = np.exp(-np.log(2) / half_life_steps)
    b = bolus.values
    iob = np.zeros(len(b))
    for i in range(len(b)):
        iob[i] = b[i] + (iob[i - 1] * decay if i > 0 else 0.0)
    return pd.Series(iob, index=bolus.index)


def _compute_cob(carbs: pd.Series, peak_steps: int = 12, tail_steps: int = 48) -> pd.Series:
    weights = np.array([
        k / peak_steps if k < peak_steps
        else (tail_steps - k) / (tail_steps - peak_steps) if k < tail_steps
        else 0.0
        for k in range(tail_steps)
    ], dtype=np.float64)
    c = carbs.values.astype(np.float64)
    cob = np.zeros(len(c))
    for i in range(len(c)):
        w = min(i + 1, tail_steps)
        cob[i] = np.dot(c[i - w + 1 : i + 1][::-1], weights[:w])
    return pd.Series(cob, index=carbs.index)


# ─── Load artifacts ──────────────────────────────────────────────────────────

def load_artifacts(model_path="nhits_model.pt",
                   classifier_path="classifier.pkl",
                   scaler_path="scaler.pkl",
                   device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NHiTS(input_size=INPUT_SIZE, n_features=N_FEATURES, horizon=HORIZON).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    clf    = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)

    return model, clf, scaler, device


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Applies the same feature engineering used in Data_Prep.py.
    Expects columns: time, patient_id, glucose, heart_rate, calories,
                     steps, basal_rate, bolus_volume_delivered, carb_input
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["patient_id", "time"]).reset_index(drop=True)

    df["bolus_volume_delivered"] = df["bolus_volume_delivered"].clip(lower=0)

    df["iob"] = df.groupby("patient_id")["bolus_volume_delivered"].transform(_compute_iob)
    df["cob"] = df.groupby("patient_id")["carb_input"].transform(_compute_cob)

    df["glucose_roc"]   = df.groupby("patient_id")["glucose"].diff().fillna(0)
    df["glucose_accel"] = df.groupby("patient_id")["glucose_roc"].diff().fillna(0)
    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)

    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

    return df


# ─── Build sliding-window sequences ─────────────────────────────────────────

def build_sequences(df: pd.DataFrame):
    sequences = []
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("time").reset_index(drop=True)
        X     = group[FEATURE_COLS].values.astype(np.float32)
        times = group["time"].values

        for i in range(len(group) - INPUT_SIZE):
            x_seq  = X[i : i + INPUT_SIZE]
            last_t = times[i + INPUT_SIZE - 1]
            sequences.append((x_seq, {"patient_id": pid, "window_end_time": last_t}))

    return sequences


# ─── Core prediction ─────────────────────────────────────────────────────────

def predict(df: pd.DataFrame,
            model_path="nhits_model.pt",
            classifier_path="classifier.pkl",
            scaler_path="scaler.pkl"):
    """
    End-to-end prediction on a raw (unscaled) DataFrame.

    Returns a DataFrame with columns:
        patient_id, window_end_time,
        forecast_t1 … forecast_t12  (scaled glucose units),
        hypo_prob                   (probability of hypoglycaemia in next HORIZON steps)
    """
    model, clf, scaler, device = load_artifacts(model_path, classifier_path, scaler_path)

    df_proc   = preprocess(df, scaler)
    sequences = build_sequences(df_proc)

    if not sequences:
        raise ValueError("No complete sequences found — need at least "
                         f"{INPUT_SIZE} consecutive rows per patient.")

    results    = []
    batch_size = 256

    for start in range(0, len(sequences), batch_size):
        batch    = sequences[start : start + batch_size]
        x_batch  = np.stack([s[0] for s in batch])
        x_tensor = torch.tensor(x_batch, device=device)

        with torch.no_grad():
            forecast, _ = model(x_tensor)

        hidden_list = []
        residual = x_tensor.clone()
        with torch.no_grad():
            for block in model.blocks:
                backcast, _, h = block(residual)
                residual = residual - backcast.unsqueeze(-1).expand_as(residual)
                hidden_list.append(h.cpu().numpy())

        features   = np.concatenate(hidden_list, axis=-1)
        hypo_probs = clf.predict_proba(features)[:, 1]
        forecasts  = forecast.cpu().numpy()

        for i, (_, meta) in enumerate(batch):
            row = {**meta}
            for t in range(HORIZON):
                row[f"forecast_t{t+1}"] = float(forecasts[i, t])
            row["hypo_prob"] = float(hypo_probs[i])
            results.append(row)

    return pd.DataFrame(results)


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    HERE = Path(__file__).parent

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "test_df.csv"
    print(f"Loading data from: {csv_path}")

    df_raw = pd.read_csv(csv_path, parse_dates=["time"])

    preds = predict(
        df_raw,
        model_path=str(HERE / "nhits_model.pt"),
        classifier_path=str(HERE / "classifier.pkl"),
        scaler_path=str(HERE / "scaler.pkl"),
    )

    print(f"\nGenerated {len(preds)} predictions.")
    print(preds[["patient_id", "window_end_time",
                 "forecast_t1", "forecast_t12", "hypo_prob"]].head(10).to_string(index=False))

    out_path = "predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"\nFull predictions saved to: {out_path}")
