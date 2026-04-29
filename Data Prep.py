import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DATA_DIR   = r"C:\Users\djpal\OneDrive\Desktop\Bioinformetch ML Term Project\Preprocessed"
OUTPUT_DIR = r"C:\Users\djpal\OneDrive\Desktop\Bioinformetch ML Term Project"

HYPO_THRESHOLD     = 70
PREDICTION_HORIZON = 12

SCALE_COLS = ["glucose", "heart_rate", "calories", "steps",
              "basal_rate", "bolus_volume_delivered", "carb_input",
              "iob", "cob"]


def _compute_iob(bolus: pd.Series, half_life_steps: int = 11) -> pd.Series:
    """Insulin On Board via exponential decay (half-life = 55 min = 11 steps)."""
    decay = np.exp(-np.log(2) / half_life_steps)
    b = bolus.values
    iob = np.zeros(len(b))
    for i in range(len(b)):
        iob[i] = b[i] + (iob[i - 1] * decay if i > 0 else 0.0)
    return pd.Series(iob, index=bolus.index)


def _compute_cob(carbs: pd.Series, peak_steps: int = 12, tail_steps: int = 48) -> pd.Series:
    """Carbs On Board: triangular absorption peaking at 60 min, zero after 4 hr."""
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


# ── Load data ────────────────────────────────────────────────────────────────
patients = []
for file in os.listdir(DATA_DIR):
    df = pd.read_csv(os.path.join(DATA_DIR, file), sep=";")
    df["patient_id"] = file
    patients.append(df)

df_all = pd.concat(patients).reset_index(drop=True)
df_all["time"] = pd.to_datetime(df_all["time"])
df_all = df_all.sort_values(["patient_id", "time"]).reset_index(drop=True)
df_all["bolus_volume_delivered"] = df_all["bolus_volume_delivered"].clip(lower=0)

# ── Feature engineering ──────────────────────────────────────────────────────
print("Computing IOB and COB (this may take a moment)...")
df_all["iob"] = df_all.groupby("patient_id")["bolus_volume_delivered"].transform(_compute_iob)
df_all["cob"] = df_all.groupby("patient_id")["carb_input"].transform(_compute_cob)

df_all["glucose_roc"]   = df_all.groupby("patient_id")["glucose"].diff().fillna(0)
df_all["glucose_accel"] = df_all.groupby("patient_id")["glucose_roc"].diff().fillna(0)
df_all["hour_sin"] = np.sin(2 * np.pi * df_all["time"].dt.hour / 24)
df_all["hour_cos"] = np.cos(2 * np.pi * df_all["time"].dt.hour / 24)

# ── Hypoglycemia label ───────────────────────────────────────────────────────
df_all["hypo_label"] = (
    df_all.groupby("patient_id")["glucose"]
          .transform(lambda x: x.shift(-PREDICTION_HORIZON)
                                .rolling(window=PREDICTION_HORIZON, min_periods=1).min())
    < HYPO_THRESHOLD
).astype(int)

# ── Train / test split (by patient) ─────────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df_all, groups=df_all["patient_id"]))

train_df = df_all.iloc[train_idx].copy()
test_df  = df_all.iloc[test_idx].copy()

# ── Normalize — fit on train only ────────────────────────────────────────────
scaler = StandardScaler()
train_df[SCALE_COLS] = scaler.fit_transform(train_df[SCALE_COLS])
test_df[SCALE_COLS]  = scaler.transform(test_df[SCALE_COLS])

# ── Save ─────────────────────────────────────────────────────────────────────
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_df.csv"), index=False)
test_df.to_csv( os.path.join(OUTPUT_DIR, "test_df.csv"),  index=False)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print(f"Train: {train_df['patient_id'].nunique()} patients, {len(train_df)} rows")
print(f"Test:  {test_df['patient_id'].nunique()} patients, {len(test_df)} rows")
print(f"Hypo event rate (train): {train_df['hypo_label'].mean():.2%}")
print(f"Hypo event rate (test):  {test_df['hypo_label'].mean():.2%}")
print("Data prep complete. Files saved.")
