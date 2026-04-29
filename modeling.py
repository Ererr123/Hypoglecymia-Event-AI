import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from nhits import NHiTS


# ─── Dataset ────────────────────────────────────────────────────────────────

class CGMDataset(Dataset):
    def __init__(self, df, input_size=72, horizon=12):
        self.samples = []
        feature_cols = ["glucose", "heart_rate", "calories", "steps",
                        "basal_rate", "bolus_volume_delivered", "carb_input",
                        "iob", "cob", "glucose_roc", "glucose_accel",
                        "hour_sin", "hour_cos"]

        for _, group in df.groupby("patient_id"):
            group = group.sort_values("time").reset_index(drop=True)
            X      = group[feature_cols].values.astype(np.float32)
            y      = group["glucose"].values.astype(np.float32)
            labels = group["hypo_label"].values.astype(np.float32)

            for i in range(len(group) - input_size - horizon):
                x_seq = X[i : i + input_size]
                y_seq = y[i + input_size : i + input_size + horizon]
                label = labels[i + input_size + horizon - 1]
                self.samples.append((x_seq, y_seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, label = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(label)


# ─── Train regression ────────────────────────────────────────────────────────

def train_regression(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion  = nn.MSELoss()

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        forecast, _ = model(x)
        loss = criterion(forecast, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─── Extract features from backbone ─────────────────────────────────────────

def extract_features(model, loader, device):
    model.eval()
    all_features, all_labels, all_preds, all_targets = [], [], [], []

    with torch.no_grad():
        for x, y, label in loader:
            x = x.to(device)
            forecast, _ = model(x)

            residual = x.clone()
            hidden_states = []
            for block in model.blocks:
                backcast, _, h = block(residual)
                residual = residual - backcast.unsqueeze(-1).expand_as(residual)
                hidden_states.append(h.cpu().numpy())

            features = np.concatenate(hidden_states, axis=-1)
            all_features.append(features)
            all_labels.append(label.numpy())
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.numpy())

    return (np.concatenate(all_features),
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_targets))


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_SIZE = 72
    HORIZON    = 12
    N_FEATURES = 13
    EPOCHS     = 30
    BATCH_SIZE = 64
    LR         = 1e-4

    output_dir = r"C:\Users\djpal\OneDrive\Desktop\Bioinformetch ML Term Project"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(fr"{output_dir}\train_df.csv", parse_dates=["time"])
    test_df  = pd.read_csv(fr"{output_dir}\test_df.csv",  parse_dates=["time"])

    print("Building datasets...")
    train_ds     = CGMDataset(train_df, INPUT_SIZE, HORIZON)
    test_ds      = CGMDataset(test_df,  INPUT_SIZE, HORIZON)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = NHiTS(input_size=INPUT_SIZE, n_features=N_FEATURES, horizon=HORIZON).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3)

    # ── Phase 1: train NHiTS ──
    print("\nPhase 1: Training NHiTS for glucose forecasting...\n")
    best_loss, patience_counter, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        loss = train_regression(model, train_loader, optimizer, device)
        scheduler.step(loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 7:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # ── Phase 2: features + classifier ──
    print("\nPhase 2: Extracting features and training classifier...\n")

    train_feats, train_labels, train_preds, train_targets = extract_features(
        model, train_loader, device)
    test_feats, test_labels, test_preds, test_targets = extract_features(
        model, test_loader, device)

    # Regression metrics in mg/dL
    scaler      = joblib.load(fr"{output_dir}\scaler.pkl")
    glucose_std = scaler.scale_[0]
    mae_norm  = mean_absolute_error(test_targets[:, -1], test_preds[:, -1])
    rmse_norm = mean_squared_error( test_targets[:, -1], test_preds[:, -1]) ** 0.5
    mae  = mae_norm  * glucose_std
    rmse = rmse_norm * glucose_std
    print(f"Regression → MAE: {mae_norm:.3f} ≈ {mae:.2f} mg/dL | RMSE: {rmse_norm:.3f} ≈ {rmse:.2f} mg/dL")

    # Balanced sample weights so GBM treats both classes equally
    sample_weights = compute_sample_weight("balanced", train_labels)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        max_features="sqrt", random_state=42)
    clf.fit(train_feats, train_labels, sample_weight=sample_weights)

    auc = roc_auc_score(test_labels, clf.predict_proba(test_feats)[:, 1])
    print(f"Classification → AUC-ROC: {auc:.4f}")

    torch.save(model.state_dict(), fr"{output_dir}\nhits_model.pt")
    joblib.dump(clf,               fr"{output_dir}\classifier.pkl")
    print("\nModels saved.")
    from sklearn.metrics import classification_report, confusion_matrix

    test_probs = clf.predict_proba(test_feats)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    print(classification_report(test_labels, test_preds, target_names=["No Hypo", "Hypo"]))
    print(confusion_matrix(test_labels, test_preds))
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (test_probs >= threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        p = precision_score(test_labels, preds)
        r = recall_score(test_labels, preds)
        f = f1_score(test_labels, preds)
        print(f"Threshold {threshold:.1f} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f}")