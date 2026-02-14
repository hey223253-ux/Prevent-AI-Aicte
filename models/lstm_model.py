"""
PreventAI – LSTM Time-Series Model
=====================================
PyTorch LSTM for predicting disease risk from 30-day lifestyle sequences
combined with static patient features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


class PatientDataset(Dataset):
    """Dataset combining static features and time-series sequences."""

    def __init__(self, static_features, timeseries_data, labels):
        """
        Args:
            static_features: np.array of shape (N, num_static_features)
            timeseries_data: np.array of shape (N, sequence_length, num_ts_features)
            labels: np.array of shape (N,)
        """
        self.static = torch.FloatTensor(static_features)
        self.timeseries = torch.FloatTensor(timeseries_data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.static[idx], self.timeseries[idx], self.labels[idx]


class LSTMRiskPredictor(nn.Module):
    """
    Hybrid LSTM model combining:
    - LSTM layers for time-series lifestyle data
    - Dense layers for static patient features
    - Combined prediction head
    """

    def __init__(
        self,
        ts_input_size,
        static_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.ts_input_size = ts_input_size
        self.static_input_size = static_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for time-series data
        self.lstm = nn.LSTM(
            input_size=ts_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combined prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, static, timeseries):
        # Process time-series through LSTM
        lstm_out, (h_n, _) = self.lstm(timeseries)
        ts_features = h_n[-1]  # Last hidden state from final layer

        # Process static features
        static_features = self.static_encoder(static)

        # Combine and predict
        combined = torch.cat([ts_features, static_features], dim=1)
        output = self.classifier(combined)
        return output.squeeze(-1)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


def prepare_lstm_data(static_df, ts_df, target_col, feature_cols, scaler=None):
    """
    Prepare data for LSTM model.

    Returns:
        static_features: np.array (N, num_static_features)
        timeseries_data: np.array (N, 30, 4) for [heart_rate, steps, sleep, stress]
        labels: np.array (N,)
    """
    from sklearn.preprocessing import StandardScaler

    # Static features
    patient_ids = static_df['patient_id'].values if 'patient_id' in static_df.columns else np.arange(len(static_df))
    static_feature_cols = [c for c in feature_cols
                           if c in static_df.columns and c != 'patient_id']
    static_features = static_df[static_feature_cols].values.astype(np.float32)

    # Handle NaN in static features
    col_means = np.nanmean(static_features, axis=0)
    nan_mask = np.isnan(static_features)
    static_features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Time-series data
    ts_features = ['heart_rate', 'steps', 'sleep_hours', 'stress_level']
    n_patients = len(static_df)
    seq_len = 30
    n_ts_features = len(ts_features)

    timeseries_data = np.zeros((n_patients, seq_len, n_ts_features), dtype=np.float32)

    if ts_df is not None and len(ts_df) > 0:
        ts_grouped = ts_df.groupby('patient_id')
        for i, pid in enumerate(patient_ids):
            if pid in ts_grouped.groups:
                group = ts_grouped.get_group(pid).sort_values('day')
                for j, feat in enumerate(ts_features):
                    if feat in group.columns:
                        vals = group[feat].values[:seq_len]
                        timeseries_data[i, :len(vals), j] = vals

    # Scale time-series features
    ts_scaler = StandardScaler()
    original_shape = timeseries_data.shape
    ts_flat = timeseries_data.reshape(-1, n_ts_features)
    ts_flat = ts_scaler.fit_transform(ts_flat)
    timeseries_data = ts_flat.reshape(original_shape)

    # Labels
    labels = static_df[target_col].values.astype(np.float32) if target_col in static_df.columns else np.zeros(n_patients, dtype=np.float32)

    return static_features, timeseries_data, labels, ts_scaler


def train_lstm_model(
    static_features, timeseries_data, labels,
    val_static=None, val_ts=None, val_labels=None,
    epochs=30, batch_size=64, lr=0.001, device=None
):
    """
    Train the LSTM risk predictor.

    Returns trained model and training history.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_dataset = PatientDataset(static_features, timeseries_data, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_static is not None:
        val_dataset = PatientDataset(val_static, val_ts, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    ts_input_size = timeseries_data.shape[2]
    static_input_size = static_features.shape[1]

    model = LSTMRiskPredictor(
        ts_input_size=ts_input_size,
        static_input_size=static_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for static, ts, target in train_loader:
            static, ts, target = static.to(device), ts.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(static, ts)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for static, ts, target in val_loader:
                    static, ts, target = static.to(device), ts.to(device), target.to(device)
                    output = model(static, ts)
                    loss = criterion(output, target)
                    val_losses.append(loss.item())
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            try:
                val_auc = roc_auc_score(val_targets, val_preds)
            except ValueError:
                val_auc = 0.5

            history['val_loss'].append(avg_val_loss)
            history['val_auc'].append(val_auc)
            scheduler.step(avg_val_loss)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f}")
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"  ✓ Restored best model (AUC: {best_auc:.4f})")

    model.eval()
    return model, history


def predict_lstm(model, static_features, timeseries_data, device=None):
    """Get predictions from trained LSTM model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        static = torch.FloatTensor(static_features).to(device)
        ts = torch.FloatTensor(timeseries_data).to(device)
        probs = model(static, ts).cpu().numpy()

    preds = (probs > 0.5).astype(int)
    return preds, probs
