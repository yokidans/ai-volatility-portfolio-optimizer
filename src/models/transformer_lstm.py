import argparse
import json
import logging
import math
import os
import warnings
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


class FinancialDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int = 63):
        # Input validation
        if len(X.shape) != 3:
            raise ValueError(
                f"X must be 3D array (samples, timesteps, features), got {len(X.shape)}D"
            )

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.window_size = window_size

        if self.window_size > X.shape[1]:
            raise ValueError(
                f"window_size {window_size} cannot be larger than sequence length {X.shape[1]}"
            )

    def __len__(self) -> int:
        return len(self.X) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset with length {len(self)}"
            )

        # Get single window of data - shape (window_size, num_features)
        x = self.X[idx]
        y = self.y[idx + self.window_size]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.3,
        num_quantiles: int = 5,
        positional_encoding: bool = True,
    ):
        super().__init__()
        if hidden_size % nhead != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by nhead {nhead}"
            )
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles
        self.positional_encoding = positional_encoding

        self.input_proj = nn.Linear(input_size, hidden_size)

        if positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.quantile_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(num_quantiles)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() < 2:
                    continue
                if "lstm" in name.lower():
                    nn.init.orthogonal_(param)
                elif "transformer" in name.lower():
                    if "self_attn" in name.lower():
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.kaiming_normal_(param)
                else:
                    nn.init.kaiming_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        # Debugging: Print input shape
        print(f"Input shape: {x.shape}")

        # Ensure proper input dimensions
        if x.dim() == 4:
            if x.size(-1) == 1:  # If last dim is 1, squeeze it
                x = x.squeeze(-1)
            else:
                raise ValueError(f"Unexpected 4D input shape: {x.shape}")
        elif x.dim() != 3:
            raise ValueError(f"Input must be 3D (batch, seq, features), got {x.dim()}D")

        # Rest of forward pass remains the same
        x = self.input_proj(x)
        if self.positional_encoding:
            x = self.pos_encoder(x)
        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        if mc_dropout:
            lstm_out = self.dropout(lstm_out)
        outputs = [head(lstm_out) for head in self.quantile_heads]
        return torch.cat(outputs, dim=1)


def quantile_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: List[float]
) -> torch.Tensor:
    if y_true.dim() != 2 or y_true.shape[1] != 1:
        raise ValueError(f"y_true must be (batch_size, 1), got {y_true.shape}")
    if y_pred.dim() != 2 or y_pred.shape[1] != len(quantiles):
        raise ValueError(
            f"y_pred must be (batch_size, {len(quantiles)}), got {y_pred.shape}"
        )

    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, i : i + 1]
        losses.append(torch.max((q - 1) * errors, q * errors))
    return torch.mean(torch.cat(losses, dim=1))


def save_predictions_plot(outputs, targets, quantiles, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label="Actual", color="black", alpha=0.7)
    for i, q in enumerate(quantiles):
        plt.plot(outputs[:, i], label=f"Q{q:.2f}", alpha=0.7)
    plt.title("Quantile Predictions vs Actual")
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(args):
    # Setup reporting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (
        f"transformer_lstm_{args.input_size}feat_{args.hidden_size}hid_{timestamp}"
    )
    report_dir = os.path.join("reports", model_name)
    os.makedirs(report_dir, exist_ok=True)

    # Save config
    with open(os.path.join(report_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    # Data preparation
    seq_len = 63
    num_samples = 1000
    X_train = np.random.randn(num_samples, seq_len, args.input_size).astype(np.float32)
    y_train = np.random.randn(num_samples).astype(np.float32)
    X_val = np.random.randn(200, seq_len, args.input_size).astype(np.float32)
    y_val = np.random.randn(200).astype(np.float32)

    train_dataset = FinancialDataset(X_train, y_train)
    val_dataset = FinancialDataset(X_val, y_val)

    # Verify dataset output shapes
    sample_x, sample_y = train_dataset[0]
    print(f"Sample X shape: {sample_x.shape}")  # Should be (63, 8)
    print(f"Sample y shape: {sample_y.shape}")  # Should be ()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    model = TransformerLSTM(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        num_quantiles=len(DEFAULT_QUANTILES),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    metrics = []

    logger.info(f"Starting training on {device}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(X_batch)
            loss = quantile_loss(y_batch, outputs, DEFAULT_QUANTILES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True).unsqueeze(-1)
                outputs = model(X_val)
                val_loss += quantile_loss(y_val, outputs, DEFAULT_QUANTILES).item()
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(y_val.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Save epoch predictions
        epoch_outputs = np.concatenate(all_outputs)
        epoch_targets = np.concatenate(all_targets)
        np.savez(
            os.path.join(report_dir, f"epoch_{epoch+1}_predictions.npz"),
            outputs=epoch_outputs,
            targets=epoch_targets,
            quantiles=DEFAULT_QUANTILES,
        )

        # Save plot for last epoch or best epoch
        if val_loss < best_val_loss or epoch == args.epochs - 1:
            save_predictions_plot(
                epoch_outputs[:200],  # Plot first 200 samples for clarity
                epoch_targets[:200],
                DEFAULT_QUANTILES,
                os.path.join(report_dir, f"predictions_epoch{epoch+1}.png"),
            )

        # Track metrics
        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Save metrics periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            pd.DataFrame(metrics).to_csv(
                os.path.join(report_dir, "training_metrics.csv"), index=False
            )

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(report_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

    # Final save
    pd.DataFrame(metrics).to_csv(
        os.path.join(report_dir, "training_metrics.csv"), index=False
    )
    logger.info(f"Training complete. All results saved to {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer-LSTM for Financial Forecasting"
    )
    parser.add_argument(
        "--input_size", type=int, required=True, help="Number of input features"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden size of the model"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layers in Transformer and LSTM",
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )

    args = parser.parse_args()

    if args.input_size <= 0:
        raise ValueError("input_size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if args.nhead <= 0:
        raise ValueError("nhead must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    train_model(args)
