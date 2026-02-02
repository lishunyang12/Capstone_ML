"""
Training Script

Usage:
    python train.py --data-dir data --epochs 100 --batch-size 16
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path

from model import SiameseLSTM, count_parameters
from dataset import create_dataloaders


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for seq1, seq2, labels in dataloader:
        seq1 = seq1.to(device)
        seq2 = seq2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(seq1, seq2)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq1, seq2, labels in dataloader:
            seq1 = seq1.to(device)
            seq2 = seq2.to(device)
            labels = labels.to(device)

            pred = model(seq1, seq2)
            loss = criterion(pred, labels)

            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae = np.mean(np.abs(all_preds - all_labels))
    accuracy = np.mean((all_preds > 0.5) == (all_labels > 0.5))

    return total_loss / len(dataloader), mae, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Siamese LSTM')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='trained_model.pth', help='Output path')
    args = parser.parse_args()

    print("=" * 60)
    print("Training Siamese LSTM")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print(f"\nLoading data from {args.data_dir}/...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size
    )

    # Model
    model = SiameseLSTM().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_acc = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Val Acc: {val_acc:.2%}")

    print("-" * 60)
    print(f"Training complete. Best model saved to {args.output}")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load(args.output))
    test_loss, test_mae, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test Acc:  {test_acc:.2%}")


if __name__ == '__main__':
    main()
