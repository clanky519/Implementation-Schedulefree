
import torch
from model import SmallResNet
from data import train_loader, val_loader
from train_utils import train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallResNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

print("Starting one-epoch training test...")

train_metrics = train_one_epoch(
    model, train_loader, optimizer, device
)

val_metrics = evaluate(
    model, val_loader, device
)

print("\nTraining results:")
print(f"Loss: {train_metrics['loss']:.4f}")
print(f"Accuracy: {train_metrics['accuracy']:.2f}%")
print(f"Time: {train_metrics['time']:.2f} seconds")

print("\nValidation results:")
print(f"Loss: {val_metrics['loss']:.4f}")
print(f"Accuracy: {val_metrics['accuracy']:.2f}%")
