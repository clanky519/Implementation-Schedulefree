
import torch

from model import SmallResNet
from data import train_loader, val_loader
from train_utils import train_one_epoch, evaluate
from optimizers import ScheduleFreeAdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallResNet().to(device)

optimizer = ScheduleFreeAdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    warmup_steps=100
)

print("Testing Schedule-Free AdamW on CIFAR-10")

for epoch in range(3):
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, device
    )

    val_metrics = evaluate(
        model, val_loader, device
    )

    print(f"\nEpoch {epoch + 1}/3")
    print(f"Train loss: {train_metrics['loss']:.4f}")
    print(f"Train accuracy: {train_metrics['accuracy']:.2f}%")
    print(f"Validation loss: {val_metrics['loss']:.4f}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"Time: {train_metrics['time']:.2f}s")
