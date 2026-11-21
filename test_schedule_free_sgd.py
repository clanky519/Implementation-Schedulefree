
import torch

from model import SmallResNet
from data import train_loader, val_loader
from train_utils import train_one_epoch, evaluate
from optimizers import ScheduleFreeSGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
model = SmallResNet().to(device)

optimizer = ScheduleFreeSGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    warmup_steps=100
)

print("Testing Schedule-Free SGD on CIFAR-10")

for epoch in range(3):
    train = train_one_epoch(model, train_loader, optimizer, device)
    val = evaluate(model, val_loader, device)

    print(f"\nEpoch {epoch + 1}/3")
    print(f"Train loss: {train['loss']:.4f}")
    print(f"Train accuracy: {train['accuracy']:.2f}%")
    print(f"Validation loss: {val['loss']:.4f}")
    print(f"Validation accuracy: {val['accuracy']:.2f}%")
    print(f"Time: {train['time']:.2f}s")
