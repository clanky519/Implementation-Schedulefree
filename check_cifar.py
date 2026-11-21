
import torch
import torchvision
from torchvision import datasets, transforms

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("\nDataset downloaded successfully")
print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print("Image shape:", train_data[0][0].shape)
print("Number of classes:", len(train_data.classes))
print("Classes:", train_data.classes)
