
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Reproducibility
SEED = 42
torch.manual_seed(SEED)

# CIFAR-10 normalization values
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation and test: no random augmentation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load separate copies so each can use its own transform
full_train_aug = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=train_transform
)

full_train_eval = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=eval_transform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=eval_transform
)

# Identical split indices for both training dataset copies
generator = torch.Generator().manual_seed(SEED)

train_subset, val_subset = random_split(
    range(len(full_train_aug)),
    [45000, 5000],
    generator=generator
)

# Apply the appropriate transforms to each split
from torch.utils.data import Subset

train_data = Subset(full_train_aug, train_subset.indices)
val_data = Subset(full_train_eval, val_subset.indices)

# Data loaders
BATCH_SIZE = 128

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))
print("Training batches:", len(train_loader))
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# Check one batch
images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
