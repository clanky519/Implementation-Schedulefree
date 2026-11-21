
import csv
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SmallResNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEEDS = [14, 22, 123]

METHODS = [
    "sgd_constant",
    "sgd_cosine",
    "adamw_cosine",
    "official_schedule_free_sgd",
    "official_schedule_free_adamw"
]

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# Test dataset
# --------------------------------------------------

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=test_transform
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    start = time.time()

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    elapsed = time.time() - start

    return {
        "test_loss": total_loss / total,
        "test_accuracy": 100.0 * correct / total,
        "evaluation_time_seconds": elapsed
    }


# --------------------------------------------------
# Evaluate all checkpoints
# --------------------------------------------------

print("Device:", DEVICE)
print("Test images:", len(test_data))
print("Checkpoints to evaluate:", len(SEEDS) * len(METHODS))

all_results = []

for seed in SEEDS:
    for method in METHODS:

        filename = f"{method}_seed{seed}.pt"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

        print(f"\nEvaluating {filename}...")

        model = SmallResNet().to(DEVICE)

        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(state_dict)

        metrics = evaluate(model, test_loader)

        result = {
            "method": method,
            "seed": seed,
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"],
            "evaluation_time_seconds": metrics["evaluation_time_seconds"]
        }

        all_results.append(result)

        print(
            f"Test accuracy: {metrics['test_accuracy']:.2f}% | "
            f"Test loss: {metrics['test_loss']:.4f}"
        )

        del model
        del state_dict

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------
# Save individual results
# --------------------------------------------------

individual_path = os.path.join(
    RESULTS_DIR,
    "final_test_results.csv"
)

with open(individual_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "method",
            "seed",
            "test_accuracy",
            "test_loss",
            "evaluation_time_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# Calculate mean and sample standard deviation
# --------------------------------------------------

summary = []

for method in METHODS:

    accuracies = [
        row["test_accuracy"]
        for row in all_results
        if row["method"] == method
    ]

    losses = [
        row["test_loss"]
        for row in all_results
        if row["method"] == method
    ]

    accuracy_tensor = torch.tensor(accuracies, dtype=torch.float64)
    loss_tensor = torch.tensor(losses, dtype=torch.float64)

    summary.append({
        "method": method,
        "mean_test_accuracy": accuracy_tensor.mean().item(),
        "std_test_accuracy": accuracy_tensor.std(unbiased=True).item(),
        "mean_test_loss": loss_tensor.mean().item()
    })


summary_path = os.path.join(
    RESULTS_DIR,
    "final_test_summary.csv"
)

with open(summary_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "method",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_loss"
        ]
    )

    writer.writeheader()
    writer.writerows(summary)


# --------------------------------------------------
# Display final summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL CIFAR-10 TEST RESULTS")
print("=" * 65)

for row in summary:
    print(
        f"{row['method']:32s} "
        f"{row['mean_test_accuracy']:.2f}% "
        f"+/- {row['std_test_accuracy']:.2f}"
    )

print("\nIndividual results:", individual_path)
print("Summary results:", summary_path)
print("\nALL TEST EVALUATIONS COMPLETED")
