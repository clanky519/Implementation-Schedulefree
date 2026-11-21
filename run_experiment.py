
import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from model import SmallResNet
from data import train_loader, val_loader
from train_utils import train_one_epoch, evaluate
#from optimizers import ScheduleFreeSGD, ScheduleFreeAdamW


# ---------------- CONFIGURATION ----------------

import sys
import schedulefree


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42


EPOCHS = 30
WARMUP_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

EXPERIMENTS = [
    {
        "name": "sgd_constant",
        "type": "sgd",
        "lr": 0.1,
        "weight_decay": 0.0005,
        "schedule": "constant"
    },
    {
        "name": "sgd_cosine",
        "type": "sgd",
        "lr": 0.1,
        "weight_decay": 0.0005,
        "schedule": "cosine"
    },
    {
        "name": "adamw_cosine",
        "type": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "schedule": "cosine"
    },
    {
        "name": "official_schedule_free_sgd",
        "type": "sf_sgd",
        "lr": 0.1,
        "weight_decay": 0.0005,
        "schedule": "none"
    },
    {
        "name": "official_schedule_free_adamw",
        "type": "sf_adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "schedule": "none"
    }
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer_old(model, config):
    common = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"]
    }

    if config["type"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=0.9,
            **common
        )

    if config["type"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            betas=(0.9, 0.999),
            **common
        )

    if config["type"] == "sf_sgd":
        return ScheduleFreeSGD(
            model.parameters(),
            momentum=0.9,
            warmup_steps=WARMUP_EPOCHS * len(train_loader),
            **common
        )

    if config["type"] == "sf_adamw":
        return ScheduleFreeAdamW(
            model.parameters(),
            betas=(0.9, 0.999),
            warmup_steps=WARMUP_EPOCHS * len(train_loader),
            **common
        )


def build_optimizer(model, config):
    common = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"]
    }

    if config["type"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=0.9,
            **common
        )

    if config["type"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            betas=(0.9, 0.999),
            **common
        )

    if config["type"] == "sf_sgd":
        return schedulefree.SGDScheduleFree(
            model.parameters(),
            momentum=0.9,
            warmup_steps=WARMUP_EPOCHS * len(train_loader),
            **common
        )

    if config["type"] == "sf_adamw":
        return schedulefree.AdamWScheduleFree(
            model.parameters(),
            betas=(0.9, 0.999),
            warmup_steps=WARMUP_EPOCHS * len(train_loader),
            **common
        )

    raise ValueError(f"Unknown optimizer type: {config['type']}")



def run_experiment(config):
    print("\n" + "=" * 60)
    print("EXPERIMENT:", config["name"])
    print("=" * 60)

    set_seed(SEED)

    model = SmallResNet().to(DEVICE)
    optimizer = build_optimizer(model, config)

    scheduler = None
    if config["schedule"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )

    csv_path = f"results/{config['name']}_seed{SEED}.csv"

    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_time",
        "learning_rate"
    ]

    best_accuracy = 0.0
    best_epoch = 0
    total_start = time.time()

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            train = train_one_epoch(
                model, train_loader, optimizer, DEVICE
            )

            val = evaluate(
                model, val_loader, DEVICE, optimizer
            )

            if scheduler is not None:
                scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "epoch_time": train["time"],
                "learning_rate": lr_now
            }

            writer.writerow(row)
            file.flush()

            if val["accuracy"] > best_accuracy:
                best_accuracy = val["accuracy"]
                best_epoch = epoch

                torch.save(
                    model.state_dict(),
                    f"checkpoints/{config['name']}_seed{SEED}.pt"
                )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train loss {train['loss']:.4f} | "
                f"Train acc {train['accuracy']:.2f}% | "
                f"Val loss {val['loss']:.4f} | "
                f"Val acc {val['accuracy']:.2f}% | "
                f"Time {train['time']:.1f}s"
            )

    total_time = time.time() - total_start

    print("\nCompleted:", config["name"])
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Total time: {total_time / 60:.1f} minutes")

    return {
        "method": config["name"],
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "total_time_seconds": total_time
    }


# ---------------- RUN ALL EXPERIMENTS ----------------

print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Seed:", SEED)
print("Number of experiments:", len(EXPERIMENTS))

summary = []

for experiment in EXPERIMENTS:
    result = run_experiment(experiment)
    summary.append(result)

#summary_path = "results/benchmark_summary.csv"
summary_path = f"results/official_benchmark_summary_seed{SEED}.csv"

with open(summary_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "method",
            "best_validation_accuracy",
            "best_epoch",
            "total_time_seconds"
        ]
    )
    writer.writeheader()
    writer.writerows(summary)

print("\nALL EXPERIMENTS COMPLETED")
print("Summary saved to:", summary_path)

for result in summary:
    print(result)
