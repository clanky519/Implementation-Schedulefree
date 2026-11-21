
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

SEEDS = [14, 22, 123]

METHODS = {
    "sgd_constant": "SGD (Constant LR)",
    "sgd_cosine": "SGD (Cosine LR)",
    "adamw_cosine": "AdamW (Cosine LR)",
    "official_schedule_free_sgd": "Schedule-Free SGD",
    "official_schedule_free_adamw": "Schedule-Free AdamW"
}

os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# Load all experiment histories
# --------------------------------------------------

all_data = []

for method, label in METHODS.items():
    for seed in SEEDS:

        filename = f"{method}_seed{seed}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}")

        df = pd.read_csv(filepath)

        df["method"] = method
        df["method_label"] = label
        df["seed"] = seed

        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("Loaded experiment records:", len(data))
print("Methods:", data["method"].nunique())
print("Seeds:", data["seed"].nunique())
print("Epochs per run:", data["epoch"].nunique())


# --------------------------------------------------
# Calculate mean and standard deviation per epoch
# --------------------------------------------------

metrics = [
    ("train_loss", "Training Loss", "Loss"),
    ("val_loss", "Validation Loss", "Loss"),
    ("val_accuracy", "Validation Accuracy", "Accuracy (%)"),
    ("train_accuracy", "Training Accuracy", "Accuracy (%)")
]

summary_data = {}

for metric, title, ylabel in metrics:

    summary_data[metric] = {}

    for method, label in METHODS.items():

        subset = data[data["method"] == method]

        grouped = subset.groupby("epoch")[metric].agg(
            ["mean", "std"]
        )

        summary_data[metric][method] = grouped


# --------------------------------------------------
# Plot mean curves with standard deviation bands
# --------------------------------------------------

for metric, title, ylabel in metrics:

    plt.figure(figsize=(10, 6))

    for method, label in METHODS.items():

        grouped = summary_data[metric][method]

        epochs = grouped.index.to_numpy()
        mean = grouped["mean"].to_numpy()
        std = grouped["std"].fillna(0).to_numpy()

        plt.plot(
            epochs,
            mean,
            linewidth=2,
            label=label
        )

        plt.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.12
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{title} Across Three Seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = os.path.join(PLOT_DIR, f"{metric}_comparison.png")

    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


# --------------------------------------------------
# Training time comparison
# --------------------------------------------------

time_data = data.groupby(
    ["method", "seed"]
)["epoch_time"].sum().reset_index()

time_summary = time_data.groupby("method")["epoch_time"].agg(
    ["mean", "std"]
)

plt.figure(figsize=(10, 6))

labels = [METHODS[m] for m in time_summary.index]
means = time_summary["mean"].to_numpy()
stds = time_summary["std"].fillna(0).to_numpy()

plt.bar(
    labels,
    means,
    yerr=stds,
    capsize=5
)

plt.ylabel("Total Training Time (seconds)")
plt.title("Training Time Comparison Across Three Seeds")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output = os.path.join(PLOT_DIR, "training_time_comparison.png")
plt.savefig(output, dpi=300)
plt.close()

print("Saved:", output)


# --------------------------------------------------
# Export mean and standard deviation tables
# --------------------------------------------------

for metric, title, ylabel in metrics:

    rows = []

    for method, label in METHODS.items():

        grouped = summary_data[metric][method]

        for epoch, values in grouped.iterrows():
            rows.append({
                "method": method,
                "epoch": epoch,
                "mean": values["mean"],
                "std": values["std"]
            })

    output_df = pd.DataFrame(rows)

    output = os.path.join(
        RESULTS_DIR,
        f"{metric}_mean_std.csv"
    )

    output_df.to_csv(output, index=False)
    print("Saved:", output)


# --------------------------------------------------
# Final test accuracy comparison
# --------------------------------------------------

test_summary_path = os.path.join(
    RESULTS_DIR,
    "final_test_summary.csv"
)

test_summary = pd.read_csv(test_summary_path)

plt.figure(figsize=(10, 6))

labels = [
    METHODS[m] for m in test_summary["method"]
]

plt.bar(
    labels,
    test_summary["mean_test_accuracy"],
    yerr=test_summary["std_test_accuracy"],
    capsize=5
)

plt.ylabel("Test Accuracy (%)")
plt.title("Final CIFAR-10 Test Accuracy Across Three Seeds")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output = os.path.join(PLOT_DIR, "final_test_accuracy.png")
plt.savefig(output, dpi=300)
plt.close()

print("Saved:", output)


print("\n" + "=" * 55)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("Output directory:", PLOT_DIR)
print("=" * 55)
