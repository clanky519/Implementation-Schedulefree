
import torch
import torch.nn as nn
from optimizers import ScheduleFreeAdamW

# Reproducibility
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Generate a simple two-class dataset
n = 1000

class_0 = torch.randn(n // 2, 2) * 0.7 + torch.tensor([-2.0, -2.0])
class_1 = torch.randn(n // 2, 2) * 0.7 + torch.tensor([2.0, 2.0])

X = torch.cat([class_0, class_1]).to(device)
y = torch.cat([
    torch.zeros(n // 2, dtype=torch.long),
    torch.ones(n // 2, dtype=torch.long)
]).to(device)

# Shuffle the dataset
indices = torch.randperm(n, device=device)
X = X[indices]
y = y[indices]

# Split into training and testing sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Small neural network
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = ScheduleFreeAdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=0.001,
    warmup_steps=20
)

# Training
print("\nTraining started")

for epoch in range(1, 101):
    optimizer.prepare_for_training()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f}")

# Evaluation
model.eval()

with torch.no_grad():
    predictions = model(X_test).argmax(dim=1)
    accuracy = (predictions == y_test).float().mean().item()

print("\nTest accuracy:", round(accuracy * 100, 2), "%")
