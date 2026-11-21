
import torch
from optimizers import ScheduleFreeAdamW

# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# A single trainable parameter, initialized at 0
w = torch.nn.Parameter(torch.tensor([0.0], device=device))

optimizer = ScheduleFreeAdamW(
    [w],
    lr=0.1,
    weight_decay=0.0,
    warmup_steps=10,
)

target = torch.tensor([5.0], device=device)

print("\nStarting optimization")

for step in range(1, 501):

    # Put parameters at y_t before evaluating the loss
    optimizer.prepare_for_training()

    # Simple quadratic objective
    loss = ((w - target) ** 2).sum()

    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.8f} | "
            f"Parameter: {w.item():.6f}"
        )

print("\nExpected parameter: 5.0")
print("Final parameter:", w.item())
