
import torch
from optimizers import ScheduleFreeSGD

# One scalar parameter, starting at 1.0
p = torch.nn.Parameter(torch.tensor(1.0))

optimizer = ScheduleFreeSGD(
    [p],
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0,
    warmup_steps=0
)

# Step 1
optimizer.prepare_for_training()
p.grad = torch.tensor(2.0)
optimizer.step()

print("After step 1:", p.item())
assert abs(p.item() - 0.9) < 1e-6

# Step 2
optimizer.prepare_for_training()
p.grad = torch.tensor(2.0)
optimizer.step()

print("After step 2:", p.item())
assert abs(p.item() - 0.8) < 1e-6

print("Schedule-Free SGD update test PASSED")
