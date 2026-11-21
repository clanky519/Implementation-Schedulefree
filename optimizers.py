
import torch
from torch.optim import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """
    Student implementation of Schedule-Free AdamW.

    Follows Algorithm 1 from:
    "The Road Less Scheduled" (NeurIPS 2024).

    Parameters are stored as the evaluation sequence x.
    z is maintained separately in optimizer state.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        warmup_steps=100,
    ):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if eps <= 0:
            raise ValueError("Epsilon must be positive.")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Betas must be in [0, 1).")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

        self.global_step = 0
        self.sum_lr_squared = {}

    @torch.no_grad()
    def prepare_for_training(self):
        """
        Move model parameters to y_t before forward/backward.
        Call this once before each training forward pass.
        """
        for group in self.param_groups:
            beta1 = group["betas"][0]

            for p in group["params"]:
                if p.grad is not None:
                    raise RuntimeError(
                        "Call prepare_for_training before computing gradients."
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["z"] = p.detach().clone()
                    state["x"] = p.detach().clone()
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                    state["sum_lr_squared"] = 0.0

                x = state["x"]
                z = state["z"]

                # y_t = (1 - beta1) z_t + beta1 x_t
                y = z.lerp(x, beta1)
                p.copy_(y)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Update z and then update the weighted average x.
        Gradients must have been computed at y_t.
        """
        #loss = None
        #if closure is not None:
        #    with torch.enable_grad():
        #       loss = closure()

        if closure is not None:
            raise RuntimeError("Closures are not supported.")
        
        self.global_step += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            warmup_steps = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    raise RuntimeError(
                        "Call prepare_for_training before the forward pass."
                    )

                z = state["z"]
                x = state["x"]
                v = state["v"]

                state["step"] += 1
                t = state["step"]

                # Adam second-moment estimate
                v.mul_(beta2).addcmul_(
                    grad, grad, value=1.0 - beta2
                )

                # Bias correction and linear warmup
                bias_factor = (1.0 - beta2 ** t) ** 0.5
                warmup_factor = 1.0

                if warmup_steps > 0:
                    warmup_factor = min(1.0, t / warmup_steps)

                lr_t = lr * bias_factor * warmup_factor

                # Weighted averaging coefficient
                state["sum_lr_squared"] += lr_t ** 2
                c = (
                    lr_t ** 2 / state["sum_lr_squared"]
                    if state["sum_lr_squared"] > 0
                    else 1.0
                )

                # Gradient update and decoupled weight decay at y_t
                denominator = v.sqrt().add(eps)

                z_new = z - lr_t * grad / denominator
                z_new = z_new - lr_t * weight_decay * p

                # x_(t+1) = (1-c)x_t + c*z_(t+1)
                x.mul_(1.0 - c).add_(z_new, alpha=c)

                z.copy_(z_new)

                # Keep model parameters at x for evaluation
                p.copy_(x)

                p.grad = None

        #return loss
        return None


class ScheduleFreeSGD_OLD(Optimizer):
    """
    Schedule-Free SGD with momentum-style interpolation.

    Parameters are stored at x.
    z is the underlying gradient-descent sequence.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_steps=100,
    ):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def prepare_for_training(self):
        for group in self.param_groups:
            beta = group["momentum"]

            for p in group["params"]:
                if p.grad is not None:
                    raise RuntimeError(
                        "Call prepare_for_training before computing gradients."
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["z"] = p.detach().clone()
                    state["x"] = p.detach().clone()
                    state["step"] = 0
                    state["sum_lr_squared"] = 0.0

                z = state["z"]
                x = state["x"]

                # y_t = (1-beta) z_t + beta x_t
                p.copy_(z.lerp(x, beta))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("Closures are not supported.")

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            warmup_steps = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    raise RuntimeError(
                        "Call prepare_for_training before the forward pass."
                    )

                z = state["z"]
                x = state["x"]

                state["step"] += 1
                t = state["step"]

                warmup_factor = 1.0
                if warmup_steps > 0:
                    warmup_factor = min(1.0, t / warmup_steps)

                lr_t = lr * warmup_factor

                state["sum_lr_squared"] += lr_t ** 2
                c = (
                    lr_t ** 2 / state["sum_lr_squared"]
                    if state["sum_lr_squared"] > 0
                    else 1.0
                )

                # Gradient update and weight decay at y_t
                z_new = z - lr_t * grad
                z_new = z_new - lr_t * weight_decay * p

                # Weighted average
                x.mul_(1.0 - c).add_(z_new, alpha=c)

                z.copy_(z_new)
                p.copy_(x)
                p.grad = None


class ScheduleFreeSGD(Optimizer):
    """
    Schedule-Free SGD following the original formulation
    in The Road Less Scheduled.

    x: evaluation sequence
    z: base SGD sequence
    y: gradient evaluation sequence
    """

    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_steps=100,
    ):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def prepare_for_training(self):
        for group in self.param_groups:
            beta = group["momentum"]

            for p in group["params"]:
                if p.grad is not None:
                    raise RuntimeError(
                        "Call prepare_for_training before computing gradients."
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["z"] = p.detach().clone()
                    state["x"] = p.detach().clone()
                    state["step"] = 0

                z = state["z"]
                x = state["x"]

                # y_t = (1-beta)z_t + beta*x_t
                p.copy_(z.lerp(x, beta))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("Closures are not supported.")

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            weight_decay = group["weight_decay"]
            warmup_steps = group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    raise RuntimeError(
                        "Call prepare_for_training before the forward pass."
                    )

                z = state["z"]
                x = state["x"]

                state["step"] += 1
                t = state["step"]

                # Linear learning-rate warmup
                warmup_factor = 1.0
                if warmup_steps > 0:
                    warmup_factor = min(1.0, t / warmup_steps)

                lr_t = lr * warmup_factor

                # Base SGD update at y_t
                z_new = z - lr_t * grad
                z_new = z_new - lr_t * weight_decay * p

                # Original Schedule-Free SGD averaging rule
                c = 1.0 / (t + 1)

                # x_(t+1) = (1-c)x_t + c*z_(t+1)
                x.mul_(1.0 - c).add_(z_new, alpha=c)

                z.copy_(z_new)

                # Return parameters to evaluation sequence
                p.copy_(x)
                p.grad = None
