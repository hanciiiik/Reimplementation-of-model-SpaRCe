# PyTorch reimplementation of SpaRCe (trainable threshold + readout)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


# Configuration of model SpaRCe
@dataclass
class SpaRCeConfig:
    N: int = 1000   # number of neurons in reservoir
    T: int = 28     # number of time stamps
    input_dim: int = 28     # dimension of input
    alpha: float = 0.17     # leak rate - how quickly reservoir states change over time
    rho: float = 0.97       # spectral radius od matrix W
    gamma: float = 0.1      # input scaling - regulates the influence of the external input on the reservoir
    pER: float = 0.01       # Erdős–Rényi connection probability of edges

    activation: str = "tanh"

    # SpaRCe thresholding
    percentile: float = 95.0        # percentile threshold - 95% of reservoir will be sparse
    relu_power: int = 1

    # Parameters to optimise
    lr_wout: float = 2e-3
    lr_theta: float = 2e-4
    weight_decay: float = 0.0

    # Training
    batch_size: int = 20
    epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Train/val split
    val_size: int = 10_000

    # For percentile computation (RAM heavy for N=1000 -> D=28000)
    max_samples_for_percentile: Optional[int] = 20_000

    seed: int = 42


# def set_seed(seed: int) -> None:
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# Set seed to reproduce results
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Get activation function according to its name
def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {name}")

# Convert class labels into one-hot representation
def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


# Reservoir
class FixedReservoir(nn.Module):
    """
    Fixed reservoir dynamics (leaky integrator).

    V(t+1) = (1 - alpha) V(t) + alpha f( gamma W_in u(t) + W V(t) )

    W is sparse ER(pER) and scaled to spectral radius rho.
    """
    def __init__(self, cfg: SpaRCeConfig):
        super().__init__()
        self.cfg = cfg
        self.f = get_activation(cfg.activation)

        # Input matrix W_in: [N, input_dim] scaled by parameter gamma
        W_in = torch.empty(cfg.N, cfg.input_dim).uniform_(-1.0, 1.0)
        W_in *= (cfg.gamma / math.sqrt(cfg.input_dim))

        # Reservoir matrix W: [N, N] with initial ER sparsity pER
        W = torch.empty(cfg.N, cfg.N).uniform_(-1.0, 1.0)
        if cfg.pER is not None and cfg.pER < 1.0:
            mask = (torch.rand(cfg.N, cfg.N) < cfg.pER).float()
            W = W * mask

        # Scale W to spectral radius rho
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(W).abs()
            sr = eigvals.max().clamp_min(1e-6)
            W = (W / sr) * cfg.rho

        # Fix parameters
        self.register_buffer("W_in", W_in)
        self.register_buffer("W", W)

    #
    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        # u_seq: [B, T, input_dim]
        # B: batch size, T: number of time stamps, input_dim: length of one timestamp
        B, T, Din = u_seq.shape
        assert T == self.cfg.T, f"Expected T={self.cfg.T}, got {T}"
        assert Din == self.cfg.input_dim, f"Expected input_dim={self.cfg.input_dim}, got {Din}"

        V = torch.zeros(B, self.cfg.N, device=u_seq.device)
        V_seq = []

        for t in range(T):
            u_t = u_seq[:, t, :]
            pre = (u_t @ self.W_in.t()) + (V @ self.W.t())
            # compute reservoir states
            V = (1.0 - self.cfg.alpha) * V + self.cfg.alpha * self.f(pre)
            V_seq.append(V)

        # returns V_seq: [B, T, N]
        return torch.stack(V_seq, dim=1)


# MNIST dataset preparation
class MNISTColumnSequenceEncoder(nn.Module):
    # column by column MNIST encoding
    # x_img: [B, 1, 28, 28] -> u_seq: [B, 28, 28] - time step t corresponds to the t-th column
    def __init__(self):
        super().__init__()

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        x = x_img.squeeze(1)      # [B, 28, 28] (H,W)
        x = x.transpose(1, 2)     # [B, 28(time=columns), 28(features=rows)]
        return x


# SpaRCe readout
class SpaRCeReadout(nn.Module):
    def __init__(self, cfg: SpaRCeConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        # dimension of the feature vector after “flattening”
        self.D = cfg.N * cfg.T
        self.num_classes = num_classes

        # Fix parameter theta_g
        self.register_buffer("theta_g", torch.zeros(1, self.D))
        # Trainable parameter theta_thilde
        self.theta_thilde = nn.Parameter(torch.randn(1, self.D) / float(cfg.N))
        # Trainable parameter output matrix
        self.W_out = nn.Parameter(torch.empty(self.D, num_classes).uniform_(0.0, 1.0) / float(cfg.N))

    # Precalculated and fixed neural threshold
    def set_theta_g(self, theta_g: torch.Tensor) -> None:
        if theta_g.dim() == 1:
            theta_g = theta_g.unsqueeze(0)
        assert theta_g.shape == (1, self.D)
        self.theta_g.data.copy_(theta_g)

    def forward(self, V_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, N = V_seq.shape
        assert (T, N) == (self.cfg.T, self.cfg.N)

        # V_thilde before sparsification
        V_thilde = V_seq.reshape(B, T * N)

        # Formulas 4, 5
        theta = self.theta_g + self.theta_thilde

        a = torch.abs(V_thilde) - theta
        z = F.relu(a)
        if self.cfg.relu_power != 1:
            z = z.pow(self.cfg.relu_power)
        x = torch.sign(V_thilde) * z

        logits = x @ self.W_out
        return V_thilde, x, logits


# The whole model SpaRCe
class SpaRCeModel(nn.Module):
    def __init__(self, cfg: SpaRCeConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = MNISTColumnSequenceEncoder()
        self.reservoir = FixedReservoir(cfg)
        self.readout = SpaRCeReadout(cfg, num_classes)

    def forward(self, x_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        u_seq = self.encoder(x_img)
        V_seq = self.reservoir(u_seq)
        V_thilde, x, logits = self.readout(V_seq)
        return {"V_thilde": V_thilde, "x": x, "logits": logits}


# Pre-training: computing theta_g (fixed threshold)
@torch.no_grad()
def fit_theta_g_percentiles(
    model: SpaRCeModel,
    train_loader: DataLoader,
    percentile: float,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    all_abs = []
    seen = 0

    for x_img, _y in train_loader:
        x_img = x_img.to(device)
        V_thilde = model(x_img)["V_thilde"]
        all_abs.append(V_thilde.abs().cpu())
        seen += V_thilde.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    abs_mat = torch.cat(all_abs, dim=0)
    if max_samples is not None and abs_mat.shape[0] > max_samples:
        abs_mat = abs_mat[:max_samples]

    q = float(percentile) / 100.0
    theta_g = torch.quantile(abs_mat, q=q, dim=0)
    return theta_g.unsqueeze(0)                    # shape [1, D]


# Evaluation of the model
@torch.no_grad()
def evaluate_bce(model: SpaRCeModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    # Returns (avg_loss, avg_accuracy) on loader using BCE-with-logits + one-hot.
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x_img, y in loader:
        x_img = x_img.to(device)
        y = y.to(device)

        logits = model(x_img)["logits"]
        y_oh = one_hot(y, num_classes=logits.shape[1])
        loss = bce(logits, y_oh)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


# -----------------------------
# 9) Training loop (VAL during training, TEST once at the end)
# -----------------------------
def train_sparce(
    cfg: SpaRCeConfig,
    model: SpaRCeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> None:
    device = torch.device(cfg.device)
    model.to(device)

    # theta_g from TRAIN ONLY
    theta_g = fit_theta_g_percentiles(
        model=model,
        train_loader=train_loader,
        percentile=cfg.percentile,
        device=device,
        max_samples=cfg.max_samples_for_percentile,
    ).to(device)
    model.readout.set_theta_g(theta_g)

    # (B) optimizers
    opt_wout = torch.optim.Adam([model.readout.W_out], lr=cfg.lr_wout, weight_decay=cfg.weight_decay)
    opt_theta = torch.optim.Adam([model.readout.theta_thilde], lr=cfg.lr_theta, weight_decay=0.0)

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n_batches = 0

        for x_img, y in train_loader:
            x_img = x_img.to(device)
            y = y.to(device)

            logits = model(x_img)["logits"]
            y_oh = one_hot(y, num_classes=logits.shape[1])
            loss = bce(logits, y_oh)

            opt_wout.zero_grad(set_to_none=True)
            opt_theta.zero_grad(set_to_none=True)
            loss.backward()
            opt_wout.step()
            opt_theta.step()

            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            tr_loss += loss.item()
            tr_acc += acc
            n_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc /= max(1, n_batches)

        # Validation (NOT test)
        va_loss, va_acc = evaluate_bce(model, val_loader, device)

        # Coding level (just as a sanity check)
        with torch.no_grad():
            x_img0, _ = next(iter(train_loader))
            x_img0 = x_img0.to(device)
            x0 = model(x_img0)["x"]
            coding = (x0 != 0).float().mean().item()

        print(
            f"[Epoch {epoch+1}/{cfg.epochs}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}"
        )

# Testing the model with test dataset
@torch.no_grad()
def final_test(cfg: SpaRCeConfig, model: SpaRCeModel, test_loader: DataLoader) -> None:
    device = torch.device(cfg.device)
    test_loss, test_acc = evaluate_bce(model, test_loader, device)
    print(f"[FINAL TEST] loss={test_loss:.4f} acc={test_acc:.4f}")


# Running the model with MNIST dataset
def main_mnist_example():
    set_seed(42)

    from torchvision import datasets, transforms

    tfm = transforms.ToTensor()
    full_train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)  # 60k
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)      # 10k

    cfg = SpaRCeConfig(
        N=1000,
        T=28,
        input_dim=28,
        alpha=0.17,
        rho=0.97,
        gamma=0.1,
        pER=0.01,
        lr_wout=2e-3,
        lr_theta=2e-4,
        batch_size=20,
        epochs=5,
        percentile=95.0,
        val_size=10_000,
        max_samples_for_percentile=20_000,
        seed=42,
    )

    # Reproducible split: 50k train, 10k val
    train_size = len(full_train_ds) - cfg.val_size
    val_size = cfg.val_size
    assert train_size > 0 and val_size > 0

    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SpaRCeModel(cfg, num_classes=10)

    # Train with validation monitoring
    train_sparce(cfg, model, train_loader, val_loader)

    # Final report on test set (exactly once)
    final_test(cfg, model, test_loader)

    print("finished")

if __name__ == "__main__":
    main_mnist_example()
