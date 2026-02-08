# SpaRCe_MNIST.py
# SpaRCe (PyTorch) – simplified, "student style" version.
# - keeps the same equations / logic
# - avoids extra dependencies (no NumPy / torchvision required)
# - MNIST is read from IDX files (downloaded automatically)

from __future__ import annotations

import os
import gzip
import struct
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Simple config (edit values here)
# -----------------------------
class Cfg:
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MNIST
    mnist_dir = "./MNIST_data"
    T = 28            # time steps (columns)
    input_dim = 28    # input per step
    n_classes = 10

    # Reservoir (single for MNIST)
    N = 1000
    dt = 0.01
    tau_m = 0.03
    tau_M = 2.0
    diluition = 0.99  # keep author's spelling to match original code

    # Precompute / eval
    batch_size_states = 256
    batch_size_eval = 512

    # Training (iterations)
    lr_wout = 0.002
    lr_theta_factor = 0.1
    adam_eps = 1e-7


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# MNIST download + IDX reader
# -----------------------------
MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_GZ_FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte",
}


def ensure_mnist_idx_present(mnist_dir: str) -> None:
    os.makedirs(mnist_dir, exist_ok=True)

    # already extracted?
    if all(os.path.exists(os.path.join(mnist_dir, out_name)) for out_name in MNIST_GZ_FILES.values()):
        return

    print(f"[MNIST] Downloading into: {os.path.abspath(mnist_dir)}")
    for gz_name, out_name in MNIST_GZ_FILES.items():
        gz_path = os.path.join(mnist_dir, gz_name)
        out_path = os.path.join(mnist_dir, out_name)

        if not os.path.exists(gz_path):
            url = MNIST_BASE_URL + gz_name
            print(f"[MNIST]  download {gz_name}")
            urllib.request.urlretrieve(url, gz_path)

        if not os.path.exists(out_path):
            print(f"[MNIST]  decompress {gz_name} -> {out_name}")
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                f_out.write(f_in.read())


def read_idx(path: str) -> torch.Tensor:
    """Reads MNIST .idx files into torch tensors (uint8)."""
    with open(path, "rb") as f:
        data = f.read()

    (magic,) = struct.unpack(">I", data[0:4])

    # images: 2051
    if magic == 2051:
        n, rows, cols = struct.unpack(">III", data[4:16])
        buf = bytearray(memoryview(data[16:]))  # make it writable for frombuffer
        t = torch.frombuffer(buf, dtype=torch.uint8)
        return t.reshape(n, rows, cols)

    # labels: 2049
    if magic == 2049:
        (n,) = struct.unpack(">I", data[4:8])
        buf = bytearray(memoryview(data[8:]))
        t = torch.frombuffer(buf, dtype=torch.uint8)
        return t.reshape(n)

    raise ValueError(f"Unknown IDX magic {magic} in {path}")


class MNISTIDX(Dataset):
    def __init__(self, images_path: str, labels_path: str):
        self.images = read_idx(images_path)
        self.labels = read_idx(labels_path)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("MNIST images/labels length mismatch")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx].to(torch.float32) / 255.0  # [28,28]
        x = x.unsqueeze(0)  # [1,28,28]
        y = self.labels[idx].to(torch.int64)
        return x, y


def load_mnist(cfg: Cfg):
    ensure_mnist_idx_present(cfg.mnist_dir)

    tr_images = os.path.join(cfg.mnist_dir, "train-images-idx3-ubyte")
    tr_labels = os.path.join(cfg.mnist_dir, "train-labels-idx1-ubyte")
    te_images = os.path.join(cfg.mnist_dir, "t10k-images-idx3-ubyte")
    te_labels = os.path.join(cfg.mnist_dir, "t10k-labels-idx1-ubyte")

    ds_train = MNISTIDX(tr_images, tr_labels)  # full 60k
    ds_test = MNISTIDX(te_images, te_labels)

    # shuffle=False because we precompute in fixed order (same as original)
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size_states, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size_states, shuffle=False, num_workers=0)
    return train_loader, test_loader


# -----------------------------
# ESN helpers
# -----------------------------
def alpha_pho(dt: float, tau_m: float, tau_M: float):
    alpha = dt / (2.0 * tau_m)
    pho = 1.0 - 2.0 * tau_m / tau_M
    return float(alpha), float(pho)


@torch.no_grad()
def spectral_radius(W: torch.Tensor) -> float:
    # keep eigenvalue-based scaling (like the TF/Numpy version)
    ev = torch.linalg.eigvals(W.detach().cpu())
    return float(max(ev.abs().max().item(), 1e-8))


@torch.no_grad()
def make_sparse_W(N: int, diluition: float, pho: float, seed: int, device: torch.device) -> torch.Tensor:
    # random dense weights in [-1,1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.rand((N, N), generator=g) * 2.0 - 1.0).to(torch.float32)

    # mask: keep ~ (1 - diluition) fraction
    D = (torch.rand((N, N), generator=g) > diluition).to(torch.float32)
    W = W * D

    # scale to spectral radius pho
    sr = spectral_radius(W)
    W = pho * W / sr
    return W.to(device)


@torch.no_grad()
def mnist_to_sequence(x: torch.Tensor) -> torch.Tensor:
    # input x: [B,1,28,28] -> sequence: [B,28,28]
    return x.squeeze(1).to(torch.float32)


class ESN1:
    def __init__(self, cfg: Cfg, seed: int, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.alpha, pho = alpha_pho(cfg.dt, cfg.tau_m, cfg.tau_M)
        self.W = make_sparse_W(cfg.N, cfg.diluition, pho, seed + 1, device)

        g = torch.Generator(device="cpu").manual_seed(seed + 2)
        W_in = torch.randn((cfg.N, cfg.input_dim), generator=g).to(torch.float32)
        self.W_in = (0.1 * W_in.t()).to(device)  # (28, N)

    @torch.no_grad()
    def forward_states(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B,28,28] ; we use columns as time steps -> u_t = X[:,:,t]
        B = X.shape[0]
        v = torch.zeros((B, self.cfg.N), device=self.device, dtype=torch.float32)

        states = []
        for t in range(self.cfg.T):
            u = X[:, :, t]  # [B,28]
            v = (1.0 - self.alpha) * v + self.alpha * torch.tanh(v @ self.W + u @ self.W_in)
            states.append(v)

        return torch.stack(states, dim=2)  # [B,N,T]


# -----------------------------
# SpaRCe readout (soft-threshold + linear classifier)
# -----------------------------
class SpaRCeReadout(nn.Module):
    def __init__(self, D: int, N_scale: int, n_classes: int, theta_g_flat: torch.Tensor, seed: int):
        super().__init__()

        g = torch.Generator(device="cpu").manual_seed(seed + 100)
        self.W_out = nn.Parameter(torch.rand((D, n_classes), generator=g).to(torch.float32) / float(N_scale))

        g = torch.Generator(device="cpu").manual_seed(seed + 101)
        self.theta_i = nn.Parameter(torch.randn((1, D), generator=g).to(torch.float32) / float(N_scale))

        # theta_g is fixed (computed from training set)
        self.register_buffer("theta_g", theta_g_flat.reshape(1, D).to(torch.float32))

    def forward(self, s_flat: torch.Tensor) -> torch.Tensor:
        thr = self.theta_g + self.theta_i
        s_sparse = torch.sign(s_flat) * F.relu(s_flat.abs() - thr)
        return s_sparse @ self.W_out


# -----------------------------
# Precompute states + theta_g
# -----------------------------
@torch.no_grad()
def precompute_states(reservoir: ESN1, loader: DataLoader, device: torch.device):
    X_all = []
    y_all = []
    states_all = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        Xseq = mnist_to_sequence(xb)
        st = reservoir.forward_states(Xseq)  # [B,N,T]

        states_all.append(st.cpu())
        y_all.append(yb.cpu())

    states = torch.cat(states_all, dim=0).contiguous()  # [S,N,T]
    y = torch.cat(y_all, dim=0).contiguous()            # [S]

    S, N, T = states.shape
    X_flat = states.reshape(S, N * T).contiguous()
    return X_flat, y, states


@torch.no_grad()
def theta_g_from_train(states_train: torch.Tensor, Pn: float) -> torch.Tensor:
    # theta_g is the Pn-th percentile of |state| per (neuron, time)
    S, N, T = states_train.shape
    q = float(Pn) / 100.0

    abs_states = states_train.abs()
    theta = torch.zeros((N, T), dtype=torch.float32)

    for t in range(T):
        theta[:, t] = torch.quantile(abs_states[:, :, t], q=q, dim=0)

    return theta.reshape(1, N * T)


# -----------------------------
# Training utils
# -----------------------------
def one_hot(y: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    return F.one_hot(y.to(device), num_classes=n_classes).to(torch.float32)


@torch.no_grad()
def accuracy(readout: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, bs: int) -> float:
    readout.eval()
    correct = 0
    total = 0

    for i in range(0, y.shape[0], bs):
        xb = X[i : i + bs].to(device)
        yb = y[i : i + bs].to(device)

        pred = readout(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

    return correct / max(total, 1)


def train_iterations(
    readout: SpaRCeReadout,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    n_episodes: int,
    batch_size_train: int,
    n_check: int,
    cfg: Cfg,
    device: torch.device,
) -> None:
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(
        [
            {"params": [readout.W_out], "lr": cfg.lr_wout},
            {"params": [readout.theta_i], "lr": cfg.lr_wout * cfg.lr_theta_factor},
        ],
        eps=cfg.adam_eps,
    )

    n = int(ytr.shape[0])
    check_every = max(n_episodes // max(int(n_check), 1), 1)

    for it in range(1, n_episodes + 1):
        idx = torch.randint(0, n, (batch_size_train,))
        xb = Xtr[idx].to(device)
        yb = ytr[idx].to(device)
        yoh = one_hot(yb, cfg.n_classes, device)

        readout.train()
        opt.zero_grad(set_to_none=True)
        logits = readout(xb)
        loss = loss_fn(logits, yoh)
        loss.backward()
        opt.step()

        if it % check_every == 0 or it == 1:
            tr_acc = accuracy(readout, Xtr, ytr, device, cfg.batch_size_eval)
            print(f"    iter {it:>7d}/{n_episodes} | train_acc={tr_acc:.4f}")


@torch.no_grad()
def predict_logits(readout: nn.Module, X: torch.Tensor, device: torch.device, bs: int) -> torch.Tensor:
    readout.eval()
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(readout(X[i : i + bs].to(device)).cpu())
    return torch.cat(outs, dim=0)


# -----------------------------
# Main experiment (MNIST)
# -----------------------------
def run_mnist():
    # --- settings you typically change ---
    Pn_list = [85.0, 90.0, 95.0]
    runs_to_average = 10
    n_episodes = 300_000
    batch_size_train = 20
    n_check = 100
    # ------------------------------------

    cfg = Cfg()
    device = torch.device(cfg.device)

    train_loader, test_loader = load_mnist(cfg)

    results = []

    for Pn in Pn_list:
        print("\n==============================")
        print(f"Pn = {Pn}")
        print("==============================")

        test_logits_sum = None
        test_y_ref = None
        test_accs = []

        for r in range(runs_to_average):
            seed = 123 + r
            set_seed(seed)

            reservoir = ESN1(cfg, seed=seed, device=device)

            print(f"  Run {r + 1}/{runs_to_average} | seed={seed}")
            print("    precompute TRAIN states...")
            Xtr, ytr, Str = precompute_states(reservoir, train_loader, device)

            print("    compute theta_g (full train)...")
            theta_g = theta_g_from_train(Str, Pn)

            print("    precompute TEST states...")
            Xte, yte, _ = precompute_states(reservoir, test_loader, device)
            if test_y_ref is None:
                test_y_ref = yte

            D = int(Xtr.shape[1])
            readout = SpaRCeReadout(D=D, N_scale=cfg.N, n_classes=cfg.n_classes, theta_g_flat=theta_g, seed=seed).to(device)

            print("    train (iterations)...")
            train_iterations(
                readout=readout,
                Xtr=Xtr,
                ytr=ytr,
                n_episodes=n_episodes,
                batch_size_train=batch_size_train,
                n_check=n_check,
                cfg=cfg,
                device=device,
            )

            te_acc = accuracy(readout, Xte, yte, device, cfg.batch_size_eval)
            test_accs.append(te_acc)
            print(f"    test_acc={te_acc:.4f}")

            logits = predict_logits(readout, Xte, device, cfg.batch_size_eval)
            test_logits_sum = logits if test_logits_sum is None else (test_logits_sum + logits)

        mean_test = sum(test_accs) / len(test_accs)
        ensemble_logits = test_logits_sum / float(runs_to_average)
        ensemble_acc = float((ensemble_logits.argmax(dim=1) == test_y_ref).to(torch.float32).mean().item())

        print("\n  ---- Summary ----")
        print(f"  test accs: {[f'{a:.4f}' for a in test_accs]}")
        print(f"  mean test acc: {mean_test:.4f}")
        print(f"  ensemble(avg logits) test acc: {ensemble_acc:.4f}")

        results.append((Pn, mean_test, ensemble_acc))

    print("\n==============================")
    print("All Pn results:")
    for pn, mean_test, ens in results:
        print(f"  Pn={pn:>5} | mean_test={mean_test:.4f} | ensemble_test={ens:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    run_mnist()
