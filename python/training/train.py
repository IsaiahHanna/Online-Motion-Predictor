"""
train.py - Offline Training Script
Online Human Intent Predictor with Adaptive Learning

Trains the IntentPredictor transformer on preprocessed AMASS windows.

Usage
-----
python python/training/train.py -- data_dir [data_dir]

# Override hyperparams:
    python python/training/train.py  \
        --data_dir    Data/output    \
        --output_dir  models/        \
        --hidden      128            \
        --n_layers    4              \
        --n_heads     4              \
        --batch_size  256            \
        --epochs      50             \
        --lr          1e-3           \
        --vel_weight  0.1            \
        --num_workers 4              

Outputs (all written to --output_dir)
-------
    checkpoints/best.pt             - state dict of the best val-loss epoch
    checkpoints/last.pt             - state dict of the final epoch
    torchscript/intent_models.pt    - TorchScript export (for C++ runtime)
    loss_curves.png                 - train + val loss per epoch
    metrics.json                    - final test ADE / FDE * param counts
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

#Local imports (same package)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from model import IntentPredictor, build_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE over all keypoints and future frames."""
    return nn.functional.mse_loss(pred, target)

def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE on frame-to-frame displacements.
    Encourages temporally smooth predictions without a separate
    smoothness constraint. pred/target: [B, K, 99]
    """
    pred_vel   = pred[:, 1:, :]   - pred[:, :-1, :]  # [B, K-1, 99]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    return nn.functional.mse_loss(pred_vel, target_vel)

def combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        vel_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = MSE + vel_weight * velocity_loss.
    Returns (total, mse_component, vel_component) for logging.
    """
    l_mse = mse_loss(pred,target)
    l_vel = velocity_loss(pred, target)
    total = l_mse + vel_weight * l_vel
    return total, l_mse, l_vel

# ---------------------------------------------------------------------------
# Metrics: ADE and FDE (in millimeters)
# ---------------------------------------------------------------------------

def compute_ade_fde(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Tuple[float,float]:
    """
    Average Displacement Error and Final Displacement Error.

    ADE: mean L2 distance (mm) across all K future frames and all joints.
    FDE: mean L2 distance (mm) at the final predicted frame only.

    pred / target : [B, K, 99] (root-relative meters)
    Returns (ADE_mm, FDE_mm)
    """

    # Reshape to [B, K, 33, 3]
    B, K, D = pred.shape
    pred_3D   = pred.detach().view(B, K, D // 3, 3)
    target_3D = target.detach().view(B, K, D // 3, 3)

    # Per-joint L2 error: [B, K, 33]
    joint_err = (pred_3D - target_3D).norm(dim=-1)

    # ADE: mean over joints, frames and batch
    ade = joint_err.mean().item() * 1000.0 # meters -> mm

    # FDE: mean over joints and batch at final frame
    fde = joint_err[:, -1, :].mean().item() * 1000.0

    return ade, fde


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vel_weight: float,
    epoch: int
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    total_vel  = 0.0
    n_batches  = len(loader)

    for batch_idx, (x,y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss, l_mse, l_vel = combined_loss(pred, y, vel_weight)
        loss.backward()

        # Gradient clipping - prevents occasional large gradient spikes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse  += l_mse.item()
        total_vel  += l_vel.item()

        if batch_idx % max(1, n_batches // 5) == 0:
            log.info(
                "   Epoch %d [%d/%d] loss=%.4f mse=%.4f vel=%.4f",
                epoch, batch_idx, n_batches,
                loss.item(), l_mse.item(), l_vel.item()
            ) 
    
    return {
        "loss": total_loss / n_batches,
        "mse":  total_mse / n_batches,
        "vel":  total_vel / n_batches
    }

# ---------------------------------------------------------------------------
# Validation / test evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    vel_weight: float
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ade  = 0.0
    total_fde  = 0.0
    n_batches  = len(loader)

    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss, _, _ = combined_loss(pred, y, vel_weight)
        ade, fde   = compute_ade_fde(pred, y)

        total_loss += loss.item()
        total_ade  += ade
        total_fde  += fde

    return {
        "loss": total_loss / n_batches,
        "ade":  total_ade  / n_batches,
        "fde":  total_fde  / n_batches
    }


# ---------------------------------------------------------------------------
# Loss curve plot
# ---------------------------------------------------------------------------

def save_loss_curves(
    train_losses: list,
    val_losses: list,
    output_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(train_losses) + 1)

        axes[0].plot(epochs, train_losses, label="train")
        axes[0].plot(epochs, val_losses,   label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE + vel)")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Log-scale version - easier to see plateau
        axes[1].semilogy(epochs, train_losses, label="train")
        axes[1].semilogy(epochs, val_losses,   label="val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss (log scale)")
        axes[1].set_title("Loss (log scale)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        log.info("Loss curves saved to %s", output_path)
    except ImportError:
        log.warning("matplotlib not available - loss curves not saved.")

# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ts_dir     = output_dir / "torchscript"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    log.info("Device: %s", device)

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    log.info("Loading data from %s ...", args.data_dir)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type in ("cuda","mps")),
        augment_train=True
    )
    log.info(
        "Train: %d batches  Val: %d batches  Test:  %d batches",
        len(train_loader), len(val_loader), len(test_loader)
    )

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    cfg = {
        "in_dim":   99,
        "hidden":   args.hidden,
        "n_heads":  args.n_heads,
        "n_layers": args.n_layers,
        "W":        30,
        "K":        15,
        "dropout":  args.dropout
    }
    model = build_model(cfg).to(device)
    counts = model.count_parameters()
    log.info(
        "Model: hidden=%d  layers=%d  heads=%d  total_params=%s",
        args.hidden, args.n_layers, args.n_heads,
        f"{counts['total']:,}"
    )

    # ---------------------------------------------------------------------------
    # Optimizer and LR schedule 
    # ---------------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    best_val_loss   = math.inf
    train_losses    = []
    val_losses      = []
    start_time      = time.time()

    log.info("Starting training for %d epochs ...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.vel_weight, epoch
        )
        val_metrics = evaluate(model, val_loader, device, args.vel_weight)
        scheduler.step()

        elapsed = time.time() - epoch_start
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])

        log.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f   "
            "val_ADE=%.1f mm   val_FDE=%.1f mm  lr=%.2e   (%.1fs)",
            epoch, args.epochs,
            train_metrics["loss"], val_metrics["loss"],
            val_metrics["ade"], val_metrics["fde"],
            scheduler.get_last_lr()[0],
            elapsed
        )

        # Checkpoint: best val loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            log.info("  -> New best val loss - saved checkpoint.")
    
    # Last checkpoint
    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    total_time = time.time() - start_time
    log.info("Training complete in %.1f s (%.1f min)", total_time, total_time / 60)

    # ---------------------------------------------------------------------------
    # Final test evaluation (use best checkpoint)
    # ---------------------------------------------------------------------------
    log.info("Evaluating best checkpoint on test set ...")
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, args.vel_weight)

    log.info(
        "TEST   loss=%.4f   ADE=%.2f mm  FDE=%.2f mm",
        test_metrics["loss"], test_metrics["ade"], test_metrics["fde"]
    )

    # Target metrics for early phase 
    ade_target, fde_target = 50.0, 80.0
    if test_metrics["ade"] < ade_target:
        log.info("  ADE %.2f mm < target %.0f mm", test_metrics["ade"], ade_target)
    else:
        log.warning(
            "  ADE %.2f mm exceeds target %.0f mm - consider more epochs or larger model",
            test_metrics["ade"], ade_target
        )
    
    if test_metrics["fde"] < fde_target:
        log.info("  FDE %.2f mm < target %.0f mm", test_metrics["fde"], fde_target)
    else:
        log.warning(
            "  FDE %.2f mm exceeds target %.0f mm",
            test_metrics["fde"], fde_target
        )
    
    # ---------------------------------------------------------------------------
    # Adapter parameter count (for the metrics.json record)
    # ---------------------------------------------------------------------------
    r = 4
    n_adapt = 2
    adapter_params = 2 * n_adapt * 2 * args.hidden * r  # 2 adapters/layer * 2 matrices
    param_pct = adapter_params / counts["total"] * 100

    # ---------------------------------------------------------------------------
    # Save loss curves and metrics 
    # ---------------------------------------------------------------------------
    save_loss_curves(
        train_losses, val_losses,
        output_dir / "loss_curves.png"
    )

    metrics = {
        "test_ade_mm":          round(test_metrics["ade"], 3),
        "test_fde_mm":          round(test_metrics["fde"], 3),
        "best_val_loss":        round(best_val_loss, 6),
        "total_params":         counts["total"],
        "adapter_params_r4_n2": adapter_params,
        "adapter_param_pct":    round(param_pct, 3),
        "training_time_s":      round(total_time, 1),
        "model_config":         cfg
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)

    # ---------------------------------------------------------------------------
    # TorchScript export - run by export_torchscript.py, but also here 
    # for convenience so training is fully self-contained
    # ---------------------------------------------------------------------------
    log.info("Exporting TorchScript model ...")
    model.eval()
    try:
        scripted = torch.jit.script(model)
        ts_path  = ts_dir / "intent_model.pt"
        scripted.save(str(ts_path))
        log.info("TorchScript saved to %s", ts_path)

        # Verify round-trip
        reloaded = torch.jit.load(str(ts_path))
        dummy    = torch.randn(1, 30, 99, device=device)
        out_orig = model(dummy)
        out_ts   = reloaded(dummy)
        max_diff = (out_orig - out_ts).abs().max().item()
        assert max_diff < 1e-4, f"TorchScript round-trip diff too large: {max_diff}"
        log.info("TorchScript round-trip verified (max diff %.2e) ", max_diff)

        # Save a test I/O tensor for C++ verification
        test_io_path = ts_dir / "test_io.pt"
        torch.save({"x": dummy.cpy(), "y": out_ts.cpu()}, test_io_path)
        log.info("Test I/O tensor saved to %s (Used in C++ test)", test_io_path)

    except Exception as e:
        log.error("TorchScript export failed: %s", e)
        log.error("Model will need manual export via export_torchscript.py")
    
    
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train IntentPredictor on preprocessed AMASS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_dir", required=True,
                   help="Directory containing X_train.npy, Y_train.npy, etc")
    p.add_argument("--output_dir", default="models",
                   help="Where to save checkpoints, TorchScript, and plots")
    
    # Model architecture 
    p.add_argument("--hidden",    type=int,   default=128,  help="Transformer hidden dim.")
    p.add_argument("--n_layers",  type=int,   default=4,    help="Number of encoder layers.")
    p.add_argument("--n_heads",   type=int,   default=4,    help="Number of attention heads.")
    p.add_argument("--dropout",   type=float, default=0.1,  help="Dropout rate.")
 
    # Training (Phase 2 spec: AdamW, lr=1e-3, batch=256, 50 epochs)
    p.add_argument("--epochs",      type=int,   default=50,   help="Training epochs.")
    p.add_argument("--batch_size",  type=int,   default=256,  help="Batch size.")
    p.add_argument("--lr",          type=float, default=1e-3, help="Peak learning rate.")
    p.add_argument("--vel_weight",  type=float, default=0.1,
                   help="Weight of velocity consistency loss.")
    p.add_argument("--num_workers", type=int,   default=4,
                   help="DataLoader worker processes (set 0 on Windows).")
    return p.parse_args()
 
 
if __name__ == "__main__":
    train(parse_args())
