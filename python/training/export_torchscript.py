"""
export_torchscript.py — TorchScript Export & Verification
Online Human Intent Predictor with Adaptive Learning

Standalone script that:
  1. Loads a trained checkpoint (best.pt or last.pt)
  2. Exports the model as a TorchScript .pt file
  3. Verifies the round-trip: scripted output == eager output within 1e-4
  4. Saves a test_io.pt tensor pair for use in later phase

Run this separately from train.py if the TorchScript export inside the
training script failed, or to re-export with a different checkpoint.

Usage
-----
    python python/training/export_torchscript.py \
        --checkpoint  models/checkpoints/best.pt  \
        --output      models/torchscript/intent_model.pt \
        --test_io     models/torchscript/test_io.pt

    # With explicit model arch flags (must match the trained checkpoint):
        --hidden   128 --n_layers 4 --n_heads 4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def export(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load checkpoint into model
    # ------------------------------------------------------------------
    cfg = {
        "in_dim":   99,
        "hidden":   args.hidden,
        "n_heads":  args.n_heads,
        "n_layers": args.n_layers,
        "W":        30,
        "K":        15,
        "dropout":  0.0,   # set to 0 for export; dropout is no-op in eval()
    }

    log.info("Building model with config: %s", cfg)
    model = build_model(cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded checkpoint: %s", ckpt_path)

    counts = model.count_parameters()
    log.info("Model parameters: %s total", f"{counts['total']:,}")

    # ------------------------------------------------------------------
    # TorchScript export
    # ------------------------------------------------------------------
    log.info("Scripting model with torch.jit.script() …")
    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        log.error("torch.jit.script() failed: %s", e)
        log.error("Common causes: dynamic shapes, non-Tensor branches, unsupported ops.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    log.info("Saved TorchScript model to %s  (%.1f MB)", output_path, size_mb)

    # ------------------------------------------------------------------
    # Round-trip verification
    # ------------------------------------------------------------------
    log.info("Verifying round-trip …")
    # torch.jit.load() does not accept map_location in PyTorch 2.x
    # — load then move to device.
    reloaded = torch.jit.load(str(output_path))
    reloaded = reloaded.to(device)
    reloaded.eval()

    # Use a batch of 4 to catch any batch-dim issues
    dummy = torch.randn(4, 30, 99, device=device)

    with torch.no_grad():
        out_eager    = model(dummy)
        out_scripted = reloaded(dummy)

    max_diff = (out_eager - out_scripted).abs().max().item()
    mean_diff = (out_eager - out_scripted).abs().mean().item()

    log.info("Round-trip  max_diff=%.2e  mean_diff=%.2e", max_diff, mean_diff)

    if max_diff < 1e-4:
        log.info("Round-trip verified  ✓  (max diff %.2e < 1e-4)", max_diff)
    else:
        log.error(
            "Round-trip FAILED: max diff %.2e exceeds 1e-4 tolerance. "
            "The .pt file may produce incorrect results in C++.",
            max_diff,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Export method check
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred_hidden_eager, hidden_eager = model.forward_with_hidden(dummy)
        pred_hidden_scripted, hidden_scripted = reloaded.forward_with_hidden(dummy)

    pred_diff = (
        pred_hidden_eager - pred_hidden_scripted
    ).abs().max().item()

    hidden_diff = (
        hidden_eager - hidden_scripted
    ).abs().max().item()

    assert pred_hidden_scripted.shape == (4, 15, 99)
    assert hidden_scripted.shape      == (4, args.hidden)

    assert pred_diff   < 1e-4
    assert hidden_diff < 1e-4

    # Also verify that forward_with_hidden produces the exact same prediction as ordinary forward()
    forward_consistency = (
        out_scripted - pred_hidden_scripted
    ).abs().max().item()

    assert forward_consistency < 1e-4

    log.info(
        "forward_with_hidden verified: pred=%s hidden=%s",
        tuple(pred_hidden_scripted.shape),
        tuple(hidden_scripted.shape),
    )

    # ------------------------------------------------------------------
    # Output shape check
    # ------------------------------------------------------------------
    assert out_scripted.shape == (4, 15, 99), (
        f"Unexpected output shape: {out_scripted.shape}, expected (4, 15, 99)"
    )
    log.info("Output shape: %s  ✓", tuple(out_scripted.shape))

    # ------------------------------------------------------------------
    # Save test I/O for C++ verification 
    #
    # In later phase, will load this file in C++ and assert that
    # torch::jit::load(model).forward(x) matches y within 1e-4.
    # Use a batch size of 1 to match the C++ runtime's inference call.
    # ------------------------------------------------------------------
    test_input  = torch.randn(1, 30, 99, device=device)
    with torch.no_grad():
        test_output = reloaded(test_input)

        test_pred_with_hidden, test_hidden = reloaded.forward_with_hidden(test_input)

    test_io_path = Path(args.test_io)
    test_io_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x":             test_input.cpu(),   # [1, 30, 99]
            "y":             test_output.cpu(),  # [1, 15, 99]
            "hidden":        test_hidden.cpu(),  # [1, 128]
            "y_with_hidden": test_pred_with_hidden.cpu(),
            "model_config":  cfg,
        },
        test_io_path,
    )
    log.info("Test I/O saved to %s", test_io_path)
    log.info("  x shape: %s", tuple(test_input.shape))
    log.info("  y shape: %s", tuple(test_output.shape))
    log.info("  Load in C++ with: torch::load('%s')", test_io_path)

    # ------------------------------------------------------------------
    # Inference latency (single-sample, GPU if available)
    # ------------------------------------------------------------------
    log.info("Measuring inference latency …")
    single = torch.randn(1, 30, 99, device=device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = reloaded(single)
    if device.type == "cuda":
        torch.cuda.synchronize()

    import time
    N_REPS = 200
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        with torch.no_grad():
            _ = reloaded(single)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / N_REPS * 1000

    log.info("Inference latency (single sample, %s): %.3f ms", device, elapsed_ms)
    if elapsed_ms < 5.0:
        log.info("  ✓ < 5 ms target for C++ inference phase")
    else:
        log.warning(
            "  %.3f ms exceeds 5 ms Python target — may still meet C++ target "
            "due to TorchScript optimisation in the C++ runtime.",
            elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("EXPORT COMPLETE")
    print("=" * 55)
    print(f"  TorchScript : {output_path}")
    print(f"  Test I/O    : {test_io_path}")
    print(f"  Params      : {counts['total']:,}")
    print(f"  Round-trip  : max_diff = {max_diff:.2e}  ✓")
    print(f"  Latency     : {elapsed_ms:.3f} ms  ({device})")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export trained IntentPredictor to TorchScript.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="models/checkpoints/best.pt",
        help="Path to trained model state dict (.pt from train.py).",
    )
    p.add_argument(
        "--output",
        default="models/torchscript/intent_model.pt",
        help="Output path for the TorchScript module.",
    )
    p.add_argument(
        "--test_io",
        default="models/torchscript/test_io.pt",
        help="Output path for the test I/O tensor pair.",
    )
    # Must match the architecture used during training
    p.add_argument("--hidden",   type=int, default=128, help="Hidden dim (must match training).")
    p.add_argument("--n_layers", type=int, default=4,   help="Encoder layers (must match training).")
    p.add_argument("--n_heads",  type=int, default=4,   help="Attention heads (must match training).")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())