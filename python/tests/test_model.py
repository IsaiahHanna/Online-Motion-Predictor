"""
test_model.py — Model Architecture & Adapter Insertion Tests
Online Human Intent Predictor with Adaptive Learning
Phase 2 completion / Phase 5 & 6 preparation

Validates the Python reference implementation of LoRA adapter insertion.
Run this BEFORE implementing the C++ adapter in Phases 5 & 6 — if these
tests pass, you have a known-good reference to compare your C++ output
against.

Usage:
    python python/tests/test_model.py
    python python/tests/test_model.py --checkpoint models/checkpoints/best.pt

Exit code 0 = all tests passed. Non-zero = at least one failure.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

# Allow running from repo root or from python/tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from model import IntentPredictor, LoRAAdapter, AdaptedEncoderLayer, build_model

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW= "\033[93m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

def _ok(msg):   return f"{_GREEN}  PASS{_RESET}  {msg}"
def _fail(msg): return f"{_RED}  FAIL{_RESET}  {msg}"
def _warn(msg): return f"{_YELLOW}  WARN{_RESET}  {msg}"
def _head(msg): return f"\n{_BOLD}{msg}{_RESET}"

passes = 0
failures = 0

def check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    global passes, failures
    if condition:
        print(_ok(pass_msg))
        passes += 1
    else:
        print(_fail(fail_msg))
        failures += 1
    return condition


# ---------------------------------------------------------------------------
# Test 1 — LoRAAdapter module structure and initialisation
# ---------------------------------------------------------------------------

def test_lora_adapter_init():
    print(_head("1. LoRAAdapter initialisation"))

    hidden, rank = 128, 4
    adapter = LoRAAdapter(hidden_dim=hidden, rank=rank)

    # Correct parameter shapes
    check(
        tuple(adapter.W_down.weight.shape) == (rank, hidden),
        f"W_down shape is ({rank}, {hidden}) ✓",
        f"W_down shape {tuple(adapter.W_down.weight.shape)} != ({rank}, {hidden})"
    )
    check(
        tuple(adapter.W_up.weight.shape) == (hidden, rank),
        f"W_up shape is ({hidden}, {rank}) ✓",
        f"W_up shape {tuple(adapter.W_up.weight.shape)} != ({hidden}, {rank})"
    )

    # W_up must be initialised to zero — adapter starts as identity residual
    w_up_max = adapter.W_up.weight.abs().max().item()
    check(
        w_up_max == 0.0,
        f"W_up initialised to zero (max={w_up_max:.2e}) ✓",
        f"W_up is NOT zero at init (max={w_up_max:.2e}) — adapter won't start as identity"
    )

    # W_down should be non-zero (kaiming uniform)
    w_down_max = adapter.W_down.weight.abs().max().item()
    check(
        w_down_max > 0.0,
        f"W_down initialised non-zero (max={w_down_max:.4f}) ✓",
        f"W_down is zero — bad initialisation"
    )

    # Forward output must be zero when W_up=0, regardless of input
    x = torch.randn(2, 30, hidden)
    out = adapter(x)
    out_max = out.abs().max().item()
    check(
        out_max == 0.0,
        f"Adapter output is exactly zero at init (W_up=0) ✓",
        f"Adapter output non-zero at init: max={out_max:.2e} — identity residual broken"
    )

    # Output shape must match input shape
    check(
        out.shape == x.shape,
        f"Adapter output shape {tuple(out.shape)} matches input ✓",
        f"Adapter output shape {tuple(out.shape)} != input {tuple(x.shape)}"
    )

    # Parameter count: 2 * hidden * rank (no bias)
    expected_params = 2 * hidden * rank
    actual_params = sum(p.numel() for p in adapter.parameters())
    check(
        actual_params == expected_params,
        f"Adapter param count {actual_params} = 2 × {hidden} × {rank} ✓",
        f"Adapter param count {actual_params} != expected {expected_params}"
    )


# ---------------------------------------------------------------------------
# Test 2 — LoRAAdapter reset
# ---------------------------------------------------------------------------

def test_lora_adapter_reset():
    print(_head("2. LoRAAdapter reset()"))

    adapter = LoRAAdapter(hidden_dim=128, rank=4)

    # Manually set W_up to non-zero to simulate trained state
    nn.init.normal_(adapter.W_up.weight)
    pre_reset_max = adapter.W_up.weight.abs().max().item()

    adapter.reset()
    post_reset_max = adapter.W_up.weight.abs().max().item()

    check(
        pre_reset_max > 0,
        f"W_up non-zero before reset (max={pre_reset_max:.4f}) ✓",
        "W_up was already zero before reset — test inconclusive"
    )
    check(
        post_reset_max == 0.0,
        f"W_up zeroed after reset() ✓",
        f"reset() did not zero W_up (max={post_reset_max:.2e})"
    )

    # After reset, forward output should again be zero
    x = torch.randn(2, 30, 128)
    out = adapter(x)
    check(
        out.abs().max().item() == 0.0,
        "Adapter output zero after reset() ✓",
        "Adapter output non-zero after reset() — reset broken"
    )


# ---------------------------------------------------------------------------
# Test 3 — Base model forward pass (sanity before adapter insertion)
# ---------------------------------------------------------------------------

def test_base_model_forward():
    print(_head("3. Base model forward pass"))

    model = IntentPredictor(hidden=128, n_layers=4, n_heads=4)
    model.eval()

    x = torch.randn(4, 30, 99)
    with torch.no_grad():
        out = model(x)

    check(
        out.shape == (4, 15, 99),
        f"Output shape (4, 15, 99) ✓",
        f"Output shape {tuple(out.shape)} != (4, 15, 99)"
    )
    check(
        out.dtype == torch.float32,
        "Output dtype float32 ✓",
        f"Output dtype {out.dtype} != float32"
    )
    check(
        torch.isfinite(out).all().item(),
        "Output contains no NaN/Inf ✓",
        "Output contains NaN or Inf values"
    )

    counts = model.count_parameters()
    check(
        counts["total"] > 0,
        f"Total parameters: {counts['total']:,} ✓",
        "Model has zero parameters"
    )
    check(
        counts["trainable"] == counts["total"],
        f"All {counts['total']:,} params trainable before freezing ✓",
        f"Only {counts['trainable']:,} / {counts['total']:,} trainable before freeze"
    )


# ---------------------------------------------------------------------------
# Test 4 — Freezing base parameters
# ---------------------------------------------------------------------------

def test_freeze_base():
    print(_head("4. Freezing base model parameters"))

    model = IntentPredictor(hidden=128, n_layers=4)
    model.freeze_base()

    counts = model.count_parameters()
    check(
        counts["trainable"] == 0,
        f"All {counts['total']:,} params frozen after freeze_base() ✓",
        f"{counts['trainable']} params still trainable after freeze_base()"
    )

    # When all params are frozen, the output has no grad_fn, so backward()
    # would raise "element 0 does not require grad". Instead verify directly
    # that every parameter has requires_grad=False and grad=None.
    params_still_trainable = [
        name for name, p in model.named_parameters()
        if p.requires_grad
    ]
    check(
        len(params_still_trainable) == 0,
        "All params have requires_grad=False after freeze_base() ✓",
        f"{len(params_still_trainable)} params still have requires_grad=True: "
        f"{params_still_trainable[:3]}"
    )

    params_with_existing_grad = [
        name for name, p in model.named_parameters()
        if p.grad is not None
    ]
    check(
        len(params_with_existing_grad) == 0,
        "All params have grad=None after freeze_base() ✓",
        f"{len(params_with_existing_grad)} params have stale gradients"
    )


# ---------------------------------------------------------------------------
# Test 5 — Adapter insertion: only adapter params trainable
# ---------------------------------------------------------------------------

def test_adapter_insertion_requires_grad():
    print(_head("5. Adapter insertion — requires_grad"))

    model = IntentPredictor(hidden=128, n_layers=4)
    n_adapt, rank = 2, 4
    adapters = model.insert_adapters(n_adapt=n_adapt, rank=rank)

    # Collect all parameter names and their trainability
    base_trainable = [
        name for name, p in model.named_parameters()
        if p.requires_grad and "adapter" not in name
    ]
    adapter_frozen = [
        name for name, p in model.named_parameters()
        if not p.requires_grad and "adapter" in name
    ]
    adapter_trainable = [
        name for name, p in model.named_parameters()
        if p.requires_grad and "adapter" in name
    ]

    check(
        len(base_trainable) == 0,
        "No non-adapter parameters are trainable ✓",
        f"{len(base_trainable)} base params still trainable: {base_trainable[:3]}"
    )
    check(
        len(adapter_frozen) == 0,
        "No adapter parameters are frozen ✓",
        f"{len(adapter_frozen)} adapter params are frozen: {adapter_frozen[:3]}"
    )

    # Expected: n_adapt layers × 2 adapters per layer × 2 matrices each
    expected_adapter_count = n_adapt * 2  # attn + ff per adapted layer
    check(
        len(adapters) == expected_adapter_count,
        f"Returned {len(adapters)} adapter modules (expected {expected_adapter_count}) ✓",
        f"Got {len(adapters)} adapter modules, expected {expected_adapter_count}"
    )

    # Expected trainable param count: 2 * hidden * rank per adapter
    hidden = 128
    expected_trainable = len(adapters) * 2 * hidden * rank
    counts = model.count_parameters()
    check(
        counts["trainable"] == expected_trainable,
        f"Trainable params = {counts['trainable']} "
        f"= {len(adapters)} adapters × 2 matrices × {hidden} × {rank} ✓",
        f"Trainable params = {counts['trainable']}, expected {expected_trainable}"
    )

    # Adapter params < 2% of total (Phase 2 spec requirement)
    pct = counts["trainable"] / counts["total"] * 100
    check(
        pct < 2.0,
        f"Adapter params = {pct:.3f}% of total (< 2% target) ✓",
        f"Adapter params = {pct:.3f}% of total — exceeds 2% target"
    )


# ---------------------------------------------------------------------------
# Test 6 — Adapter identity at initialisation
# ---------------------------------------------------------------------------

def test_adapter_identity_at_init():
    """
    After insertion, the adapted model must produce IDENTICAL output to the
    base model on the same input, because W_up=0 means all adapter residuals
    are zero. This is the key property that makes online adaptation safe to
    start at any point.
    """
    print(_head("6. Adapter identity at initialisation (output unchanged)"))

    # Base model output (no adapters)
    model_base = IntentPredictor(hidden=128, n_layers=4)
    model_base.eval()

    # Adapted model — same weights, adapters zeroed
    model_adapted = IntentPredictor(hidden=128, n_layers=4)
    model_adapted.load_state_dict(model_base.state_dict())
    model_adapted.insert_adapters(n_adapt=2, rank=4)
    model_adapted.eval()

    x = torch.randn(3, 30, 99)
    with torch.no_grad():
        out_base    = model_base(x)
        out_adapted = model_adapted(x)

    max_diff = (out_base - out_adapted).abs().max().item()
    check(
        max_diff == 0.0,
        f"Adapted model output identical to base at init (max_diff={max_diff:.2e}) ✓",
        f"Adapted model output differs from base at init: max_diff={max_diff:.2e} "
        f"— W_up initialisation may not be zero"
    )


# ---------------------------------------------------------------------------
# Test 7 — 10 gradient steps: adapter weights change, base weights do not
# (This is the exact test specified in the Phase 2 spec)
# ---------------------------------------------------------------------------

def test_online_adaptation_loop():
    """
    Run 10 gradient steps on a short sequence (simulating online adaptation).
    Confirm:
      - Adapter weights change
      - Base model weights do NOT change
      - Prediction error decreases (adapter is actually learning something)
    """
    print(_head("7. Online adaptation — 10 gradient steps"))

    model = IntentPredictor(hidden=128, n_layers=4)
    adapters = model.insert_adapters(n_adapt=2, rank=4)
    model.train()

    optimizer = torch.optim.Adam(
        [p for a in adapters for p in a.parameters()],
        lr=1e-3
    )

    # Snapshot base weights before adaptation
    base_weight_snapshots = {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if "adapter" not in name
    }

    # Snapshot adapter weights before adaptation
    adapter_weight_before = {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if "adapter" in name
    }

    # Simulate a short observed sequence (as in the real online loop)
    x      = torch.randn(4, 30, 99)
    target = torch.randn(4, 15, 99)

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # --- Check 1: base weights unchanged ---
    base_changed = []
    for name, original in base_weight_snapshots.items():
        current = dict(model.named_parameters())[name].data
        if not torch.equal(original, current):
            base_changed.append(name)

    check(
        len(base_changed) == 0,
        "Base model weights unchanged after 10 gradient steps ✓",
        f"{len(base_changed)} base params changed: {base_changed[:3]}"
    )

    # --- Check 2: adapter weights changed ---
    adapter_unchanged = []
    for name, original in adapter_weight_before.items():
        current = dict(model.named_parameters())[name].data
        if torch.equal(original, current):
            adapter_unchanged.append(name)

    # W_up starts at zero and should move; W_down starts non-zero and also moves
    check(
        len(adapter_unchanged) == 0,
        f"All adapter weights changed after 10 gradient steps ✓",
        f"{len(adapter_unchanged)} adapter params did NOT change: {adapter_unchanged[:3]}"
    )

    # --- Check 3: loss decreased over the 10 steps ---
    loss_decreased = losses[-1] < losses[0]
    check(
        loss_decreased,
        f"Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f} ✓",
        f"Loss did NOT decrease: {losses[0]:.4f} → {losses[-1]:.4f} "
        f"— adapter may not be receiving gradients"
    )

    # --- Check 4: W_up is no longer zero (it moved from its init) ---
    w_up_norms = []
    for a in adapters:
        w_up_norms.append(a.W_up.weight.norm().item())
    all_w_up_moved = all(n > 0 for n in w_up_norms)
    check(
        all_w_up_moved,
        f"All W_up matrices non-zero after adaptation "
        f"(norms: {[f'{n:.4f}' for n in w_up_norms]}) ✓",
        f"Some W_up matrices still zero: {w_up_norms}"
    )


# ---------------------------------------------------------------------------
# Test 8 — Gradient isolation: no cross-contamination
# ---------------------------------------------------------------------------

def test_gradient_isolation():
    """
    After backward(), base params must have grad=None (not just grad=0).
    grad=None means PyTorch never allocated a gradient buffer for them,
    which is more efficient and confirms requires_grad=False is working
    at the autograd level, not just being zeroed.
    """
    print(_head("8. Gradient isolation — base params have grad=None"))

    model = IntentPredictor(hidden=128, n_layers=4)
    adapters = model.insert_adapters(n_adapt=2, rank=4)
    model.train()

    x   = torch.randn(2, 30, 99)
    out = model(x)
    loss = out.mean()
    loss.backward()

    base_with_grad = [
        name for name, p in model.named_parameters()
        if "adapter" not in name and p.grad is not None
    ]
    check(
        len(base_with_grad) == 0,
        "Base params have grad=None after backward() ✓",
        f"{len(base_with_grad)} base params have non-None grad: {base_with_grad[:3]}"
    )

    adapter_with_grad = [
        name for name, p in model.named_parameters()
        if "adapter" in name and p.grad is not None
    ]
    check(
        len(adapter_with_grad) > 0,
        f"{len(adapter_with_grad)} adapter params have non-None grad ✓",
        "No adapter params received gradients — adaptation will not work"
    )


# ---------------------------------------------------------------------------
# Test 9 — Adapter reset reverts to base model output
# ---------------------------------------------------------------------------

def test_adapter_reset_reverts_output():
    """
    After adaptation (W_up != 0), resetting the adapters must make the
    model output identical to the pre-adaptation base model output again.
    This validates the 'switching subjects' mechanism.
    """
    print(_head("9. Adapter reset reverts output to base model"))

    model = IntentPredictor(hidden=128, n_layers=4)
    adapters = model.insert_adapters(n_adapt=2, rank=4)

    x = torch.randn(2, 30, 99)

    # Record base output (adapters are zero, so this IS the base output)
    model.eval()
    with torch.no_grad():
        out_before = model(x).clone()

    # Simulate adaptation: manually set W_up to non-zero
    for a in adapters:
        nn.init.normal_(a.W_up.weight, std=0.1)

    with torch.no_grad():
        out_adapted = model(x)
    diff_after_adapt = (out_before - out_adapted).abs().max().item()

    check(
        diff_after_adapt > 0,
        f"Output differs after adaptation (max_diff={diff_after_adapt:.4f}) ✓",
        "Output unchanged after adaptation — adapter not contributing to forward pass"
    )

    # Reset all adapters
    for a in adapters:
        a.reset()

    with torch.no_grad():
        out_reset = model(x)
    diff_after_reset = (out_before - out_reset).abs().max().item()

    check(
        diff_after_reset == 0.0,
        f"Output reverts to base after reset() (max_diff={diff_after_reset:.2e}) ✓",
        f"Output does NOT revert after reset(): max_diff={diff_after_reset:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 10 — Load trained checkpoint and insert adapters
# ---------------------------------------------------------------------------

def test_adapter_insertion_on_checkpoint(checkpoint_path: str):
    print(_head("10. Adapter insertion on trained checkpoint"))

    path = Path(checkpoint_path)
    if not path.exists():
        print(_warn(f"Checkpoint not found at {path} — test skipped"))
        print(_warn("Run with --checkpoint models/checkpoints/best.pt to enable this test"))
        return

    model = build_model({"hidden": 128, "n_layers": 4, "n_heads": 4})
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # Record output before adapter insertion
    model.eval()
    x = torch.randn(2, 30, 99)
    with torch.no_grad():
        out_pretrained = model(x).clone()

    # Insert adapters
    adapters = model.insert_adapters(n_adapt=2, rank=4)

    # Output must be identical (adapters start as identity)
    with torch.no_grad():
        out_with_adapters = model(x)
    max_diff = (out_pretrained - out_with_adapters).abs().max().item()

    check(
        max_diff == 0.0,
        f"Trained checkpoint + adapters: output unchanged at init (max_diff={max_diff:.2e}) ✓",
        f"Trained checkpoint + adapters: output changed at init (max_diff={max_diff:.2e})"
    )

    # Run adaptation and confirm ADE improves
    model.train()
    optimizer = torch.optim.Adam(
        [p for a in adapters for p in a.parameters()], lr=1e-3
    )
    target = torch.randn(2, 15, 99)
    initial_loss = nn.functional.mse_loss(model(x), target).item()

    for _ in range(10):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

    final_loss = nn.functional.mse_loss(model(x), target).item()
    check(
        final_loss < initial_loss,
        f"Loss on trained checkpoint decreases after 10 steps: "
        f"{initial_loss:.4f} → {final_loss:.4f} ✓",
        f"Loss did not decrease on trained checkpoint: {initial_loss:.4f} → {final_loss:.4f}"
    )

    counts = model.count_parameters()
    pct = counts["trainable"] / counts["total"] * 100
    print(f"        Adapter param %: {pct:.3f}% of {counts['total']:,} total")
    check(True, f"Checkpoint loaded and adapter-wrapped successfully ✓", "")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(checkpoint: str) -> bool:
    print(f"\n{_BOLD}{'=' * 62}{_RESET}")
    print(f"{_BOLD}  OHIPA Model & Adapter Test Suite{_RESET}")
    print(f"  Phase 2 completion / Phase 5 & 6 preparation")
    print(f"{'=' * 62}{_RESET}")

    test_lora_adapter_init()
    test_lora_adapter_reset()
    test_base_model_forward()
    test_freeze_base()
    test_adapter_insertion_requires_grad()
    test_adapter_identity_at_init()
    test_online_adaptation_loop()
    test_gradient_isolation()
    test_adapter_reset_reverts_output()
    test_adapter_insertion_on_checkpoint(checkpoint)

    print(f"\n{'=' * 62}")
    print(
        f"  {_BOLD}Tests run: {passes + failures}   "
        f"{_GREEN}Passed: {passes}{_RESET}   "
        f"{_RED}Failed: {failures}{_RESET}"
    )
    print(f"{'=' * 62}")

    if failures == 0:
        print(f"{_GREEN}{_BOLD}  ✓ All tests passed — adapter logic ready for C++ (Phase 5){_RESET}")
    else:
        print(f"{_RED}{_BOLD}  ✗ {failures} test(s) failed — fix before proceeding to Phase 5{_RESET}")

    return failures == 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate IntentPredictor adapter insertion (Phase 2 / 5 & 6 prep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="models/checkpoints/best.pt",
        help="Trained checkpoint to validate adapter insertion against.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run(args.checkpoint)
    sys.exit(0 if ok else 1)