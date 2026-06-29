"""
test_preprocess.py — Preprocessing Output Validation
Online Human Intent Predictor with Adaptive Learning

Validates that the preprocessing pipeline produced correct, model-ready
outputs. Run this after preprocess.py completes before moving to training.

Usage:
    # From the repo root, point at your processed data directory:
    python python/tests/test_preprocess.py --data_dir Data/output

    # Or with explicit args to override the defaults:
    python python/tests/test_preprocess.py \
        --data_dir  Data/output \
        --W         30 \
        --K         15 \
        --joints    33 \
        --min_total 10000

Exit code 0 = all tests passed. Non-zero = at least one failure.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Colour helpers (no external deps)
# ---------------------------------------------------------------------------
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"{_GREEN}  PASS{_RESET}  {msg}"
def _fail(msg: str) -> str: return f"{_RED}  FAIL{_RESET}  {msg}"
def _warn(msg: str) -> str: return f"{_YELLOW}  WARN{_RESET}  {msg}"
def _head(msg: str) -> str: return f"\n{_BOLD}{msg}{_RESET}"


# ---------------------------------------------------------------------------
# Individual test functions
# Each returns (passed: bool, message: str)
# ---------------------------------------------------------------------------

def test_files_exist(data_dir: Path) -> List[Tuple[bool, str]]:
    """All expected output files are present."""
    results = []
    expected = [
        "X_train.npy", "Y_train.npy",
        "X_val.npy",   "Y_val.npy",
        "X_test.npy",  "Y_test.npy",
        "X.npy",       "Y.npy",
        "subject_ids.npy",
        "subject_ids_train.npy",
        "subject_ids_val.npy",
        "subject_ids_test.npy",
    ]
    for fname in expected:
        p = data_dir / fname
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            results.append((True, f"{fname} exists ({size_mb:.1f} MB)"))
        else:
            results.append((False, f"{fname} NOT FOUND in {data_dir}"))
    return results


def test_shapes(
    arrays: Dict[str, np.ndarray],
    W: int,
    K: int,
    D: int,
) -> List[Tuple[bool, str]]:
    """X arrays are [N, W, D] and Y arrays are [N, K, D]."""
    results = []
    for split in ("train", "val", "test"):
        X = arrays[f"X_{split}"]
        Y = arrays[f"Y_{split}"]

        # X shape
        if X.ndim == 3 and X.shape[1] == W and X.shape[2] == D:
            results.append((True,  f"X_{split} shape {X.shape} ✓"))
        else:
            results.append((False, f"X_{split} shape {X.shape} — expected (N, {W}, {D})"))

        # Y shape
        if Y.ndim == 3 and Y.shape[1] == K and Y.shape[2] == D:
            results.append((True,  f"Y_{split} shape {Y.shape} ✓"))
        else:
            results.append((False, f"Y_{split} shape {Y.shape} — expected (N, {K}, {D})"))

        # Matching window counts
        if X.shape[0] == Y.shape[0]:
            results.append((True,  f"{split}: X and Y have matching window counts ({X.shape[0]:,})"))
        else:
            results.append((False, f"{split}: X has {X.shape[0]:,} windows but Y has {Y.shape[0]:,}"))

    # Full arrays
    X_full = arrays["X"]
    Y_full = arrays["Y"]
    total  = X_full.shape[0]
    split_total = (arrays["X_train"].shape[0]
                   + arrays["X_val"].shape[0]
                   + arrays["X_test"].shape[0])
    if total == split_total:
        results.append((True,  f"Full array total ({total:,}) == sum of splits ({split_total:,})"))
    else:
        results.append((False, f"Full array total ({total:,}) != sum of splits ({split_total:,})"))

    return results


def test_dtype(arrays: Dict[str, np.ndarray]) -> List[Tuple[bool, str]]:
    """All X/Y arrays are float32 (what the transformer DataLoader expects)."""
    results = []
    for key, arr in arrays.items():
        if key.startswith("subject_ids"):
            continue
        if arr.dtype == np.float32:
            results.append((True,  f"{key}: dtype float32 ✓"))
        else:
            results.append((False, f"{key}: dtype is {arr.dtype}, expected float32"))
    return results


def test_no_nan_inf(arrays: Dict[str, np.ndarray]) -> List[Tuple[bool, str]]:
    """No NaN or Inf values in any array."""
    results = []
    for key, arr in arrays.items():
        if key.startswith("subject_ids"):
            continue
        bad = ~np.isfinite(arr)
        n_bad = int(bad.sum())
        if n_bad == 0:
            results.append((True,  f"{key}: no NaN/Inf ✓"))
        else:
            results.append((False, f"{key}: {n_bad:,} NaN/Inf values detected"))
    return results


def test_root_normalisation(arrays: Dict[str, np.ndarray]) -> List[Tuple[bool, str]]:
    """
    After root-relative normalisation the pelvis joint (dims 0-2) should
    be ~0 in every frame of every window.
    """
    results = []
    tol = 1e-3
    for split in ("train", "val", "test"):
        X = arrays[f"X_{split}"]
        if X.shape[0] == 0:
            results.append((None, f"X_{split}: skipped (empty split)"))
            continue
        pelvis = X[:, :, :3]          # [N, W, 3]
        max_val = float(np.abs(pelvis).max())
        if max_val <= tol:
            results.append((True,  f"X_{split}: pelvis ~0 after normalisation (max={max_val:.2e}) ✓"))
        else:
            results.append((False, f"X_{split}: pelvis max = {max_val:.5f} (expected < {tol}) — "
                                   "root normalisation may have failed"))
    return results


def test_value_range(arrays: Dict[str, np.ndarray]) -> List[Tuple[bool, str]]:
    """
    Joint positions should be within a human-plausible range.
    After root normalisation all joints are relative to the pelvis,
    so values should stay within ±2.5 m for any realistic motion.
    """
    results = []
    limit = 2.5
    for split in ("train", "val", "test"):
        X = arrays[f"X_{split}"]
        Y = arrays[f"Y_{split}"]
        if X.shape[0] == 0:
            results.append((None, f"{split}: skipped (empty split)"))
            continue
        max_x = float(np.abs(X).max())
        max_y = float(np.abs(Y).max())
        worst = max(max_x, max_y)
        if worst <= limit:
            results.append((True,  f"{split}: max joint displacement {worst:.4f} m (< {limit} m) ✓"))
        else:
            results.append((False, f"{split}: max joint displacement {worst:.4f} m exceeds {limit} m "
                                   "— possible bad sequence or normalisation error"))
    return results


def test_split_sizes(
    arrays: Dict[str, np.ndarray],
    min_total: int,
) -> List[Tuple[bool, str]]:
    """
    All three splits are non-empty and proportions are approximately 80/10/10.
    Also checks that the total exceeds the minimum expected window count.
    """
    results = []
    n_train = arrays["X_train"].shape[0]
    n_val   = arrays["X_val"].shape[0]
    n_test  = arrays["X_test"].shape[0]
    total   = n_train + n_val + n_test

    # Non-empty
    for name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        if n > 0:
            results.append((True,  f"{name} split: {n:,} windows ✓"))
        else:
            results.append((False, f"{name} split is EMPTY — subject ID inference may have failed"))

    # Approximate proportions (generous tolerance: ±15%)
    if total > 0:
        for name, n, target in [("train", n_train, 0.80),
                                  ("val",   n_val,   0.10),
                                  ("test",  n_test,  0.10)]:
            actual = n / total
            lo, hi = target - 0.15, target + 0.15
            if lo <= actual <= hi:
                results.append((True,  f"{name} fraction {actual:.1%} (target ~{target:.0%}) ✓"))
            else:
                results.append((False, f"{name} fraction {actual:.1%} is far from target {target:.0%}"))

    # Minimum total
    if total >= min_total:
        results.append((True,  f"Total windows {total:,} >= minimum {min_total:,} ✓"))
    else:
        results.append((False, f"Total windows {total:,} < minimum {min_total:,} "
                               "— dataset may be too small for training"))

    return results


def test_no_subject_leakage(
    arrays: Dict[str, np.ndarray],
    data_dir: Path,
) -> List[Tuple[bool, str]]:
    """
    No subject ID appears in more than one split.

    Requires subject_ids_train/val/test.npy — saved by preprocess.py
    alongside X/Y for each split. subject_ids.npy (the full pre-split
    array) is in file-scan order, not split order, so it cannot be
    sliced by split sizes to reconstruct membership.
    """
    results = []

    per_split_files = {
        "train": data_dir / "subject_ids_train.npy",
        "val":   data_dir / "subject_ids_val.npy",
        "test":  data_dir / "subject_ids_test.npy",
    }

    missing = [str(p) for p in per_split_files.values() if not p.exists()]
    if missing:
        results.append((None,
            "Per-split subject ID files not found — re-run preprocess.py "
            "to generate subject_ids_train/val/test.npy. "
            f"Missing: {missing}"))
        return results

    split_sids = {
        name: set(np.load(path, allow_pickle=False).tolist())
        for name, path in per_split_files.items()
    }

    pairs = [
        ("train∩val",  "train", "val"),
        ("train∩test", "train", "test"),
        ("val∩test",   "val",   "test"),
    ]
    for label, a, b in pairs:
        overlap = split_sids[a] & split_sids[b]
        if not overlap:
            results.append((True,  f"No subject leakage in {label} ✓"))
        else:
            results.append((False,
                f"Subject(s) appear in both {label}: {sorted(str(s) for s in overlap)[:5]}"))

    all_sids = split_sids["train"] | split_sids["val"] | split_sids["test"]
    n_unique = len(all_sids)
    results.append((True, f"Total unique subjects across all splits: {n_unique}"))

    # Sanity: per-split window counts should match X array lengths
    for split in ("train", "val", "test"):
        sid_arr = np.load(per_split_files[split], allow_pickle=False)
        x_len   = arrays[f"X_{split}"].shape[0]
        if len(sid_arr) == x_len:
            results.append((True,
                f"subject_ids_{split} length ({len(sid_arr):,}) matches X_{split} ✓"))
        else:
            results.append((False,
                f"subject_ids_{split} length ({len(sid_arr):,}) != X_{split} length ({x_len:,})"))

    return results


def test_temporal_continuity(arrays: Dict[str, np.ndarray]) -> List[Tuple[bool, str]]:
    """
    X[-1] (last input frame) and Y[0] (first target frame) of each window
    should be adjacent — i.e. the target immediately follows the input.
    Checks a random sample of 1000 windows from the training set.
    """
    results = []
    X_train = arrays["X_train"]
    Y_train = arrays["Y_train"]

    if X_train.shape[0] == 0:
        results.append((None, "train split empty — continuity check skipped"))
        return results

    rng = np.random.default_rng(0)
    n_check = min(1000, X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n_check, replace=False)

    # The last frame of X and the first frame of Y must NOT be identical
    # (they are adjacent, not overlapping), but the displacement between
    # them should be small (< 0.5 m per joint on average) for smooth motion.
    last_input  = X_train[idx, -1, :]   # [n_check, 99]
    first_target = Y_train[idx,  0, :]  # [n_check, 99]
    delta = np.abs(first_target - last_input).reshape(n_check, 33, 3)
    mean_per_joint = delta.mean(axis=2)                  # [n_check, 33]
    max_jump = float(mean_per_joint.max())
    avg_jump = float(mean_per_joint.mean())

    if max_jump < 0.5:
        results.append((True,  f"X→Y temporal boundary: avg jump {avg_jump:.4f} m, "
                               f"max {max_jump:.4f} m (< 0.5 m) ✓"))
    else:
        results.append((False, f"X→Y temporal boundary: max joint jump {max_jump:.4f} m "
                               "— windows may not be contiguous"))

    # Also confirm last input != first target (not accidentally duplicated)
    identical = np.all(last_input == first_target, axis=1).sum()
    if identical == 0:
        results.append((True,  "No windows where last input frame == first target frame ✓"))
    else:
        results.append((False, f"{identical} windows have identical last-input and first-target frames"))

    return results


def test_velocity_statistics(arrays: Dict[str, np.ndarray]) -> List[Tuple[bool, str]]:
    """
    Frame-to-frame velocity within windows should be small and non-zero.
    Checks:
      - mean velocity > 0 (motion is actually happening)
      - mean velocity < 0.1 m/frame (no teleportation artifacts at 30fps)
    """
    results = []
    X = arrays["X_train"]
    if X.shape[0] == 0:
        results.append((None, "train split empty — velocity check skipped"))
        return results

    sample = X[:min(5000, X.shape[0])]        # keep it fast
    vel = np.diff(sample, axis=1)              # [N, W-1, 99]
    mean_vel = float(np.abs(vel).mean())
    max_vel  = float(np.abs(vel).max())

    if mean_vel > 1e-6:
        results.append((True,  f"Mean per-frame velocity {mean_vel:.5f} m > 0 (motion present) ✓"))
    else:
        results.append((False, f"Mean velocity is ~0 — all frames may be identical (static data?)"))

    if max_vel < 0.5:
        results.append((True,  f"Max per-frame velocity {max_vel:.4f} m < 0.5 m ✓"))
    else:
        results.append((False, f"Max per-frame velocity {max_vel:.4f} m is large "
                               "— possible FPS resampling issue or outlier sequence"))

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_arrays(data_dir: Path) -> Dict[str, np.ndarray]:
    arrays = {}
    keys = ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test",
            "X", "Y", "subject_ids",
            "subject_ids_train", "subject_ids_val", "subject_ids_test"]
    for key in keys:
        path = data_dir / f"{key}.npy"
        if path.exists():
            arrays[key] = np.load(path, allow_pickle=False)
    return arrays


def run_all(data_dir: Path, W: int, K: int, joints: int, min_total: int) -> bool:
    D = joints * 3
    print(f"{_BOLD}{'='*62}{_RESET}")
    print(f"{_BOLD}  OHIPA Preprocessing Test Suite{_RESET}")
    print(f"  Data directory : {data_dir.resolve()}")
    print(f"  Expected shape : X=(N,{W},{D})  Y=(N,{K},{D})")
    print(f"{'='*62}{_RESET}")

    passes = 0
    failures = 0
    warnings = 0

    def report(results: List[Tuple]):
        nonlocal passes, failures, warnings
        for item in results:
            passed, msg = item[0], item[1]
            if passed is True:
                print(_ok(msg));   passes += 1
            elif passed is False:
                print(_fail(msg)); failures += 1
            else:
                print(_warn(msg)); warnings += 1

    # 1. Files exist
    print(_head("1. Output files"))
    file_results = test_files_exist(data_dir)
    report(file_results)
    any_missing = any(not r[0] for r in file_results)
    if any_missing:
        print(f"\n{_RED}Critical files missing — cannot continue remaining tests.{_RESET}")
        print(f"\n{_BOLD}Result: {failures} failed, {passes} passed, {warnings} warnings{_RESET}")
        return False

    # Load all arrays
    arrays = load_arrays(data_dir)

    # 2. Shapes
    print(_head("2. Array shapes"))
    report(test_shapes(arrays, W, K, D))

    # 3. Dtype
    print(_head("3. Data types"))
    report(test_dtype(arrays))

    # 4. NaN / Inf
    print(_head("4. NaN / Inf check"))
    report(test_no_nan_inf(arrays))

    # 5. Root normalisation
    print(_head("5. Root-relative normalisation (pelvis ≈ 0)"))
    report(test_root_normalisation(arrays))

    # 6. Value range
    print(_head("6. Joint position range (< 2.5 m from pelvis)"))
    report(test_value_range(arrays))

    # 7. Split sizes
    print(_head("7. Train / val / test split sizes"))
    report(test_split_sizes(arrays, min_total))

    # 8. Subject leakage
    print(_head("8. No subject leakage across splits"))
    report(test_no_subject_leakage(arrays, data_dir))

    # 9. Temporal continuity
    print(_head("9. X → Y temporal continuity"))
    report(test_temporal_continuity(arrays))

    # 10. Velocity statistics
    print(_head("10. Per-frame velocity statistics"))
    report(test_velocity_statistics(arrays))

    # Summary
    total = passes + failures + warnings
    print(f"\n{'='*62}")
    print(f"  {_BOLD}Tests run: {total}   "
          f"{_GREEN}Passed: {passes}{_RESET}   "
          f"{_RED}Failed: {failures}{_RESET}   "
          f"{_YELLOW}Warnings: {warnings}{_RESET}")
    print(f"{'='*62}")

    if failures == 0:
        print(f"{_GREEN}{_BOLD}  ✓ All tests passed — ready for Phase 2 (training){_RESET}")
    else:
        print(f"{_RED}{_BOLD}  ✗ {failures} test(s) failed — fix before training{_RESET}")

    return failures == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate OHIPA preprocessing outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing X_train.npy, Y_train.npy, etc.",
    )
    p.add_argument("--W",         type=int, default=30,     help="Expected input window length.")
    p.add_argument("--K",         type=int, default=15,     help="Expected prediction horizon.")
    p.add_argument("--joints",    type=int, default=33,     help="Expected number of joints.")
    p.add_argument("--min_total", type=int, default=50_000, help="Minimum acceptable total windows.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run_all(
        data_dir=Path(args.data_dir),
        W=args.W,
        K=args.K,
        joints=args.joints,
        min_total=args.min_total,
    )
    sys.exit(0 if ok else 1)