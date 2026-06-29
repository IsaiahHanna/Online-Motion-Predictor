"""
preprocess.py — AMASS Dataset Preprocessing Pipeline
Online Human Intent Predictor with Adaptive Learning
Author: Isaiah | SFU Data Science

Converts AMASS motion capture sequences (SMPL+H pose parameters) into
sliding windows of 3D joint positions ready for transformer training.

Output shapes:
    X.npy — [N, 30, 99]   input windows  (W=30 frames, 33 joints × 3 coords)
    Y.npy — [N, 15, 99]   target windows (K=15 future frames)

Directory layout expected (matches your actual download):
    Data/
      amass/               <- --amass_root  (contains CMU/ subdirectory)
      body_models/
        smpl/
          models/
            basicmodel_f_lbs_10_207_0_v1.*.pkl
            basicmodel_m_lbs_10_207_0_v1.*.pkl
            basicmodel_neutral_lbs_10_207_0_v1.*.pkl   <- used
        dmpls/          

Usage:
    python preprocess.py \
        --amass_root      ./Data/amass \
        --smpl_model_dir  ./Data/body_models \
        --output_dir      ./data \
        [--window_size 30] \
        [--horizon     15] \
        [--stride       5] \
        [--joints      33] \
        [--target_fps  30] \
        [--split_seed  42]

Notes on downloaded data
------------------------
* AMASS "CMU SMPL+H G" stores 156-dim poses (SMPL-H, hands included).
  We read only the first 72 dims (global orient + body pose) and run
  them through the plain SMPL model — hand params are intentionally
  discarded so the joint set matches what MediaPipe produces at runtime.
 
* SMPL v1.1.0 ships the neutral model as:
      body_models/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.x.x.pkl
  smplx.create() expects the file to be named SMPL_NEUTRAL.pkl.
  build_body_model() finds the actual file automatically and passes its
  path directly — no manual renaming required.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import numpy as np

# Compatibility patch for old chumpy/SMPL pickles with modern NumPy.
# Use NumPy scalar types where possible; do not set np.bool = bool.
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "unicode"):
    np.unicode = str
if not hasattr(np, "str"):
    np.str = str
    
import torch
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# smplx import guard
# ---------------------------------------------------------------------------
try:
    import smplx
except ImportError:
    sys.exit(
        "smplx is not installed.\n"
        "Run:  pip install smplx chumpy\n"
        "and confirm you have the SMPL model .pkl files downloaded."
    )


# ---------------------------------------------------------------------------
# Body model
# ---------------------------------------------------------------------------

def _ensure_smpl_neutral_pkl(model_dir: str) -> str:
    """
    Ensure smplx can find the neutral SMPL model and return the directory
    to pass as model_path to smplx.create().
 
    smplx.create(model_type='smpl') internally constructs:
        <model_path>/smpl/SMPL_NEUTRAL.pkl
    and hard-asserts it exists. SMPL v1.1.0 ships the file as:
        smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
    so the name never matches and the assertion fails.
    """
    root = Path(model_dir)
    expected = root / "smpl" / "SMPL_NEUTRAL.pkl"
 
    if expected.exists() or expected.is_symlink():
        log.info("Using SMPL neutral model: %s", expected.resolve())
        return str(root)
 
    # Search for the actual neutral SMPL pkl (exclude smplx/smplh variants)
    candidates = sorted(root.rglob("*.pkl"))
    neutral_candidates = [
        p for p in candidates
        if "neutral" in p.name.lower()
        and "smplx" not in p.name.lower()
        and "smplh" not in p.name.lower()
    ]
 
    if not neutral_candidates:
        found = [str(p) for p in candidates[:10]]
        raise FileNotFoundError(
            f"Could not find a neutral SMPL .pkl under '{model_dir}'.\n"
            f"Files found: {found}\n"
            "Make sure SMPL v1.1.0 is extracted under Data/body_models/smpl/."
        )
 
    actual = neutral_candidates[0].resolve()
    log.info("Found SMPL neutral model at: %s", actual)
 
    # Create symlink: Data/body_models/smpl/SMPL_NEUTRAL.pkl -> actual file
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.symlink_to(actual)
    log.info("Created symlink: %s -> %s", expected, actual)
 
    return str(root)
 
 
def build_body_model(model_dir: str, device: torch.device) -> smplx.SMPL:
    """
    Load SMPL (neutral) from *model_dir*.
 
    Handles SMPL v1.1.0's non-standard filename (basicmodel_neutral_lbs_*.pkl)
    by automatically symlinking it to SMPL_NEUTRAL.pkl on first run,
    which is the name smplx.create() requires internally.
 
    We use model_type='smpl' (not 'smplh') because AMASS SMPL+H poses are
    truncated to 72 dims (body only) before the forward pass.
    """
    smpl_model_path = _ensure_smpl_neutral_pkl(model_dir)
 
    model = smplx.create(
        model_path=smpl_model_path,
        model_type="smpl",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    ).to(device)
    model.eval()
    return model
 

# ---------------------------------------------------------------------------
# FPS resampling
# ---------------------------------------------------------------------------

def resample_sequence(seq: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """
    Temporally resample *seq* from *src_fps* to *tgt_fps* using linear
    interpolation along the time axis.

    Parameters
    ----------
    seq     : [T, D]  float32
    src_fps : original capture framerate (from npz['mocap_framerate'])
    tgt_fps : desired output framerate (--target_fps, default 30)

    Returns
    -------
    resampled : [T', D]  where T' = round(T * tgt_fps / src_fps)
    """
    if abs(src_fps - tgt_fps) < 0.5:          # already close enough, skip
        return seq

    T, D = seq.shape
    T_new = max(2, round(T * tgt_fps / src_fps))

    src_times = np.linspace(0.0, 1.0, T)
    tgt_times = np.linspace(0.0, 1.0, T_new)

    # Interpolate each dimension independently
    resampled = np.empty((T_new, D), dtype=np.float32)
    for d in range(D):
        resampled[:, d] = np.interp(tgt_times, src_times, seq[:, d])

    return resampled


# ---------------------------------------------------------------------------
# NPZ → joint positions
# ---------------------------------------------------------------------------

def npz_to_joints(
    npz_path: Path,
    body_model: smplx.SMPL,
    n_joints: int,
    device: torch.device,
    target_fps: float,
) -> Optional[np.ndarray]:
    """
    Convert a single AMASS .npz to root-relative 3-D joint positions,
    optionally resampled to *target_fps*.

    AMASS SMPL+H files contain:
        poses            — [T, 156]  (first 72: global orient + body pose)
        trans            — [T, 3]    (world translation — discarded)
        betas            — [16]      (shape params)
        mocap_framerate  — scalar    (original capture FPS)

    Returns
    -------
    joints : np.ndarray [T', n_joints * 3]  float32, root-relative
             None if the file is unreadable or too short.
    """
    # --- load ---
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        log.warning("Cannot load %s: %s", npz_path, exc)
        return None

    if "poses" not in data:
        log.warning("No 'poses' key in %s — skipping.", npz_path)
        return None

    raw_poses = data["poses"]          # [T, 156] for SMPL+H, [T, 72] for SMPL
    T = raw_poses.shape[0]
    if T < 2:
        return None

    # SMPL+H uses 156-dim poses; we take the first 72 (body only, no hands)
    poses = torch.tensor(raw_poses[:, :72], dtype=torch.float32, device=device)

    # --- shape params ---
    # betas may be 1-D [16] or 2-D [1, 16] depending on the AMASS subset
    if "betas" in data:
        betas_np = np.array(data["betas"]).flatten()[:10].astype(np.float32)
        betas = (
            torch.tensor(betas_np, dtype=torch.float32, device=device)
            .unsqueeze(0)            # [1, 10]
            .expand(T, -1)           # [T, 10]
        )
    else:
        betas = torch.zeros(T, 10, dtype=torch.float32, device=device)

    # --- SMPL forward pass (chunked to avoid GPU OOM) ---
    chunk = 512
    all_joints: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            out = body_model(
                body_pose=poses[start:end, 3:],       # [C, 69]
                global_orient=poses[start:end, :3],   # [C, 3]
                betas=betas[start:end],               # [C, 10]
                return_verts=False,
            )
            # out.joints: [C, 45, 3] for SMPL (indices 0-23 are body joints)
            joints_chunk = out.joints[:, :n_joints, :]   # [C, n_joints, 3]
            all_joints.append(joints_chunk.cpu())

    joints = torch.cat(all_joints, dim=0).numpy()    # [T, n_joints, 3]

    # --- root-relative normalisation: subtract pelvis (joint 0) ---
    pelvis = joints[:, 0:1, :]      # [T, 1, 3]
    joints = joints - pelvis        # broadcast over n_joints

    # --- flatten to [T, n_joints * 3] ---
    joints = joints.reshape(T, -1).astype(np.float32)

    # --- FPS resampling ---
    src_fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else target_fps
    if src_fps <= 0:
        src_fps = target_fps
    joints = resample_sequence(joints, src_fps, target_fps)

    return joints


# ---------------------------------------------------------------------------
# Sliding window extraction
# ---------------------------------------------------------------------------

def extract_windows(
    seq: np.ndarray,
    W: int,
    K: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a (W + K)-frame window over *seq* with the given *stride*.

    Parameters
    ----------
    seq    : [T, D]
    W      : input window length  (30)
    K      : prediction horizon   (15)
    stride : step between windows  (5)

    Returns
    -------
    X : [n_windows, W, D]
    Y : [n_windows, K, D]
    """
    T, D = seq.shape
    total = W + K
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for start in range(0, T - total + 1, stride):
        window = seq[start : start + total]   # [W+K, D]
        X_list.append(window[:W])
        Y_list.append(window[W:])

    if not X_list:
        return (
            np.empty((0, W, D), dtype=np.float32),
            np.empty((0, K, D), dtype=np.float32),
        )

    X = np.stack(X_list, axis=0)    # [n, W, D]
    Y = np.stack(Y_list, axis=0)    # [n, K, D]
    return X, Y


# ---------------------------------------------------------------------------
# Subject-aware train / val / test split
# ---------------------------------------------------------------------------

def subject_split(
    subject_ids: np.ndarray,
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split window indices by subject ID so no subject leaks across partitions.

    Returns boolean masks for train, val, test.
    """
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    rng.shuffle(unique_subjects)

    n       = len(unique_subjects)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects   = set(unique_subjects[n_train : n_train + n_val])
    test_subjects  = set(unique_subjects[n_train + n_val :])

    train_mask = np.array([s in train_subjects for s in subject_ids])
    val_mask   = np.array([s in val_subjects   for s in subject_ids])
    test_mask  = np.array([s in test_subjects  for s in subject_ids])

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(X: np.ndarray, Y: np.ndarray, n_joints: int) -> None:
    """Basic integrity checks on the assembled dataset."""
    D = n_joints * 3
    assert X.ndim == 3 and X.shape[2] == D, \
        f"X shape mismatch: expected [N, W, {D}], got {X.shape}"
    assert Y.ndim == 3 and Y.shape[2] == D, \
        f"Y shape mismatch: expected [N, K, {D}], got {Y.shape}"
    assert X.shape[0] == Y.shape[0], "X and Y window counts must match"

    if np.any(~np.isfinite(X)):
        log.warning("NaN/Inf detected in X — check preprocessing pipeline.")
    if np.any(~np.isfinite(Y)):
        log.warning("NaN/Inf detected in Y — check preprocessing pipeline.")

    # After root-normalisation, pelvis (dims 0-2) should be ~0 every frame
    root_coords = X[:, :, :3]
    root_max = float(np.abs(root_coords).max())
    if root_max > 1e-3:
        log.warning(
            "Root joint max absolute value = %.5f (expected ~0 after normalisation).",
            root_max,
        )

    pos_range = float(np.abs(X).max())
    log.info(
        "Joint position range (absolute): %.4f  (expect < 2.0 m for humans)",
        pos_range,
    )
    log.info("Sanity check passed — X %s  Y %s  dtype=%s", X.shape, Y.shape, X.dtype)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_npz_paths(amass_root: Path) -> List[Tuple[Path, str]]:
    """
    Recursively find all .npz files under *amass_root*.

    Subject ID is inferred from the first subdirectory level, e.g.:
        CMU/01/01_01_stances_poses.npz  →  subject '01'
    """
    paths = []
    for npz in sorted(amass_root.rglob("*.npz")):
        rel = npz.relative_to(amass_root)
        subject_id = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        paths.append((npz, subject_id))
    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    amass_root = Path(args.amass_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    W          = args.window_size   # 30
    K          = args.horizon       # 15
    stride     = args.stride        # 5
    n_joints   = args.joints        # 33
    target_fps = args.target_fps    # 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading SMPL body model from: %s", args.smpl_model_dir)
    body_model = build_body_model(args.smpl_model_dir, device)
    log.info("SMPL model loaded.")

    npz_files = collect_npz_paths(amass_root)
    log.info("Found %d .npz files under %s", len(npz_files), amass_root)

    if not npz_files:
        sys.exit(
            f"No .npz files found under {amass_root}.\n"
            "Check that --amass_root points to the correct directory."
        )

    all_X:   List[np.ndarray] = []
    all_Y:   List[np.ndarray] = []
    all_sid: List[str]        = []
    failed = 0

    for idx, (npz_path, subject_id) in enumerate(npz_files):
        if idx % 50 == 0:
            log.info(
                "Processing file %d / %d  (subject %s) …",
                idx + 1, len(npz_files), subject_id,
            )

        joints = npz_to_joints(npz_path, body_model, n_joints, device, target_fps)
        if joints is None:
            failed += 1
            continue

        X_seq, Y_seq = extract_windows(joints, W=W, K=K, stride=stride)

        if X_seq.shape[0] == 0:
            log.debug("Sequence too short for any windows: %s", npz_path.name)
            continue

        all_X.append(X_seq)
        all_Y.append(Y_seq)
        all_sid.extend([subject_id] * X_seq.shape[0])

    log.info(
        "Finished. Skipped %d / %d files (corrupt or too short).",
        failed, len(npz_files),
    )

    if not all_X:
        sys.exit("No valid windows generated. Check dataset path and .npz contents.")

    X           = np.concatenate(all_X,  axis=0)   # [N, W, D]
    Y           = np.concatenate(all_Y,  axis=0)   # [N, K, D]
    subject_ids = np.array(all_sid)                # [N]

    log.info("Total windows: %d", X.shape[0])

    sanity_check(X, Y, n_joints)

    train_mask, val_mask, test_mask = subject_split(
        subject_ids, seed=args.split_seed
    )

    splits = {
        "train": (X[train_mask], Y[train_mask]),
        "val":   (X[val_mask],   Y[val_mask]),
        "test":  (X[test_mask],  Y[test_mask]),
    }

    for split_name, (X_split, Y_split) in splits.items():
        log.info("  %-6s  %6d windows", split_name, X_split.shape[0])
        np.save(output_dir / f"X_{split_name}.npy", X_split)
        np.save(output_dir / f"Y_{split_name}.npy", Y_split)
        log.info("    Saved → X_%s.npy  Y_%s.npy", split_name, split_name)

    np.save(output_dir / "X.npy",           X)
    np.save(output_dir / "Y.npy",           Y)
    np.save(output_dir / "subject_ids.npy", subject_ids)
    log.info("Full arrays saved: X.npy %s  Y.npy %s", X.shape, Y.shape)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Total windows    : {X.shape[0]:,}")
    print(f"  X shape          : {X.shape}  (float32)")
    print(f"  Y shape          : {Y.shape}  (float32)")   
    print(f"  Train windows    : {train_mask.sum():,}")
    print(f"  Val   windows    : {val_mask.sum():,}")
    print(f"  Test  windows    : {test_mask.sum():,}")
    print(f"  Unique subjects  : {len(np.unique(subject_ids))}")  
    print(f"  Window size (W)  : {W} frames")
    print(f"  Horizon    (K)   : {K} frames")
    print(f"  Stride           : {stride} frames")
    print(f"  Target FPS       : {target_fps}")
    print(f"  Joints           : {n_joints}  →  {n_joints * 3} dims")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess AMASS motion capture data for IntentPredictor training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--amass_root",
        required=True,
        help=(
            "Root of the AMASS dataset, e.g. ./Data/amass  "
            "(recurses into all subdirectories, so you can point at the top-level "
            "Data/amass folder or directly at Data/amass/CMU)."
        ),
    )
    p.add_argument(
        "--smpl_model_dir",
        required=True,
        help=(
            "Directory to search for the neutral SMPL .pkl, e.g. ./Data/body_models  "
            "(finds basicmodel_neutral_lbs_*.pkl automatically — "
            "no renaming to SMPL_NEUTRAL.pkl required)."
        ),
    )
    p.add_argument(
        "--output_dir",
        default="./data",
        help="Where to write X.npy, Y.npy, and the train/val/test splits.",
    )
    p.add_argument("--window_size", type=int,   default=30,   help="Input window length W (frames).")
    p.add_argument("--horizon",     type=int,   default=15,   help="Prediction horizon K (frames).")
    p.add_argument("--stride",      type=int,   default=5,    help="Sliding window stride (frames).")
    p.add_argument("--joints",      type=int,   default=33,   help="Number of joints to keep per frame.")
    p.add_argument("--target_fps",  type=float, default=30.0, help="Resample all sequences to this FPS.")
    p.add_argument("--split_seed",  type=int,   default=42,   help="RNG seed for subject-level split.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())