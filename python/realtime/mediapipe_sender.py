"""
mediapipe_sender.py — Pose Estimation & ZMQ Sender
Online Human Intent Predictor with Adaptive Learning

Uses the MediaPipe Tasks PoseLandmarker API to run pose estimation on a
webcam or video file, extract 33 keypoint positions per frame, and send
them to the C++ runtime over ZMQ.

Wire format (must match pose_receiver.cpp exactly):
    [uint32_t seq | double timestamp | float32[99] keypoints]
    = 4 + 8 + 396 = 408 bytes per message

Coordinate system:
    MediaPipe outputs x, y in normalized image coordinates [0, 1] and z as a
    relative depth estimate. We take only (x, y, z) per landmark and discard
    visibility, giving 33 * 3 = 99 floats per frame.

Normalization:
    The pose is made root-relative by subtracting the midpoint of the left and
    right hip landmarks (23 and 24) from every landmark.

Usage:
    # Webcam:
    python python/realtime/mediapipe_sender.py \
        --model models/pose_landmarker_full.task \
        --endpoint "tcp://*:5556"

    # Video file:
    python python/realtime/mediapipe_sender.py \
        --source path/to/video.mp4 \
        --model models/pose_landmarker_full.task \
        --endpoint "tcp://*:5556"

    # Synthetic test mode:
    python python/realtime/mediapipe_sender.py \
        --synthetic \
        --endpoint "tcp://*:5556"

Dependencies:
    pip install mediapipe opencv-python pyzmq numpy
"""

import argparse
import struct
import time
import sys
from pathlib import Path

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# MediaPipe / OpenCV import guard
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import cv2
except ImportError:
    sys.exit(
        "mediapipe or opencv-python not installed.\n"
        "Run: pip install mediapipe opencv-python"
    )


# ---------------------------------------------------------------------------
# Constants — must match C++ side exactly
# ---------------------------------------------------------------------------
N_LANDMARKS = 33
N_DIMS      = 3
D           = N_LANDMARKS * N_DIMS   # 99
K           = 15                     # prediction horizon (for reference)
TARGET_FPS  = 30.0

LEFT_HIP_IDX  = 23
RIGHT_HIP_IDX = 24


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def root_normalise(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints: [33, 3]

    Returns root-relative [33, 3] coordinates using the midpoint of the
    MediaPipe left/right hip landmarks as the pelvis/root.
    """
    pelvis = (keypoints[LEFT_HIP_IDX] + keypoints[RIGHT_HIP_IDX]) / 2.0
    return keypoints - pelvis


# ---------------------------------------------------------------------------
# Extract keypoints from MediaPipe Tasks PoseLandmarker output
# ---------------------------------------------------------------------------
def extract_keypoints(pose_landmarks) -> np.ndarray | None:
    """
    pose_landmarks:
        A list of 33 NormalizedLandmark objects for one detected pose.

    Returns:
        [33, 3] float32 array of x, y, z, or None if no pose is available.
    """
    if pose_landmarks is None or len(pose_landmarks) != N_LANDMARKS:
        return None

    return np.array(
        [[lm.x, lm.y, lm.z] for lm in pose_landmarks],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Build the wire-format message
# ---------------------------------------------------------------------------
def build_message(seq: int, keypoints: np.ndarray) -> bytes:
    """
    keypoints: [33, 3] float32, already root-normalised

    Returns:
        408-byte message ready to send over ZMQ.
    """
    header  = struct.pack("<Id", seq, time.time())
    payload = keypoints.flatten().astype(np.float32).tobytes()

    return header + payload


# ---------------------------------------------------------------------------
# Synthetic sender
# ---------------------------------------------------------------------------
def run_synthetic(sock: zmq.Socket, args) -> None:
    """
    Sends smooth synthetic frames at TARGET_FPS.

    Useful for testing the ZMQ + C++ pipeline without a webcam or MediaPipe
    model.
    """
    print("[Sender] Synthetic mode — sending fake frames at 30fps.")
    print("[Sender] Ctrl+C to stop.")

    seq       = 0
    frame_dur = 1.0 / TARGET_FPS

    try:
        while True:
            t = time.time()

            phase_offsets = np.linspace(
                0,
                2 * np.pi,
                N_LANDMARKS * N_DIMS,
            )

            flat = np.sin(
                2 * np.pi * 0.5 * t + phase_offsets
            ).astype(np.float32)

            keypoints = flat.reshape(
                N_LANDMARKS,
                N_DIMS,
            ) * 0.3

            keypoints = root_normalise(keypoints)

            msg = build_message(seq, keypoints)

            try:
                sock.send(msg, zmq.NOBLOCK)
            except zmq.Again:
                # Drop frame if C++ consumer cannot keep up.
                pass

            seq += 1

            if seq % 150 == 0:
                print(f"[Sender] Synthetic frames sent: {seq}")

            time.sleep(frame_dur)

    except KeyboardInterrupt:
        print(
            f"\n[Sender] Synthetic stopped. "
            f"Total frames sent: {seq}"
        )


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------
def parse_source(source):
    """
    argparse returns --source as a string when supplied explicitly.

    Converts webcam values such as "0" into int for cv2.VideoCapture.
    Otherwise returns the value as a video path.
    """
    if isinstance(source, int):
        return source

    try:
        return int(source)
    except (TypeError, ValueError):
        return source


# ---------------------------------------------------------------------------
# Real MediaPipe sender
# ---------------------------------------------------------------------------
def run_mediapipe(sock: zmq.Socket, args) -> None:
    """
    Opens a webcam or video file, runs MediaPipe PoseLandmarker, and sends
    root-normalised poses over ZMQ.

    VIDEO running mode is used because frames are processed synchronously
    with monotonically increasing timestamps.

    No visualization is performed here. Prediction visualization is handled
    separately by zmq_client.py.
    """
    model_path = Path(args.model)

    if not model_path.exists():
        sys.exit(
            f"Pose Landmarker model not found: {model_path}\n"
            "Pass --model /path/to/pose_landmarker_full.task"
        )

    base_options = mp_python.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    source = parse_source(args.source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        sys.exit(
            f"Error: Could not open video capture source: {source}"
        )

    seq            = 0
    frame_dur      = 1.0 / TARGET_FPS
    frames_sent    = 0
    frames_skipped = 0

    print(f"[Sender] Opening source: {source}")
    print(f"[Sender] Pose model: {model_path}")
    print(f"[Sender] Sending to: {args.endpoint}")
    print("[Sender] Ctrl+C to stop.")

    start_time        = time.monotonic()
    last_timestamp_ms = -1

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                t_loop_start = time.time()

                # MediaPipe expects RGB.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                rgb_frame = np.ascontiguousarray(
                    rgb_frame,
                    dtype=np.uint8,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # VIDEO mode requires monotonically increasing timestamps.
                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if not result.pose_landmarks:
                    frames_skipped += 1

                    elapsed = time.time() - t_loop_start
                    time.sleep(
                        max(0.0, frame_dur - elapsed)
                    )
                    continue

                landmarks = result.pose_landmarks[0]

                keypoints = extract_keypoints(landmarks)

                if keypoints is None:
                    frames_skipped += 1
                    continue

                keypoints = root_normalise(keypoints)

                msg = build_message(
                    seq,
                    keypoints,
                )

                try:
                    sock.send(
                        msg,
                        zmq.NOBLOCK,
                    )
                except zmq.Again:
                    # Drop instead of blocking to preserve freshest data.
                    pass

                seq += 1
                frames_sent += 1

                # Maintain target FPS.
                elapsed = time.time() - t_loop_start
                time.sleep(
                    max(0.0, frame_dur - elapsed)
                )

                # Periodic stats.
                if frames_sent % 150 == 0:
                    print(
                        f"[Sender] Frames sent:    "
                        f"{frames_sent}"
                    )
                    print(
                        f"[Sender] Frames skipped: "
                        f"{frames_skipped}"
                    )

    except KeyboardInterrupt:
        print("\n[Sender] Stopped.")

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe pose sender for OHIPA C++ runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--source",
        default=0,
        help="Webcam index (e.g. 0) or path to a video file.",
    )

    p.add_argument(
        "--endpoint",
        default="tcp://localhost:5555",
        help="ZMQ endpoint for the PUSH socket to bind to.",
    )

    p.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help=(
            "Path to a MediaPipe Pose Landmarker .task model. "
            "Not required when --synthetic is used."
        ),
    )

    p.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Send synthetic sinusoidal data instead of "
            "camera/video poses."
        ),
    )

    p.add_argument(
        "--hwm",
        type=int,
        default=4,
        help=(
            "ZMQ send high-water mark. "
            "Keep small for freshness."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    ctx = zmq.Context()

    sock = ctx.socket(zmq.PUSH)

    sock.setsockopt(
        zmq.SNDHWM,
        args.hwm,
    )

    sock.bind(args.endpoint)

    print(
        f"[Sender] Bound to {args.endpoint} "
        f"(SNDHWM={args.hwm})"
    )

    try:
        if args.synthetic:
            run_synthetic(
                sock,
                args,
            )
        else:
            run_mediapipe(
                sock,
                args,
            )

    except KeyboardInterrupt:
        print("\n[Sender] Stopped.")

    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()