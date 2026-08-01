"""
mediapipe_sender.py — Pose Estimation & ZMQ Sender
Online Human Intent Predictor with Adaptive Learning

Runs MediaPipe Pose on a webcam or video file, extracts 33 keypoint
positions per frame, and sends them to the C++ runtime over ZMQ.

Wire format (must match pose_receiver.cpp exactly):
    [uint32_t seq | double timestamp | float32[99] keypoints]
    = 4 + 8 + 396 = 408 bytes per message

Coordinate system:
    MediaPipe outputs x, y in normalised image coords [0,1] and z as a
    relative depth estimate. We take only (x, y, z) per landmark and
    discard visibility — giving 33 * 3 = 99 floats per frame.

Usage:
    # Webcam (default device 0):
    python python/realtime/mediapipe_sender.py --endpoint tcp://VM_IP:5555

    # Video file:
    python python/realtime/mediapipe_sender.py \
        --source path/to/video.mp4 \
        --endpoint tcp://VM_IP:5555

    # Synthetic test mode (no webcam needed — sends random frames):
    python python/realtime/mediapipe_sender.py --synthetic

Dependencies:
    pip install mediapipe opencv-python pyzmq numpy
"""

import argparse
import struct
import time
import sys

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# MediaPipe import guard
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
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
N_DIMS      = 3        # x, y, z (drop visibility)
D           = N_LANDMARKS * N_DIMS   # 99
K           = 15       # prediction horizon (for reference)
TARGET_FPS  = 30.0


# ---------------------------------------------------------------------------
# Normalisation
#
# Subtract pelvis (landmark 0) from all landmarks to produce root-relative
# coordinates. This must match the normalisation in preprocess.py exactly.
# ---------------------------------------------------------------------------
def root_normalise(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints: [33, 3]
    Returns root-relative [33, 3] with pelvis at origin.
    """
    return keypoints - keypoints[0]


# ---------------------------------------------------------------------------
# Extract keypoints from a MediaPipe result
# ---------------------------------------------------------------------------
def extract_keypoints(landmarks) -> np.ndarray:
    """
    Convert a MediaPipe NormalizedLandmarkList into a [33, 3] float32 array.
    Extracts (x, y, z) only — drops visibility.

    Returns None if landmarks is None (no person detected).
    """
    if landmarks is None:
        return None

    return np.array([[l.x,l.y,l.z] for l in landmarks.landmark],dtype=np.float32)
    


# ---------------------------------------------------------------------------
# Build the wire-format message
# ---------------------------------------------------------------------------
def build_message(seq: int, keypoints: np.ndarray) -> bytes:
    """
    keypoints: [33, 3] float32, already root-normalised
    Returns: 408-byte message ready to send over ZMQ
    """
    header  = struct.pack('<Id',seq, time.time())
    payload = keypoints.flatten().astype(np.float32).tobytes()
    return header + payload


# ---------------------------------------------------------------------------
# Synthetic sender 
#
# Sends plausible-looking sinusoidal keypoint data at TARGET_FPS.
# Use this to confirm the C++ pipeline is working before involving a webcam.
# ---------------------------------------------------------------------------
def run_synthetic(sock: zmq.Socket, args) -> None:
    """
    Sends synthetic frames at TARGET_FPS indefinitely.
    Keypoints oscillate sinusoidally so the ring buffer sees smooth motion
    rather than pure noise — closer to real data and easier to spot bugs.
    """
    print("[Sender] Synthetic mode — sending fake frames at 30fps.")
    print("[Sender] Ctrl+C to stop.")

    seq       = 0
    frame_dur = 1.0 / TARGET_FPS

    try:
        while True:
            t = time.time()

            # Sinusiodal keypoints: each joint oscillates at a slightly
            # different frequent so the motion looks varied
            phase_offsets = np.linspace(0, 2 * np.pi, N_LANDMARKS * N_DIMS)
            flat = np.sin(2 * np.pi * 0.5 * t + phase_offsets).astype(np.float32)
            keypoints = flat.reshape(N_LANDMARKS, N_DIMS) * 0.3 # scale to meters (roughly)

            keypoints = root_normalise(keypoints)
            msg = build_message(seq, keypoints)

            try: 
                sock.send(msg, zmq.NOBLOCK)
            except zmq.Again:
                pass # drop if C++ isn't consuming fast enough

            seq += 1

            if seq % 150 == 0:
                print(f"[Sender] Synthetic frames sent: {seq}")

            time.sleep(frame_dur)
            
    except KeyboardInterrupt:
        print(f"\n[Sender] Synthetic stopped. Total frames sent: {seq}")


# ---------------------------------------------------------------------------
# Real MediaPipe sender
# ---------------------------------------------------------------------------
def run_mediapipe(sock: zmq.Socket, args) -> None:
    """
    Opens a webcam or video file, runs MediaPipe Pose, and sends frames.

    MediaPipe Pose docs:
        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

    Key decisions:
    - model_complexity=1 (balanced accuracy/speed; 0 is fastest, 2 is best)
    - min_detection_confidence=0.5
    - When no person is detected, DO NOT send — the C++ side handles
      absence of frames gracefully (ring buffer just doesn't advance).
    """
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        sys.exit(1)

    seq            = 0
    frame_dur      = 1.0 / TARGET_FPS
    frames_sent    = 0
    frames_skipped = 0

    print(f"[Sender] Opening source: {args.source}")
    print(f"[Sender] Sending to: {args.endpoint}")
    print("[Sender] Ctrl+C to stop.")

    with mp_pose.Pose(model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            t_loop_start = time.time()

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(rgb_frame)

            kp = extract_keypoints(results.pose_landmarks)
            if kp is None: 
                frames_skipped += 1
                continue

            kp = root_normalise(kp)

            msg = build_message(seq, kp)
            try:
                sock.send(msg, zmq.NOBLOCK)
            except zmq.Again:
                pass

            seq         += 1
            frames_sent += 1

            # Optional: draw skeleton overlay and show with cv2.imshow
            # Useful for debugging - shows you what MediaPipe sees
            mp_drawing.draw_landmarks(frame, results.pose_landmarks)
            cv2.imshow('OHIPA Sender', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Maintain target FPS
            elapsed = time.time() - t_loop_start
            time.sleep(max(0, frame_dur - elapsed))

            # Stats: Every 150 frames
            if frames_sent % 150 == 0:
                print(f"[Sender] Frames sent:    {frames_sent}")
                print(f"[Sender] Frames skipped: {frames_skipped}")

    cap.release()
    cv2.destroyAllWindows()



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
        help="Webcam index (int) or path to video file (str).",
    )
    p.add_argument(
        "--endpoint",
        default="tcp://localhost:5555",
        help="ZMQ endpoint to bind to. Use VM IP for remote C++ process.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Send synthetic sinusoidal data instead of real MediaPipe output.",
    )
    p.add_argument(
        "--hwm",
        type=int,
        default=4,
        help="ZMQ send high-water mark. Keep small for freshness policy.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Set up ZMQ PUSH socket
    # ------------------------------------------------------------------
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, args.hwm)
    sock.bind(args.endpoint)

    print(f"[Sender] Bound to {args.endpoint} (SNDHWM={args.hwm})")

    try:
        if args.synthetic:
            run_synthetic(sock, args)
        else:
            run_mediapipe(sock, args)
    except KeyboardInterrupt:
        print("\n[Sender] Stopped.")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()