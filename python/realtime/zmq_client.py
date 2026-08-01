"""
zmq_client.py — Prediction Visualizer
Online Human Intent Predictor with Adaptive Learning

Subscribes to the C++ runtime's PUB socket and visualizes:
  - The current observed pose (received via a second channel or inferred)
  - The predicted future trajectory (K=15 frames)

Wire format (output from C++):
    float32[15 * 99] — predicted keypoints for next 15 frames
    shape after reshape: [15, 33, 3]

Usage:
    python python/realtime/zmq_client.py --endpoint tcp://VM_IP:5556

    # Headless (no display, e.g. on the VM):
    python python/realtime/zmq_client.py --endpoint tcp://VM_IP:5556 --text_only

Dependencies:
    pip install pyzmq numpy matplotlib
"""

import argparse
import sys
import time

import numpy as np
import zmq

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    sys.exit("matplotlib not installed. Run: pip install matplotlib")

# ---------------------------------------------------------------------------
# Constants — must match C++ side
# ---------------------------------------------------------------------------
K           = 15     # prediction horizon frames
N_LANDMARKS = 33     # Number of joints
N_DIMS      = 3
D           = N_LANDMARKS * N_DIMS   # 99
MSG_SIZE    = K * D * 4              # bytes expected per message


# ---------------------------------------------------------------------------
# PredictionReceiver
#
# Runs in the main thread — receives predictions from the C++ PUB socket
# and stores the latest one for the visualizer to render.
# ---------------------------------------------------------------------------
class PredictionReceiver:
    """
    Wraps a ZMQ SUB socket. Call update() each frame to poll for the
    latest prediction without blocking.
    """

    def __init__(self, endpoint: str):
        self.context = zmq.Context()
        self.socket  = self.context.socket(zmq.SUB)

        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 0)

        self.socket.connect(endpoint)

        self.latest: np.ndarray | None = None   # [K, 33, 3] or None
        self.frames_received = 0

    def update(self) -> bool:
        """
        Non-blocking poll. Returns True if a new prediction was received.
        Sets self.latest to the new [K, 33, 3] array.
        """
        try:
            msg = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return False

        if len(msg) != MSG_SIZE:
            print(f"[Visualizer] Unexpected message size: {len(msg)} (expected {MSG_SIZE})")
            return False
        
        self.latest = np.frombuffer(msg, dtype=np.float32).reshape(K, N_LANDMARKS, N_DIMS)
        self.frames_received += 1
        return True



    def close(self):
        self.socket.close()
        self.context.term()
        


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
class Visualizer:
    """
    Animates the predicted future trajectory using matplotlib.
    """

    def __init__(self, receiver: PredictionReceiver):
        self.receiver = receiver

        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Predicted Future Trajectory")

        self.scatter = self.ax.scatter([],[],[])
        self.anim    = None # assigned in run() - must be stored to prevent GC
        

    def _update_frame(self, frame_idx):
        """
        Called by FuncAnimation each tick. Polls the receiver and redraws
        if a new prediction is available.

        frame_idx is provided by FuncAnimation — you don't call this directly.
        """
        self.receiver.update()

        if self.receiver.latest is not None:
            # Show the final predicted frame [33, 3]
            frame = self.receiver.latest[-1]

            self.scatter._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])

            self.ax.set_title(
                f"Predicted Future Trajectory | "
                f"frames received: {self.receiver.frames_received}"
            )

        return (self.scatter, )

    def run(self, interval_ms: int = 33):
        """
        Start the animation loop. Blocks until the window is closed.
        interval_ms: target refresh interval (33ms ≈ 30fps).
        """
        self.anim = animation.FuncAnimation(
            self.fig, self._update_frame, interval=interval_ms, blit=False
        )
        plt.show()

# ---------------------------------------------------------------------------
# Text-only mode (headless)
# ---------------------------------------------------------------------------
def run_text_only(receiver: PredictionReceiver) -> None:
    """
    Headless mode — prints prediction stats without opening a window.
    """
    print("[Visualizer] Text-only mode. Ctrl+C to stop.")

    try:
        while True:
            new_pred = receiver.update()
            if new_pred and receiver.latest is not None:
                latest = receiver.latest
                print(f"[Visualizer] shape={latest.shape}  "
                      f"mean={latest.mean():.4f}  "
                      f"max={latest.max():.4f}  "
                      f"frames={receiver.frames_received}")

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize OHIPA predictions from the C++ runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--endpoint",
        default="tcp://localhost:5556",
        help="ZMQ endpoint of the C++ PUB socket.",
    )
    p.add_argument(
        "--text_only",
        action="store_true",
        help="Print raw prediction stats to stdout instead of plotting. "
             "Useful when running headless (no display).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    receiver = PredictionReceiver(args.endpoint)
    print(f"[Visualizer] Subscribed to {args.endpoint}")

    try:
        if args.text_only:
            run_text_only(receiver)
        else:
            viz = Visualizer(receiver)
            viz.run()
    except KeyboardInterrupt:
        print("\n[Visualizer] Stopped.")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()