"""
zmq_client.py — Prediction Visualizer
Online Human Intent Predictor with Adaptive Learning

Subscribes to the C++ runtime's PUB socket and visualizes the predicted
future human pose trajectory.

Wire format (output from C++):
    float32[15 * 99] — predicted keypoints for next 15 frames
    shape after reshape: [15, 33, 3]

Visualization:
    - Draws several intermediate predicted poses with low opacity
    - Highlights the final predicted pose
    - Draws MediaPipe skeleton connections
    - Draws trajectories for important joints through the prediction horizon
    - Automatically scales the 3D axes around the prediction
    - Keeps equal X/Y/Z scaling to avoid distortion
    - Drops stale queued predictions and renders the newest prediction

Usage:
    python python/realtime/zmq_client.py \
        --endpoint tcp://localhost:5557

    # Headless:
    python python/realtime/zmq_client.py \
        --endpoint tcp://localhost:5557 \
        --text_only

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
K           = 15
N_LANDMARKS = 33
N_DIMS      = 3

D        = N_LANDMARKS * N_DIMS
MSG_SIZE = K * D * 4


# ---------------------------------------------------------------------------
# MediaPipe Pose landmark connections
# ---------------------------------------------------------------------------
POSE_CONNECTIONS = [
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    (9, 10),

    # Upper body
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Left hand
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),

    # Right arm
    (12, 14),
    (14, 16),

    # Right hand
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Left foot
    (27, 29),
    (29, 31),
    (27, 31),

    # Right leg
    (24, 26),
    (26, 28),

    # Right foot
    (28, 30),
    (30, 32),
    (28, 32),
]


# Important joints whose motion through time is useful to visualize.
TRAJECTORY_JOINTS = [
    0,      # nose
    15, 16, # wrists
    23, 24, # hips
    27, 28, # ankles
]


# Only draw these full skeleton frames.
# Drawing all 15 skeletons makes the display too cluttered.
SKELETON_FRAMES = [0, 4, 9, 14]


# ---------------------------------------------------------------------------
# PredictionReceiver
# ---------------------------------------------------------------------------
class PredictionReceiver:
    """
    Wraps a ZMQ SUB socket.

    update() drains all currently queued predictions and keeps only the
    newest one so the visualizer stays close to real time.
    """

    def __init__(self, endpoint: str):
        self.context = zmq.Context()
        self.socket  = self.context.socket(zmq.SUB)

        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 2)

        self.socket.connect(endpoint)

        self.latest: np.ndarray | None = None
        self.frames_received = 0

    def update(self) -> bool:
        """
        Drain queued predictions and keep only the newest.

        Returns True if at least one new valid prediction was received.
        """
        newest = None
        received_any = False

        while True:
            try:
                msg = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if len(msg) != MSG_SIZE:
                print(
                    f"[Visualizer] Unexpected message size: "
                    f"{len(msg)} (expected {MSG_SIZE})"
                )
                continue

            newest = np.frombuffer(
                msg,
                dtype=np.float32
            ).reshape(
                K,
                N_LANDMARKS,
                N_DIMS
            ).copy()

            self.frames_received += 1
            received_any = True

        if newest is not None:
            self.latest = newest

        return received_any

    def close(self):
        self.socket.close()
        self.context.term()


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
class Visualizer:
    """
    Visualizes the predicted future pose trajectory.

    Intermediate future poses are faint.
    The final predicted pose is emphasized.
    """

    def __init__(self, receiver: PredictionReceiver):
        self.receiver = receiver

        self.fig = plt.figure(figsize=(10, 9))
        self.ax  = self.fig.add_subplot(111, projection="3d")

        self.anim = None

        self._configure_axes()

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    def _configure_axes(self):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_title("Predicted Future Trajectory")

        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)

        try:
            self.ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    def _set_equal_axes(self, points: np.ndarray):
        """
        Auto-scale around all predicted points while maintaining equal
        physical scale on X, Y and Z.
        """
        if len(points) == 0:
            return

        mins = points.min(axis=0)
        maxs = points.max(axis=0)

        center = (mins + maxs) / 2.0
        ranges = maxs - mins

        max_range = float(np.max(ranges))

        # Avoid extreme zoom if the prediction nearly collapses.
        max_range = max(max_range, 0.25)

        margin = 0.15 * max_range
        half   = max_range / 2.0 + margin

        self.ax.set_xlim(
            center[0] - half,
            center[0] + half
        )

        self.ax.set_ylim(
            center[1] - half,
            center[1] + half
        )

        self.ax.set_zlim(
            center[2] - half,
            center[2] + half
        )

        try:
            self.ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Skeleton drawing
    # ------------------------------------------------------------------
    def _draw_skeleton(
        self,
        pose: np.ndarray,
        alpha: float,
        linewidth: float,
        point_size: float,
        color,
    ):
        """
        Draw one [33, 3] MediaPipe-style pose.
        """
        self.ax.scatter(
            pose[:, 0],
            pose[:, 1],
            pose[:, 2],
            s=point_size,
            alpha=alpha,
            color=color,
            depthshade=True,
        )

        for a, b in POSE_CONNECTIONS:
            self.ax.plot(
                [pose[a, 0], pose[b, 0]],
                [pose[a, 1], pose[b, 1]],
                [pose[a, 2], pose[b, 2]],
                alpha=alpha,
                linewidth=linewidth,
                color=color,
            )

    # ------------------------------------------------------------------
    # Joint trajectories
    # ------------------------------------------------------------------
    def _draw_joint_trajectories(self, prediction: np.ndarray):
        """
        Draw the path of selected joints across all K predicted frames.
        """
        for joint_idx in TRAJECTORY_JOINTS:
            traj = prediction[:, joint_idx, :]

            self.ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                linewidth=1.3,
                alpha=0.55,
            )

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------
    def _update_frame(self, frame_idx):
        self.receiver.update()

        if self.receiver.latest is None:
            return ()

        prediction = self.receiver.latest

        # Ignore invalid numeric points when computing display range.
        all_points = prediction.reshape(-1, 3)

        finite_mask = np.all(
            np.isfinite(all_points),
            axis=1
        )

        valid_points = all_points[finite_mask]

        if len(valid_points) == 0:
            return ()

        # Redraw the scene from scratch.
        self.ax.cla()
        self._configure_axes()

        # --------------------------------------------------------------
        # Draw selected intermediate skeletons
        # --------------------------------------------------------------
        for frame_id in SKELETON_FRAMES[:-1]:

            progress = frame_id / (K - 1)

            alpha = 0.10 + progress * 0.20

            self._draw_skeleton(
                prediction[frame_id],
                alpha=alpha,
                linewidth=0.8,
                point_size=8,
                color="gray",
            )

        # --------------------------------------------------------------
        # Draw trajectories for important joints
        # --------------------------------------------------------------
        self._draw_joint_trajectories(prediction)

        # --------------------------------------------------------------
        # Draw final pose prominently
        # --------------------------------------------------------------
        final_pose = prediction[-1]

        self._draw_skeleton(
            final_pose,
            alpha=1.0,
            linewidth=2.5,
            point_size=32,
            color="tab:red",
        )

        # --------------------------------------------------------------
        # Auto-scale
        # --------------------------------------------------------------
        self._set_equal_axes(valid_points)

        # --------------------------------------------------------------
        # Display useful prediction statistics
        # --------------------------------------------------------------
        prediction_min = float(np.nanmin(prediction))
        prediction_max = float(np.nanmax(prediction))
        prediction_mean = float(np.nanmean(prediction))

        self.ax.set_title(
            "Predicted Future Trajectory\n"
            f"Predictions received: {self.receiver.frames_received} | "
            f"range: [{prediction_min:.3f}, {prediction_max:.3f}] | "
            f"mean: {prediction_mean:.3f}"
        )

        return ()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, interval_ms: int = 33):
        """
        Start the visualization loop.

        33 ms ≈ 30 FPS.
        """
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Text-only mode
# ---------------------------------------------------------------------------
def run_text_only(receiver: PredictionReceiver) -> None:
    """
    Print prediction statistics without opening a window.
    """
    print("[Visualizer] Text-only mode. Ctrl+C to stop.")

    try:
        while True:

            new_pred = receiver.update()

            if new_pred and receiver.latest is not None:

                latest = receiver.latest

                print(
                    f"[Visualizer] "
                    f"shape={latest.shape}  "
                    f"mean={latest.mean():.4f}  "
                    f"min={latest.min():.4f}  "
                    f"max={latest.max():.4f}  "
                    f"std={latest.std():.4f}  "
                    f"frames={receiver.frames_received}"
                )

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
        default="tcp://localhost:5557",
        help="ZMQ endpoint of the C++ PUB socket.",
    )

    p.add_argument(
        "--text_only",
        action="store_true",
        help=(
            "Print prediction statistics instead of plotting. "
            "Useful when running headless."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:

    args = parse_args()

    receiver = PredictionReceiver(
        args.endpoint
    )

    print(
        f"[Visualizer] Subscribed to "
        f"{args.endpoint}"
    )

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