# Online Human Intent Predictor with Adaptive Learning

Real-time human motion prediction from pose estimates using a frozen temporal transformer with online LoRA-style adapter fine-tuning.

---

## Overview

This is a real-time inference system for predicting short-horizon human body pose from live skeletal observations. The system is designed around two core ideas:

1. **Offline-trained base model**: A temporal transformer trained on large-scale motion capture data provides a strong prior over human motion dynamics.
2. **Online adaptation**: Lightweight LoRA-style adapter modules update continuously from live observations, enabling person-specific prediction without modifying frozen base weights.

The pipeline is built in C++ using libtorch and TorchScript for inference, with ZMQ-based interprocess communication and a thread-safe ring buffer for real-time skeletal data ingestion from MediaPipe.

---

## Architecture
MediaPipe Pose (Python)
│
│  ZMQ (push/pull)
▼
Ring Buffer (C++)
│
▼
TorchScript Transformer (frozen)
│
▼
LoRA Adapters (online update)
│
▼
Predicted Pose (next N frames)

--- 

**Base model**: Temporal transformer trained on AMASS motion capture sequences. Takes a fixed-length window of past pose observations and predicts future joint positions.

**Online adapters**: LoRA-style low-rank adapter modules (0.4% of total parameters) inserted into the frozen base model. Updated continuously from live pose observations to adapt to the current subject's motion patterns without catastrophic forgetting of the base prior.

**Inference pipeline**: C++ process using libtorch to load a frozen TorchScript model. A thread-safe ring buffer stores incoming skeletal frames. ZMQ handles IPC between the Python MediaPipe process and the C++ inference engine.

---

## Base Model Training

The temporal transformer was trained offline on the [AMASS](https://amass.is.tue.mpg.de/) motion capture dataset.

| Metric | Value |
|--------|-------|
| ADE (Average Displacement Error) | 47mm |
| FDE (Final Displacement Error) | 78mm |

Evaluated on held-out AMASS test sequences not seen during training.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose estimation | MediaPipe Pose |
| IPC | ZMQ (push/pull) |
| Inference engine | C++, libtorch, TorchScript |
| Base model training | Python, PyTorch |
| Online adaptation | LoRA-style adapters (custom) |
| Build system | CMake |
| Platform | Ubuntu Linux (GCP) |

---

## Project Status

| Component | Status |
|-----------|--------|
| AMASS data pipeline | Complete |
| Temporal transformer training | Complete |
| TorchScript export | Complete |
| C++ libtorch inference engine | Complete |
| ZMQ IPC layer | Complete |
| Ring buffer implementation | Complete |
| Online LoRA adapter modules | In progress |
| MediaPipe integration | In progress |
| End-to-end pipeline | In progress |

---

## Background

Human motion prediction is a core capability for safe human-robot interaction. Offline models trained on large datasets generalize well across motion types but cannot adapt to individual users in real-time. This project addresses this by keeping the base model frozen and learning lightweight person-specific residuals online — similar in spirit to test-time adaptation but operating continuously at inference time.

The system targets robotics and HRI applications where predicting a person's next movement 300–500ms ahead enables safer, more natural robot responses.

---

## References

- [AMASS: Archive of Motion Capture as Surface Shapes](https://amass.is.tue.mpg.de/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [libtorch C++ API](https://pytorch.org/cppdocs/)
