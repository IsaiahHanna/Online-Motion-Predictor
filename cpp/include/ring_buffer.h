// ring_buffer.h
// Online Human Intent Predictor with Adaptive Learning
//
// Declaration of RingBuffer — a thread-safe circular buffer that holds
// the last W=30 frames of pose keypoints for the inference window.

#pragma once

#include <array>
#include <mutex>
#include <condition_variable>
#include <atomic>

// ---------------------------------------------------------------------------
// Constants
// Buffer capacity must be strictly greater than W so a full window always
// fits even as new frames are being written concurrently.
// ---------------------------------------------------------------------------

constexpr int W   = 30;   // inference window size (frames)
constexpr int K   = 15;   // prediction horizon
constexpr int D   = 99;   // keypoint dimensions (33 joints × 3 coords)
constexpr int BUF = 64;   // ring buffer capacity — must be > W

using Frame  = std::array<float, D>;
using Window = std::array<Frame, W>;

// ---------------------------------------------------------------------------
// RingBuffer
// ---------------------------------------------------------------------------

class RingBuffer
{
public:
    void push(const Frame& frame);
    uint64_t get_window(Window& out, uint64_t after_seq, const std::atomic<bool>& running);

private:
    uint64_t head_ = 0;       // total frames written (monotonic)
    std::array<Frame, BUF> buf_;
    std::mutex mutex_;
    std::condition_variable cv_;
};