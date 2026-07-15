// ring_buffer.cpp
// Online Human Intent Predictor with Adaptive Learning
//
// Implements a thread-safe circular buffer for incoming pose keypoint frames.
// The writer thread (ZMQ receiver) calls push() when a new frame arrives.
// The reader thread (inference loop) calls get_window() to retrieve the last
// W=30 frames as a contiguous window for model input.
//
// Key concepts to understand before implementing:
//   - A circular buffer wraps its write head back to index 0 when it reaches
//     the end of the backing array. Use modulo arithmetic: head = (head + 1) % BUF
//   - std::mutex guards buf_, write_head_, and count_ from concurrent access
//   - std::condition_variable lets get_window() block cheaply until enough
//     frames are available, rather than busy-polling
//   - The buffer must hold BUF >= W frames so a full window always fits

#include "ring_buffer.h"

// ---------------------------------------------------------------------------
// push()
//
// Called by the ZMQ receiver thread each time a new frame arrives.
// Writes the frame into buf_ at write_head_, advances write_head_ with
// wraparound, and increments count_ (capped at BUF).
//
// Thread safety: hold mtx_ for the entire write.
// After writing, call cv_.notify_one() so a waiting get_window() can wake up.
// ---------------------------------------------------------------------------
void RingBuffer::push(const Frame& frame)
{
    {
    std::lock_guard<std::mutex> lock(mutex_);
    buf_[head_ % BUF] = frame;          // copy 396 bytes
    ++head_;
    }
    cv_.notify_one();
}

// ---------------------------------------------------------------------------
// get_window()
//
// Called by the inference thread. Blocks until at least W frames are in the
// buffer AND head_ > after_seq (at least one new frame since last call).
// Copies the last W frames into `out` in chronological order (oldest first).
// Returns the sequence number of the newest frame in the window — pass this
// back as after_seq on the next call to prevent re-processing the same window.
//
// How to reconstruct chronological order from a circular buffer:
//   With a monotonic head_, the last W frames are logical indices
//   [head_ - W, head_). The physical slot for logical index i is i % BUF.
//   So the oldest frame sits at physical slot (head_ - W) % BUF, and you
//   iterate W steps forward, wrapping with % BUF each step.
//
// Thread safety: hold mtx_ while reading head_ and buf_.
// Use cv_.wait() with a lambda predicate so the lock is released while
// sleeping and automatically re-acquired before the predicate is re-checked.
// ---------------------------------------------------------------------------
uint64_t RingBuffer::get_window(Window& out, uint64_t after_seq)
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]{return head_ >= W && head_ > after_seq; });
    const uint64_t newest = head_;               // frames [newest-W, newest]

    for (int i = 0; i < W; ++i) {
        out[i] = buf_[(newest - W + i) % BUF]; 
    }
    return newest;
}