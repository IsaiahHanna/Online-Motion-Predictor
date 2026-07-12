// pose_receiver.h
// Online Human Intent Predictor with Adaptive Learning
//
// PoseReceiver runs on its own thread. It connects to the Python MediaPipe
// PUSH socket, receives one frame per message, and pushes each frame into
// the shared RingBuffer.
//
// Wire format:
//   [uint32_t seq | double timestamp | float[99] keypoints]
//   = 4 + 8 + 396 = 408 bytes per message
//
// The seq number helps to detect dropped frames.
// The timestamp enables true end-to-end latency measurement later on.

#pragma once

#include "ring_buffer.h"
#include <atomic>
#include <string>
#include <thread>

// Packed header prepended to every ZMQ message.
// __attribute__((packed)) prevents compiler padding between the uint32
// and the double (which would silently make the struct 16 bytes instead
// of 12, shifting all the offsets).
struct FrameHeader
{
    uint32_t seq;   // monotonically increasing frame counter
    double   ts;    // capture timestamp (seconds since epoch)
} __attribute__((packed));

static_assert(sizeof(FrameHeader) == 12, "FrameHeader must be 12 bytes");

class PoseReceiver
{
public:

    // Bind the receiver to `endpoint` and start the background thread.
    // `running` is a shared flag — set it to false to stop the thread.
    PoseReceiver(RingBuffer& buf,
                 const std::string& endpoint,
                 std::atomic<bool>& running);

    // Joins the background thread. Call before destroying the object.
    ~PoseReceiver();

private:

    // The receive loop — runs on thread_.
    // Connects the ZMQ PULL socket, then loops:
    //   1. recv() with a timeout so it can check `running_`
    //   2. validate message size
    //   3. memcpy header and keypoints out of the message
    //   4. buf_.push(frame)
    void receive_loop(const std::string& endpoint);

    RingBuffer&        buf_;
    std::atomic<bool>& running_;
    std::thread        thread_;
};