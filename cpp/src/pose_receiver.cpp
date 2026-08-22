// pose_receiver.cpp
// Online Human Intent Predictor with Adaptive Learning

#include "pose_receiver.h"
#include <zmq.hpp>
#include <cstring>
#include <iostream>

// ---------------------------------------------------------------------------
// Constructor
//
// Store references, then launch receive_loop on a background thread.
// ---------------------------------------------------------------------------
PoseReceiver::PoseReceiver(RingBuffer& buf,
                           const std::string& endpoint,
                           std::atomic<bool>& running)
    : buf_(buf), running_(running)
{
    // TODO: launch thread_ on receive_loop with std::ref(endpoint)
    //       remember: endpoint is a local — pass by value or ensure lifetime
    thread_ = std::thread(&PoseReceiver::receive_loop, this, endpoint);
}

// ---------------------------------------------------------------------------
// Destructor
//
// Join the thread before the object is destroyed.
// ---------------------------------------------------------------------------
PoseReceiver::~PoseReceiver()
{
    // TODO: join thread_
    thread_.join();
}

// ---------------------------------------------------------------------------
// receive_loop()
//
// Runs on thread_. Lifecycle:
//   1. Create a zmq::context_t (1 I/O thread is plenty)
//   2. Create a PULL socket, set RCVHWM and RCVTIMEO, then connect
//   3. Loop while running_.load():
//      a. recv() — returns empty optional on timeout, check flag and continue
//      b. Validate msg.size() == sizeof(FrameHeader) + D * sizeof(float)
//         If wrong size: log and skip (protocol violation / no person detected)
//      c. memcpy the header (first 12 bytes)
//      d. memcpy the keypoints (remaining 396 bytes) into a Frame
//      e. buf_.push(frame)
// ---------------------------------------------------------------------------
void PoseReceiver::receive_loop(const std::string& endpoint)
{
    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, zmq::socket_type::pull);
    sock.set(zmq::sockopt::rcvhwm,4);
    sock.set(zmq::sockopt::rcvtimeo, 100);
    sock.connect(endpoint);

    constexpr size_t EXPECTED_SIZE = sizeof(FrameHeader) + D * sizeof(float);
    Frame frame;
    uint32_t last_seq = 0;
    uint32_t frames_received = 0;

    std::cout << "[PoseReceiver] Connected to " << endpoint << "\n";

    while (running_.load())
    {
        zmq::message_t msg; 
        auto res = sock.recv(msg, zmq::recv_flage::none);
        
        if (!res){
            // Timeout — normal, just recheck the running_ flag.
            // Uncomment below if you want to see heartbeat logs:
            // std::cout << "[PoseReceiver] Waiting for frames...\n";
            continue;
        }

        if (msg.size() != EXPECTED_SIZE) {
            std::cerr << "[PoseReceiver] Unexpected message size"
                      << msg.size() << " bytes (expected " << EXPECTED_SIZE
                      << "). Skipping - no person detected or protocol mismatch.\n";
            continue;
        }

        FrameHeader hdr;
        std::memcpy(&hdr, msg.data(), sizeof(hdr));
        std::memcpy(frame.data(), static_cast<const char*>(msg.data()) + sizeof(hdr), D * sizeof(float));

        // Detect dropped frames
        if (frames_received > 0 && hdr.seq != last_seq + 1) {
            std::cerr << "[PoseReceiver] Dropped " << (hdr.seq - last_seq - 1)<< " frame(s) between seq " << last_seq<< " and " << hdr.seq << "\n"; 
        } 
        last_seq = hdr_seq;
        ++frames_received;

        // Periodic heartbeat log
        if (frames_received % 150 == 0) {
            std::cout << "[PoseReceiver] Received " << frames_received << " frames. Last_seq: " << hdr.seq << "\n"
        }
        buf_.push(frame);
    }
    std::cout << "[PoseReceiver] Shutting down. Total frames received: " << frames_received << "\n";
}