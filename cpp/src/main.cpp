// main.cpp
// Online Human Intent Predictor with Adaptive Learning
// Runtime Pipeline Test Version
//
// Pipeline:
//   [Python MediaPipe] --ZMQ PUSH--> PoseReceiver thread
//                                         |
//                                    RingBuffer
//                                         |
//                               [Inference thread] (main)
//                                    Predictor::infer()
//                                         |
//                                  ZMQ PUB out --> Python visualizer

#include "ring_buffer.h"
#include "pose_receiver.h"
#include "predictor.h"

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <zmq.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Shutdown flag
// ---------------------------------------------------------------------------
std::atomic<bool> running{true};

void handle_sigint(int) { running = false; }

// ---------------------------------------------------------------------------
// window_to_tensor()
// ---------------------------------------------------------------------------
torch::Tensor window_to_tensor(const Window& w, torch::Device device)
{
    return torch::from_blob(
        const_cast<float*>(w[0].data()), {1, W, D}, torch::kFloat32
    ).to(device);
}

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------
    // 0. Arguments
    // ------------------------------------------------------------------
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.pt> [endpoint]\n";
        std::cerr << "  endpoint default: tcp://localhost:5555\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string endpoint   = (argc >= 3) ? argv[2] : "tcp://localhost:5555";

    // ------------------------------------------------------------------
    // 1. SIGINT handler for clean shutdown 
    // ------------------------------------------------------------------
    std::signal(SIGINT, handle_sigint);

    // ------------------------------------------------------------------
    // 2. Device selection
    // ------------------------------------------------------------------
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA available. Using GPU.\n";
    } else {
        std::cout << "CUDA unavailable. Using CPU.\n";
    }

    // ------------------------------------------------------------------
    // 3. Construct pipeline components
    // ------------------------------------------------------------------
    RingBuffer   buf;
    PoseReceiver poseReceiver(buf, endpoint, running);
    Predictor    predictor(model_path, device);

    // ------------------------------------------------------------------
    // 4. Warm-up
    // ------------------------------------------------------------------
    for (int i = 0; i < 10; i++) {
        predictor.infer(torch::zeros({1, W, D}, device));
    }
    torch::cuda::synchronize();
    std::cout << "Warm-up complete.\n";

    // ------------------------------------------------------------------
    // 5. Output socket
    // ------------------------------------------------------------------
    zmq::context_t pub_ctx(1);
    zmq::socket_t  pub_sock(pub_ctx, zmq::socket_type::pub);
    pub_sock.set(zmq::sockopt::sndhwm, 4);
    pub_sock.bind("tcp://*:5556");
    std::cout << "Publishing predictions on tcp://*:5556\n";

    // ------------------------------------------------------------------
    // 6. Latency tracking
    // ------------------------------------------------------------------
    std::vector<double> frame_times_ms;
    frame_times_ms.reserve(10000);

    uint64_t last_seq    = 0;
    uint64_t frame_count = 0;

    // ------------------------------------------------------------------
    // 7. Main inference loop
    // ------------------------------------------------------------------
    std::cout << "Pipeline running. Ctrl+C to stop.\n";

    while (running.load())
    {
        auto t_frame_start = std::chrono::steady_clock::now();

        // 7a. Get window from ring buffer
        Window window;
        last_seq = buf.get_window(window, last_seq);

        // 7b. Build tensor and run inference
        auto x         = window_to_tensor(window, device);
        auto base_pred = predictor.infer(x);   // [1, 15, 99]

        // 7c. Publish prediction
        auto pred_cpu = base_pred.to(torch::kCPU).contiguous();
        pub_sock.send(
            zmq::buffer(pred_cpu.data_ptr<float>(), 15 * D * sizeof(float)),
            zmq::send_flags::dontwait
        );

        // 7d. Latency bookkeeping
        if (device.is_cuda()) torch::cuda::synchronize();
        auto t_frame_end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(
                        t_frame_end - t_frame_start).count();
        frame_times_ms.push_back(ms);
        ++frame_count;

        if (frame_count % 150 == 0) {
            std::cout << "Frames: " << frame_count
                      << "  Last latency: " << ms << " ms\n";
        }
    }

    // ------------------------------------------------------------------
    // 8. Shutdown and stats
    // ------------------------------------------------------------------
    std::cout << "\nShutting down...\n";

    if (!frame_times_ms.empty()) {
        std::sort(frame_times_ms.begin(), frame_times_ms.end());
        size_t n   = frame_times_ms.size();
        double p50 = frame_times_ms[n * 0.50];
        double p99 = frame_times_ms[n * 0.99];
        std::cout << "Total frames : " << frame_count << "\n";
        std::cout << "Latency p50  : " << p50 << " ms\n";
        std::cout << "Latency p99  : " << p99 << " ms\n";
    }

    return 0;
}