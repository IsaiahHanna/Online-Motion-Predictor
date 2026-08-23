// main.cpp
// Online Human Intent Predictor with Adaptive Learning
//
// Pipeline:
//   [Python MediaPipe] --ZMQ PUSH--> PoseReceiver thread
//                                         |
//                                    RingBuffer
//                                         |
//                               [Inference thread] (main)
//                                    Predictor::infer()
//                                    LoRAAdapter::forward()
//                                    trajectory scorer
//                                         |
//                                  ZMQ PUB out --> Python visualizer

#include "ring_buffer.h"
#include "pose_receiver.h"
#include "predictor.h"
#include "lora_adapter.h"

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <zmq.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <adapted_model.pt>  <base_model.pt> [endpoint]\n";
        std::cerr << "  endpoint default: tcp://localhost:5555\n";
        return 1;
    }

    const std::string adapted_model_path = argv[1];
    const std::string base_model_path    = argv[2];
    const std::string endpoint   =
        (argc >= 4) ? argv[3] : "tcp://localhost:5555";

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
    RingBuffer         buf;
    PoseReceiver       poseReceiver(buf, endpoint, running);
    Predictor          predictor(adapted_model_path, device);
    LoRAAdapterManager loraManager(predictor.module(), 1e-4f);

    // For logging performance comparison
    Predictor          base_predictor(base_model_path, device);

    // ------------------------------------------------------------------
    // 4. Warm-up
    // ------------------------------------------------------------------
    for (int i = 0; i < 10; i++) {
        predictor.infer(
            torch::zeros({1, W, D}, device)
        );
    }

    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }

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
    // 6. Latency and Adaptation tracking
    // ------------------------------------------------------------------
    std::vector<double> frame_times_ms;
    frame_times_ms.reserve(10000);

    uint64_t last_seq     = 0;
    uint64_t frame_count  = 0;
    uint64_t update_count = 0;

    constexpr double ADAPT_SECONDS = 60.0;

    bool adaptation_started = false;
    bool adapter_frozen     = false;

    std::chrono::steady_clock::time_point adaptation_start;

    // MSE over the complete 60-second adaptation period
    double adapted_mse_sum = 0.0;
    double base_mse_sum    = 0.0;
    uint64_t mse_count     = 0;

    // MSE over the most recent 10 updates
    double recent_adapted_mse_sum = 0.0;
    double recent_base_mse_sum    = 0.0;
    uint64_t recent_mse_count     = 0;

    // Separate post-freeze statistics
    double post_adapted_mse_sum = 0.0;
    double post_base_mse_sum    = 0.0;
    uint64_t post_mse_count     = 0;

    // ------------------------------------------------------------------
    // Record initial base-model parameter norm.
    //
    // This lets us confirm at shutdown that the base model stayed frozen.
    // ------------------------------------------------------------------
    double base_norm_before = 0.0;

    {
        torch::NoGradGuard no_grad;

        for (const auto& item : base_predictor.named_parameters()) {
            base_norm_before +=
                item.second
                    .detach()
                    .pow(2)
                    .sum()
                    .item<double>();
        }

        base_norm_before =
            std::sqrt(base_norm_before);
    }

    std::cout
        << "Initial base model norm: "
        << base_norm_before
        << "\n";

    std::cout
        << "Initial adapter norm: "
        << loraManager.weight_norm()
        << "\n";

    // Delayed label buffer
    // The target for the prediction made at frame T is the
    // actual observed keypoints at frame T+K (0.5 s later).
    std::deque<
        std::pair<uint64_t, torch::Tensor>
    > pending_labels;

    // ------------------------------------------------------------------
    // 7. Main inference loop
    // ------------------------------------------------------------------
    std::cout << "Pipeline running. Ctrl+C to stop.\n";

    while (running.load())
    {
        auto t_frame_start =
            std::chrono::steady_clock::now();

        // 7a. Get window from ring buffer
        Window window;

        last_seq =
            buf.get_window(
                window,
                last_seq,
                running
            );

        if (!running.load()) {
            break;
        }

        // 7b. Build tensor and run base inference
        auto x =
            window_to_tensor(
                window,
                device
            );

        auto predictor_output =
            predictor.infer_with_hidden(x);



        auto final_pred = predictor_output.prediction;

        // 7c. Publish prediction
        auto pred_cpu =
            final_pred
                .to(torch::kCPU)
                .contiguous();

        pub_sock.send(
            zmq::buffer(
                pred_cpu.data_ptr<float>(),
                K * D * sizeof(float)
            ),
            zmq::send_flags::dontwait
        );

        // Store this frame's input for when its label matures.
        pending_labels.push_back({
            last_seq,
            x.detach().clone()
        });

        // Check if the oldest pending label has matured.
        while (
            !pending_labels.empty() &&
            last_seq >= pending_labels.front().first + K
        ) {
            auto [old_seq, old_x] =
                pending_labels.front();

            pending_labels.pop_front();

            // The current window is the exact K-frame future only when
            // this prediction matures at precisely old_seq + K.
            if (last_seq != old_seq + K) {
                continue;
            }

            auto window_tensor =
                window_to_tensor(
                    window,
                    device
                );

            auto target =
                window_tensor
                    .slice(1, W - K, W)
                    .contiguous();

            auto old_output =
                predictor.infer_with_hidden(old_x);

            auto base_eval =
                old_output.prediction.detach();

            // ----------------------------------------------------------
            // Measure base-model and adapted-model MSE BEFORE this update.
            // ----------------------------------------------------------
            double base_mse    = 0.0;
            double adapted_mse = 0.0;

            {
                torch::NoGradGuard no_grad;
                
                auto base_only = base_predictor.infer(old_x);
                base_mse    = torch::mse_loss(base_only.detach(),target).item<double>();
                adapted_mse = torch::mse_loss(old_output.prediction.detach(),target).item<double>();
            }

            // ----------------------------------------------------------
            // Start the 60-second timer only when the first real delayed
            // label becomes available.
            // ----------------------------------------------------------
            if (!adaptation_started) {
                adaptation_started = true;

                adaptation_start =
                    std::chrono::steady_clock::now();

                std::cout
                    << "[Adapt] Starting 60-second adaptation period.\n";
            }

            double elapsed_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    adaptation_start
                ).count();

            // ----------------------------------------------------------
            // Adapt for the first 60 seconds.
            // ----------------------------------------------------------
            if (!adapter_frozen)
            {
                base_mse_sum +=
                    base_mse;

                adapted_mse_sum +=
                    adapted_mse;

                ++mse_count;

                recent_base_mse_sum +=
                    base_mse;

                recent_adapted_mse_sum +=
                    adapted_mse;

                ++recent_mse_count;

                auto grad_output    = predictor.forward_for_update(old_x);
                auto pred_with_grad = grad_output.prediction;       // adapter grad flows automatically

                loraManager.update(
                    pred_with_grad,
                    target
                );

                ++update_count;

                // ------------------------------------------------------
                // Validation log every 10 updates.
                // ------------------------------------------------------
                if (update_count % 10 == 0)
                {
                    std::cout
                        << "Update "
                        << update_count

                        << "  adapter norm: "
                        << loraManager.weight_norm()

                        << "  recent base MSE: "
                        << recent_base_mse_sum /
                           recent_mse_count

                        << "  recent adapted MSE: "
                        << recent_adapted_mse_sum /
                           recent_mse_count

                        << "  elapsed: "
                        << elapsed_seconds
                        << " s\n";

                    recent_base_mse_sum =
                        0.0;

                    recent_adapted_mse_sum =
                        0.0;

                    recent_mse_count =
                        0;
                }

                // ------------------------------------------------------
                // After 60 seconds, stop updating the adapter.
                // ------------------------------------------------------
                if (elapsed_seconds >= ADAPT_SECONDS)
                {
                    adapter_frozen = true;

                    std::cout
                        << "\n"
                        << "60 SECOND ADAPTATION COMPLETE\n"

                        << "Adapter norm: "
                        << loraManager.weight_norm()
                        << "\n"

                        << "Training base MSE: "
                        << base_mse_sum / mse_count
                        << "\n"

                        << "Training adapted MSE: "
                        << adapted_mse_sum / mse_count
                        << "\n"

                        << "Adapter frozen.\n\n";
                }
            }

            // ----------------------------------------------------------
            // After 60 seconds, compare the frozen personalized adapter
            // with the unchanged base model on future samples.
            // ----------------------------------------------------------
            else
            {
                post_base_mse_sum +=
                    base_mse;

                post_adapted_mse_sum +=
                    adapted_mse;

                ++post_mse_count;

                if (post_mse_count % 30 == 0)
                {
                    std::cout
                        << "[Frozen comparison]"
                        << " samples="
                        << post_mse_count

                        << "  base MSE="
                        << post_base_mse_sum /
                           post_mse_count

                        << "  adapted MSE="
                        << post_adapted_mse_sum /
                           post_mse_count

                        << "\n";
                }
            }
        }

        // 7d. Latency bookkeeping
        if (device.is_cuda()) {
            torch::cuda::synchronize();
        }

        auto t_frame_end =
            std::chrono::steady_clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(
                t_frame_end -
                t_frame_start
            ).count();

        frame_times_ms.push_back(ms);

        ++frame_count;

        if (frame_count % 150 == 0) {
            std::cout
                << "Frames: "
                << frame_count
                << "  Last latency: "
                << ms
                << " ms\n"

                << "   LoRAAdapter weight norm: "
                << loraManager.weight_norm()
                << "\n";
        }
    }

    // ------------------------------------------------------------------
    // 8. Shutdown and stats
    // ------------------------------------------------------------------
    std::cout << "\nShutting down...\n";

    if (!frame_times_ms.empty()) {
        std::sort(
            frame_times_ms.begin(),
            frame_times_ms.end()
        );

        size_t n =
            frame_times_ms.size();

        double p50 =
            frame_times_ms[n * 0.50];

        double p99 =
            frame_times_ms[n * 0.99];

        std::cout
            << "Total frames : "
            << frame_count
            << "\n";

        std::cout
            << "Latency p50  : "
            << p50
            << " ms\n";

        std::cout
            << "Latency p99  : "
            << p99
            << " ms\n";
    }

    // ------------------------------------------------------------------
    // Verify that the frozen base model did not change.
    // ------------------------------------------------------------------
    double base_norm_after = 0.0;

    {
        torch::NoGradGuard no_grad;

        for (const auto& item : base_predictor.named_parameters()) {
            base_norm_after +=
                item.second
                    .detach()
                    .pow(2)
                    .sum()
                    .item<double>();
        }

        base_norm_after =
            std::sqrt(base_norm_after);
    }

    std::cout
        << "Base model norm before: "
        << base_norm_before
        << "\n";

    std::cout
        << "Base model norm after : "
        << base_norm_after
        << "\n";

    std::cout
        << "Base model norm delta : "
        << std::abs(
               base_norm_after -
               base_norm_before
           )
        << "\n";

    // ------------------------------------------------------------------
    // Final frozen-adapter comparison.
    // ------------------------------------------------------------------
    if (post_mse_count > 0)
    {
        double post_base_avg =
            post_base_mse_sum /
            post_mse_count;

        double post_adapted_avg =
            post_adapted_mse_sum /
            post_mse_count;

        std::cout
            << "Post-freeze samples     : "
            << post_mse_count
            << "\n";

        std::cout
            << "Post-freeze base MSE    : "
            << post_base_avg
            << "\n";

        std::cout
            << "Post-freeze adapted MSE : "
            << post_adapted_avg
            << "\n";

        if (post_base_avg > 0.0) {
            double improvement =
                100.0 *
                (
                    post_base_avg -
                    post_adapted_avg
                ) /
                post_base_avg;

            std::cout
                << "Post-freeze improvement: "
                << improvement
                << "%\n";
        }
    }

    return 0;
}