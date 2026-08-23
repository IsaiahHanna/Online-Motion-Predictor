// predictor.h
// Online Human Intent Predictor with Adaptive Learning
//
// Wraps the TorchScript module and exposes three inference entry points:
//   infer()               — fast prediction-only path
//   infer_with_hidden()   — prediction + hidden-state path
//   forward_for_update()  — gradient-enabled path for adaptation

#pragma once

#include <string>
#include <torch/script.h>
#include <torch/torch.h>

struct PredictorOutput
{
    torch::Tensor prediction;
    torch::Tensor hidden;
};

class Predictor
{
public:

    // Load the TorchScript module from disk and move it to the given device.
    // Calls module_.eval() — dropout must be off for deterministic inference.
    // Throws std::runtime_error if the file cannot be loaded.
    explicit Predictor(const std::string& model_path, torch::Device device);

    // Fast inference path. No gradient graph is built.
    // x: [1, W, D] on the same device as the model.
    // Returns: [1, K, D]
    torch::Tensor infer(const torch::Tensor& x);

    PredictorOutput infer_with_hidden(const torch::Tensor& x);

    // Gradient-enabled forward pass for the online adaptation loop.
    // NOT wrapped in NoGradGuard — gradients must flow.
    // x: [1, W, D] on device.
    PredictorOutput forward_for_update(const torch::Tensor& x);

    // Access named parameters for the optimizer.
    // Returns references into the live module — updating them updates the model.
    // Use the "adapter" substring to partition trainable from frozen params.
    std::vector<std::pair<std::string, torch::Tensor>> named_parameters();

    // Public accessor for module
    torch::jit::script::Module& module() {return module_;}

private:

    torch::jit::script::Module module_;
    torch::Device device_;
};