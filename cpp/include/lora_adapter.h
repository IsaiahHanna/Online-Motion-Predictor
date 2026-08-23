// lora_adapter.h
// Online Human Intent Predictor with Adaptive Learning
//
// Manages the LoRA adapter parameters that live INSIDE adapted_model.pt.
// Unlike HeadAdapter which held its own weight tensor, this class
// holds REFERENCES into the TorchScript module's parameter table — so
// updating them directly updates the live model.
//
// Why this is different from HeadAdapter:
//   HeadAdapter owned w_ as a standalone tensor outside the model.
//   LoRAAdapterManager holds pointers INTO the loaded TorchScript module.
//   This means: no separate forward() call needed — adapters run
//   automatically inside module_.forward(). The update path just needs
//   to call loss.backward() and optimizer_.step() on the adapter params.

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>

class LoRAAdapterManager
{
public:

    // Construct from a loaded TorchScript module.
    // Finds all parameters whose name contains "adapter", partitions them
    // as trainable, and builds an Adam optimizer over them.
    //
    // Throws std::runtime_error if no adapter parameters are found —
    // this means the wrong model file was loaded (base model instead of
    // adapted_model.pt).
    //
    // lr: Adam learning rate (1e-4)
    explicit LoRAAdapterManager(torch::jit::script::Module& module,
                                float lr = 1e-4f);

    // One online update step.
    // pred:   [1, K, 99] — output from module_.forward(), carries grad graph
    //         (must come from forward_for_update(), NOT infer())
    // target: [1, K, 99] — ground truth future frames
    //
    // Gradient clipping applied before step
    void update(const torch::Tensor& pred, const torch::Tensor& target);

    // Reset all adapter parameters to zero (identity residual).
    // Use when switching to a new subject.
    //
    // In-place zero_, NOT reassignment.
    // Also re-constructs optimizer_ to clear Adam moment buffers.
    // (Carrying m/v from the previous person contaminates early updates for the new person)
    void reset();

    // Returns ||adapter_params||_F for logging.
    float weight_norm() const;

    // Returns the number of adapter parameters.
    int64_t param_count() const;

    // Access adapter params directly (e.g. for saving/loading snapshots).
    const std::vector<torch::Tensor>& params() const { return adapter_params_; }

private:

    // Live references into the TorchScript module's parameter storage.
    // These are NOT copies — writing to them writes to the model.
    std::vector<torch::Tensor>             adapter_params_;

    std::unique_ptr<torch::optim::Adam>    optimizer_;

    float lr_;
};