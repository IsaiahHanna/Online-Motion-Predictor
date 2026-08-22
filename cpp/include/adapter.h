// adapter.h
// Online Human Intent Predictor with Adaptive Learning
//
// HeadAdapter
// ---------------------
// A single residual linear layer added to the output of the frozen prediction
// head. The simplest possible adapter — its purpose is to prove the online
// adaptation loop (predict -> delayed label -> backward -> update) works in
// C++ with libtorch before the adapter architecture becomes a variable.

#pragma once

#include <torch/torch.h>
#include <memory>

class HeadAdapter
{
public:

    // Construct on `device`. hidden: transformer hidden dim (128).
    // out_dim: K * 99 = 15 * 99 = 1485.
    HeadAdapter(int hidden, int out_dim, torch::Device device);

    // Compute the residual contribution: hidden_vec @ w_
    // hidden: [1, hidden_dim] — the last-timestep encoder output
    // Returns: [1, out_dim] — added to the base head output
    torch::Tensor forward(const torch::Tensor& hidden);

    // One online update step.
    // pred:   combined output (base + adapter residual), [1, K, 99]
    // target: ground truth future poses,                 [1, K, 99]
    void update(const torch::Tensor& pred, const torch::Tensor& target);

    // Reset adapter to identity (zero residual).
    // Use this when switching to a new subject.
    void reset();

    // Return the adapter weight norm for logging
    float weight_norm() const;

private:

    torch::Tensor                        w_;          // leaf tensor, on device
    std::unique_ptr<torch::optim::Adam>  optimizer_;
    torch::Device                        device_;
};