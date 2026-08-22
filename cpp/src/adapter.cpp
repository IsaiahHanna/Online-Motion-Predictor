// adapter.cpp
// Online Human Intent Predictor with Adaptive Learning

#include "adapter.h"

// ---------------------------------------------------------------------------
// Constructor
//
// Create w_ as a leaf tensor directly on `device` with requires_grad=true.
//
// Zero init gives the identity residual property: adapter output is exactly
// zero at t=0, so the combined prediction equals the base model prediction.
//
// Build the Adam optimizer after w_ is constructed and pass it a reference
// to w_ (not a copy). lr=1e-4 as specified.
// ---------------------------------------------------------------------------
HeadAdapter::HeadAdapter(int hidden, int out_dim, torch::Device device)
    : device_(device)
{
    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device)
                    .requires_grad(true);
    w_ = torch::zeros({hidden, out_dim}, opts);
    optimizer_ = std::make_unique<torch::optim::Adam> (std::vector<torch::Tensor>{w_}, torch::optim::AdamOptions(1e-4));
}

// ---------------------------------------------------------------------------
// forward()
//
// Compute the adapter's residual contribution via a matrix multiply.
// hidden: [1, H] — the encoder's last-timestep output (before the base head)
// Returns: [1, out_dim] — added to the base head's output in main.cpp
//
// This must be called with a live autograd graph during adaptation
// (i.e. NOT under NoGradGuard) so gradients can flow back to w_.
// During pure inference (no update) it can be called under NoGradGuard.
// ---------------------------------------------------------------------------
torch::Tensor HeadAdapter::forward(const torch::Tensor& hidden)
{
    return torch::matmul(hidden, w_);
}

// ---------------------------------------------------------------------------
// update()
//
// One online gradient step.
//
// pred and target must both be on the same device as w_.
// pred must carry an autograd graph — it must come from forward_for_update(),
// not from infer(). If pred was produced under NoGradGuard, loss.backward()
// will throw "element 0 does not require grad".
// ---------------------------------------------------------------------------
void HeadAdapter::update(const torch::Tensor& pred, const torch::Tensor& target)
{
    optimizer_->zero_grad();
    auto loss = torch::mse_loss(pred, target);
    loss.backward();
    torch::nn::utils::clip_grad_norm_({w_},1.0);
    optimizer_->step();
}

// ---------------------------------------------------------------------------
// reset()
//
// Zero w_ in-place, reverting the adapter to the identity residual.
// Use this when switching to a new subject.
// ---------------------------------------------------------------------------
void HeadAdapter::reset()
{
    torch::NoGradGuard g; 
    w_.zero_();
    optimizer_ = std::make_unique<torch::optim::Adam> (std::vector<torch::Tensor>{w_}, torch::optim::AdamOptions(1e-4));
}

// ---------------------------------------------------------------------------
// weight_norm()
//
// Returns ||w_||_F as a float for logging.
// ---------------------------------------------------------------------------
float HeadAdapter::weight_norm() const
{
    return w_.norm().item<float>();
}