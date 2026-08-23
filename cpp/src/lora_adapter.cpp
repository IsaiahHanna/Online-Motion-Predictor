// lora_adapter.cpp
// Online Human Intent Predictor with Adaptive Learning

#include "lora_adapter.h"
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
// Constructor
//
// Walk module_.named_parameters() and collect every parameter whose name
// contains "adapter". Set requires_grad=true on those, false on everything
// else. Build an Adam optimizer over the adapter params only.
//
// Note: p.value is a live reference — it IS the parameter tensor in the
// module, not a copy. This is what makes the optimizer update actually
// change the model's weights.
// ---------------------------------------------------------------------------
LoRAAdapterManager::LoRAAdapterManager(
    torch::jit::script::Module& module,
    float lr)
    : lr_(lr)
{
    for (const auto& p : module.named_parameters()) {
        bool is_adapter = p.name.find("adapter") != std::string::npos;
        p.value.set_requires_grad(is_adapter);
        if (is_adapter) adapter_params_.push_back(p.value);
    }

    assert (adapter_params_.size() > 0);
    std::cout << "Number of adapters: " << adapter_params_.size() << "\n";

    auto options = torch::optim::AdamOptions(lr);
    optimizer_ = std::make_unique<torch::optim::Adam>(adapter_params_, options);
}

// ---------------------------------------------------------------------------
// update()
//
// One online gradient step on the adapter parameters.
//
// pred must carry a live autograd graph — it must come from
// predictor.forward_for_update(), not predictor.infer().
// The base model weights have requires_grad=false so backward()
// stops there; only adapter gradients accumulate.
//
// Gradient clipping prevents a single bad frame from
// destroying the adapter. 1.0 is a reasonable threshold.
//
// torch::nn::utils::clip_grad_norm_(adapter_params_, 1.0);
// ---------------------------------------------------------------------------
void LoRAAdapterManager::update(
    const torch::Tensor& pred,
    const torch::Tensor& target)
{
    optimizer_->zero_grad();
    auto loss = torch::mse_loss(pred, target);
    loss.backward();
    torch::nn::utils::clip_grad_norm_(adapter_params_, 1.0);
    optimizer_->step();
}

// ---------------------------------------------------------------------------
// reset()
//
// Zero all adapter parameters in-place and re-construct the optimizer.
//
// WHY in-place (same as HeadAdapter):
// If you reassign a tensor, the optimizer's internal reference points to
// the OLD storage. In-place zeroing preserves the reference.
//
// WHY re-construct optimizer:
// Adam's moment buffers (m, v) encode the previous subject's gradient
// history. Re-constructing clears them.
// ---------------------------------------------------------------------------
void LoRAAdapterManager::reset()
{
    for (const auto& p: adapter_params_) {
        {torch::NoGradGuard g; p.zero_();}
    }
    optimizer_ = std::make_unique<torch::optim::Adam>(adapter_params_, torch::optim::AdamOptions(lr_));
}

// ---------------------------------------------------------------------------
// weight_norm()
//
// Frobenius norm of all adapter parameters concatenated.
// Used for logging and validation (norm should rise from 0,
// then plateau as the adapter converges).
// ---------------------------------------------------------------------------
float LoRAAdapterManager::weight_norm() const
{
    float sq_sum = 0.0;
    for (const auto& p : adapter_params_){
        sq_sum = sq_sum + p.norm().pow(2).item<float>();
    }
    return sqrt(sq_sum);

}

// ---------------------------------------------------------------------------
// param_count()
// ---------------------------------------------------------------------------
int64_t LoRAAdapterManager::param_count() const
{
    int64_t param_count = 0;
    for (const auto& p : adapter_params_){
        param_count = param_count + p.numel();
    }
    return param_count;
}