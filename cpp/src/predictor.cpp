// predictor.cpp
// Online Human Intent Predictor with Adaptive Learning

#include "predictor.h"
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
//
// Load the TorchScript module, move it to device, set eval mode.
// ---------------------------------------------------------------------------
Predictor::Predictor(const std::string& model_path, torch::Device device)
    : device_(device)
{
    try {
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        module_.eval();
    }
    catch (const c10::Error & e) {
        throw std::runtime_error("Module failed to load from: " + model_path + "\n" + e.what());
    }
}

// ---------------------------------------------------------------------------
// infer()
//
// Fast path. Use torch::NoGradGuard to prevent autograd from building a
// computation graph — without it, memory grows each call and latency rises.
//
// Box x into a std::vector<torch::jit::IValue>, call module_.forward(),
// then unbox the result with .toTensor().
// ---------------------------------------------------------------------------
torch::Tensor Predictor::infer(const torch::Tensor& x)
{
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {x};
    return module_.forward(inputs).toTensor(); // If model returns a tuple, use .toTuple()->elements()[0].toTensor() 
}

torch::Tensor Predictor::infer_with_hidden(const torch::Tensor& x)
{
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {x};
    
    auto method = module_.get_method("forward_with_hidden");
    auto result = method(inputs);
    auto tuple  = result.toTuple();

    const auto& elements = tuple->elements();
    if (elements.size != 2) {
        throw std::runtime_error(
            "forward_with_hidden expected 2 inputs"
        );
    }

    torch::Tensor prediction = elements[0].toTensor();
    torch::Tensor hidden     = elements[1].toTensor();

    return {
        hidden,
        prediction
    };

}

// ---------------------------------------------------------------------------
// forward_for_update()
//
// Gradient-enabled path for the adaptation loop.
//
// The call structure is identical to infer(); only the grad context differs.
// Keeping them as separate named functions makes it impossible to
// accidentally use the wrong one.
// ---------------------------------------------------------------------------
torch::Tensor Predictor::forward_for_update(const torch::Tensor& x)
{
    std::vector<torch::jit::IValue> inputs = {x};
    return module_.forward(inputs).toTensor();
}

// ---------------------------------------------------------------------------
// named_parameters()
//
// Returns the live parameter references from the loaded module.
// Use these to partition adapter params from base params, and to build
// the optimizer's parameter list.
// ---------------------------------------------------------------------------
std::vector<std::pair<std::string, torch::Tensor>> Predictor::named_parameters()
{
    std::vector<std::pair<std::string, torch::Tensor>> params;
    for (const auto& param:module_.named_parameters()) {
        params.push_back({param.name, param.value});
    }
    return params;
}