// test_inference.cpp — Phase 3 C++ Inference Verification
// Online Human Intent Predictor with Adaptive Learning
//
// Loads intent_model.pt and test_io.pt, runs a forward pass, and verifies
// the C++ output matches the Python-generated reference within 1e-4 tolerance.
//
// Usage:
//   ./test_inference <path/to/intent_model.pt> <path/to/test_io.pt>
//
// Expected output on success:
//   Model loaded.
//   Test I/O loaded.  x: [1, 30, 99]  y: [1, 15, 99]
//   Model moved to CUDA.
//   Inference complete.
//   max_diff : 0.0000e+00
//   mean_diff: 0.0000e+00
//   PASS — C++ output matches Python within 1e-4 tolerance.
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#include <torch/torch.h>     // torch::jit::load, torch::jit::script::Module
#include <torch/script.h>    // torch::Tensor, torch::alllose, etc.
#include <c10/util/Exception.h>



int main(int argc, char* argv[])
{
    // ----------------------------------------------------
    // 0. Argument Check
    // ----------------------------------------------------
    if (argc > 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <intent_model.pt> <test_io.pt>\n";
        return 1;
    }

    // ----------------------------------------------------
    // 1. Device Selection
    // ----------------------------------------------------
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()){
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA available - using GPU.\n"
    } else {
        std::cout << "CUDA unavailable - using CPU.\n"
    }

    // ----------------------------------------------------
    // 2. Load the TorchScript model
    // Declare model before the try block, so it stays in score below.
    // ----------------------------------------------------
    torch::jit::script::Module model;

    try {
        model = torch::jit::load(argv[1]);
        model.to(device);
        model.eval();
        std::cout << "Loaded the model.\n";
    }
    catch (const c10::Error& e){
        std::cerr << "Error loading in the model: " << e.what()  << "\n";
        return 1;
    }

    // ----------------------------------------------------
    // 3. Load test_io.pt
    //
    //    test_io.pt was saved from Python with:
    //        torch.save({"x": x_cpu, "y": y_cpu}, path)
    //
    //    In C++, torch.save() dicts are loaded via torch::pickle_load()
    //    after reading the raw bytes. The result is a c10::IValue
    //    containing a generic dict keyed by string IValues.
    //
    //    Declare x and y BEFORE the try block so they stay in scope.
    // ----------------------------------------------------
    torch::Tensor x,y;
    try {
        // Read the file into a byte buffer
        std::fstream file(argv[2], std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Cannot open test_io.pt: " << argv[2] << "\n";
            return 1;
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Failed to read test_io.pt\n";
            return 1;
        }

        // Deserialize the pickle dict
        c10::IValue loaded = torch::pickle_load(buffer);
        auto data_dict = loaded.toGenericDict();

        x = data_dict.at("x").toTensor();
        y = data_dict.at("y").toTensor();

        std::cout << "Test I/O loaded."
                  << "  x: " << x.sizes()
                  << "  y: " << y.sizes() << "\n";
    }
    catch (const c10::Error& e){
        std::cerr << "Error loading in test_io.pt:" << e.what() << "\n";
        return 1;
    }

    // ----------------------------------------------------
    // 4. Move tensors to the same device as the model
    // ----------------------------------------------------
    x = x.to(device);
    y = y.to(device);
    std::cout << "Tensors moved to " << device << "\n";

    // ----------------------------------------------------
    // 5. Run inference and time it
    // ----------------------------------------------------
    torch::Tensor output;
    try {
        // Synchronize before timing so GPU kernels from previous ops don't
        // bleed into the environment.
        if (device.is_cuda()) torch::cuda::synchronize();

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<torch::jit::IValue> inputs = { x };
        output = model.forward(inputs).toTensor();   // [1, 15, 99]

        if (device.is_cuda()) torch::cuda::synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = 
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        std::cout << "Inference complete. latency: " << elapsed_ms << " ms\n";
        
        if (elapsed_ms < 5.0) {
            std::cout << "PASS - latency " << elapsed_ms << " ms exceeds 5 ms target.\n";
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during the inference: " << e.what() << "\n";
        return 1;
    }

    // ----------------------------------------------------
    // 6. Verify output matches Python reference within 1e-4
    //
    //    .equal() does exact bitwise comparison — almost always false for
    //    float32 ops across platforms. Use torch::allclose() instead,
    //    which checks  |a - b| <= atol + rtol * |b|  elementwise.
    // ----------------------------------------------------
    torch::Tensor diff = (output - y).abs();
    float max_diff     = diff.max().item<float>();
    float mean_diff    = diff.mean().item<float>();

    std::cout << std::scientific;
    std::cout << "max_diff : " << max_diff << "\n";
    std::cout << "mean_diff : " << mean_diff << "\n";
    std::cout << std::defaultfloat;
    
    // Output shape check
    if (output.sizes() != y.sizes()) {
        std::cerr << "FAIL - output shape " << output.sizes() 
                  << " != expected " << y.sizes() << "\n";
        return 1;
    }

    // Tolerance check (spec: within 1e-4)
    bool pass = torch::allclose(output, y, 1e-4, 1e-4) // rtol and atol equal 1e-4

    if (pass) {
        std::cout << "PASS - C++ output matches Python within 1e-4 tolerance. \n";
        return 0;
    } else {
        std::cerr << "FAIL - max_diff " << max_diff << " exceeds 1e-4 tolerance.\n";
        return 1;
    }
}