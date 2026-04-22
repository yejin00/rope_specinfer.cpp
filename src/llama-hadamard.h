#pragma once

#include "llama.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_tensor;
struct llama_model;
struct llama_cparams;

struct llama_hadamard_tensors {
    bool enabled = false;

    uint32_t seed = 0;
    enum llama_hadamard_granularity granularity = LLAMA_HADAMARD_GRANULARITY_HEAD;
    int32_t head_dim = 0;

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;

    ggml_tensor * matrix = nullptr;
    std::vector<ggml_tensor *> q_signs;
    std::vector<ggml_tensor *> k_signs;
};

std::unique_ptr<llama_hadamard_tensors> llama_hadamard_init(
        const llama_model   & model,
        const llama_cparams & cparams);
