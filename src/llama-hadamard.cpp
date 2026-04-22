#include "llama-hadamard.h"

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

static bool is_power_of_two(uint32_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static float hadamard_sign(uint32_t seed, uint32_t il, uint32_t head, uint32_t dim) {
    const uint64_t key =
            (uint64_t(seed) << 32) ^
            (uint64_t(il + 1) << 24) ^
            (uint64_t(head + 1) << 12) ^
            uint64_t(dim + 1);

    return (splitmix64(key) & 1ULL) ? 1.0f : -1.0f;
}

static uint32_t bit_parity(uint32_t x) {
    uint32_t parity = 0;
    while (x != 0) {
        parity ^= x & 1U;
        x >>= 1;
    }

    return parity;
}

static ggml_backend_buffer_type_t hadamard_select_buft(const llama_model & model) {
    const uint32_t n_layers = model.hparams.n_layer;

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (n_layers == 0) {
        return buft;
    }

    try {
        ggml_backend_buffer_type_t layer_buft = model.select_buft(0);
        bool same_buft_all_layers = true;

        for (uint32_t il = 1; il < n_layers; ++il) {
            if (model.select_buft(il) != layer_buft) {
                same_buft_all_layers = false;
                break;
            }
        }

        if (same_buft_all_layers) {
            buft = layer_buft;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_WARN("%s: failed to choose Hadamard backend buffer type (%s), using CPU\n",
                __func__, err.what());
    }

    return buft;
}

} // namespace

std::unique_ptr<llama_hadamard_tensors> llama_hadamard_init(
        const llama_model   & model,
        const llama_cparams & cparams) {
    if (!cparams.hadamard) {
        return nullptr;
    }

    const auto & hparams = model.hparams;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t head_dim = hparams.n_embd_head_k;

    if (!is_power_of_two(head_dim)) {
        throw std::runtime_error(format(
                "Hadamard rotation requires a power-of-two head dimension, got %u", head_dim));
    }

    auto hadamard = std::make_unique<llama_hadamard_tensors>();
    hadamard->enabled = true;
    hadamard->seed = cparams.hadamard_seed;
    hadamard->granularity = cparams.hadamard_granularity;
    hadamard->head_dim = (int32_t) head_dim;
    hadamard->q_signs.resize(n_layer);
    hadamard->k_signs.resize(n_layer);

    const size_t n_meta = size_t(1 + 2*n_layer + 8);
    hadamard->ctx.reset(ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * n_meta,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }));

    if (!hadamard->ctx) {
        throw std::runtime_error("failed to create Hadamard tensor context");
    }

    hadamard->matrix = ggml_new_tensor_2d(hadamard->ctx.get(), GGML_TYPE_F32, head_dim, head_dim);
    ggml_set_name(hadamard->matrix, "hadamard_matrix");

    for (uint32_t il = 0; il < n_layer; ++il) {
        hadamard->q_signs[il] = ggml_new_tensor_2d(
                hadamard->ctx.get(), GGML_TYPE_F32, head_dim, hparams.n_head(il));
        hadamard->k_signs[il] = ggml_new_tensor_2d(
                hadamard->ctx.get(), GGML_TYPE_F32, head_dim, hparams.n_head_kv(il));

        ggml_set_name(hadamard->q_signs[il], format("hadamard_q_signs_%u", il).c_str());
        ggml_set_name(hadamard->k_signs[il], format("hadamard_k_signs_%u", il).c_str());
    }

    ggml_backend_buffer_type_t buft = hadamard_select_buft(model);
    LLAMA_LOG_INFO("%s: enabled = true, seed = %u, granularity = %s, head_dim = %u\n",
            __func__,
            hadamard->seed,
            hadamard->granularity == LLAMA_HADAMARD_GRANULARITY_HEAD ? "head" : "layer",
            head_dim);
    LLAMA_LOG_INFO("%s: allocating Hadamard tensors on device: %s\n",
            __func__, ggml_backend_buft_name(buft));

    hadamard->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(hadamard->ctx.get(), buft));
    if (!hadamard->buf) {
        throw std::runtime_error("failed to allocate Hadamard tensors");
    }

    std::vector<float> matrix_data(size_t(head_dim) * head_dim);
    const float inv_sqrt = 1.0f / std::sqrt((float) head_dim);
    for (uint32_t row = 0; row < head_dim; ++row) {
        for (uint32_t col = 0; col < head_dim; ++col) {
            const uint32_t parity = bit_parity(row & col);
            matrix_data[row * head_dim + col] = parity ? -inv_sqrt : inv_sqrt;
        }
    }
    ggml_backend_tensor_set(hadamard->matrix, matrix_data.data(), 0, matrix_data.size() * sizeof(float));

    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t n_head_q  = hparams.n_head(il);
        const uint32_t n_head_kv = hparams.n_head_kv(il);

        std::vector<float> k_signs(size_t(head_dim) * n_head_kv);
        for (uint32_t hk = 0; hk < n_head_kv; ++hk) {
            const uint32_t pattern_head = cparams.hadamard_granularity == LLAMA_HADAMARD_GRANULARITY_HEAD ? hk : 0;
            for (uint32_t d = 0; d < head_dim; ++d) {
                k_signs[hk * head_dim + d] = hadamard_sign(hadamard->seed, il, pattern_head, d);
            }
        }

        std::vector<float> q_signs(size_t(head_dim) * n_head_q);
        for (uint32_t hq = 0; hq < n_head_q; ++hq) {
            const uint32_t hk = n_head_kv == 0 ? 0 : (hq * n_head_kv) / n_head_q;
            for (uint32_t d = 0; d < head_dim; ++d) {
                q_signs[hq * head_dim + d] = k_signs[hk * head_dim + d];
            }
        }

        ggml_backend_tensor_set(hadamard->q_signs[il], q_signs.data(), 0, q_signs.size() * sizeof(float));
        ggml_backend_tensor_set(hadamard->k_signs[il], k_signs.data(), 0, k_signs.size() * sizeof(float));

        LLAMA_LOG_INFO("%s: layer %3u prepared signs for Q heads=%u, K heads=%u\n",
                __func__, il, n_head_q, n_head_kv);
    }

    return hadamard;
}
