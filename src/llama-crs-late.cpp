/**
 * Late Dimension CRS (Channel-wise Row Scaling) Implementation
 *
 * Loads calibration scales for post-RoPE outlier channels and uploads
 * sparse per-head index/scale tensors consumed by GGML_OP_CRS_SPARSE_MUL.
 */

#include "llama-model.h"
#include "llama-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#define CRS_LATE_MAGIC 0x53435253  // "SCRS"

struct crs_late_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_dims;
    uint32_t top_k;
};

struct crs_late_head_data {
    std::vector<int32_t> indices;
    std::vector<float> scales;
};

void llama_model::load_crs_late_scales(const std::string & scales_path) {
    if (scales_path.empty()) {
        LLAMA_LOG_INFO("%s: no late CRS scales file specified, skipping\n", __func__);
        return;
    }

    LLAMA_LOG_INFO("%s: loading late CRS scales from '%s'\n", __func__, scales_path.c_str());

    std::ifstream file(scales_path, std::ios::binary);
    if (!file) {
        LLAMA_LOG_WARN("%s: failed to open CRS scales file '%s', skipping late CRS\n", __func__, scales_path.c_str());
        return;
    }

    crs_late_file_header header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));

    if (!file || header.magic != CRS_LATE_MAGIC) {
        LLAMA_LOG_ERROR("%s: invalid CRS scales file '%s'\n", __func__, scales_path.c_str());
        return;
    }

    const uint32_t n_layers = header.n_layers;
    const uint32_t n_heads  = header.n_heads;
    const uint32_t n_dims   = header.n_dims;
    const uint32_t top_k    = header.top_k;

    LLAMA_LOG_INFO("%s: CRS scales: layers=%u, heads=%u, dims=%u, top_k=%u\n",
            __func__, n_layers, n_heads, n_dims, top_k);

    std::vector<std::vector<crs_late_head_data>> layer_data(n_layers);

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        layer_data[layer].resize(n_heads);
        for (uint32_t head = 0; head < n_heads; ++head) {
            auto & hd = layer_data[layer][head];
            hd.indices.resize(top_k);
            hd.scales.resize(top_k);
            file.read(reinterpret_cast<char *>(hd.indices.data()), top_k * sizeof(int32_t));
            file.read(reinterpret_cast<char *>(hd.scales.data()),  top_k * sizeof(float));
        }
    }

    if (!file) {
        LLAMA_LOG_ERROR("%s: failed while reading late CRS file '%s'\n", __func__, scales_path.c_str());
        return;
    }

    crs_late_q_indices.resize(n_layers, nullptr);
    crs_late_q_scales.resize(n_layers, nullptr);
    crs_late_k_indices.resize(n_layers, nullptr);
    crs_late_k_scales.resize(n_layers, nullptr);

    crs_late_q_indices_data.resize(n_layers);
    crs_late_q_scales_data.resize(n_layers);
    crs_late_k_indices_data.resize(n_layers);
    crs_late_k_scales_data.resize(n_layers);

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        const uint32_t model_n_head    = hparams.n_head(layer);
        const uint32_t model_n_head_kv = hparams.n_head_kv(layer);

        if (layer == 0 && n_heads != model_n_head) {
            LLAMA_LOG_WARN("%s: scale file heads (%u) != model heads (%u) for layer 0, remapping heads\n",
                    __func__, n_heads, model_n_head);
        }

        auto & q_idx_layer = crs_late_q_indices_data[layer];
        auto & q_scl_layer = crs_late_q_scales_data[layer];
        auto & k_idx_layer = crs_late_k_indices_data[layer];
        auto & k_scl_layer = crs_late_k_scales_data[layer];

        q_idx_layer.assign((size_t) top_k * model_n_head, -1);
        q_scl_layer.assign((size_t) top_k * model_n_head, 1.0f);
        k_idx_layer.assign((size_t) top_k * model_n_head_kv, -1);
        k_scl_layer.assign((size_t) top_k * model_n_head_kv, 1.0f);

        for (uint32_t h = 0; h < model_n_head; ++h) {
            const uint32_t file_h = (h * n_heads) / model_n_head;
            const auto & head_data = layer_data[layer][file_h];

            for (uint32_t i = 0; i < top_k; ++i) {
                const int32_t idx = head_data.indices[i];
                const float scale = head_data.scales[i];
                if (idx >= 0 && idx < (int32_t) n_dims && scale > 0.0f) {
                    q_idx_layer[h * top_k + i] = idx;
                    q_scl_layer[h * top_k + i] = scale;
                }
            }
        }

        for (uint32_t h = 0; h < model_n_head_kv; ++h) {
            const uint32_t file_h = (h * n_heads) / model_n_head_kv;
            const auto & head_data = layer_data[layer][file_h];

            for (uint32_t i = 0; i < top_k; ++i) {
                const int32_t idx = head_data.indices[i];
                const float scale = head_data.scales[i];
                if (idx >= 0 && idx < (int32_t) n_dims && scale > 1e-6f) {
                    k_idx_layer[h * top_k + i] = idx;
                    k_scl_layer[h * top_k + i] = 1.0f / scale;
                }
            }
        }
    }

    const size_t n_tensor_meta = (size_t) n_layers * 4 + 8;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensor_meta,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx_crs_late = ggml_init(params);
    if (!ctx_crs_late) {
        LLAMA_LOG_ERROR("%s: failed to create ggml context for late CRS tensors\n", __func__);
        return;
    }

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        const uint32_t model_n_head    = hparams.n_head(layer);
        const uint32_t model_n_head_kv = hparams.n_head_kv(layer);

        crs_late_q_indices[layer] = ggml_new_tensor_2d(ctx_crs_late, GGML_TYPE_I32, top_k, model_n_head);
        crs_late_q_scales[layer]  = ggml_new_tensor_2d(ctx_crs_late, GGML_TYPE_F32, top_k, model_n_head);
        crs_late_k_indices[layer] = ggml_new_tensor_2d(ctx_crs_late, GGML_TYPE_I32, top_k, model_n_head_kv);
        crs_late_k_scales[layer]  = ggml_new_tensor_2d(ctx_crs_late, GGML_TYPE_F32, top_k, model_n_head_kv);

        ggml_set_name(crs_late_q_indices[layer], ("crs_late_q_indices_" + std::to_string(layer)).c_str());
        ggml_set_name(crs_late_q_scales[layer],  ("crs_late_q_scales_"  + std::to_string(layer)).c_str());
        ggml_set_name(crs_late_k_indices[layer], ("crs_late_k_indices_" + std::to_string(layer)).c_str());
        ggml_set_name(crs_late_k_scales[layer],  ("crs_late_k_scales_"  + std::to_string(layer)).c_str());
    }

    // Try to place late CRS tensors on the same backend buffer type used by the
    // repeating layers. This avoids backend scheduler split-input explosions on
    // OpenCL/mobile builds, where CPU-resident sparse index/scale tensors would
    // otherwise be copied into many GPU splits. If the model uses mixed buffer
    // types across layers, fall back to CPU for safety.
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (n_layers > 0) {
        try {
            ggml_backend_buffer_type_t layer_buft = select_buft(0);
            bool same_buft_all_layers = true;
            for (uint32_t layer = 1; layer < n_layers; ++layer) {
                if (select_buft(layer) != layer_buft) {
                    same_buft_all_layers = false;
                    break;
                }
            }
            if (same_buft_all_layers) {
                buft = layer_buft;
            }
        } catch (const std::exception & err) {
            LLAMA_LOG_WARN("%s: failed to choose late CRS backend buffer type (%s), using CPU\n",
                    __func__, err.what());
        }
    }
    LLAMA_LOG_INFO("%s: allocating late CRS tensors on device: %s\n", __func__, ggml_backend_buft_name(buft));

    buf_crs_late = ggml_backend_alloc_ctx_tensors_from_buft(ctx_crs_late, buft);
    if (!buf_crs_late) {
        LLAMA_LOG_ERROR("%s: failed to allocate buffer for late CRS tensors\n", __func__);
        ggml_free(ctx_crs_late);
        ctx_crs_late = nullptr;
        return;
    }

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        ggml_backend_tensor_set(crs_late_q_indices[layer], crs_late_q_indices_data[layer].data(), 0,
                crs_late_q_indices_data[layer].size() * sizeof(int32_t));
        ggml_backend_tensor_set(crs_late_q_scales[layer], crs_late_q_scales_data[layer].data(), 0,
                crs_late_q_scales_data[layer].size() * sizeof(float));
        ggml_backend_tensor_set(crs_late_k_indices[layer], crs_late_k_indices_data[layer].data(), 0,
                crs_late_k_indices_data[layer].size() * sizeof(int32_t));
        ggml_backend_tensor_set(crs_late_k_scales[layer], crs_late_k_scales_data[layer].data(), 0,
                crs_late_k_scales_data[layer].size() * sizeof(float));
    }

    if (n_layers > 0 && !crs_late_q_indices_data[0].empty()) {
        LLAMA_LOG_INFO("%s: sample late CRS Q entries [layer 0, head 0]:\n", __func__);
        const size_t limit = std::min<size_t>(top_k, 8);
        for (size_t i = 0; i < limit; ++i) {
            LLAMA_LOG_INFO("  [%zu] idx=%d scale=%.4f\n", i,
                    crs_late_q_indices_data[0][i],
                    crs_late_q_scales_data[0][i]);
        }
    }

    LLAMA_LOG_INFO("%s: loaded late CRS sparse tensors for %u layers\n", __func__, n_layers);
}
