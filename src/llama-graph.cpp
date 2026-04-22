#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-hadamard.h"
#include "llama-kv-sensitivity.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

// CRS (Channel-wise Row Scaling) for KV cache quantization
extern "C" {
    #include "../ggml/src/ggml-cpu/crs-scales.h"
}

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <mutex>

static std::mutex g_dump_mutex;

// Dump callback
void dump_k_callback(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    // Identity copy (src -> dst)
    memcpy(dst->data, src->data, ggml_nbytes(src));

    // Only dump if environment variable is set
    static const char * dump_prefix = std::getenv("DUMP_PREFIX");
    if (!dump_prefix) return;

    int il = (int)(intptr_t)userdata;

    // We assume input is F32 (which is usually true for k_cur before RoPE or cache)
    // k_cur shape: [n_embd_head, n_head, n_tokens]
    // Check type: Support F32, F16, BF16
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16) {
        if (il == 0 && ith == 0) {
            fprintf(stderr, "[DUMP_DEBUG] Layer %d: Unsupported type %d\n", il, src->type);
        }
        return;
    }

    std::lock_guard<std::mutex> lock(g_dump_mutex);
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_layer_%d.bin", dump_prefix, il);
    
    FILE * f = fopen(filename, "ab");
    if (f) {
        if (src->type == GGML_TYPE_F32) {
            fwrite(src->data, 1, ggml_nbytes(src), f);
        } else if (src->type == GGML_TYPE_F16) {
            int64_t ne = ggml_nelements(src);
            std::vector<float> buf(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)src->data, buf.data(), ne);
            fwrite(buf.data(), 1, ne * sizeof(float), f);
        } else if (src->type == GGML_TYPE_BF16) {
            int64_t ne = ggml_nelements(src);
            std::vector<float> buf(ne);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)src->data, buf.data(), ne);
            fwrite(buf.data(), 1, ne * sizeof(float), f);
        }
        fclose(f);
        
        static bool printed = false;
        if (!printed && il == 0) {
            fprintf(stderr, "[DUMP] Appended %ld elements to %s (Type: %s)\n", 
                (long)ggml_nelements(src), filename, ggml_type_name(src->type));
            printed = true;
        }
    }
}

// ============================================================================
// Pre-RoPE K value collection system (memory accumulate + atexit dump)
// Environment variables:
//   ROPE_DIST_VALUES_PATH - output file path
//   ROPE_DIST_TOKENS      - max tokens to collect (default: 200000)
// ============================================================================

struct rope_dist_values {
    std::vector<float> pre_values;   // [token * head * dim] flattened
    std::vector<float> post_values;  // [token * head * dim] flattened
};

static std::vector<rope_dist_values> g_rope_dist_values;  // [layer]
static std::vector<std::mutex> g_rope_dist_values_mtx;
static bool g_rope_dist_collect_values = false;
static const char * g_rope_dist_values_path = nullptr;
static int64_t g_rope_dist_layers = 0;
static int64_t g_rope_dist_heads = 0;
static int64_t g_rope_dist_dims = 0;
static int64_t g_rope_dist_max_tokens = 200000;        // total budget
static int64_t g_rope_dist_max_tokens_per_layer = 0;   // = max_tokens / n_layers
static std::vector<int64_t> g_rope_dist_layer_token_count; // per-layer collected count
static int64_t g_rope_dist_token_count = 0;            // global (for header)
static bool g_rope_dist_inited = false;

static void rope_dist_dump_values() {
    if (!g_rope_dist_collect_values || !g_rope_dist_values_path) {
        return;
    }

    FILE * fp = fopen(g_rope_dist_values_path, "wb");
    if (!fp) {
        fprintf(stderr, "rope_dist_dump_values: failed to open %s\n", g_rope_dist_values_path);
        return;
    }

    const uint32_t magic = 0x524F5056;  // 'ROPV'
    const uint32_t version = 1;
    const uint32_t layers = (uint32_t) g_rope_dist_layers;
    const uint32_t heads = (uint32_t) g_rope_dist_heads;
    const uint32_t dims = (uint32_t) g_rope_dist_dims;
    const uint32_t tokens = (uint32_t) g_rope_dist_token_count;

    fwrite(&magic,   sizeof(magic),   1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&layers,  sizeof(layers),  1, fp);
    fwrite(&heads,   sizeof(heads),   1, fp);
    fwrite(&dims,    sizeof(dims),    1, fp);
    fwrite(&tokens,  sizeof(tokens),  1, fp);

    for (size_t layer = 0; layer < g_rope_dist_values.size(); ++layer) {
        const auto & v = g_rope_dist_values[layer];
        uint32_t pre_count = (uint32_t) v.pre_values.size();
        fwrite(&pre_count, sizeof(uint32_t), 1, fp);
        if (pre_count > 0) {
            fwrite(v.pre_values.data(), sizeof(float), pre_count, fp);
        }
        uint32_t post_count = (uint32_t) v.post_values.size();
        fwrite(&post_count, sizeof(uint32_t), 1, fp);
        if (post_count > 0) {
            fwrite(v.post_values.data(), sizeof(float), post_count, fp);
        }
    }

    fclose(fp);
    fprintf(stderr, "rope_dist_dump_values: saved to %s (layers=%d, heads=%d, dims=%d, tokens=%d)\n",
            g_rope_dist_values_path, layers, heads, dims, tokens);
}

void rope_dist_init_if_needed(int64_t n_layers, int64_t n_heads, int64_t head_dim) {
    if (g_rope_dist_inited) {
        return;
    }
    g_rope_dist_inited = true;

    const char * values_path = getenv("ROPE_DIST_VALUES_PATH");
    if (!values_path || values_path[0] == '\0') {
        return;
    }

    int64_t layers = n_layers;
    int64_t heads = n_heads;
    int64_t dims = head_dim;

    const char * s_tokens = getenv("ROPE_DIST_TOKENS");
    if (s_tokens && s_tokens[0]) {
        g_rope_dist_max_tokens = atoll(s_tokens);
    }

    if (layers <= 0 || heads <= 0 || dims <= 0) {
        return;
    }

    g_rope_dist_layers = layers;
    g_rope_dist_heads = heads;
    g_rope_dist_dims = dims;

    g_rope_dist_max_tokens_per_layer = g_rope_dist_max_tokens / layers;
    if (g_rope_dist_max_tokens_per_layer < 1) g_rope_dist_max_tokens_per_layer = 1;

    g_rope_dist_collect_values = true;
    g_rope_dist_values_path = values_path;
    g_rope_dist_values.resize(layers);
    g_rope_dist_values_mtx = std::vector<std::mutex>((size_t)layers);
    g_rope_dist_layer_token_count.resize(layers, 0);

    atexit(rope_dist_dump_values);

    fprintf(stderr, "rope_dist_values: enabled (path=%s, layers=%ld, heads=%ld, dims=%ld, total_budget=%ld, per_layer=%ld)\n",
            values_path, (long)layers, (long)heads, (long)dims, (long)g_rope_dist_max_tokens, (long)g_rope_dist_max_tokens_per_layer);
}

void rope_dist_update_pre(int layer, const float * data, int64_t n_head, int64_t head_dim, int64_t n_tokens) {
    if (!g_rope_dist_collect_values) return;
    if (layer < 0 || layer >= g_rope_dist_layers) return;

    const int64_t heads_to_track = std::min(n_head, g_rope_dist_heads);
    const int64_t dims_to_track = std::min(head_dim, g_rope_dist_dims);
    const int64_t stride = heads_to_track * dims_to_track;
    const int64_t max_elems = g_rope_dist_max_tokens_per_layer * stride;

    std::lock_guard<std::mutex> lock(g_rope_dist_values_mtx[layer]);
    auto & v = g_rope_dist_values[layer];

    if ((int64_t)v.pre_values.size() >= max_elems) return;

    for (int64_t t = 0; t < n_tokens && (int64_t)v.pre_values.size() < max_elems; ++t) {
        for (int64_t h = 0; h < heads_to_track; ++h) {
            for (int64_t d = 0; d < dims_to_track; ++d) {
                float val = data[t * n_head * head_dim + h * head_dim + d];
                v.pre_values.push_back(val);
            }
        }
    }
    if (layer == 0) {
        g_rope_dist_layer_token_count[0] = (int64_t)v.pre_values.size() / stride;
    }
}

void rope_dist_update_post(int layer, const float * data, int64_t n_head, int64_t head_dim, int64_t n_tokens) {
    if (!g_rope_dist_collect_values) return;
    if (layer < 0 || layer >= g_rope_dist_layers) return;

    const int64_t heads_to_track = std::min(n_head, g_rope_dist_heads);
    const int64_t dims_to_track = std::min(head_dim, g_rope_dist_dims);
    const int64_t stride = heads_to_track * dims_to_track;
    const int64_t max_elems = g_rope_dist_max_tokens_per_layer * stride;

    std::lock_guard<std::mutex> lock(g_rope_dist_values_mtx[layer]);
    auto & v = g_rope_dist_values[layer];

    if ((int64_t)v.post_values.size() >= max_elems) return;

    for (int64_t t = 0; t < n_tokens && (int64_t)v.post_values.size() < max_elems; ++t) {
        for (int64_t h = 0; h < heads_to_track; ++h) {
            for (int64_t d = 0; d < dims_to_track; ++d) {
                float val = data[t * n_head * head_dim + h * head_dim + d];
                v.post_values.push_back(val);
            }
        }
    }
}

void rope_dist_advance_tokens(int64_t n_tokens) {
    if (g_rope_dist_collect_values) {
        g_rope_dist_token_count += n_tokens;
    }
    // no-op: per-layer counting is done inside rope_dist_update_pre
}

// ============================================================================
// Final Q/K collection system for QK logit analysis.
// Q is usually sampled, K is usually dense.
// ============================================================================

struct qk_dist_layer_values {
    std::vector<uint32_t> positions; // absolute token indices in collection order
    std::vector<float> values;       // [sample * head * dim] flattened
};

struct qk_dist_state {
    bool enabled = false;
    bool inited = false;
    const char * path = nullptr;
    const char * label = nullptr;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t dims = 0;
    int64_t max_tokens = 200000;
    int64_t prefix_tokens = 0;
    int64_t stride = 1;
    std::vector<qk_dist_layer_values> layer_values;
    std::vector<std::mutex> layer_mtx;
    std::vector<int64_t> layer_seen_tokens;
};

static qk_dist_state g_qk_dist_q;
static qk_dist_state g_qk_dist_k;
static bool g_qk_dist_dump_registered = false;

static bool qk_dist_should_keep_token(int64_t abs_token, int64_t max_tokens, int64_t prefix_tokens, int64_t stride) {
    if (abs_token < 0 || abs_token >= max_tokens) {
        return false;
    }
    if (abs_token < prefix_tokens) {
        return true;
    }
    if (stride <= 0) {
        return false;
    }
    return ((abs_token - prefix_tokens) % stride) == 0;
}

static void qk_dist_dump_state(const qk_dist_state & state) {
    if (!state.enabled || !state.path) {
        return;
    }

    FILE * fp = fopen(state.path, "wb");
    if (!fp) {
        fprintf(stderr, "qk_dist_dump_state: failed to open %s\n", state.path);
        return;
    }

    const uint32_t magic = 0x41435456; // 'ACTV'
    const uint32_t version = 1;
    const uint32_t layers = (uint32_t) state.layers;
    const uint32_t heads = (uint32_t) state.heads;
    const uint32_t dims = (uint32_t) state.dims;

    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&layers, sizeof(layers), 1, fp);
    fwrite(&heads, sizeof(heads), 1, fp);
    fwrite(&dims, sizeof(dims), 1, fp);

    size_t total_samples = 0;
    for (size_t layer = 0; layer < state.layer_values.size(); ++layer) {
        const auto & v = state.layer_values[layer];
        const uint32_t sample_count = (uint32_t) v.positions.size();
        const uint32_t value_count = (uint32_t) v.values.size();

        fwrite(&sample_count, sizeof(sample_count), 1, fp);
        if (sample_count > 0) {
            fwrite(v.positions.data(), sizeof(uint32_t), sample_count, fp);
        }

        fwrite(&value_count, sizeof(value_count), 1, fp);
        if (value_count > 0) {
            fwrite(v.values.data(), sizeof(float), value_count, fp);
        }

        total_samples += sample_count;
    }

    fclose(fp);
    fprintf(stderr,
            "qk_dist_dump_state: saved %s to %s (layers=%d, heads=%d, dims=%d, samples=%zu)\n",
            state.label ? state.label : "activations",
            state.path,
            layers,
            heads,
            dims,
            total_samples);
}

static void qk_dist_dump_all() {
    qk_dist_dump_state(g_qk_dist_q);
    qk_dist_dump_state(g_qk_dist_k);
}

static void qk_dist_init_state(
        qk_dist_state & state,
        const char * path_env,
        const char * label,
        int64_t n_layers,
        int64_t n_heads,
        int64_t head_dim,
        int64_t default_prefix_tokens,
        int64_t default_stride,
        const char * prefix_env,
        const char * stride_env) {
    if (state.inited) {
        return;
    }
    state.inited = true;

    const char * path = getenv(path_env);
    if (!path || path[0] == '\0') {
        return;
    }

    state.path = path;
    state.label = label;
    state.layers = n_layers;
    state.heads = n_heads;
    state.dims = head_dim;
    state.prefix_tokens = default_prefix_tokens;
    state.stride = default_stride;

    const char * s_max_tokens = getenv("QK_DIST_MAX_TOKENS");
    if (s_max_tokens && s_max_tokens[0]) {
        state.max_tokens = atoll(s_max_tokens);
    }

    const char * s_prefix = getenv(prefix_env);
    if (s_prefix && s_prefix[0]) {
        state.prefix_tokens = atoll(s_prefix);
    }

    const char * s_stride = getenv(stride_env);
    if (s_stride && s_stride[0]) {
        state.stride = atoll(s_stride);
    }

    state.enabled = true;
    state.layer_values.resize((size_t) n_layers);
    state.layer_mtx = std::vector<std::mutex>((size_t) n_layers);
    state.layer_seen_tokens.resize((size_t) n_layers, 0);

    if (!g_qk_dist_dump_registered) {
        atexit(qk_dist_dump_all);
        g_qk_dist_dump_registered = true;
    }

    fprintf(stderr,
            "qk_dist: enabled %s (path=%s, layers=%ld, heads=%ld, dims=%ld, max_tokens=%ld, prefix=%ld, stride=%ld)\n",
            label,
            path,
            (long) n_layers,
            (long) n_heads,
            (long) head_dim,
            (long) state.max_tokens,
            (long) state.prefix_tokens,
            (long) state.stride);
}

static void qk_dist_update_state(
        qk_dist_state & state,
        int layer,
        const float * data,
        int64_t n_head,
        int64_t head_dim,
        int64_t n_tokens) {
    if (!state.enabled) {
        return;
    }
    if (layer < 0 || layer >= state.layers) {
        return;
    }

    const int64_t heads_to_track = std::min(n_head, state.heads);
    const int64_t dims_to_track = std::min(head_dim, state.dims);

    std::lock_guard<std::mutex> lock(state.layer_mtx[(size_t) layer]);
    auto & v = state.layer_values[(size_t) layer];
    auto & seen = state.layer_seen_tokens[(size_t) layer];

    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t abs_token = seen++;
        if (!qk_dist_should_keep_token(abs_token, state.max_tokens, state.prefix_tokens, state.stride)) {
            continue;
        }

        v.positions.push_back((uint32_t) abs_token);
        for (int64_t h = 0; h < heads_to_track; ++h) {
            for (int64_t d = 0; d < dims_to_track; ++d) {
                const float val = data[t * n_head * head_dim + h * head_dim + d];
                v.values.push_back(val);
            }
        }
    }
}

void qk_dist_q_init_if_needed(int64_t n_layers, int64_t n_heads, int64_t head_dim) {
    qk_dist_init_state(
            g_qk_dist_q,
            "QK_DIST_Q_PATH",
            "Q samples",
            n_layers,
            n_heads,
            head_dim,
            16,
            128,
            "QK_DIST_Q_PREFIX_TOKENS",
            "QK_DIST_Q_STRIDE");
}

void qk_dist_k_init_if_needed(int64_t n_layers, int64_t n_heads, int64_t head_dim) {
    qk_dist_init_state(
            g_qk_dist_k,
            "QK_DIST_K_PATH",
            "K samples",
            n_layers,
            n_heads,
            head_dim,
            0,
            1,
            "QK_DIST_K_PREFIX_TOKENS",
            "QK_DIST_K_STRIDE");
}

void qk_dist_q_update(int layer, const float * data, int64_t n_head, int64_t head_dim, int64_t n_tokens) {
    qk_dist_update_state(g_qk_dist_q, layer, data, n_head, head_dim, n_tokens);
}

void qk_dist_k_update(int layer, const float * data, int64_t n_head, int64_t head_dim, int64_t n_tokens) {
    qk_dist_update_state(g_qk_dist_k, layer, data, n_head, head_dim, n_tokens);
}

// CRS scale restoration parameters
struct crs_restore_params {
    int32_t layer;
    int32_t n_head;
    int32_t n_embd_head;
};

// CRS scale application: divides K by CRS scale before quantization
static void crs_apply_op(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    
    const crs_restore_params * params = (const crs_restore_params *)userdata;
    if (!params || !g_crs_static.enabled) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return;
    }
    
    const int layer = params->layer;
    const int n_head = params->n_head;
    const int n_embd_head = params->n_embd_head;
    
    // K shape before cpy: [n_embd_head, n_head, n_tokens]
    const int64_t ne0 = src->ne[0];  // n_embd_head
    const int64_t ne1 = src->ne[1];  // n_head
    const int64_t ne2 = src->ne[2];  // n_tokens
    
    const float * src_data = (const float *)src->data;
    float * dst_data = (float *)dst->data;
    
    // Debug: print once per layer for head 1 (head 0 has no outliers)
    static int debug_count = 0;
    if (debug_count < 1 && layer == 0) {
        fprintf(stderr, "[CRS_APPLY] Layer %d, shape=[%ld,%ld,%ld], head 1 outliers: ", layer, (long)ne0, (long)ne1, (long)ne2);
        for (int d = 80; d < 90; d++) {
            float s = crs_get_static_scale(layer, 1, d);
            if (s != 1.0f) fprintf(stderr, "d%d=%.2f ", d, s);
        }
        // Debug: Analyze Block 2 (dims 64-95) to find unhandled outliers
        if (ne2 > 0 && ne1 > 1) {
            fprintf(stderr, "[DEBUG] Layer %d Head 1 Block 2 (dims 64-95) analysis:\n", layer);
            float max_val = 0;
            int max_idx = -1;
            
            for (int d = 64; d < 96; d++) {
                int64_t idx = 0 * ne1 * ne0 + 1 * ne0 + d; // token 0, head 1, dim d
                float val = fabsf(src_data[idx]);
                float scale = crs_get_static_scale(layer, 1, d);
                
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
                
                // Print any large values (>1.0) and their scale status
                if (val > 1.0f) {
                    fprintf(stderr, "  d%d: val=%.4f, scale=%.2f\n", d, val, scale);
                }
            }
            fprintf(stderr, "  => Block Max: d%d = %.4f (Scale applied: %s)\n", 
                    max_idx, max_val, crs_get_static_scale(layer, 1, max_idx) > 1.0f ? "YES" : "NO");
        }
        fprintf(stderr, "\n");
        debug_count++;
    }
    
    // Apply CRS scale per head, per channel (divide to suppress outliers)
    for (int64_t t = 0; t < ne2; t++) {
        for (int64_t h = 0; h < ne1; h++) {
            for (int64_t d = 0; d < ne0; d++) {
                int64_t idx = t * ne1 * ne0 + h * ne0 + d;
                float scale = crs_get_static_scale(layer, h, d);
                dst_data[idx] = src_data[idx] / scale;  // Divide to suppress
            }
        }
    }
}

// CRS scale restoration: multiplies K by CRS scale after dequantization
static void crs_restore_op(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    
    const crs_restore_params * params = (const crs_restore_params *)userdata;
    if (!params || !g_crs_static.enabled) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return;
    }
    
    const int layer = params->layer;
    const int n_head = params->n_head;
    const int n_embd_head = params->n_embd_head;
    
    // BEFORE permute: K shape is [n_embd_head, n_head, n_kv, n_stream]
    // ne[0] = n_embd_head, ne[1] = n_head, ne[2] = n_kv, ne[3] = n_stream
    const int64_t ne0 = src->ne[0];  // n_embd_head
    const int64_t ne1 = src->ne[1];  // n_head
    const int64_t ne2 = src->ne[2];  // n_kv
    const int64_t ne3 = src->ne[3];  // n_stream
    
    const float * src_data = (const float *)src->data;
    float * dst_data = (float *)dst->data;
    
    // Debug: print once per layer for head 1
    static int debug_count = 0;
    if (debug_count < 1 && layer == 0) {
        fprintf(stderr, "[CRS_RESTORE] Layer %d, shape=[%ld,%ld,%ld,%ld], head 1 outliers: ", layer, (long)ne0, (long)ne1, (long)ne2, (long)ne3);
        for (int d = 80; d < 90; d++) {
            float s = crs_get_static_scale(layer, 1, d);
            if (s != 1.0f) fprintf(stderr, "d%d=%.2f ", d, s);
        }
        // Print sample K value before/after restore
        if (ne2 > 0 && ne1 > 1 && ne3 > 0) {
            int64_t idx_d80 = 0 * ne2 * ne1 * ne0 + 0 * ne1 * ne0 + 1 * ne0 + 80; // stream 0, kv 0, head 1, dim 80
            float k_before = src_data[idx_d80];
            float scale_d80 = crs_get_static_scale(layer, 1, 80);
            fprintf(stderr, "| K[0,0,1,80]: %.4f * %.2f = %.4f", k_before, scale_d80, k_before*scale_d80);
        }
        fprintf(stderr, "\n");
        debug_count++;
    }
    
    // Restore CRS scale per head, per channel
    for (int64_t s = 0; s < ne3; s++) {
        for (int64_t kv = 0; kv < ne2; kv++) {
            for (int64_t h = 0; h < ne1; h++) {
                for (int64_t d = 0; d < ne0; d++) {
                    int64_t idx = s * ne2 * ne1 * ne0 + kv * ne1 * ne0 + h * ne0 + d;
                    float scale = crs_get_static_scale(layer, h, d);
                    dst_data[idx] = src_data[idx] * scale;
                }
            }
        }
    }
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        const int64_t n_embd   = embd->ne[0];
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!tokens && !params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!embd   && !params.ubatch.embd)  || (embd   &&   embd->ne[0] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + 1.0f) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
                }
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && arch == LLM_ARCH_QWEN3) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

static void print_mask(const float * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swq_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = data[i * n_kv + j];
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](float * data, int n_swa, llama_swa_type swa_type) {
        for (int h = 0; h < 1; ++h) {
            for (int i1 = 0; i1 < n_tokens; ++i1) {
                const llama_seq_id s1 = ubatch->seq_id[i1][0];
                const llama_pos    p1 = ubatch->pos[i1];

                const uint64_t idst = h*(n_kv*n_tokens) + i1*n_kv;

                for (int i0 = 0; i0 < n_tokens; ++i0) {
                    const llama_seq_id s0 = ubatch->seq_id[i0][0];
                    const llama_pos p0    = ubatch->pos[i0];

                    // mask different sequences
                    if (s0 != s1) {
                        continue;
                    }

                    // mask future tokens
                    if (cparams.causal_attn && p0 > p1) {
                        continue;
                    }

                    // apply SWA if any
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        continue;
                    }

                    data[idst + i0] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
                }
            }
        }
    };

    {
        GGML_ASSERT(self_kq_mask);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));

        float * data = (float *) self_kq_mask->data;

        std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);

        fill_mask(data, 0, LLAMA_SWA_TYPE_NONE);

        if (debug) {
            print_mask(data, n_tokens, n_kv, 0, LLAMA_SWA_TYPE_NONE);
        }
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));

        float * data = (float *) self_kq_mask_swa->data;

        std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);

        fill_mask(data, hparams.n_swa, hparams.swa_type);

        if (debug) {
            print_mask(data, n_tokens, n_kv, hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_k_cache_pos::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);
    if (pos && kv_cache) {
        kv_cache->set_input_k_cache_pos(pos, n_kv, n_pos_per_embd);
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_n_kv();
    res &= self_kq_mask->ne[1] == GGML_PAD(params.ubatch.n_tokens, GGML_KQ_MASK_PAD);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

    mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_base()->get_n_kv();
    res &= self_kq_mask->ne[1] == GGML_PAD(params.ubatch.n_tokens, GGML_KQ_MASK_PAD);

    res &= self_kq_mask_swa->ne[0] == mctx->get_swa()->get_n_kv();
    res &= self_kq_mask_swa->ne[1] == GGML_PAD(params.ubatch.n_tokens, GGML_KQ_MASK_PAD);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    float * data = (float *) cross_kq_mask->data;

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[h*(n_enc*n_tokens) + i*n_enc + j] = f;
            }
        }

        for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
            for (int j = 0; j < n_enc; ++j) {
                data[h*(n_enc*n_tokens) + i*n_enc + j] = -INFINITY;
            }
        }
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    inp_attn->set_input(ubatch);
    inp_rs->set_input(ubatch);
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_tokens      = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;

    params = {};

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    hadamard         (params.hadamard),
    kv_sensitivity   (params.kv_sensitivity),
    kv_sensitivity_active(params.kv_sensitivity_active),
    mctx             (params.mctx),
    cross            (params.cross),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_tensor * llm_graph_context::build_hadamard_rotated(
        ggml_tensor * cur,
        ggml_tensor * signs,
        const char  * name,
                int   il) const {
    if (hadamard == nullptr || !hadamard->enabled) {
        return cur;
    }

    if (cur == nullptr || signs == nullptr) {
        throw std::runtime_error(format("%s: missing Hadamard tensors for layer %d", __func__, il));
    }

    if (cur->ne[0] != hadamard->head_dim) {
        throw std::runtime_error(format(
                "%s: Hadamard head dim mismatch for %s at layer %d: expected %d, got %" PRId64,
                __func__, name, il, hadamard->head_dim, cur->ne[0]));
    }

    if (signs->ne[0] != cur->ne[0] || signs->ne[1] != cur->ne[1]) {
        throw std::runtime_error(format(
                "%s: Hadamard sign shape mismatch for %s at layer %d", __func__, name, il));
    }

    ggml_tensor * signs3 = ggml_reshape_3d(ctx0, signs, signs->ne[0], signs->ne[1], 1);
    ggml_tensor * signed_cur = ggml_mul(ctx0, cur, ggml_repeat(ctx0, signs3, cur));
    cb(signed_cur, format("%s_hadamard_sign", name).c_str(), il);

    ggml_tensor * rotated = ggml_mul_mat(ctx0, hadamard->matrix, signed_cur);
    cb(rotated, format("%s_hadamard", name).c_str(), il);

    if (rotated->type != cur->type) {
        rotated = ggml_cast(ctx0, rotated, cur->type);
        cb(rotated, format("%s_hadamard_cast", name).c_str(), il);
    }

    if (ggml_row_size(rotated->type, rotated->ne[0]) != rotated->nb[1]) {
        rotated = ggml_cont(ctx0, rotated);
        cb(rotated, format("%s_hadamard_cont", name).c_str(), il);
    }

    return rotated;
}

ggml_tensor * llm_graph_context::build_kv_sensitivity_quantized(
        ggml_tensor * cur,
          ggml_type   type,
        bool          apply_crs,
          const char * name,
                int   il) const {
    GGML_ASSERT(cur != nullptr);

    ggml_tensor * cur_f32 = cur->type == GGML_TYPE_F32 ? cur : ggml_cast(ctx0, cur, GGML_TYPE_F32);
    if (!ggml_is_contiguous(cur_f32)) {
        cur_f32 = ggml_cont(ctx0, cur_f32);
    }

    if (apply_crs && g_crs_static.enabled) {
        auto * params = new crs_restore_params{il, (int32_t) cur_f32->ne[1], (int32_t) cur_f32->ne[0]};
        cur_f32 = ggml_map_custom1(ctx0, cur_f32, crs_apply_op, 1, params);
        ggml_format_name(cur_f32, "%s_crs_applied_l%d", name, il);
    }

    ggml_tensor * quantized_storage = ggml_new_tensor(ctx0, type, GGML_MAX_DIMS, cur_f32->ne);
    ggml_format_name(quantized_storage, "%s_storage_l%d", name, il);

    ggml_tensor * quantized = ggml_cpy(ctx0, cur_f32, quantized_storage);
    ggml_tensor * dequantized = ggml_cast(ctx0, quantized, GGML_TYPE_F32);

    if (apply_crs && g_crs_static.enabled) {
        auto * params = new crs_restore_params{il, (int32_t) dequantized->ne[1], (int32_t) dequantized->ne[0]};
        dequantized = ggml_map_custom1(ctx0, dequantized, crs_restore_op, 1, params);
        ggml_format_name(dequantized, "%s_crs_restored_l%d", name, il);
    }

    return dequantized;
}

void llm_graph_context::build_kv_sensitivity_measurement(
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    if (kv_sensitivity == nullptr || !kv_sensitivity_active || !cparams.measure_kv_sensitivity || il != cparams.sensitivity_layer) {
        return;
    }

    // Skip single-token decode steps so the measurement captures prompt-prefill behavior only.
    if (n_tokens <= 1) {
        return;
    }

    auto build_branch = [&](ggml_tensor * q_in, ggml_tensor * k_in, ggml_tensor * v_in, const char * tag) {
        const bool v_trans = v_in->nb[1] > v_in->nb[2];
        const auto n_stream = k_in->ne[3];
        const auto n_q = q_in->ne[2] / n_stream;
        const auto n_k = k_in->ne[2];

        ggml_tensor * q = ggml_view_4d(ctx0, q_in, q_in->ne[0], q_in->ne[1], q_in->ne[2]/n_stream, n_stream, q_in->nb[1], q_in->nb[2], q_in->nb[3]/n_stream, 0);
        ggml_tensor * k = k_in;
        ggml_tensor * v = v_in;
        ggml_tensor * kq_mask_local = kq_mask;

        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);

        if (kq_mask_local != nullptr) {
            const int64_t padded_q = GGML_PAD(n_q, GGML_KQ_MASK_PAD);

            if (kq_mask_local->ne[0] != n_k || kq_mask_local->ne[1] != padded_q) {
                GGML_ASSERT(kq_mask_local->ne[0] >= n_k);
                GGML_ASSERT(kq_mask_local->ne[1] >= padded_q);

                kq_mask_local = ggml_view_4d(
                        ctx0,
                        kq_mask_local,
                        n_k,
                        padded_q,
                        kq_mask_local->ne[2],
                        kq_mask_local->ne[3],
                        kq_mask_local->nb[1],
                        kq_mask_local->nb[2],
                        kq_mask_local->nb[3],
                        0);
                ggml_format_name(kq_mask_local, "kv_sens_%s_kq_mask_l%d", tag, il);
            }

            if (!ggml_is_contiguous(kq_mask_local)) {
                kq_mask_local = ggml_cont(ctx0, kq_mask_local);
                ggml_format_name(kq_mask_local, "kv_sens_%s_kq_mask_cont_l%d", tag, il);
            }
        }

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        ggml_format_name(kq, "kv_sens_%s_kq_l%d", tag, il);

        if (arch == LLM_ARCH_GROK) {
            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh(ctx0, kq);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask_local, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        ggml_format_name(kq, "kv_sens_%s_score_l%d", tag, il);

        if (!v_trans) {
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
        }

        ggml_tensor * cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        ggml_format_name(cur, "kv_sens_%s_out_l%d", tag, il);

        return std::make_pair(kq, cur);
    };

    ggml_tensor * q_ref = q_cur->type == GGML_TYPE_F32 ? q_cur : ggml_cast(ctx0, q_cur, GGML_TYPE_F32);
    ggml_tensor * k_ref = k_cur->type == GGML_TYPE_F32 ? k_cur : ggml_cast(ctx0, k_cur, GGML_TYPE_F32);
    ggml_tensor * v_ref = v_cur->type == GGML_TYPE_F32 ? v_cur : ggml_cast(ctx0, v_cur, GGML_TYPE_F32);

    ggml_tensor * k_baseline = build_kv_sensitivity_quantized(k_ref, cparams.sensitivity_baseline_k_type, true,  "kv_sens_k_baseline", il);
    ggml_tensor * v_baseline = build_kv_sensitivity_quantized(v_ref, cparams.sensitivity_baseline_v_type, false, "kv_sens_v_baseline", il);
    ggml_tensor * k_probe    = build_kv_sensitivity_quantized(k_ref, cparams.sensitivity_probe_k_type,    true,  "kv_sens_k_probe",    il);
    ggml_tensor * v_probe    = build_kv_sensitivity_quantized(v_ref, cparams.sensitivity_probe_v_type,    false, "kv_sens_v_probe",    il);

    const auto ref_branch      = build_branch(q_ref, k_ref,      v_ref,      "ref");
    const auto baseline_branch = build_branch(q_ref, k_baseline, v_baseline, "baseline");
    const auto probe_branch    = build_branch(q_ref, k_probe,    v_probe,    "probe");

    auto * score_params = new llama_kv_sensitivity_capture_params{kv_sensitivity};
    ggml_tensor * score_capture = ggml_map_custom3(
            ctx0,
            ref_branch.first,
            baseline_branch.first,
            probe_branch.first,
            llama_kv_sensitivity_capture_score_op,
            1,
            score_params);
    ggml_format_name(score_capture, "kv_sensitivity_score_capture_l%d", il);
    ggml_backend_sched_set_tensor_backend(sched, score_capture, backend_cpu);
    ggml_build_forward_expand(gf, score_capture);

    auto * output_params = new llama_kv_sensitivity_capture_params{kv_sensitivity};
    ggml_tensor * output_capture = ggml_map_custom3(
            ctx0,
            ref_branch.second,
            baseline_branch.second,
            probe_branch.second,
            llama_kv_sensitivity_capture_output_op,
            1,
            output_params);
    ggml_format_name(output_capture, "kv_sensitivity_output_capture_l%d", il);
    ggml_backend_sched_set_tensor_backend(sched, output_capture, backend_cpu);
    ggml_build_forward_expand(gf, output_capture);
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    //expand here so that we can fuse ffn gate
    ggml_build_forward_expand(gf, cur);

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        scale_w,
        w_scale,
        gating_op,
        il,
        probs_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_scale_bias(ctx0, selection_groups, 0.0f, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);

    if (up_exps_b) {
        up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
        cb(up, "ffn_moe_up_biased", il);
    }

    ggml_tensor * experts = nullptr;
    if (gate_exps) {
        cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(cur, "ffn_moe_gate", il);
    } else {
        cur = up;
    }

    if (gate_exps_b) {
        cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
        cb(cur, "ffn_moe_gate_biased", il);
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate_exps) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (gate_exps) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    //expand here so that we can fuse ffn gate
    ggml_build_forward_expand(gf, cur);

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd = hparams.n_embd_inp();

    auto inp = std::make_unique<llm_graph_input_embd>();

    ggml_tensor * cur = nullptr;

    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        //cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_tokens = inp->tokens;

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }
    } else {
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        ggml_set_input(inp->embd);

        cur = inp->embd;
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_embd_fc(ggml_tensor * embd, ggml_tensor * fc, ggml_tensor * fc_b) const {
    ggml_tensor * cur = nullptr;

    ggml_tensor * hidden_states = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, ubatch.n_tokens);
    ggml_set_input(hidden_states);

    cur = ggml_concat(ctx0, embd, hidden_states, 0);
    cb(cur, "inp_concat", -1);

    cur = ggml_mul_mat(ctx0, fc, cur);
    cb(cur, "inp_matmul", -1);

    if (fc_b) {
        cur = ggml_add(ctx0, cur, fc_b);
    }

    cb(cur, "inp_embd_fc_out", -1);

    res->set_hidden_states(hidden_states);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_k_cache_pos(const llama_kv_cache * kv_cache, uint32_t n_kv, uint32_t n_stream) const {
    auto inp = std::make_unique<llm_graph_input_k_cache_pos>(hparams, cparams, kv_cache, n_kv, hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_kv * n_stream * hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interere with
    //       features that require constant topology such as pipline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    // Apply CRS scale restoration if enabled BEFORE permute
    // K shape before permute: [n_embd_head, n_head, n_kv, n_stream]
    if (g_crs_static.enabled && ggml_is_quantized(k->type)) {
        // First dequantize K to F32
        k = ggml_cast(ctx0, k, GGML_TYPE_F32);
        
        // Apply CRS scale restoration on original layout
        crs_restore_params * params = new crs_restore_params{il, (int32_t)hparams.n_head_kv(), (int32_t)hparams.n_embd_head_k};
        k = ggml_map_custom1(ctx0, k, crs_restore_op, GGML_N_TASKS_MAX, params);
        ggml_format_name(k, "k_crs_restored_l%d", il);
    }

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    if (cparams.flash_attn && kq_b == nullptr) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il);

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is ineffient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    if (hadamard && hadamard->enabled) {
        q = build_hadamard_rotated(q, hadamard->q_signs.at(il), "Qcur", il);
        k = build_hadamard_rotated(k, hadamard->k_signs.at(il), "Kcur", il);
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD), 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;

    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il,
        ggml_tensor * rope_factors) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_tensor * q = q_cur;
    ggml_tensor * k_store_src = k_cur;

    if (hadamard && hadamard->enabled) {
        q = build_hadamard_rotated(q, hadamard->q_signs.at(il), "Qcur", il);
        k_store_src = build_hadamard_rotated(k_store_src, hadamard->k_signs.at(il), "Kcur", il);
    }

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        // Apply CRS scale before storing to cache (suppress outliers)
        ggml_tensor * k_to_store = k_store_src;
        if (g_crs_static.enabled) {
            crs_restore_params * params = new crs_restore_params{il, (int32_t)hparams.n_head_kv(), (int32_t)hparams.n_embd_head_k};
            k_to_store = ggml_map_custom1(ctx0, k_store_src, crs_apply_op, GGML_N_TASKS_MAX, params);
            ggml_format_name(k_to_store, "k_crs_applied_l%d", il);
        }

        // Dump activations if DUMP_PREFIX is set
        if (std::getenv("DUMP_PREFIX")) {
            static bool logged = false;
            if (!logged) {
                fprintf(stderr, "[DUMP_GRAPH] Adding dump node for Layer %d\n", il);
                logged = true;
            }
            k_to_store = ggml_map_custom1(ctx0, k_to_store, dump_k_callback, 1, (void*)(intptr_t)il);
            ggml_format_name(k_to_store, "k_dump_l%d", il);
        }

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_to_store, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    if (kv_sensitivity_active && cparams.measure_kv_sensitivity && il == cparams.sensitivity_layer) {
        build_kv_sensitivity_measurement(q, k_store_src, v_cur, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    }

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // pre_rope: apply RoPE to K from cache at attention time
    if (cparams.pre_rope) {
        const auto n_kv = mctx_cur->get_n_kv();
        const llama_kv_cache * kv_cache = mctx_cur->get_kv_cache();

        // If K is quantized, dequantize to F32 first
        if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F32);
            cb(k, "k_dequant", il);
        }

        // If V is quantized, dequantize to F32 to match K type for flash attention
        if (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F32);
            cb(v, "v_dequant", il);
        }

        const int64_t ne0 = k->ne[0]; // n_embd_head
        const int64_t ne1 = k->ne[1]; // n_head_kv
        const int64_t ne2 = k->ne[2]; // n_kv
        const int64_t ne3 = k->ne[3]; // n_stream (from slot_info, may be < kv_cache->get_n_stream())

        if (ne3 == 1) {
            // Single stream: apply RoPE directly to 4D K (ne[2]=n_kv matches pos size)
            ggml_tensor * inp_k_pos = build_inp_k_cache_pos(kv_cache, n_kv, 1);
            k = ggml_rope_ext(
                    ctx0, k, inp_k_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(k, "k_rope", il);
            ggml_build_forward_expand(gf, k);
        } else {
            // Multi-stream: reshape to 3D so each stream gets its own positions
            // K may not be contiguous (stride gap when n_kv < kv_size)
            k = ggml_cont(ctx0, k);
            k = ggml_reshape_3d(ctx0, k, ne0, ne1, ne2*ne3);

            ggml_tensor * inp_k_pos = build_inp_k_cache_pos(kv_cache, n_kv, ne3);
            k = ggml_rope_ext(
                    ctx0, k, inp_k_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            k = ggml_reshape_4d(ctx0, k, ne0, ne1, ne2, ne3);
            cb(k, "k_rope", il);
            ggml_build_forward_expand(gf, k);
        }
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const bool is_swa = hparams.is_swa(il);

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    ggml_tensor * q = q_cur;

    if (hadamard && hadamard->enabled) {
        q = build_hadamard_rotated(q, hadamard->q_signs.at(il), "Qcur", il);
    }

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();
        ggml_tensor * k_to_store = k_cur;

        if (hadamard && hadamard->enabled) {
            k_to_store = build_hadamard_rotated(k_to_store, hadamard->k_signs.at(il), "Kcur", il);
        }

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_to_store, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    if (hadamard && hadamard->enabled) {
        q = build_hadamard_rotated(q, hadamard->q_signs.at(il), "Qcur", il);
        k = build_hadamard_rotated(k, hadamard->k_signs.at(il), "Kcur", il);
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    {
        const auto n_kv = mctx_cur->get_base()->get_n_kv();

        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD), 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        const auto n_kv = mctx_cur->get_swa()->get_n_kv();

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD), 1, n_stream);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    }

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_1d(ctx0, s, state_size*(n_rs - n_seqs), (rs_head + n_seqs)*state_size*ggml_element_size(s))));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || dense_2 == nullptr || dense_3 == nullptr) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    cur = ggml_mul_mat(ctx0, dense_2, cur);
    cur = ggml_mul_mat(ctx0, dense_3, cur);
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_tanh(ctx0, cur);
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            } break;
    }

    cb(cur, "result_embd_pooled", -1);

    ggml_build_forward_expand(gf, cur);
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
