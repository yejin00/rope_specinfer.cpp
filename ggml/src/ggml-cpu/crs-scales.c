/**
 * CRS (Channel-wise Row Scaling) Scale Management Implementation
 */

#include "crs-scales.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Global CRS state
crs_static_config_t g_crs_static = {0};
crs_online_config_t g_crs_online = {0};

void crs_init(void) {
    memset(&g_crs_static, 0, sizeof(g_crs_static));
    memset(&g_crs_online, 0, sizeof(g_crs_online));
    g_crs_static.enabled = false;
    g_crs_online.enabled = false;
}

int crs_load_static_scales(const char * path) {
    if (!path || !path[0]) {
        return -1;
    }
    
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CRS] Failed to open static scales file: %s\n", path);
        return -1;
    }
    
    // Read header
    uint32_t magic, version, n_layers, n_heads, n_dims, top_k;
    
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != CRS_MAGIC_STATIC) {
        fprintf(stderr, "[CRS] Invalid magic number in scales file\n");
        fclose(f);
        return -1;
    }
    
    fread(&version, sizeof(uint32_t), 1, f);
    fread(&n_layers, sizeof(uint32_t), 1, f);
    fread(&n_heads, sizeof(uint32_t), 1, f);
    fread(&n_dims, sizeof(uint32_t), 1, f);
    fread(&top_k, sizeof(uint32_t), 1, f);

    uint32_t stored_len = top_k;
    {
        const long header_bytes = 6L * (long) sizeof(uint32_t);
        long cur = ftell(f);
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, cur, SEEK_SET);

        if (file_size > header_bytes) {
            const long remaining = file_size - header_bytes;
            const long per_head_bytes_crs = (long) top_k * (long) sizeof(int32_t) + (long) top_k * (long) sizeof(float);
            const long per_head_bytes_prs = (long) (top_k * 2U) * (long) sizeof(int32_t) + (long) (top_k * 2U) * (long) sizeof(float);
            const long expected_crs = (long) n_layers * (long) n_heads * per_head_bytes_crs;
            const long expected_prs = (long) n_layers * (long) n_heads * per_head_bytes_prs;

            if (remaining == expected_prs) {
                stored_len = top_k * 2U;
            } else if (remaining == expected_crs) {
                stored_len = top_k;
            } else {
                stored_len = top_k * 2U;
            }
        }
    }
    
    if (n_layers > CRS_MAX_LAYERS || n_heads > CRS_MAX_HEADS || stored_len > CRS_MAX_OUTLIERS) {
        fprintf(stderr, "[CRS] Scales file dimensions exceed limits: layers=%u, heads=%u, stored_len=%u\n",
                n_layers, n_heads, stored_len);
        fclose(f);
        return -1;
    }
    
    g_crs_static.n_layers = n_layers;
    g_crs_static.n_heads = n_heads;
    g_crs_static.n_dims = n_dims;
    g_crs_static.top_k = stored_len;
    
    // Read per-layer, per-head data
    for (uint32_t l = 0; l < n_layers; l++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            crs_head_config_t * cfg = &g_crs_static.configs[l][h];
            
            int32_t indices[CRS_MAX_OUTLIERS];
            float scales[CRS_MAX_OUTLIERS];
            
            if (fread(indices, sizeof(int32_t), stored_len, f) != stored_len ||
                fread(scales, sizeof(float), stored_len, f) != stored_len) {
                fprintf(stderr, "[CRS] Failed to read scales for layer %u, head %u\n", l, h);
                fclose(f);
                return -1;
            }
            
            // Copy valid entries (indices >= 0)
            cfg->n_outliers = 0;
            for (uint32_t i = 0; i < stored_len; i++) {
                if (indices[i] >= 0 && (uint32_t) indices[i] < n_dims) {
                    cfg->indices[cfg->n_outliers] = indices[i];
                    cfg->scales[cfg->n_outliers] = scales[i];
                    cfg->n_outliers++;
                }
            }
        }
    }
    
    fclose(f);
    g_crs_static.enabled = true;
    
    fprintf(stderr, "[CRS] Loaded static scales: %u layers, %u heads, top-%u outliers\n",
            n_layers, n_heads, stored_len);
    
    return 0;
}

void crs_enable_online(uint32_t top_k, float threshold) {
    g_crs_online.enabled = true;
    g_crs_online.top_k = top_k > CRS_MAX_OUTLIERS ? CRS_MAX_OUTLIERS : top_k;
    g_crs_online.threshold = threshold;
    
    fprintf(stderr, "[CRS] Online CRS enabled: top-%u, threshold=%.2f\n", 
            g_crs_online.top_k, threshold);
}

float crs_get_static_scale(int layer, int head, int channel) {
    if (!g_crs_static.enabled || 
        layer < 0 || layer >= (int)g_crs_static.n_layers ||
        head < 0 || head >= (int)g_crs_static.n_heads) {
        return 1.0f;
    }
    
    const crs_head_config_t * cfg = &g_crs_static.configs[layer][head];
    for (int i = 0; i < cfg->n_outliers; i++) {
        if (cfg->indices[i] == channel) {
            return cfg->scales[i];
        }
    }
    
    return 1.0f;
}

bool crs_is_static_outlier(int layer, int head, int channel) {
    if (!g_crs_static.enabled ||
        layer < 0 || layer >= (int)g_crs_static.n_layers ||
        head < 0 || head >= (int)g_crs_static.n_heads) {
        return false;
    }
    
    const crs_head_config_t * cfg = &g_crs_static.configs[layer][head];
    for (int i = 0; i < cfg->n_outliers; i++) {
        if (cfg->indices[i] == channel) {
            return true;
        }
    }
    
    return false;
}

void crs_apply_static_scale(int layer, int head, float * k_data, int n_dims) {
    if (!g_crs_static.enabled || !k_data ||
        layer < 0 || layer >= (int)g_crs_static.n_layers ||
        head < 0 || head >= (int)g_crs_static.n_heads) {
        return;
    }
    
    const crs_head_config_t * cfg = &g_crs_static.configs[layer][head];
    
    // Divide outlier channels by their scale to reduce magnitude before quantization
    for (int i = 0; i < cfg->n_outliers; i++) {
        int ch = cfg->indices[i];
        if (ch >= 0 && ch < n_dims && cfg->scales[i] > 1e-6f) {
            k_data[ch] /= cfg->scales[i];
        }
    }
}

int crs_compute_online_scales(
    int layer, int head,
    const float * k_data, int n_dims,
    int32_t * out_indices, float * out_scales) 
{
    if (!g_crs_online.enabled || !k_data || !out_indices || !out_scales) {
        return 0;
    }
    
    // Compute channel-wise absmax
    float absmax[CRS_MAX_OUTLIERS * 2];  // Temporary storage
    int indices[CRS_MAX_OUTLIERS * 2];
    
    // Find top-k channels by absmax (excluding static outliers)
    int n_candidates = 0;
    for (int ch = 0; ch < n_dims && n_candidates < (int)(g_crs_online.top_k * 2); ch++) {
        // Skip static outliers - they're already handled
        if (crs_is_static_outlier(layer, head, ch)) {
            continue;
        }
        
        float val = fabsf(k_data[ch]);
        
        // Simple insertion sort to maintain top-k
        int pos = n_candidates;
        while (pos > 0 && val > absmax[pos - 1]) {
            if (pos < (int)(g_crs_online.top_k)) {
                absmax[pos] = absmax[pos - 1];
                indices[pos] = indices[pos - 1];
            }
            pos--;
        }
        
        if (pos < (int)(g_crs_online.top_k)) {
            absmax[pos] = val;
            indices[pos] = ch;
            if (n_candidates < (int)(g_crs_online.top_k)) {
                n_candidates++;
            }
        }
    }
    
    // Apply threshold filter
    int n_outliers = 0;
    float threshold = g_crs_online.threshold;
    
    // Auto threshold: use median * 2 if threshold is 0
    if (threshold <= 0 && n_candidates > 0) {
        // Simple heuristic: outlier if > 2x the median
        int mid = n_candidates / 2;
        threshold = absmax[mid] * 2.0f;
    }
    
    for (int i = 0; i < n_candidates && n_outliers < (int)(g_crs_online.top_k); i++) {
        if (absmax[i] > threshold) {
            out_indices[n_outliers] = indices[i];
            out_scales[n_outliers] = absmax[i];
            n_outliers++;
        }
    }
    
    return n_outliers;
}

void crs_apply_online_scale(
    float * k_data, int n_dims,
    int n_outliers, const int32_t * indices, const float * scales) 
{
    if (!k_data || n_outliers <= 0 || !indices || !scales) {
        return;
    }
    
    for (int i = 0; i < n_outliers; i++) {
        int ch = indices[i];
        if (ch >= 0 && ch < n_dims && scales[i] > 1e-6f) {
            k_data[ch] /= scales[i];
        }
    }
}

void crs_restore_scale(
    float * k_data, int n_dims,
    int layer, int head,
    int n_online_outliers, const int32_t * online_indices, const float * online_scales) 
{
    if (!k_data) {
        return;
    }
    
    // Restore static CRS scales
    if (g_crs_static.enabled &&
        layer >= 0 && layer < (int)g_crs_static.n_layers &&
        head >= 0 && head < (int)g_crs_static.n_heads) 
    {
        const crs_head_config_t * cfg = &g_crs_static.configs[layer][head];
        for (int i = 0; i < cfg->n_outliers; i++) {
            int ch = cfg->indices[i];
            if (ch >= 0 && ch < n_dims) {
                k_data[ch] *= cfg->scales[i];
            }
        }
    }
    
    // Restore online CRS scales
    if (n_online_outliers > 0 && online_indices && online_scales) {
        for (int i = 0; i < n_online_outliers; i++) {
            int ch = online_indices[i];
            if (ch >= 0 && ch < n_dims) {
                k_data[ch] *= online_scales[i];
            }
        }
    }
}
