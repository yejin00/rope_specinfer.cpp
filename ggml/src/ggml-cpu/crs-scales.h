/**
 * CRS (Channel-wise Row Scaling) Scale Management
 * 
 * Two-stage CRS for KV Cache Quantization:
 * 
 * Stage 1: Static CRS (Offline)
 *   - Pre-computed scales for channels with persistently large values (pre-RoPE)
 *   - Loaded from file at initialization
 *   - Applied during K cache quantization
 * 
 * Stage 2: Online CRS (Dynamic)
 *   - Runtime detection of RoPE-induced outliers
 *   - Dynamic scale computation based on channel-wise absmax
 *   - Applied per-token during inference
 */

#ifndef GGML_CRS_SCALES_H
#define GGML_CRS_SCALES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CRS_MAGIC_STATIC 0x53435253  // "SCRS" - Static CRS
#define CRS_MAX_LAYERS 64
#define CRS_MAX_HEADS 32
#define CRS_MAX_OUTLIERS 64

/**
 * Static CRS configuration per head
 */
typedef struct {
    int32_t  n_outliers;                    // Number of outlier channels
    int32_t  indices[CRS_MAX_OUTLIERS];     // Outlier channel indices
    float    scales[CRS_MAX_OUTLIERS];      // Scale values (absmax)
} crs_head_config_t;

/**
 * Static CRS configuration for entire model
 */
typedef struct {
    bool     enabled;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_dims;
    uint32_t top_k;
    crs_head_config_t configs[CRS_MAX_LAYERS][CRS_MAX_HEADS];
} crs_static_config_t;

/**
 * Online CRS state for dynamic outlier tracking
 */
typedef struct {
    bool     enabled;
    uint32_t top_k;                         // Number of channels to track
    float    threshold;                     // Absmax threshold for outlier detection
} crs_online_config_t;

// Global CRS state
extern crs_static_config_t g_crs_static;
extern crs_online_config_t g_crs_online;

/**
 * Initialize CRS system
 */
void crs_init(void);

/**
 * Load static CRS scales from file
 * 
 * @param path Path to scales_static_k.bin file
 * @return 0 on success, -1 on error
 */
int crs_load_static_scales(const char * path);

/**
 * Enable online CRS with specified parameters
 * 
 * @param top_k Number of top outlier channels to track per head
 * @param threshold Absmax threshold (0 = auto)
 */
void crs_enable_online(uint32_t top_k, float threshold);

/**
 * Get static CRS scale for a specific channel
 * 
 * @param layer Layer index
 * @param head Head index
 * @param channel Channel index
 * @return Scale value (1.0 if not an outlier)
 */
float crs_get_static_scale(int layer, int head, int channel);

/**
 * Check if a channel is a static outlier
 * 
 * @param layer Layer index
 * @param head Head index
 * @param channel Channel index
 * @return true if channel is a static outlier
 */
bool crs_is_static_outlier(int layer, int head, int channel);

/**
 * Apply static CRS scaling to K values before quantization
 * 
 * K_scaled[ch] = K[ch] / static_scale[ch]  (for outlier channels)
 * 
 * @param layer Layer index
 * @param head Head index  
 * @param k_data Pointer to K values (will be modified in-place)
 * @param n_dims Number of dimensions (channels)
 */
void crs_apply_static_scale(int layer, int head, float * k_data, int n_dims);

/**
 * Compute online CRS scales from current K values
 * 
 * For each head, computes channel-wise absmax and identifies
 * top-k outlier channels that need scaling.
 * 
 * @param layer Layer index
 * @param head Head index
 * @param k_data Pointer to K values
 * @param n_dims Number of dimensions
 * @param out_indices Output: outlier channel indices
 * @param out_scales Output: scale values
 * @return Number of outlier channels detected
 */
int crs_compute_online_scales(
    int layer, int head,
    const float * k_data, int n_dims,
    int32_t * out_indices, float * out_scales);

/**
 * Apply online CRS scaling to K values
 * 
 * @param k_data Pointer to K values (will be modified in-place)
 * @param n_dims Number of dimensions
 * @param n_outliers Number of outlier channels
 * @param indices Outlier channel indices
 * @param scales Scale values
 */
void crs_apply_online_scale(
    float * k_data, int n_dims,
    int n_outliers, const int32_t * indices, const float * scales);

/**
 * Restore CRS scaling during attention computation
 * 
 * K_restored[ch] = K_quantized[ch] * scale[ch]
 * 
 * This is called during dequantization or dot product computation.
 */
void crs_restore_scale(
    float * k_data, int n_dims,
    int layer, int head,
    int n_online_outliers, const int32_t * online_indices, const float * online_scales);

#ifdef __cplusplus
}
#endif

#endif // GGML_CRS_SCALES_H
