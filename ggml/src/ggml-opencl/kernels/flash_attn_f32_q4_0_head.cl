// =================================================================================================
// Flash Attention f32/q4_0_head - Per-head KV cache quantization (block_size=128)
// =================================================================================================
// Adapted from flash_attn_f32_q4_0.cl with QK=128 instead of 32.
// For DK=128, each head maps to exactly 1 block (1 scale per head).
// =================================================================================================

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_intel_subgroups
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_64
#define REQD_SUBGROUP_SIZE_128
#endif

typedef uchar uint8_t;

#define QK4_0_HEAD 128

struct block_q4_0_head {
    half d;
    uint8_t qs[QK4_0_HEAD / 2];
};

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_O_DATA4(x) (x)

#ifndef DK
#define DK 128
#endif

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)

#define DEC_WG_SIZE 64

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}

// =================================================================================================
// q4_0_head dequantization helpers (block_size=128, nibble split at 64)
// =================================================================================================

// Dequantize 4 consecutive elements at vec4 index 'col' (col = element_offset / 4)
// QK4_0_HEAD=128 => 32 vec4s per block, first 16 vec4s = low nibbles, last 16 = high nibbles
inline float4 dequant_q4_0_head_4(const global struct block_q4_0_head * row_blocks, int col) {
    const int bi  = col / 32;     // 128/4 = 32 vec4s per block
    const int vi  = col % 32;
    const global struct block_q4_0_head * b = &row_blocks[bi];
    const float d = (float)b->d;

    const int eb = vi * 4;
    if (vi < 16) {
        return (float4)(
            ((int)(b->qs[eb + 0] & 0x0F) - 8) * d,
            ((int)(b->qs[eb + 1] & 0x0F) - 8) * d,
            ((int)(b->qs[eb + 2] & 0x0F) - 8) * d,
            ((int)(b->qs[eb + 3] & 0x0F) - 8) * d
        );
    } else {
        const int qb = eb - 64;
        return (float4)(
            ((int)(b->qs[qb + 0] >> 4) - 8) * d,
            ((int)(b->qs[qb + 1] >> 4) - 8) * d,
            ((int)(b->qs[qb + 2] >> 4) - 8) * d,
            ((int)(b->qs[qb + 3] >> 4) - 8) * d
        );
    }
}

// =================================================================================================
// 1. Optimized Prefill Kernel - K-only Local Memory, V Global Direct Access
// =================================================================================================
__kernel void flash_attn_f32_q4_0_head_old(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid            = get_local_id(0);
    const int block_q_idx    = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global       char* o_base = (global       char*)o_void + o_offset;

    const bool valid_query = (my_query_row < n_q);

    const global MASK_DATA_TYPE* mask_ptr = NULL;
    if (mask_void != NULL && valid_query) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char* mask_base =
            (const global char*)mask_void + mask_offset
            + (ulong)mask_batch_idx * mask_nb3
            + (ulong)mask_head_idx  * mask_nb2
            + (ulong)my_query_row   * mask_nb1;
        mask_ptr = (const global MASK_DATA_TYPE*)mask_base;
    }

    float4 q_priv[DK_VEC];
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = (float4)(0.0f);
    }

    if (valid_query) {
        const ulong q_row_offset =
            (ulong)batch_idx    * q_nb3 +
            (ulong)head_idx     * q_nb2 +
            (ulong)my_query_row * q_nb1;
        const global float4* q_ptr = (const global float4*)(q_base + q_row_offset);

        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = q_ptr[i];
        }
    }

    float4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (float4)(0.0f);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    const int causal_limit = is_causal ? (n_kv - n_q + my_query_row) : n_kv;

    __local half4 l_k[BLOCK_N][DK_VEC];

    const ulong v_base_offset = (ulong)batch_idx * v_nb3 + (ulong)head_kv_idx * v_nb2;
    const ulong k_base_offset = (ulong)batch_idx * k_nb3 + (ulong)head_kv_idx * k_nb2;

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        const int k_tile_end = min(k_start + BLOCK_N, n_kv);
        const int k_tile_size = k_tile_end - k_start;

        // K tile -> local memory (q4_0_head dequant -> half4)
        #pragma unroll 1
        for (int idx = tid; idx < BLOCK_N * DK_VEC; idx += WG_SIZE) {
            const int row = idx / DK_VEC;
            const int col = idx % DK_VEC;

            half4 k_val = (half4)(0.0h);
            if (row < k_tile_size) {
                const ulong k_row_offset = k_base_offset + (k_start + row) * k_nb1;
                const global struct block_q4_0_head * k_blocks =
                    (const global struct block_q4_0_head *)(k_base + k_row_offset);

                // 32 vec4s per block, first 16 = low nibbles, last 16 = high nibbles
                const int block_idx = col / 32;
                const int vec_idx = col % 32;
                const global struct block_q4_0_head * kb = &k_blocks[block_idx];
                const half d = kb->d;

                const int elem_base = vec_idx * 4;
                if (vec_idx < 16) {
                    k_val = (half4)(
                        (half)(((int)(kb->qs[elem_base + 0] & 0x0F) - 8) * d),
                        (half)(((int)(kb->qs[elem_base + 1] & 0x0F) - 8) * d),
                        (half)(((int)(kb->qs[elem_base + 2] & 0x0F) - 8) * d),
                        (half)(((int)(kb->qs[elem_base + 3] & 0x0F) - 8) * d)
                    );
                } else {
                    const int qs_base = elem_base - 64;
                    k_val = (half4)(
                        (half)(((int)(kb->qs[qs_base + 0] >> 4) - 8) * d),
                        (half)(((int)(kb->qs[qs_base + 1] >> 4) - 8) * d),
                        (half)(((int)(kb->qs[qs_base + 2] >> 4) - 8) * d),
                        (half)(((int)(kb->qs[qs_base + 3] >> 4) - 8) * d)
                    );
                }
            }
            l_k[row][col] = k_val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (valid_query) {
            #if BLOCK_N >= 16
                #define UNROLL_FACTOR 4
            #else
                #define UNROLL_FACTOR 2
            #endif

            #pragma unroll 1
            for (int j = 0; j < k_tile_size; j += UNROLL_FACTOR) {
                if (is_causal && (k_start + j) > causal_limit) {
                    break;
                }

                float scores[UNROLL_FACTOR];

                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    const int k_row = k_start + j + w;

                    if (k_row >= n_kv || (is_causal && k_row > causal_limit) || k_row < 0) {
                        scores[w] = -INFINITY;
                    } else if (j + w < k_tile_size) {
                        float score = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < DK_VEC; ++k) {
                            score += dot(q_priv[k], convert_float4(l_k[j + w][k]));
                        }
                        scores[w] = score * scale;
                    } else {
                        scores[w] = -INFINITY;
                    }
                }

                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    if (scores[w] == -INFINITY) continue;

                    const int k_row = k_start + j + w;
                    if (mask_ptr != NULL) {
                        scores[w] += slope * (float)mask_ptr[k_row];
                    }
                    if (logit_softcap > 0.0f) {
                        scores[w] = logit_softcap * tanh(scores[w] / logit_softcap);
                    }
                }

                float m_new = m_i;
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    m_new = fmax(m_new, scores[w]);
                }

                const float scale_prev = exp(m_i - m_new);

                float p[UNROLL_FACTOR];
                float p_sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    p[w] = exp(scores[w] - m_new);
                    p_sum += p[w];
                }

                // V accumulation - direct global access with on-the-fly q4_0_head dequant
                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    float4 v_acc = (float4)(0.0f);

                    #pragma unroll
                    for (int w = 0; w < UNROLL_FACTOR; ++w) {
                        const int k_row = k_start + j + w;
                        if (k_row < n_kv && p[w] > 0.0f) {
                            const global struct block_q4_0_head * v_blocks =
                                (const global struct block_q4_0_head *)(v_base + v_base_offset + k_row * v_nb1);
                            float4 v_val = dequant_q4_0_head_4(v_blocks, i);
                            v_acc = mad((float4)(p[w]), v_val, v_acc);
                        }
                    }

                    o_acc[i] = mad(v_acc, (float4)(1.0f), o_acc[i] * scale_prev);
                }

                l_i = l_i * scale_prev + p_sum;
                m_i = m_new;
            }

            #undef UNROLL_FACTOR
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid_query && l_i > 0.0f) {
        if (sinks_void != NULL) {
            const global float* sinks_ptr =
                (const global float*)((const global char*)sinks_void + sinks_offset);
            const float m_sink  = sinks_ptr[head_idx];
            const float m_final = fmax(m_i, m_sink);

            const float scale_o = exp(m_i - m_final);
            const float sink_contrib = exp(m_sink - m_final);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * scale_o + sink_contrib;
        }

        const ulong o_row_offset =
            (ulong)batch_idx    * o_nb3 +
            (ulong)my_query_row * o_nb2 +
            (ulong)head_idx     * o_nb1;
        global float4* o_row = (global float4*)(o_base + o_row_offset);

        const float l_inv = 1.0f / l_i;

        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) {
            o_row[i] = o_acc[i] * l_inv;
        }
    } else if (valid_query) {
        const ulong o_row_offset =
            (ulong)batch_idx    * o_nb3 +
            (ulong)my_query_row * o_nb2 +
            (ulong)head_idx     * o_nb1;
        global float4* o_row = (global float4*)(o_base + o_row_offset);

        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) {
            o_row[i] = (float4)(0.0f);
        }
    }
}


// =================================================================================================
// 2. GEMM-quality Decoding Kernel (Adreno 750, wave=64)
// =================================================================================================
#define DEC_KV_UNROLL 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void flash_attn_f32_q4_0_head_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int bi  = head_batch_idx / n_head;
    const int hi  = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);

    const int d = tid * 2;

    // ===== Base pointers (batch/head baked in) =====
    const global char* k_ptr_base = (const global char*)k_void + k_offset
                                  + (ulong)bi * k_nb3 + (ulong)hkv * k_nb2;
    const global char* v_ptr_base = (const global char*)v_void + v_offset
                                  + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2;

    // ===== Q -> register (scale pre-multiply) =====
    const global float* q_ptr = (const global float*)(
        (const global char*)q_void + q_offset
        + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);

#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    // ===== Dequant constants (loop-invariant, QK=128) =====
    const int k_bi  = d / QK4_0_HEAD;
    const int k_off = d % QK4_0_HEAD;
    const int k_hi  = (k_off >= 64);
    const int k_qi  = k_hi ? (k_off - 64) : k_off;
    const int k_shift = k_hi ? 4 : 0;

    const int v_bi  = d / QK4_0_HEAD;
    const int v_off = d % QK4_0_HEAD;
    const int v_hi  = (v_off >= 64);
    const int v_qi  = v_hi ? (v_off - 64) : v_off;
    const int v_shift = v_hi ? 4 : 0;

    // ===== State =====
    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    // ===== Mask =====
    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL) {
        mask_row = (const global MASK_DATA_TYPE*)(
            (const global char*)mask_void + mask_offset
            + (ulong)(bi % mask_ne3) * mask_nb3
            + (ulong)(hi % mask_ne2) * mask_nb2);
    }

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);

    // ===== Main loop: 4 KV tokens per iteration =====
    const int kv_batched = (n_kv / DEC_KV_UNROLL) * DEC_KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += DEC_KV_UNROLL) {
        float s[DEC_KV_UNROLL];

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
#if DK == 128
            const global struct block_q4_0_head * kb =
                (const global struct block_q4_0_head *)(k_ptr_base + (ulong)(ki + w) * k_nb1) + k_bi;
            const float kd = (float)kb->d;
            const float2 kv = (float2)(
                ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
                ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
            );
            s[w] = sub_group_reduce_add(dot(qv, kv));
#else
            float partial = 0.0f;
            if (d < DK) {
                const global struct block_q4_0_head * kb =
                    (const global struct block_q4_0_head *)(k_ptr_base + (ulong)(ki + w) * k_nb1) + k_bi;
                const float kd = (float)kb->d;
                const float2 kv = (float2)(
                    ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
                    ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
                );
                partial = dot(qv, kv);
            }
            s[w] = sub_group_reduce_add(partial);
#endif
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w)
                s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w)
                s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        mb = fmax(mb, s[1]); mb = fmax(mb, s[2]); mb = fmax(mb, s[3]);
        const float m_new = fmax(mi, mb);

        if (m_new > mi) {
            const float alpha = native_exp(mi - m_new);
            oacc *= alpha;
            li   *= alpha;
            mi    = m_new;
        }

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float p = native_exp(s[w] - mi);
            li += p;

#if DV == 128
            const global struct block_q4_0_head * vb =
                (const global struct block_q4_0_head *)(v_ptr_base + (ulong)(ki + w) * v_nb1) + v_bi;
            const float vd = (float)vb->d;
            const float2 vv = (float2)(
                ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
                ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
            );
            oacc = mad((float2)(p), vv, oacc);
#else
            if (d < DV) {
                const global struct block_q4_0_head * vb =
                    (const global struct block_q4_0_head *)(v_ptr_base + (ulong)(ki + w) * v_nb1) + v_bi;
                const float vd = (float)vb->d;
                const float2 vv = (float2)(
                    ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
                    ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
                );
                oacc = mad((float2)(p), vv, oacc);
            }
#endif
        }
    }

    // ===== Tail =====
    for (; ki < n_kv; ++ki) {
#if DK == 128
        const global struct block_q4_0_head * kb =
            (const global struct block_q4_0_head *)(k_ptr_base + (ulong)ki * k_nb1) + k_bi;
        const float kd = (float)kb->d;
        const float2 kv = (float2)(
            ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
            ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
        );
        float score = sub_group_reduce_add(dot(qv, kv));
#else
        float partial = 0.0f;
        if (d < DK) {
            const global struct block_q4_0_head * kb =
                (const global struct block_q4_0_head *)(k_ptr_base + (ulong)ki * k_nb1) + k_bi;
            const float kd = (float)kb->d;
            const float2 kv = (float2)(
                ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
                ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
            );
            partial = dot(qv, kv);
        }
        float score = sub_group_reduce_add(partial);
#endif
        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = fmax(mi, score);
        if (m_new > mi) {
            const float alpha = native_exp(mi - m_new);
            oacc *= alpha;
            li   *= alpha;
            mi    = m_new;
        }
        const float p = native_exp(score - mi);

#if DV == 128
        const global struct block_q4_0_head * vb =
            (const global struct block_q4_0_head *)(v_ptr_base + (ulong)ki * v_nb1) + v_bi;
        const float vd = (float)vb->d;
        const float2 vv = (float2)(
            ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
            ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
        );
        oacc = mad((float2)(p), vv, oacc);
#else
        if (d < DV) {
            const global struct block_q4_0_head * vb =
                (const global struct block_q4_0_head *)(v_ptr_base + (ulong)ki * v_nb1) + v_bi;
            const float vd = (float)vb->d;
            const float2 vv = (float2)(
                ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
                ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
            );
            oacc = mad((float2)(p), vv, oacc);
        }
#endif
        li += p;
    }

    // ===== Sinks =====
    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = fmax(mi, ms);
        const float so = (mi > -INFINITY) ? exp(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + exp(ms - mf);
    }

    // ===== Normalize + write =====
    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)(
        (global char*)o_void + o_offset
        + (ulong)bi * o_nb3 + (ulong)hi * o_nb1);

#if DV == 128
    o_row[d]     = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}


// =================================================================================================
// 3. Prefill Kernel - Single-Q, Barrier-Free (Adreno 750, wave=64)
// =================================================================================================
#define KV_UNROLL 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void flash_attn_f32_q4_0_head(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int bhq = get_global_id(1);
    const int bh  = bhq / n_q;
    const int qr  = bhq % n_q;

    const int bi  = bh / n_head;
    const int hi  = bh % n_head;
    const int hkv = hi / (n_head / n_head_kv);

    if (qr >= n_q) return;

    const int d = tid * 2;

    // ===== Base pointers =====
    const global char* k_ptr_base = (const global char*)k_void + k_offset
                                  + (ulong)bi * k_nb3 + (ulong)hkv * k_nb2;
    const global char* v_ptr_base = (const global char*)v_void + v_offset
                                  + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2;

    // ===== Q -> register (pre-scaled) =====
    const global float* q_row = (const global float*)(
        (const global char*)q_void + q_offset
        + (ulong)bi * q_nb3 + (ulong)hi * q_nb2 + (ulong)qr * q_nb1);

#if DK == 128
    const float2 qv = (float2)(q_row[d], q_row[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_row[d] * scale; if (d + 1 < DK) qv.y = q_row[d + 1] * scale; }
#endif

    // ===== Dequant constants (loop-invariant, QK=128) =====
    const int k_bi  = d / QK4_0_HEAD;
    const int k_off = d % QK4_0_HEAD;
    const int k_shift = (k_off >= 64) ? 4 : 0;
    const int k_qi  = (k_off >= 64) ? (k_off - 64) : k_off;

    const int v_bi  = d / QK4_0_HEAD;
    const int v_off = d % QK4_0_HEAD;
    const int v_shift = (v_off >= 64) ? 4 : 0;
    const int v_qi  = (v_off >= 64) ? (v_off - 64) : v_off;

    // ===== State =====
    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    // ===== Mask =====
    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL) {
        mask_row = (const global MASK_DATA_TYPE*)(
            (const global char*)mask_void + mask_offset
            + (ulong)(bi % mask_ne3) * mask_nb3
            + (ulong)(hi % mask_ne2) * mask_nb2
            + (ulong)qr * mask_nb1);
    }

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);
    const int kv_end = is_causal ? min(n_kv, n_kv - n_q + qr + 1) : n_kv;

    // ===== Main loop: KV_UNROLL=4 =====
    const int kv_batched = (kv_end / KV_UNROLL) * KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += KV_UNROLL) {
        float s[KV_UNROLL];

        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
#if DK == 128
            const global struct block_q4_0_head * kb =
                (const global struct block_q4_0_head *)(k_ptr_base + (ulong)(ki + w) * k_nb1) + k_bi;
            const float kd = (float)kb->d;
            const float2 kv = (float2)(
                ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
                ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
            );
            s[w] = sub_group_reduce_add(dot(qv, kv));
#else
            float partial = 0.0f;
            if (d < DK) {
                const global struct block_q4_0_head * kb =
                    (const global struct block_q4_0_head *)(k_ptr_base + (ulong)(ki + w) * k_nb1) + k_bi;
                const float kd = (float)kb->d;
                const float2 kv = (float2)(
                    ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
                    ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
                );
                partial = dot(qv, kv);
            }
            s[w] = sub_group_reduce_add(partial);
#endif
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < KV_UNROLL; ++w)
                s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < KV_UNROLL; ++w)
                s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        mb = fmax(mb, s[1]); mb = fmax(mb, s[2]); mb = fmax(mb, s[3]);
        const float m_new = fmax(mi, mb);

        if (m_new > mi) {
            const float alpha = native_exp(mi - m_new);
            oacc *= alpha; li *= alpha; mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
            const float p = native_exp(s[w] - mi);
            li += p;
#if DV == 128
            const global struct block_q4_0_head * vb =
                (const global struct block_q4_0_head *)(v_ptr_base + (ulong)(ki + w) * v_nb1) + v_bi;
            const float vd = (float)vb->d;
            const float2 vv = (float2)(
                ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
                ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
            );
            oacc = mad((float2)(p), vv, oacc);
#else
            if (d < DV) {
                const global struct block_q4_0_head * vb =
                    (const global struct block_q4_0_head *)(v_ptr_base + (ulong)(ki + w) * v_nb1) + v_bi;
                const float vd = (float)vb->d;
                const float2 vv = (float2)(
                    ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
                    ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
                );
                oacc = mad((float2)(p), vv, oacc);
            }
#endif
        }
    }

    // ===== Tail =====
    for (; ki < kv_end; ++ki) {
#if DK == 128
        const global struct block_q4_0_head * kb =
            (const global struct block_q4_0_head *)(k_ptr_base + (ulong)ki * k_nb1) + k_bi;
        const float kd = (float)kb->d;
        const float2 kv = (float2)(
            ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
            ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
        );
        float score = sub_group_reduce_add(dot(qv, kv));
#else
        float partial = 0.0f;
        if (d < DK) {
            const global struct block_q4_0_head * kb =
                (const global struct block_q4_0_head *)(k_ptr_base + (ulong)ki * k_nb1) + k_bi;
            const float kd = (float)kb->d;
            const float2 kv = (float2)(
                ((int)((kb->qs[k_qi]     >> k_shift) & 0x0F) - 8) * kd,
                ((int)((kb->qs[k_qi + 1] >> k_shift) & 0x0F) - 8) * kd
            );
            partial = dot(qv, kv);
        }
        float score = sub_group_reduce_add(partial);
#endif
        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = fmax(mi, score);
        if (m_new > mi) {
            const float alpha = native_exp(mi - m_new);
            oacc *= alpha; li *= alpha; mi = m_new;
        }
        const float p = native_exp(score - mi);

#if DV == 128
        const global struct block_q4_0_head * vb =
            (const global struct block_q4_0_head *)(v_ptr_base + (ulong)ki * v_nb1) + v_bi;
        const float vd = (float)vb->d;
        const float2 vv = (float2)(
            ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
            ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
        );
        oacc = mad((float2)(p), vv, oacc);
#else
        if (d < DV) {
            const global struct block_q4_0_head * vb =
                (const global struct block_q4_0_head *)(v_ptr_base + (ulong)ki * v_nb1) + v_bi;
            const float vd = (float)vb->d;
            const float2 vv = (float2)(
                ((int)((vb->qs[v_qi]     >> v_shift) & 0x0F) - 8) * vd,
                ((int)((vb->qs[v_qi + 1] >> v_shift) & 0x0F) - 8) * vd
            );
            oacc = mad((float2)(p), vv, oacc);
        }
#endif
        li += p;
    }

    // ===== Sinks =====
    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = fmax(mi, ms);
        const float so = (mi > -INFINITY) ? exp(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + exp(ms - mf);
    }

    // ===== Normalize + write =====
    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)(
        (global char*)o_void + o_offset
        + (ulong)bi * o_nb3 + (ulong)qr * o_nb2 + (ulong)hi * o_nb1);

#if DV == 128
    o_row[d]     = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}
