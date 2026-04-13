// FlashAttention kernel for mixed precision (Q=F32, K/V=F16)
// K/V are read from image1d_buffer_t for texture cache optimization

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE float
#define Q_DATA_TYPE4 float4
#define O_DATA_TYPE float
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half

#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
#define Q1_WG_SIZE 64

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
// Prefill Kernel - K/V read from image1d_buffer_t
// =================================================================================================
__kernel void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    __read_only image1d_buffer_t k_img,  // K as image (CL_RGBA, CL_HALF_FLOAT)
    __read_only image1d_buffer_t v_img,  // V as image (CL_RGBA, CL_HALF_FLOAT)
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const int k_nb1_vec4,  // k_nb1 / (4 * sizeof(half)) = stride in texels per row
    const int k_nb2_vec4,  // k_nb2 / (4 * sizeof(half)) = stride in texels per head
    const int k_nb3_vec4,  // k_nb3 / (4 * sizeof(half)) = stride in texels per batch
    const int v_nb1_vec4,
    const int v_nb2_vec4,
    const int v_nb3_vec4,
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
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    global char* o_base = (global char*)o_void + o_offset;

    // Base texel index for K/V access
    const int kv_base_texel = batch_idx * k_nb3_vec4 + head_kv_idx * k_nb2_vec4;
    const int kv_base_texel_v = batch_idx * v_nb3_vec4 + head_kv_idx * v_nb2_vec4;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // Load Q into private registers
    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }

    // Output accumulator
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // Local memory for K/V tiles
    __local half4 l_k[BLOCK_N][DK_VEC];
    __local half4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        // Cooperative load K tile from image
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                // Read from image: texel index = base + row * row_stride + col
                int texel_idx = kv_base_texel + k_row_idx * k_nb1_vec4 + col;
                float4 k4 = read_imagef(k_img, texel_idx);
                l_k[row][col] = convert_half4(k4);
            }
        }
        // Cooperative load V tile from image
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                int texel_idx = kv_base_texel_v + v_row_idx * v_nb1_vec4 + col;
                float4 v4 = read_imagef(v_img, texel_idx);
                l_v[row][col] = convert_half4(v4);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            barrier(CLK_LOCAL_MEM_FENCE);
            continue;
        }

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], convert_float4(l_k[j][k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], convert_float4(l_k[j+1][k]), dot_acc1);
            }
            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
            }

            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p0 * convert_float4(l_v[j][i]) + p1 * convert_float4(l_v[j+1][i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
}

// =================================================================================================
// Decoding Kernel (q=1) - K/V read from image1d_buffer_t
// =================================================================================================
__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    __read_only image1d_buffer_t k_img,
    __read_only image1d_buffer_t v_img,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const int k_nb1_vec4,
    const int k_nb2_vec4,
    const int k_nb3_vec4,
    const int v_nb1_vec4,
    const int v_nb2_vec4,
    const int v_nb3_vec4,
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

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    global char* o_base = (global char*)o_void + o_offset;

    // Base texel index for K/V access
    const int kv_base_texel = batch_idx * k_nb3_vec4 + head_kv_idx * k_nb2_vec4;
    const int kv_base_texel_v = batch_idx * v_nb3_vec4 + head_kv_idx * v_nb2_vec4;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // Load Q into private registers
    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    // First pass: compute max score
    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            int texel_idx = kv_base_texel + k_idx * k_nb1_vec4 + k;
            float4 k4 = read_imagef(k_img, texel_idx);
            dot_acc = mad(q_priv[k], k4, dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    // Workgroup reduction for max
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m_i = local_m[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Second pass: compute softmax and weighted sum
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE l_i = (sinks_ptr != NULL) ? exp(sinks_ptr[head_idx] - m_i) : 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            int texel_idx = kv_base_texel + k_idx * k_nb1_vec4 + k;
            float4 k4 = read_imagef(k_img, texel_idx);
            dot_acc = mad(q_priv[k], k4, dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        ACC_TYPE p = exp(score - m_i);
        l_i += p;

        #pragma unroll
        for (int v = 0; v < DV_VEC; v++) {
            int texel_idx = kv_base_texel_v + k_idx * v_nb1_vec4 + v;
            float4 v4 = read_imagef(v_img, texel_idx);
            o_acc[v] = mad((ACC_TYPE4)(p), v4, o_acc[v]);
        }
    }

    // Workgroup reduction for l_i and o_acc
    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_partial[Q1_WG_SIZE];

    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    ACC_TYPE l_final = local_l[0];

    // Reduce each vec4 of O separately
    for (int d = 0; d < DV_VEC; d++) {
        local_o_partial[tid] = o_acc[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o_partial[tid] += local_o_partial[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Thread 0 writes output
        if (tid == 0) {
            const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
            global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
            if (l_final > 0.0f) {
                o_row[d] = CONVERT_O_DATA4(local_o_partial[0] / l_final);
            } else {
                o_row[d] = (O_DATA_TYPE4)(0.0f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}