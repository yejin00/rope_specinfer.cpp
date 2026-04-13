// =================================================================================================
// Ablation Study Kernels for Flash Attention f32/f16
// =================================================================================================
// 빌드: cmake -DGGML_OPENCL_USE_ABLATION=ON -DABLATION_VARIANT=N
// ggml-opencl.cpp: GGML_OPENCL_USE_ABLATION 시 flash_attn_f32_f16 대신 이 파일 로드
//
// 표 채우기용 Variant 매핑:
// | 적용된 최적화 기법              | Prefill | Decode | k_img | 비고 |
// |--------------------------------|---------|--------|-------|------|
// | 0 Base (Naive FA, llama.cpp)   | block_m | 2-pass | X     | 63.81/8.38 TPS 기준 |
// | 1 +1-Pass Decode & Subgroup    | block_m | 1-pass | X     | sub_group_reduce_add, V global |
// | 2 +Texture Cache for K        | K tex   | K tex  | O     | L1 cache, L2 분산 |
// | 3 +Direct Global Access for V | V direct| V direct| O    | l_v 제거 |
// | 4 +1-Query per Wave Launcher  | 1Q/wave | 동일   | O     | Prefill launch 변경 |
// | 5 +Native Math (fmax, exp)     | 동일    | 동일   | O     | 76.82/12.56 TPS 목표 |
//
// 의존성: 2→3→4→5는 누적. 1은 Decode만 변경(Prefill Base 유지).
// =================================================================================================

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SHUFFLE 1
#else
#define HAS_SHUFFLE 0
#endif

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half

#ifndef DK
#define DK 128
#endif

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
#define Q1_WG_SIZE 64

#ifndef ABLATION_VARIANT
#define ABLATION_VARIANT 0
#endif

// Native math: variant 5 이상에서 사용
#if ABLATION_VARIANT >= 5
#define ABL_FMAX(x,y) fmax((x),(y))
#define ABL_EXP(x) native_exp(x)
#else
#define ABL_FMAX(x,y) max((x),(y))
#define ABL_EXP(x) exp(x)
#endif

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) return 1.0f;
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
    return pow(base, exph);
}

// =================================================================================================
// ABLATION VARIANT 0: Base (Naive FA, llama.cpp style)
// - 2-pass: K/V local load -> barrier -> compute
// - Prefill: block_m tiling, BLOCK_N tiles
// - Decode: 2-pass (max pass + softmax pass), barrier
// - k_img, k_offset_t 등은 사용하지 않으나 ggml-opencl.cpp와 시그니처 통일을 위해 포함
// =================================================================================================
#if ABLATION_VARIANT == 0

__kernel void flash_attn_ablation_prefill(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    (void)k_img; (void)k_offset_t; (void)k_nb1_t; (void)k_nb2_t; (void)k_nb3_t;
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);
    const int my_query_row = block_q_idx * BLOCK_M + tid;
    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) q_priv[i] = q_ptr[i];
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC, col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((const global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC, col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((const global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) continue;

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j, k_row1 = k_start + j + 1;
            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f), dot_acc1 = (ACC_TYPE4)(0.0f);
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

            const ACC_TYPE m_new = ABL_FMAX(m_i, ABL_FMAX(score0, score1));
            const ACC_TYPE p0 = ABL_EXP(score0 - m_new);
            const ACC_TYPE p1 = ABL_EXP(score1 - m_new);
            const ACC_TYPE scale_prev = ABL_EXP(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i)
                o_acc[i] = o_acc[i] * scale_prev + p0 * convert_float4(l_v[j][i]) + p1 * convert_float4(l_v[j+1][i]);
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = ABL_FMAX(m_i, m_sink);
            const ACC_TYPE scale_o = ABL_EXP(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * scale_o + ABL_EXP(m_sink - m_final);
        }
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = o_acc[i] * l_inv;
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
        }
    }
}

__kernel void flash_attn_ablation_decode(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    (void)k_img; (void)k_offset_t; (void)k_nb1_t; (void)k_nb2_t; (void)k_nb3_t;
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) q_priv[i] = q_ptr[i];

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    const global ACC_TYPE* sinks_ptr = sinks_void != NULL ? (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset) : NULL;

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) dot_acc = mad(q_priv[k], convert_float4(k_ptr[k]), dot_acc);
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) score += slope * (ACC_TYPE)((const global MASK_DATA_TYPE*)mask_base)[k_idx];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);
        m_i = ABL_FMAX(m_i, score);
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = ABL_FMAX(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) dot_acc = mad(q_priv[k], convert_float4(k_ptr[k]), dot_acc);
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) score += slope * (ACC_TYPE)((const global MASK_DATA_TYPE*)mask_base)[k_idx];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);
        const ACC_TYPE p = ABL_EXP(score - m_final);
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) o_acc[i] = mad(p, convert_float4(v_ptr[i]), o_acc[i]);
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];
    if (sinks_ptr != NULL) l_final += ABL_EXP(sinks_ptr[head_idx] - m_final);

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) o_row[i] = local_o_comp[0] * l_inv;
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

#endif // ABLATION_VARIANT == 0

// =================================================================================================
// ABLATION VARIANT 1: + 1-Pass Decode & Subgroup
// - Decode: 1-pass (max+softmax+accum 동시), sub_group_reduce_add
// - Prefill: Base와 동일 (block_m tiling)
// =================================================================================================
#if ABLATION_VARIANT == 1

#define DEC_KV_UNROLL 4

__kernel void flash_attn_ablation_prefill(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    (void)k_img; (void)k_offset_t; (void)k_nb1_t; (void)k_nb2_t; (void)k_nb3_t;
    // Prefill: Base와 동일 (Variant 0 prefill 재사용 로직)
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);
    const int my_query_row = block_q_idx * BLOCK_M + tid;
    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) q_priv[i] = q_ptr[i];
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC, col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((const global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC, col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((const global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) continue;

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j, k_row1 = k_start + j + 1;
            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f), dot_acc1 = (ACC_TYPE4)(0.0f);
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

            const ACC_TYPE m_new = ABL_FMAX(m_i, ABL_FMAX(score0, score1));
            const ACC_TYPE p0 = ABL_EXP(score0 - m_new);
            const ACC_TYPE p1 = ABL_EXP(score1 - m_new);
            const ACC_TYPE scale_prev = ABL_EXP(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i)
                o_acc[i] = o_acc[i] * scale_prev + p0 * convert_float4(l_v[j][i]) + p1 * convert_float4(l_v[j+1][i]);
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = ABL_FMAX(m_i, m_sink);
            const ACC_TYPE scale_o = ABL_EXP(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * scale_o + ABL_EXP(m_sink - m_final);
        }
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = o_acc[i] * l_inv;
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
        }
    }
}

// 1-Pass Decode: sub_group_reduce_add, K/V global 직접 접근 (local 없음)
__kernel void flash_attn_ablation_decode(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    (void)k_img; (void)k_offset_t; (void)k_nb1_t; (void)k_nb2_t; (void)k_nb3_t;
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int bi = head_batch_idx / n_head;
    const int hi = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);
    const int d = tid * 2;

    const global char* k_ptr_base = (const global char*)k_void + k_offset + (ulong)bi * k_nb3 + (ulong)hkv * k_nb2;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_ptr = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);
#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);

    const int kv_batched = (n_kv / DEC_KV_UNROLL) * DEC_KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += DEC_KV_UNROLL) {
        float s[DEC_KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
#if DK == 128
            const global half* kr = (const global half*)(k_ptr_base + (ulong)(ki + w) * k_nb1);
            s[w] = sub_group_reduce_add(dot(qv, vload_half2(0, kr + d)));
#else
            const global half* kr = (const global half*)(k_ptr_base + (ulong)(ki + w) * k_nb1);
            float partial = 0.0f;
            if (d < DK) partial = dot(qv, vload_half2(0, kr + d));
            s[w] = sub_group_reduce_add(partial);
#endif
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        #pragma unroll
        for (int w = 1; w < DEC_KV_UNROLL; ++w) mb = ABL_FMAX(mb, s[w]);
        const float m_new = ABL_FMAX(mi, mb);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < n_kv; ++ki) {
#if DK == 128
        const global half* kr = (const global half*)(k_ptr_base + (ulong)ki * k_nb1);
        float score = sub_group_reduce_add(dot(qv, vload_half2(0, kr + d)));
#else
        const global half* kr = (const global half*)(k_ptr_base + (ulong)ki * k_nb1);
        float partial = 0.0f;
        if (d < DK) partial = dot(qv, vload_half2(0, kr + d));
        float score = sub_group_reduce_add(partial);
#endif
        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

#endif // ABLATION_VARIANT == 1

// =================================================================================================
// ABLATION VARIANT 2: + Texture Cache for K
// - K를 image1d_buffer_t로 읽어 L1 cache 활용
// - Decode: k_img 사용, Prefill: Base와 동일 (또는 K texture - Prefill도 적용 가능)
// =================================================================================================
#if ABLATION_VARIANT == 2

#define DEC_KV_UNROLL 4

__kernel void flash_attn_ablation_prefill(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    // Prefill: K texture 사용 (V는 아직 local - variant 3에서 direct)
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);
    const int my_query_row = block_q_idx * BLOCK_M + tid;
    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const int k_base_t = k_offset_t + batch_idx * k_nb3_t + head_kv_idx * k_nb2_t;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) q_priv[i] = q_ptr[i];
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC, col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                int texel_idx = k_base_t + k_row_idx * k_nb1_t + col;
                float4 kf = read_imagef(k_img, texel_idx);
                l_k[row][col] = convert_half4(kf);
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC, col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((const global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) continue;

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j, k_row1 = k_start + j + 1;
            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f), dot_acc1 = (ACC_TYPE4)(0.0f);
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

            const ACC_TYPE m_new = ABL_FMAX(m_i, ABL_FMAX(score0, score1));
            const ACC_TYPE p0 = ABL_EXP(score0 - m_new);
            const ACC_TYPE p1 = ABL_EXP(score1 - m_new);
            const ACC_TYPE scale_prev = ABL_EXP(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i)
                o_acc[i] = o_acc[i] * scale_prev + p0 * convert_float4(l_v[j][i]) + p1 * convert_float4(l_v[j+1][i]);
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = ABL_FMAX(m_i, m_sink);
            const ACC_TYPE scale_o = ABL_EXP(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * scale_o + ABL_EXP(m_sink - m_final);
        }
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = o_acc[i] * l_inv;
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
        }
    }
}

__kernel void flash_attn_ablation_decode(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int bi = head_batch_idx / n_head;
    const int hi = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);
    const int d = tid * 2;

    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_ptr = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);
#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);

    const int kv_batched = (n_kv / DEC_KV_UNROLL) * DEC_KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += DEC_KV_UNROLL) {
        float s[DEC_KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        #pragma unroll
        for (int w = 1; w < DEC_KV_UNROLL; ++w) mb = ABL_FMAX(mb, s[w]);
        const float m_new = ABL_FMAX(mi, mb);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < n_kv; ++ki) {
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));

        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

#endif // ABLATION_VARIANT == 2

// =================================================================================================
// ABLATION VARIANT 3: + Direct Global Access for V
// - V를 local memory 복사 없이 global에서 직접 읽기 (L2 cache만 사용)
// - Prefill: K texture, V global direct
// =================================================================================================
#if ABLATION_VARIANT == 3

#define DEC_KV_UNROLL 4

__kernel void flash_attn_ablation_prefill(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
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
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const int k_base_t = k_offset_t + batch_idx * k_nb3_t + head_kv_idx * k_nb2_t;
    const ulong v_base_offset = (ulong)batch_idx * v_nb3 + (ulong)head_kv_idx * v_nb2;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) q_priv[i] = q_ptr[i];
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    const int causal_limit = is_causal ? (n_kv - n_q + my_query_row) : n_kv;

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        const int k_tile_end = min(k_start + BLOCK_N, n_kv);
        const int k_tile_size = k_tile_end - k_start;

        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC, col = i % DK_VEC;
            if (row < k_tile_size) {
                int texel_idx = k_base_t + (k_start + row) * k_nb1_t + col;
                float4 kf = read_imagef(k_img, texel_idx);
                l_k[row][col] = convert_half4(kf);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) { barrier(CLK_LOCAL_MEM_FENCE); continue; }

        #if BLOCK_N >= 32
            #define V3_UNROLL 4
        #elif BLOCK_N >= 16
            #define V3_UNROLL 4
        #else
            #define V3_UNROLL 2
        #endif

        #pragma unroll 1
        for (int j = 0; j < k_tile_size; j += V3_UNROLL) {
            if (is_causal && (k_start + j) > causal_limit) break;

            ACC_TYPE scores[V3_UNROLL];
            #pragma unroll
            for (int w = 0; w < V3_UNROLL; ++w) {
                const int k_row = k_start + j + w;
                if (k_row >= n_kv || (is_causal && k_row > causal_limit) || k_row < 0) {
                    scores[w] = -INFINITY;
                } else if (j + w < k_tile_size) {
                    ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
                    #pragma unroll
                    for (int k = 0; k < DK_VEC; k++)
                        dot_acc = mad(q_priv[k], convert_float4(l_k[j + w][k]), dot_acc);
                    scores[w] = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
                } else {
                    scores[w] = -INFINITY;
                }
            }

            #pragma unroll
            for (int w = 0; w < V3_UNROLL; ++w) {
                const int k_row = k_start + j + w;
                if (scores[w] == -INFINITY) continue;
                if (mask_base != NULL) scores[w] += slope * (ACC_TYPE)((const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1))[k_row];
                if (logit_softcap > 0.0f) scores[w] = logit_softcap * tanh(scores[w] / logit_softcap);
            }

            ACC_TYPE m_new = m_i;
            #pragma unroll
            for (int w = 0; w < V3_UNROLL; ++w) m_new = ABL_FMAX(m_new, scores[w]);

            const ACC_TYPE scale_prev = ABL_EXP(m_i - m_new);
            ACC_TYPE p_arr[4];
            ACC_TYPE p_sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < V3_UNROLL; ++w) {
                p_arr[w] = (k_start + j + w < n_kv && j + w < k_tile_size && scores[w] > -INFINITY)
                    ? ABL_EXP(scores[w] - m_new) : 0.0f;
                p_sum += p_arr[w];
            }

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                float4 v_acc = (float4)(0.0f);
                #pragma unroll
                for (int w = 0; w < V3_UNROLL; ++w) {
                    const int k_row = k_start + j + w;
                    if (k_row < n_kv && j + w < k_tile_size && p_arr[w] > 0.0f) {
                        const global half4* vp = (const global half4*)(v_base + v_base_offset + k_row * v_nb1);
                        v_acc = mad((float4)(p_arr[w]), convert_float4(vp[i]), v_acc);
                    }
                }
                o_acc[i] = mad(v_acc, (float4)(1.0f), o_acc[i] * scale_prev);
            }
            l_i = l_i * scale_prev + p_sum;
            m_i = m_new;
        }
        #undef V3_UNROLL
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = ABL_FMAX(m_i, m_sink);
            const ACC_TYPE scale_o = ABL_EXP(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * scale_o + ABL_EXP(m_sink - m_final);
        }
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = o_acc[i] * l_inv;
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
        }
    }
}

__kernel void flash_attn_ablation_decode(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int bi = head_batch_idx / n_head;
    const int hi = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);
    const int d = tid * 2;

    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_ptr = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);
#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);

    const int kv_batched = (n_kv / DEC_KV_UNROLL) * DEC_KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += DEC_KV_UNROLL) {
        float s[DEC_KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        #pragma unroll
        for (int w = 1; w < DEC_KV_UNROLL; ++w) mb = ABL_FMAX(mb, s[w]);
        const float m_new = ABL_FMAX(mi, mb);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < n_kv; ++ki) {
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));

        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

#endif // ABLATION_VARIANT == 3

// =================================================================================================
// ABLATION VARIANT 4: + 1-Query per Wave Launcher
// - Prefill: 1 subgroup(64 lanes) = 1 (batch, head, query_row), barrier-free
// - Decode: 동일 (이미 1 query per wave)
// =================================================================================================
#if ABLATION_VARIANT == 4

#define DEC_KV_UNROLL 4
#define KV_UNROLL 4

__kernel void flash_attn_ablation_prefill(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    const int tid = get_local_id(0);
    const int bhq = get_global_id(1);
    const int bh = bhq / n_q;
    const int qr = bhq % n_q;

    const int bi = bh / n_head;
    const int hi = bh % n_head;
    const int hkv = hi / (n_head / n_head_kv);

    if (qr >= n_q) return;

    const int d = tid * 2;

    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_row = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2 + (ulong)qr * q_nb1);

#if DK == 128
    const float2 qv = (float2)(q_row[d], q_row[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_row[d] * scale; if (d + 1 < DK) qv.y = q_row[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2 + (ulong)qr * mask_nb1);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);
    const int kv_end = is_causal ? min(n_kv, n_kv - n_q + qr + 1) : n_kv;

    const int kv_batched = (kv_end / KV_UNROLL) * KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += KV_UNROLL) {
        float s[KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float m_new = ABL_FMAX(mi, s[0]);
        #pragma unroll
        for (int w = 1; w < KV_UNROLL; ++w) m_new = ABL_FMAX(m_new, s[w]);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < kv_end; ++ki) {
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));

        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)qr * o_nb2 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

__kernel void flash_attn_ablation_decode(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int bi = head_batch_idx / n_head;
    const int hi = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);
    const int d = tid * 2;

    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_ptr = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);
#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);

    const int kv_batched = (n_kv / DEC_KV_UNROLL) * DEC_KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += DEC_KV_UNROLL) {
        float s[DEC_KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        #pragma unroll
        for (int w = 1; w < DEC_KV_UNROLL; ++w) mb = ABL_FMAX(mb, s[w]);
        const float m_new = ABL_FMAX(mi, mb);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < n_kv; ++ki) {
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));

        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

#endif // ABLATION_VARIANT == 4

// =================================================================================================
// ABLATION VARIANT 5: + Native Math (fmax, native_exp)
// - 최종: 모든 최적화 + native_exp, fmax
// =================================================================================================
#if ABLATION_VARIANT == 5

#define DEC_KV_UNROLL 4
#define KV_UNROLL 4

__kernel void flash_attn_ablation_prefill(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    const int tid = get_local_id(0);
    const int bhq = get_global_id(1);
    const int bh = bhq / n_q;
    const int qr = bhq % n_q;

    const int bi = bh / n_head;
    const int hi = bh % n_head;
    const int hkv = hi / (n_head / n_head_kv);

    if (qr >= n_q) return;

    const int d = tid * 2;

    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_row = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2 + (ulong)qr * q_nb1);

#if DK == 128
    const float2 qv = (float2)(q_row[d], q_row[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_row[d] * scale; if (d + 1 < DK) qv.y = q_row[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2 + (ulong)qr * mask_nb1);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);
    const int kv_end = is_causal ? min(n_kv, n_kv - n_q + qr + 1) : n_kv;

    const int kv_batched = (kv_end / KV_UNROLL) * KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += KV_UNROLL) {
        float s[KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float m_new = ABL_FMAX(mi, s[0]);
        #pragma unroll
        for (int w = 1; w < KV_UNROLL; ++w) m_new = ABL_FMAX(m_new, s[w]);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < kv_end; ++ki) {
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));

        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)qr * o_nb2 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

__kernel void flash_attn_ablation_decode(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q, const int n_kv, const int is_causal, const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap, const int n_head_kv,
    const global void* mask_void, const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    const global void* sinks_void, const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t, const int k_nb1_t, const int k_nb2_t, const int k_nb3_t
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int bi = head_batch_idx / n_head;
    const int hi = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);
    const int d = tid * 2;

    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
    const global half* vr_base = (const global half*)((const global char*)v_void + v_offset + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1);

    const global float* q_ptr = (const global float*)((const global char*)q_void + q_offset + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);
#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    const global MASK_DATA_TYPE* mask_row = NULL;
    if (mask_void != NULL)
        mask_row = (const global MASK_DATA_TYPE*)((const global char*)mask_void + mask_offset + (ulong)(bi % mask_ne3) * mask_nb3 + (ulong)(hi % mask_ne2) * mask_nb2);

    const float slope = get_alibi_slope(max_bias, hi, n_head_log2, m0, m1);

    const int kv_batched = (n_kv / DEC_KV_UNROLL) * DEC_KV_UNROLL;
    int ki = 0;

    for (; ki < kv_batched; ki += DEC_KV_UNROLL) {
        float s[DEC_KV_UNROLL];
        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
        }

        if (mask_row) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] += slope * (float)mask_row[ki + w];
        }
        if (logit_softcap > 0.0f) {
            #pragma unroll
            for (int w = 0; w < DEC_KV_UNROLL; ++w) s[w] = logit_softcap * tanh(s[w] / logit_softcap);
        }

        float mb = s[0];
        #pragma unroll
        for (int w = 1; w < DEC_KV_UNROLL; ++w) mb = ABL_FMAX(mb, s[w]);
        const float m_new = ABL_FMAX(mi, mb);

        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
            const float p = ABL_EXP(s[w] - mi);
            li += p;
#if DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
    }

    for (; ki < n_kv; ++ki) {
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));

        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = ABL_FMAX(mi, score);
        if (m_new > mi) {
            const float alpha = ABL_EXP(mi - m_new);
            oacc *= alpha;
            li *= alpha;
            mi = m_new;
        }
        const float p = ABL_EXP(score - mi);
#if DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
        li += p;
    }

    if (sinks_void != NULL) {
        const global float* sp = (const global float*)((const global char*)sinks_void + sinks_offset);
        const float ms = sp[hi];
        const float mf = ABL_FMAX(mi, ms);
        const float so = (mi > -INFINITY) ? ABL_EXP(mi - mf) : 0.0f;
        oacc *= so;
        li = li * so + ABL_EXP(ms - mf);
    }

    oacc = (li > 0.0f) ? oacc * (1.0f / li) : (float2)(0.0f);

    global float* o_row = (global float*)((global char*)o_void + o_offset + (ulong)bi * o_nb3 + (ulong)hi * o_nb1);
#if DV == 128
    o_row[d] = oacc.x;
    o_row[d + 1] = oacc.y;
#else
    if (d < DV) { o_row[d] = oacc.x; if (d + 1 < DV) o_row[d + 1] = oacc.y; }
#endif
}

#endif // ABLATION_VARIANT == 5