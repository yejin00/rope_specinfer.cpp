// FlashAttention-2 OpenCL Kernel for llama.cpp (Adreno 750-oriented)
// - Prefill: Causal trimming (block-level + per-row), optional V caching
// - Decode(q1): subgroup reduce + dimension-parallel output (2 dims per thread)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half

#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

#ifndef DK
#define DK 128
#endif

#ifndef DV
#define DV 128
#endif

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)

#define WG_SIZE (BLOCK_M)
#define Q1_WG_SIZE 64

// 1이면 V도 local에 캐싱(글로벌 트래픽 감소, local 사용량 증가)
// Adreno에서 BLOCK_N이 크거나 BLOCK_M이 크면 occupancy가 떨어질 수 있음.
// 실측으로 0/1 둘 다 비교해봐.
#ifndef FA2_CACHE_V
#define FA2_CACHE_V 1
#endif

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) return 1.0f;
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
    return pow(base, exph);
}

// -----------------------------------------------------------------------------------------------
// Prefill kernel (keep host launch scheme: thread = one query row)
// Major win on causal self-attn: avoid loading tiles beyond what the block can ever use.
// -----------------------------------------------------------------------------------------------
__kernel void flash_attn_f32_f16_fa2(
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

    // mask pointer를 "row 기준"으로 한 번만 잡아둠
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

    // Q: register(private) 캐싱
    ACC_TYPE4 q_priv[DK_VEC];
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) q_priv[i] = (ACC_TYPE4)(0.0f);

    if (valid_query) {
        const ulong q_row_offset =
            (ulong)batch_idx    * q_nb3 +
            (ulong)head_idx     * q_nb2 +
            (ulong)my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    // O accumulator (register): 기존 구조 유지
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // local cache: K는 항상 캐시
    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];

#if FA2_CACHE_V
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];
#endif

    // base offsets (byte)
    const ulong k_base_offset = (ulong)batch_idx * k_nb3 + (ulong)head_kv_idx * k_nb2;
    const ulong v_base_offset = (ulong)batch_idx * v_nb3 + (ulong)head_kv_idx * v_nb2;

    // -----------------------------------------
    // 핵심: causal이면 "이 블록이 절대 사용하지 않을 KV tail"은 로드 자체를 하지 않음
    // -----------------------------------------
    int kv_max_block = n_kv;
    if (is_causal) {
        // 블록 내 최대 query row (inclusive)
        const int q_block_start = block_q_idx * BLOCK_M;
        const int q_block_end   = min(q_block_start + BLOCK_M, n_q);   // exclusive
        const int q_max         = q_block_end - 1;

        // causal_limit(q) = n_kv - n_q + q
        const int causal_limit_max = (n_kv - n_q + q_max);
        kv_max_block = clamp(causal_limit_max + 1, 0, n_kv); // exclusive upper bound
    }

    // Main KV loop
    for (int k_start = 0; k_start < kv_max_block; k_start += BLOCK_N) {
        const int k_tile_end  = min(k_start + BLOCK_N, kv_max_block);
        const int k_tile_size = k_tile_end - k_start;

        // ---- load K tile (cooperative) ----
        #pragma unroll 1
        for (int idx = tid; idx < k_tile_size * DK_VEC; idx += WG_SIZE) {
            const int row = idx / DK_VEC;
            const int col = idx % DK_VEC;
            const int k_row_idx = k_start + row;
            const ulong k_row_offset = k_base_offset + (ulong)k_row_idx * k_nb1;
            l_k[row][col] = ((const global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
        }

#if FA2_CACHE_V
        // ---- load V tile (cooperative) ----
        #pragma unroll 1
        for (int idx = tid; idx < k_tile_size * DV_VEC; idx += WG_SIZE) {
            const int row = idx / DV_VEC;
            const int col = idx % DV_VEC;
            const int v_row_idx = k_start + row;
            const ulong v_row_offset = v_base_offset + (ulong)v_row_idx * v_nb1;
            l_v[row][col] = ((const global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        if (valid_query) {
            // per-row causal limit: tile 내부에서도 j_end까지만 처리
            int j_end = k_tile_size;
            if (is_causal) {
                const int causal_limit = (n_kv - n_q + my_query_row);
                // tile 내에서 유효한 j 개수 = (causal_limit - k_start + 1)
                j_end = clamp(causal_limit - k_start + 1, 0, k_tile_size);
            }

            // 2개씩 처리(기존 스타일 유지) + 마지막 odd 처리
            int j = 0;
            for (; j + 1 < j_end; j += 2) {
                // dot(Q, K)
                ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);

                #pragma unroll
                for (int kk = 0; kk < DK_VEC; ++kk) {
                    dot_acc0 = mad(q_priv[kk], CONVERT_KV_ACC4(l_k[j][kk]),   dot_acc0);
                    dot_acc1 = mad(q_priv[kk], CONVERT_KV_ACC4(l_k[j+1][kk]), dot_acc1);
                }

                ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
                ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

                // mask/alibi
                if (mask_ptr != NULL) {
                    const int k0 = k_start + j;
                    const int k1 = k_start + j + 1;
                    score0 += slope * (ACC_TYPE)mask_ptr[k0];
                    score1 += slope * (ACC_TYPE)mask_ptr[k1];
                }

                // softcap
                if (logit_softcap > 0.0f) {
                    score0 = logit_softcap * tanh(score0 / logit_softcap);
                    score1 = logit_softcap * tanh(score1 / logit_softcap);
                }

                // online softmax update
                const ACC_TYPE m_new = fmax(m_i, fmax(score0, score1));
                const ACC_TYPE scale_prev = exp(m_i - m_new);

                const ACC_TYPE p0 = exp(score0 - m_new);
                const ACC_TYPE p1 = exp(score1 - m_new);

                #pragma unroll
                for (int ii = 0; ii < DV_VEC; ++ii) {
#if FA2_CACHE_V
                    const ACC_TYPE4 v0 = CONVERT_KV_ACC4(l_v[j][ii]);
                    const ACC_TYPE4 v1 = CONVERT_KV_ACC4(l_v[j+1][ii]);
#else
                    const int k0 = k_start + j;
                    const int k1 = k_start + j + 1;
                    const global KV_DATA_TYPE4* v_ptr0 =
                        (const global KV_DATA_TYPE4*)(v_base + v_base_offset + (ulong)k0 * v_nb1);
                    const global KV_DATA_TYPE4* v_ptr1 =
                        (const global KV_DATA_TYPE4*)(v_base + v_base_offset + (ulong)k1 * v_nb1);
                    const ACC_TYPE4 v0 = CONVERT_KV_ACC4(v_ptr0[ii]);
                    const ACC_TYPE4 v1 = CONVERT_KV_ACC4(v_ptr1[ii]);
#endif
                    o_acc[ii] = o_acc[ii] * scale_prev + (ACC_TYPE4)(p0) * v0 + (ACC_TYPE4)(p1) * v1;
                }

                l_i = l_i * scale_prev + p0 + p1;
                m_i = m_new;
            }

            // odd 마지막 1개
            if (j < j_end) {
                ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int kk = 0; kk < DK_VEC; ++kk) {
                    dot_acc = mad(q_priv[kk], CONVERT_KV_ACC4(l_k[j][kk]), dot_acc);
                }
                ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;

                if (mask_ptr != NULL) {
                    const int k0 = k_start + j;
                    score += slope * (ACC_TYPE)mask_ptr[k0];
                }
                if (logit_softcap > 0.0f) {
                    score = logit_softcap * tanh(score / logit_softcap);
                }

                const ACC_TYPE m_new = fmax(m_i, score);
                const ACC_TYPE scale_prev = exp(m_i - m_new);
                const ACC_TYPE p = exp(score - m_new);

                #pragma unroll
                for (int ii = 0; ii < DV_VEC; ++ii) {
#if FA2_CACHE_V
                    const ACC_TYPE4 v0 = CONVERT_KV_ACC4(l_v[j][ii]);
#else
                    const int k0 = k_start + j;
                    const global KV_DATA_TYPE4* v_ptr0 =
                        (const global KV_DATA_TYPE4*)(v_base + v_base_offset + (ulong)k0 * v_nb1);
                    const ACC_TYPE4 v0 = CONVERT_KV_ACC4(v_ptr0[ii]);
#endif
                    o_acc[ii] = o_acc[ii] * scale_prev + (ACC_TYPE4)(p) * v0;
                }

                l_i = l_i * scale_prev + p;
                m_i = m_new;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write out
    if (valid_query) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr =
                (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink  = sinks_ptr[head_idx];
            const ACC_TYPE m_final = fmax(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int ii = 0; ii < DV_VEC; ++ii) o_acc[ii] *= scale_o;

            l_i = l_i * scale_o + exp(m_sink - m_final);
            m_i = m_final;
        }

        const ulong o_row_offset =
            (ulong)batch_idx    * o_nb3 +
            (ulong)my_query_row * o_nb2 +
            (ulong)head_idx     * o_nb1;

        global O_DATA_TYPE4* o_row = (global O_DATA_TYPE4*)(o_base + o_row_offset);

        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int ii = 0; ii < DV_VEC; ++ii) o_row[ii] = CONVERT_O_DATA4(o_acc[ii] * l_inv);
        } else {
            #pragma unroll
            for (int ii = 0; ii < DV_VEC; ++ii) o_row[ii] = (O_DATA_TYPE4)(0.0f);
        }
    }
}

// -----------------------------------------------------------------------------------------------
// Decode(q1) kernel: Adreno 750-friendly (subgroup reduce + dim-parallel output)
// local_size should be 64 on Adreno (wave64).
// -----------------------------------------------------------------------------------------------
__kernel void flash_attn_f32_f16_fa2_q1(
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
    // dim-parallel: 64 threads, each thread handles 2 output dims (total 128 dims)
    const int tid = get_local_id(0); // 0..63 expected
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    // local Q (half) with padding to reduce bank conflict
    __local half l_q[DK + 32];

    const global char* q_base = (const global char*)q_void + q_offset;
    const ulong q_row_offset = (ulong)batch_idx * q_nb3 + (ulong)head_idx * q_nb2;
    const global float* q_ptr = (const global float*)(q_base + q_row_offset);

    // cooperative load Q: store as half to reduce local footprint
    for (int i = tid; i < DK; i += Q1_WG_SIZE) {
        l_q[i] = (half)q_ptr[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;

    const ulong k_base_offset = (ulong)batch_idx * k_nb3 + (ulong)head_kv_idx * k_nb2;
    const ulong v_base_offset = (ulong)batch_idx * v_nb3 + (ulong)head_kv_idx * v_nb2;

    // mask base (q1은 row offset이 없음)
    const global half* mask_ptr = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char* mask_base =
            (const global char*)mask_void + mask_offset
            + (ulong)mask_batch_idx * mask_nb3
            + (ulong)mask_head_idx  * mask_nb2;
        mask_ptr = (const global half*)mask_base;
    }

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global float* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global float*)((const global char*)sinks_void + sinks_offset);
    }

    // online softmax state (each thread has same m,l logically; but we update using identical score)
    float m_local = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    float l_local = 0.0f;

    const int my_dim_base = tid * 2; // 0..126

    float2 my_o_acc = (float2)(0.0f, 0.0f);

    // iterate KV tokens
    for (int k_idx = 0; k_idx < n_kv; ++k_idx) {
        // causal check (decode에서는 거의 항상 true지만, 일반성을 위해 유지)
        if (is_causal) {
            // q=0일 때 causal_limit = n_kv - n_q + 0
            const int causal_limit = n_kv - n_q;
            if (k_idx > causal_limit) break;
        }

        // partial dot for my 2 dims
        float my_part = 0.0f;
        if (my_dim_base < DK) {
            // Q local (half2) -> float2
            half2 q_h2 = *(__local half2*)&l_q[my_dim_base];
            float2 q_f2 = convert_float2(q_h2);

            // K global (half2) -> float2
            const ulong k_row_offset = k_base_offset + (ulong)k_idx * k_nb1;
            const global half* k_ptr_half = (const global half*)(k_base + k_row_offset);
            float2 k_f2 = vload_half2(0, k_ptr_half + my_dim_base);

            my_part = dot(q_f2, k_f2);
        }

        // subgroup reduce sum to get full score
        float score = sub_group_reduce_add(my_part);
        score *= scale;

        if (mask_ptr != NULL) {
            score += slope * (float)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // online softmax update
        float m_prev = m_local;
        m_local = fmax(m_prev, score);

        float scale_prev = (m_prev > -INFINITY) ? exp(m_prev - m_local) : 0.0f;
        float p = exp(score - m_local);

        l_local = l_local * scale_prev + p;

        // accumulate V for my 2 dims
        if (my_dim_base < DK) {
            const ulong v_row_offset = v_base_offset + (ulong)k_idx * v_nb1;
            const global half* v_ptr_half = (const global half*)(v_base + v_row_offset);
            float2 v_f2 = vload_half2(0, v_ptr_half + my_dim_base);

            my_o_acc = mad((float2)(p), v_f2, my_o_acc * scale_prev);
        }
    }

    // finalize normalize
    if (l_local > 0.0f) {
        my_o_acc *= (1.0f / l_local);
    } else {
        my_o_acc = (float2)(0.0f, 0.0f);
    }

    // write out (each thread writes its 2 dims)
    global char* o_base = (global char*)o_void + o_offset;
    const ulong o_row_offset = (ulong)batch_idx * o_nb3 + (ulong)head_idx * o_nb1;
    global float* o_ptr = (global float*)(o_base + o_row_offset);

    if (my_dim_base < DK) {
        o_ptr[my_dim_base] = my_o_acc.x;
        if (my_dim_base + 1 < DK) o_ptr[my_dim_base + 1] = my_o_acc.y;
    }
}