#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SHUFFLE 1
#else
#define HAS_SHUFFLE 0
#endif

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

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

// 매크로가 정의되어 있지 않을 경우를 대비한 안전장치 (보통 컴파일 옵션으로 넘어옴)
#ifndef DK
#define DK 128
#endif

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)

// Decoding Kernel을 위한 전용 Wave Size (Adreno 최적화: 64)
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
// Optimized Prefill Kernel - Memory Efficient Version
// =================================================================================================
// 최적화 기법:
// 1. K만 local memory 캐싱 - local memory 사용량 50% 감소
// 2. V는 global memory 직접 접근 - compute-while-load 패턴
// 3. Q를 float으로 유지 - 불필요한 변환 제거
// 4. Adaptive unrolling - BLOCK_N 크기에 따라 최적화
// 5. Simplified indexing - division/modulo 연산 최소화
// 6. Better memory coalescing - 연속 메모리 접근 패턴
// 7. Reduced barriers - 동기화 오버헤드 최소화
// =================================================================================================
__kernel void flash_attn_f32_f16_old(
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

    // Mask pointer 사전 계산
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

    // Q를 float precision으로 레지스터에 캐시
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

    // Output accumulator
    float4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (float4)(0.0f);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    const int causal_limit = is_causal ? (n_kv - n_q + my_query_row) : n_kv;

    // K만 local memory에 캐싱 (메모리 사용량 50% 감소)
    __local half4 l_k[BLOCK_N][DK_VEC];

    // Precompute base offsets
    const ulong v_base_offset = (ulong)batch_idx * v_nb3 + (ulong)head_kv_idx * v_nb2;
    const ulong k_base_offset = (ulong)batch_idx * k_nb3 + (ulong)head_kv_idx * k_nb2;

    // Main KV loop
    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        const int k_tile_end = min(k_start + BLOCK_N, n_kv);
        const int k_tile_size = k_tile_end - k_start;
        
        // K 타일을 협력적으로 로드 (coalesced access)
        #pragma unroll 1
        for (int idx = tid; idx < BLOCK_N * DK_VEC; idx += WG_SIZE) {
            const int row = idx / DK_VEC;
            const int col = idx % DK_VEC;
            
            half4 k_val = (half4)(0.0h);
            if (row < k_tile_size) {
                const global half4* k_row_ptr = 
                    (const global half4*)(k_base + k_base_offset + (k_start + row) * k_nb1);
                k_val = k_row_ptr[col];
            }
            l_k[row][col] = k_val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (valid_query) {
            // Process KV tokens - adaptive unrolling based on BLOCK_N
            #if BLOCK_N >= 32
                #define UNROLL_FACTOR 4
            #elif BLOCK_N >= 16  
                #define UNROLL_FACTOR 4
            #else
                #define UNROLL_FACTOR 2
            #endif
            
            #pragma unroll 1
            for (int j = 0; j < k_tile_size; j += UNROLL_FACTOR) {
                // decode와 달리 prefill은 causal 삼각형 구조:
                // 이 query row에 대해 더 이상 유효한 k_row가 없으면 조기 종료
                if (is_causal && (k_start + j) > causal_limit) {
                    break;
                }

                // Compute attention scores
                float scores[UNROLL_FACTOR];
                
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    const int k_row = k_start + j + w;

                    // 이 위치가 사각형 바깥(=삼각형 마스크 영역)이면 dot 자체를 건너뜀
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

                // Apply masking and bias (여기서는 causal 제외한 mask/alibi/softcap만 적용)
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    const int k_row = k_start + j + w;
                    
                    if (scores[w] == -INFINITY) {
                        continue;
                    }

                    if (mask_ptr != NULL) {
                        scores[w] += slope * (float)mask_ptr[k_row];
                    }
                    if (logit_softcap > 0.0f) {
                        scores[w] = logit_softcap * tanh(scores[w] / logit_softcap);
                    }
                }

                // Online softmax update
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

                // V accumulation - direct global memory access
                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    float4 v_acc = (float4)(0.0f);
                    
                    #pragma unroll
                    for (int w = 0; w < UNROLL_FACTOR; ++w) {
                        const int k_row = k_start + j + w;
                        if (k_row < n_kv && p[w] > 0.0f) {
                            const global half4* v_ptr = 
                                (const global half4*)(v_base + v_base_offset + k_row * v_nb1);
                            v_acc = mad((float4)(p[w]), convert_float4(v_ptr[i]), v_acc);
                        }
                    }
                    
                    o_acc[i] = mad(v_acc, (float4)(1.0f), o_acc[i] * scale_prev);
                }

                // Update normalizer
                l_i = l_i * scale_prev + p_sum;
                m_i = m_new;
            }
            
            #undef UNROLL_FACTOR
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Output write
    if (valid_query && l_i > 0.0f) {
        // Sink 처리
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

        // Normalize & write
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
        // Handle zero attention case
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
//
// Prefill 커널과 동일한 최적화 패턴 적용:
//   - K/V base 포인터 사전 계산
//   - Q에 scale 사전 곱셈
//   - DK=128 경로: bounds-check 전부 제거
//   - KV_UNROLL=4 배치: rescale 횟수 75% 감소, exp 37.5% 감소
//   - 조건부 rescale: max가 변할 때만 exp(alpha) 계산
// =================================================================================================
#ifdef ADRENO_GPU
#define DEC_KV_UNROLL 8
#else
#define DEC_KV_UNROLL 4
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void flash_attn_f32_f16_q1(
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
    const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t,
    const int k_nb1_t,
    const int k_nb2_t,
    const int k_nb3_t
) {
    const int tid = get_local_id(0); // 0..63
    const int head_batch_idx = get_global_id(1);
    const int bi  = head_batch_idx / n_head;
    const int hi  = head_batch_idx % n_head;
    const int hkv = hi / (n_head / n_head_kv);

    const int d = tid * 2;

    // ===== Base pointers =====
#ifdef ADRENO_GPU
    // K → texture L1, V → buffer L2 (cache separation)
    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
#else
    const global char* k_ptr_base = (const global char*)k_void + k_offset
                                  + (ulong)bi * k_nb3 + (ulong)hkv * k_nb2;
#endif
    // V → 32-bit stride (avoid expensive ulong multiply per V read)
    const global half* vr_base = (const global half*)(
        (const global char*)v_void + v_offset
        + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1); // stride in halves (32-bit)

    // ===== Q → register (scale 사전 곱셈) =====
    const global float* q_ptr = (const global float*)(
        (const global char*)q_void + q_offset
        + (ulong)bi * q_nb3 + (ulong)hi * q_nb2);

#if DK == 128
    const float2 qv = (float2)(q_ptr[d], q_ptr[d + 1]) * scale;
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_ptr[d] * scale; if (d + 1 < DK) qv.y = q_ptr[d + 1] * scale; }
#endif

    // ===== State =====
    float2 oacc = (float2)(0.0f);
    float mi = -INFINITY;
    float li = 0.0f;

    // ===== Mask (decode: Q=1이므로 query_row 차원 없음) =====
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

#ifdef ADRENO_GPU
        prefetch(vr_base + ki * v_stride_h, 8);
#endif

        #pragma unroll
        for (int w = 0; w < DEC_KV_UNROLL; ++w) {
#ifdef ADRENO_GPU
            const float4 k4 = read_imagef(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const float2 kv = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add(dot(qv, kv));
#elif DK == 128
            const global half* kr = (const global half*)(k_ptr_base + (ulong)(ki + w) * k_nb1);
            s[w] = sub_group_reduce_add(dot(qv, vload_half2(0, kr + d)));
#else
            const global half* kr = (const global half*)(k_ptr_base + (ulong)(ki + w) * k_nb1);
            float partial = 0.0f;
            if (d < DK) { partial = dot(qv, vload_half2(0, kr + d)); }
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
        #pragma unroll
        for (int w = 1; w < DEC_KV_UNROLL; ++w) mb = fmax(mb, s[w]);
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
#ifdef ADRENO_GPU
            // Coalesced vload_half4: 2 lanes share 1 load (half4 = 8 bytes)
            const global half* vp = vr_base + (ki + w) * v_stride_h + (tid >> 1) * 4;
            float4 v4 = vload_half4(0, vp);
            float2 vv = (tid & 1) ? v4.zw : v4.xy;
            oacc = mad((float2)(p), vv, oacc);
#else
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
#else
            if (d < DV) { oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc); }
#endif
        }
#ifdef ADRENO_GPU
        if (ki + DEC_KV_UNROLL < kv_batched)
            prefetch(vr_base + (ki + DEC_KV_UNROLL) * v_stride_h, 8);
#endif
    }

    // ===== Tail =====
    for (; ki < n_kv; ++ki) {
#ifdef ADRENO_GPU
        const float4 k4 = read_imagef(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const float2 kv = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add(dot(qv, kv));
#elif DK == 128
        const global half* kr = (const global half*)(k_ptr_base + (ulong)ki * k_nb1);
        float score = sub_group_reduce_add(dot(qv, vload_half2(0, kr + d)));
#else
        const global half* kr = (const global half*)(k_ptr_base + (ulong)ki * k_nb1);
        float partial = 0.0f;
        if (d < DK) { partial = dot(qv, vload_half2(0, kr + d)); }
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
#ifdef ADRENO_GPU
        const global half* vp = vr_base + ki * v_stride_h + (tid >> 1) * 4;
        float4 v4 = vload_half4(0, vp);
        float2 vv = (tid & 1) ? v4.zw : v4.xy;
        oacc = mad((float2)(p), vv, oacc);
#else
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#endif
#else
        if (d < DV) { oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc); }
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
// 3. Mobile-optimized Prefill Kernel (single-Q, barrier-free, Adreno wave=64)
//    - 1 subgroup(64 lanes) = 1 (batch, head, query_row)
//    - lane 당 dv의 2차원(d0, d1)을 전담 (DV=128 기준: 64 lanes * 2 = 128)
//    - online softmax, barrier-free, sub_group_reduce_add만 사용
//    - Optional: local K 타일링으로 K 재사용 (K_TILE=8, DK=128 기준 2KB)
// =================================================================================================
#define K_TILE   8
#define DK_HALF2 (DK/2)
__kernel void flash_attn_f32_f16_k8(
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
    const int tid = get_local_id(0); // 0..63, subgroup lane

    // gid1: (batch, head, query_row) 인덱싱
    const int bhq_idx = get_global_id(1);
    const int head_batch_idx = bhq_idx / n_q;
    const int my_query_row   = bhq_idx % n_q;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    if (my_query_row >= n_q) {
        return;
    }

    // lane별 dv 차원 2개 전담
    const int my_dim_base = tid * 2;

    // [1] Q 로드 (float precision, register-resident)
    const global char* q_base = (const global char*)q_void + q_offset;
    const ulong q_row_offset =
        (ulong) batch_idx    * q_nb3 +
        (ulong) head_idx     * q_nb2 +
        (ulong) my_query_row * q_nb1;
    const global float* q_ptr = (const global float*)(q_base + q_row_offset);

    float2 q_val = (float2)(0.0f, 0.0f);
    if (my_dim_base < DK) {
        q_val.x = q_ptr[my_dim_base];
        if (my_dim_base + 1 < DK) {
            q_val.y = q_ptr[my_dim_base + 1];
        }
    }

    // [2] 출력 accumulator / softmax 상태
    float2 my_o_acc = (float2)(0.0f, 0.0f);

    float m_local = -INFINITY;
    float l_local = 0.0f;

    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;

    // Prefill(Q>1)에서는 mask가 (batch, head, query, k_idx) 형태라고 가정
    const global MASK_DATA_TYPE* mask_ptr = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char* mask_base =
            (const global char*)mask_void +
            mask_offset +
            (ulong) mask_batch_idx * mask_nb3 +
            (ulong) mask_head_idx  * mask_nb2 +
            (ulong) my_query_row   * mask_nb1;
        mask_ptr = (const global MASK_DATA_TYPE*) mask_base;
    }

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // 이 query row에 대한 causal limit (prefill 시 삼각형 구조)
    const int causal_limit = is_causal ? (n_kv - n_q + my_query_row) : n_kv - 1;

    // [3] Main KV loop (1-pass, barrier-free, K 타일링 적용)
    __local float2 l_k[K_TILE][DK_HALF2];

    for (int k_start = 0; k_start < n_kv; k_start += K_TILE) {
        const int k_tile_end  = min(k_start + K_TILE, n_kv);
        const int k_tile_size = k_tile_end - k_start;

        // 이 타일 전체가 causal limit 밖이면 조기 종료
        if (is_causal && k_start > causal_limit) {
            break;
        }

        // 3-1. K 타일을 local로 로드
        const int lane_stride = 64;
        for (int idx = tid; idx < k_tile_size * DK_HALF2; idx += lane_stride) {
            const int row = idx / DK_HALF2;
            const int col = idx % DK_HALF2;

            const int k_idx = k_start + row;
            const ulong k_row_offset =
                (ulong) batch_idx   * k_nb3 +
                (ulong) head_kv_idx * k_nb2 +
                (ulong) k_idx       * k_nb1;
            const global half* k_ptr_half = (const global half*)(k_base + k_row_offset);

            const int dim0 = col * 2;
            float2 kv = (float2)(0.0f, 0.0f);
            if (dim0 + 1 < DK) {
                kv = vload_half2(0, k_ptr_half + dim0);
            }
            l_k[row][col] = kv;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3-2. 타일 내부 k 처리
        for (int t = 0; t < k_tile_size; ++t) {
            const int k_idx = k_start + t;

            if (is_causal && k_idx > causal_limit) {
                // 이후 타일에서도 더 이상 유효한 k 없음
                k_start = n_kv;
                break;
            }

            // A. partial dot product (local K 사용)
            float2 k_val = (float2)(0.0f, 0.0f);
            if (my_dim_base < DK) {
                const int col = my_dim_base / 2;
                k_val = l_k[t][col];
            }

            float my_score_part = 0.0f;
            if (my_dim_base < DK) {
                my_score_part = dot(q_val, k_val);
            }

            float score = sub_group_reduce_add(my_score_part);

            // B. mask / alibi / softcap
            score *= scale;
            if (mask_ptr != NULL) {
                score += slope * (float) mask_ptr[k_idx];
            }
            if (logit_softcap > 0.0f) {
                score = logit_softcap * tanh(score / logit_softcap);
            }

            // C. online softmax
            const float m_prev = m_local;
            m_local = fmax(m_prev, score);

            float p = 0.0f;
            float scale_prev = 1.0f;
            if (m_local > -INFINITY) {
                p = exp(score - m_local);
                scale_prev = (m_prev > -INFINITY) ? exp(m_prev - m_local) : 0.0f;
            }

            l_local = l_local * scale_prev + p;

            // D. V accumulation (dimension-parallel)
            const ulong v_row_offset =
                (ulong) batch_idx   * v_nb3 +
                (ulong) head_kv_idx * v_nb2 +
                (ulong) k_idx       * v_nb1;
            const global half* v_ptr_half = (const global half*)(v_base + v_row_offset);

            if (my_dim_base < DV) {
                float2 v_val = vload_half2(0, v_ptr_half + my_dim_base);
                my_o_acc = mad((float2)(p), v_val, my_o_acc * scale_prev);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // [4] Sink 처리 + 정규화 + 출력
    const global float* sinks_ptr =
        sinks_void != NULL ? (const global float*)((const global char*)sinks_void + sinks_offset) : NULL;

    float m_final = m_local;
    float l_final = l_local;

    if (sinks_ptr != NULL) {
        const float m_sink  = sinks_ptr[head_idx];
        m_final = fmax(m_local, m_sink);

        const float scale_o      = (m_local > -INFINITY) ? exp(m_local - m_final) : 0.0f;
        const float sink_contrib = exp(m_sink - m_final);

        my_o_acc *= scale_o;
        l_final   = l_local * scale_o + sink_contrib;
    }

    if (l_final > 0.0f) {
        const float l_inv = 1.0f / l_final;
        my_o_acc *= l_inv;
    } else {
        my_o_acc = (float2)(0.0f, 0.0f);
    }

    // 출력 쓰기
    const global char* o_base = (const global char*)o_void + o_offset;
    const ulong o_row_offset =
        (ulong) batch_idx    * o_nb3 +
        (ulong) my_query_row * o_nb2 +
        (ulong) head_idx     * o_nb1;
    global float* o_ptr = (global float*)(o_base + o_row_offset);

    if (my_dim_base < DV) {
        o_ptr[my_dim_base] = my_o_acc.x;
        if (my_dim_base + 1 < DV) {
            o_ptr[my_dim_base + 1] = my_o_acc.y;
        }
    }
}

// =================================================================================================
// 3. Prefill Kernel — Single-Q, Barrier-Free (Adreno 750, wave=64)
//
// 최적 아키텍처: 1 subgroup(64 lanes) = 1 (batch, head, query_row)
//   - Barrier 0회 (subgroup intrinsic만 사용)
//   - LDS 0 bytes (Q register, K/V global → L2 cache)
//   - Adreno: Load half4(넓게), Compute half2(잘게): 64 lanes 전부 활성, V half4 load 2 lanes 공유
//   - Q pre-scaled, DK=128 branch elimination, conditional rescale
// =================================================================================================
#ifdef ADRENO_GPU
#define KV_UNROLL 4
#else
#define KV_UNROLL 4
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void flash_attn_f32_f16(
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
    const ulong sinks_offset,
    __read_only image1d_buffer_t k_img,
    const int k_offset_t,
    const int k_nb1_t,
    const int k_nb2_t,
    const int k_nb3_t
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
#ifdef ADRENO_GPU
    // K → texture L1, V → buffer L2 (cache separation)
    const int k_base_t = k_offset_t + bi * k_nb3_t + hkv * k_nb2_t;
#else
    const global char* k_ptr_base = (const global char*)k_void + k_offset
                                  + (ulong)bi * k_nb3 + (ulong)hkv * k_nb2;
#endif
    // V → 32-bit stride (avoid expensive ulong multiply per V read)
    const global half* vr_base = (const global half*)(
        (const global char*)v_void + v_offset
        + (ulong)bi * v_nb3 + (ulong)hkv * v_nb2);
    const int v_stride_h = (int)(v_nb1 >> 1); // stride in halves (32-bit)

    // ===== Q → register (pre-scaled) =====
    const global float* q_row = (const global float*)(
        (const global char*)q_void + q_offset
        + (ulong)bi * q_nb3 + (ulong)hi * q_nb2 + (ulong)qr * q_nb1);

#if DK == 128
    const float2 qv = (float2)(q_row[d], q_row[d + 1]) * scale;
#if defined(ADRENO_GPU)
    const half2 qv_h2 = convert_half2(qv);
#endif
#else
    float2 qv = (float2)(0.0f);
    if (d < DK) { qv.x = q_row[d] * scale; if (d + 1 < DK) qv.y = q_row[d + 1] * scale; }
#endif

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

#ifdef ADRENO_GPU
        prefetch(vr_base + ki * v_stride_h, 8);
        if (ki + KV_UNROLL * 2 < kv_batched)
            prefetch(vr_base + (ki + KV_UNROLL * 2) * v_stride_h, 8);
#endif

        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
#if defined(ADRENO_GPU) && DK == 128
            const half4 k4 = read_imageh(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const half2 k2 = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add((float)dot(qv_h2, k2));
#elif defined(ADRENO_GPU)
            const half4 k4 = read_imageh(k_img, k_base_t + (ki + w) * k_nb1_t + (tid >> 1));
            const half2 k2 = (tid & 1) ? k4.zw : k4.xy;
            s[w] = sub_group_reduce_add((float)dot(convert_half2(qv), k2));
#elif DK == 128
            const global half* kr = (const global half*)(k_ptr_base + (ulong)(ki + w) * k_nb1);
            s[w] = sub_group_reduce_add(dot(qv, vload_half2(0, kr + d)));
#else
            const global half* kr = (const global half*)(k_ptr_base + (ulong)(ki + w) * k_nb1);
            float partial = 0.0f;
            if (d < DK) { partial = dot(qv, vload_half2(0, kr + d)); }
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

        float m_new = fmax(mi, s[0]);
        #pragma unroll
        for (int w = 1; w < KV_UNROLL; ++w) m_new = fmax(m_new, s[w]);

        if (m_new > mi) {
            const float alpha = native_exp(mi - m_new);
            oacc *= alpha;
            li *= alpha; mi = m_new;
        }

        #pragma unroll
        for (int w = 0; w < KV_UNROLL; ++w) {
            const float p = native_exp(s[w] - mi);
            li += p;
#if defined(ADRENO_GPU) && DV == 128
            const half2 v_h2 = convert_half2(vload_half2(0, vr_base + (ki + w) * v_stride_h + d));
            oacc = mad((float2)(p), convert_float2(v_h2), oacc);
#elif DV == 128
            oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#else
            if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + (ki + w) * v_stride_h + d), oacc);
#endif
        }
#ifdef ADRENO_GPU
        if (ki + KV_UNROLL < kv_batched)
            prefetch(vr_base + (ki + KV_UNROLL) * v_stride_h, 8);
#endif
    }

    // ===== Tail =====
    for (; ki < kv_end; ++ki) {
#if defined(ADRENO_GPU) && DK == 128
        const half4 k4 = read_imageh(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const half2 k2 = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add((float)dot(qv_h2, k2));
#elif defined(ADRENO_GPU)
        const half4 k4 = read_imageh(k_img, k_base_t + ki * k_nb1_t + (tid >> 1));
        const half2 k2 = (tid & 1) ? k4.zw : k4.xy;
        float score = sub_group_reduce_add((float)dot(convert_half2(qv), k2));
#elif DK == 128
        const global half* kr = (const global half*)(k_ptr_base + (ulong)ki * k_nb1);
        float score = sub_group_reduce_add(dot(qv, vload_half2(0, kr + d)));
#else
        const global half* kr = (const global half*)(k_ptr_base + (ulong)ki * k_nb1);
        float partial = 0.0f;
        if (d < DK) { partial = dot(qv, vload_half2(0, kr + d)); }
        float score = sub_group_reduce_add(partial);
#endif
        if (mask_row) score += slope * (float)mask_row[ki];
        if (logit_softcap > 0.0f) score = logit_softcap * tanh(score / logit_softcap);

        const float m_new = fmax(mi, score);
        if (m_new > mi) {
            const float alpha = native_exp(mi - m_new);
            oacc *= alpha;
            li *= alpha; mi = m_new;
        }
        const float p = native_exp(score - mi);

#if defined(ADRENO_GPU) && DV == 128
        const half2 v_h2 = convert_half2(vload_half2(0, vr_base + ki * v_stride_h + d));
        oacc = mad((float2)(p), convert_float2(v_h2), oacc);
#elif DV == 128
        oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
#else
        if (d < DV) oacc = mad((float2)(p), vload_half2(0, vr_base + ki * v_stride_h + d), oacc);
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