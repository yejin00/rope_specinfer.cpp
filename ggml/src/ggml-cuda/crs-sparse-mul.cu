#include "crs-sparse-mul.cuh"
#include "convert.cuh"

#include <cstdlib>

static bool ggml_cuda_crs_use_index_path() {
    const char * env = getenv("LLAMA_CRS_INDEX_PATH");
    return env != nullptr && atoi(env) == 1;
}

template <typename src_t>
static __global__ void k_crs_sparse_mul_dense(const src_t * __restrict__ src0,
                                              const int32_t * __restrict__ src1,
                                              const float * __restrict__ src2,
                                              src_t * __restrict__ dst,
                                              const int64_t ne0,
                                              const int64_t ne1,
                                              const int64_t ne2,
                                              const int64_t ne3,
                                              const int64_t top_k,
                                              const int64_t s00,
                                              const int64_t s01,
                                              const int64_t s02,
                                              const int64_t s03,
                                              const int64_t s10,
                                              const int64_t s11,
                                              const int64_t s20,
                                              const int64_t s21,
                                              const int64_t sd0,
                                              const int64_t sd1,
                                              const int64_t sd2,
                                              const int64_t sd3) {
    const int64_t d  = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t h  = blockIdx.y;
    const int64_t it = blockIdx.z;

    if (d >= ne0 || h >= ne1 || it >= ne2 * ne3) {
        return;
    }

    const int64_t i2 = it % ne2;
    const int64_t i3 = it / ne2;

    float total_scale = 1.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t idx = *(src1 + k * s10 + h * s11);
        if (idx != d) {
            continue;
        }

        total_scale *= *(src2 + k * s20 + h * s21);
    }

    const src_t * src_ptr = src0 + d * s00 + h * s01 + i2 * s02 + i3 * s03;
    src_t * dst_ptr = dst + d * sd0 + h * sd1 + i2 * sd2 + i3 * sd3;
    *dst_ptr = ggml_cuda_cast<src_t>(ggml_cuda_cast<float>(*src_ptr) * total_scale);
}

template <typename src_t>
static __global__ void k_crs_sparse_mul_index(const src_t * __restrict__ src0,
                                              const int32_t * __restrict__ src1,
                                              const float * __restrict__ src2,
                                              src_t * __restrict__ dst,
                                              const int64_t ne0,
                                              const int64_t ne1,
                                              const int64_t ne2,
                                              const int64_t ne3,
                                              const int64_t top_k,
                                              const int64_t s00,
                                              const int64_t s01,
                                              const int64_t s02,
                                              const int64_t s03,
                                              const int64_t s10,
                                              const int64_t s11,
                                              const int64_t s20,
                                              const int64_t s21,
                                              const int64_t sd0,
                                              const int64_t sd1,
                                              const int64_t sd2,
                                              const int64_t sd3) {
    const int64_t k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t h  = blockIdx.y;
    const int64_t it = blockIdx.z;

    if (k >= top_k || h >= ne1 || it >= ne2 * ne3) {
        return;
    }

    const int64_t i2 = it % ne2;
    const int64_t i3 = it / ne2;

    const int32_t idx = *(src1 + k * s10 + h * s11);
    if (idx < 0 || idx >= ne0) {
        return;
    }

    const float scale = *(src2 + k * s20 + h * s21);
    const src_t * src_ptr = src0 + idx * s00 + h * s01 + i2 * s02 + i3 * s03;
    src_t * dst_ptr = dst + idx * sd0 + h * sd1 + i2 * sd2 + i3 * sd3;
    *dst_ptr = ggml_cuda_cast<src_t>(ggml_cuda_cast<float>(*src_ptr) * scale);
}

template <typename src_t>
static void crs_sparse_mul_cuda(const src_t * src0_d,
                                const int32_t * src1_d,
                                const float * src2_d,
                                src_t * dst_d,
                                const int64_t ne0,
                                const int64_t ne1,
                                const int64_t ne2,
                                const int64_t ne3,
                                const int64_t top_k,
                                const size_t nb00,
                                const size_t nb01,
                                const size_t nb02,
                                const size_t nb03,
                                const size_t nb10,
                                const size_t nb11,
                                const size_t nb20,
                                const size_t nb21,
                                const size_t nb0,
                                const size_t nb1,
                                const size_t nb2,
                                const size_t nb3,
                                cudaStream_t stream,
                                const bool use_index_path) {
    const int64_t s00 = nb00 / sizeof(src_t);
    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(int32_t);
    const int64_t s11 = nb11 / sizeof(int32_t);
    const int64_t s20 = nb20 / sizeof(float);
    const int64_t s21 = nb21 / sizeof(float);
    const int64_t sd0 = nb0 / sizeof(src_t);
    const int64_t sd1 = nb1 / sizeof(src_t);
    const int64_t sd2 = nb2 / sizeof(src_t);
    const int64_t sd3 = nb3 / sizeof(src_t);

    const int64_t it_total = ne2 * ne3;
    if (ne1 <= 0 || it_total <= 0) {
        return;
    }

    const dim3 block_size(256, 1, 1);
    if (use_index_path) {
        const dim3 grid_size((top_k + block_size.x - 1) / block_size.x, ne1, it_total);
        k_crs_sparse_mul_index<<<grid_size, block_size, 0, stream>>>(
                src0_d, src1_d, src2_d, dst_d,
                ne0, ne1, ne2, ne3, top_k,
                s00, s01, s02, s03,
                s10, s11,
                s20, s21,
                sd0, sd1, sd2, sd3);
    } else {
        const dim3 grid_size((ne0 + block_size.x - 1) / block_size.x, ne1, it_total);
        k_crs_sparse_mul_dense<<<grid_size, block_size, 0, stream>>>(
                src0_d, src1_d, src2_d, dst_d,
                ne0, ne1, ne2, ne3, top_k,
                s00, s01, s02, s03,
                s10, s11,
                s20, s21,
                sd0, sd1, sd2, sd3);
    }
}

void ggml_cuda_op_crs_sparse_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(src2 != nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == src0->type);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const int64_t top_k = src1->ne[0];

    const bool use_index_path = ggml_cuda_crs_use_index_path();
    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_F32) {
        crs_sparse_mul_cuda(
                static_cast<const float *>(src0->data),
                static_cast<const int32_t *>(src1->data),
                static_cast<const float *>(src2->data),
                static_cast<float *>(dst->data),
                ne0, ne1, ne2, ne3, top_k,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->nb[0], src1->nb[1],
                src2->nb[0], src2->nb[1],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                stream, use_index_path);
    } else {
        crs_sparse_mul_cuda(
                static_cast<const half *>(src0->data),
                static_cast<const int32_t *>(src1->data),
                static_cast<const float *>(src2->data),
                static_cast<half *>(dst->data),
                ne0, ne1, ne2, ne3, top_k,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->nb[0], src1->nb[1],
                src2->nb[0], src2->nb[1],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                stream, use_index_path);
    }
}
